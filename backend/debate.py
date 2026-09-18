"""Petla debaty: rundy agentow, sedzia, uzupelnianie zrodel po lukach.

Po kazdej rundzie zbieramy twierdzenia typu information_gap (agent zglosil, ze
brakuje danej) i probujemy dociagnac zrodla, ktore te luki zasypia. Dociagniete
dokumenty wchodza do WSPOLNEGO pakietu zrodel, wiec w nastepnej rundzie widza je
wszyscy agenci.

discover_urls przyjmuje dowolna liczbe "odkrywaczy" - funkcji
(gaps, asset, asset_kind) -> URL-e. Na start jest jeden: EDGAR po tickerze,
dzialajacy tylko dla akcji. Dolozenie API wyszukiwania to napisanie jednej
takiej funkcji i wrzucenie jej na liste.
"""

import json
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4
from typing import Callable, Optional

import httpx

from agents import run_bear, run_bull, run_quant
from config import CREDIBILITY_BUDGET, DEFAULT_TIERS, ROUNDS
from defillama import fetch_chain_doc, fetch_fees_doc, fetch_protocol_doc
from judge import answer_question, apply_rulings, final_verdict, judge_round
from models import AgentState, AssetKind, Claim, DebateState, Verdict
from sources import (
    CACHE_DIR,
    SourceDoc,
    allowlist_for,
    asset_reference_doc,
    fetch_sources,
    user_agent_for,
)

# Odkrywacz zrodel: z luk, tickera i rodzaju aktywa robi liste URL-i do pobrania.
Discoverer = Callable[[list[Claim], str, AssetKind], list[str]]

# Odbiorca zdarzen debaty. Wolany W TRAKCIE, nie na koncu - arena ma sie
# zapelniac na oczach widza (patrz api.py).
EventSink = Callable[[dict], None]

ASSET_FILE = Path(__file__).resolve().parent.parent / "data" / "assets.json"

_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_TICKERS_CACHE = CACHE_DIR / "edgar_company_tickers.json"
_EDGAR_FORMS = ("10-K", "10-Q", "8-K")
_EDGAR_MAX_FILINGS = 3
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def asset_meta(ticker: str) -> dict:
    """Wpis z data/assets.json dla tickera. Brak pliku albo wpisu -> pusty dict.

    Ten sam plik karmi picker i panel aktywa we frontendzie (GET /assets), wiec
    slug DefiLlama i seed_urls wpisuje sie w jednym miejscu.
    """
    try:
        data = json.loads(ASSET_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        print(f"[assets] nie moge odczytac {ASSET_FILE.name} ({type(err).__name__})")
        return {}
    return data.get(ticker.strip().upper(), {})


def _sec_get(url: str) -> Optional[dict]:
    """GET na SEC z wymaganym przez nich UA. None przy dowolnym bledzie."""
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": user_agent_for(url), "Accept": "application/json"},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as err:
        print(f"[discover] EDGAR nie odpowiedzial ({type(err).__name__}): {url}")
        return None


def _cik_for_ticker(ticker: str) -> Optional[str]:
    """Mapuje ticker na 10-cyfrowy CIK. Mapa tickerow leci do cache na dysku."""
    mapping = None
    if _EDGAR_TICKERS_CACHE.exists():
        try:
            mapping = json.loads(_EDGAR_TICKERS_CACHE.read_text(encoding="utf-8"))
        except ValueError:
            mapping = None

    if mapping is None:
        mapping = _sec_get(_EDGAR_TICKERS_URL)
        if mapping is None:
            return None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _EDGAR_TICKERS_CACHE.write_text(json.dumps(mapping), encoding="utf-8")

    wanted = ticker.strip().upper()
    for entry in mapping.values():
        if entry.get("ticker", "").upper() == wanted:
            return str(entry["cik_str"]).zfill(10)

    print(f"[discover] {wanted} nie ma w mapie tickerow EDGAR (nie spolka z USA?)")
    return None


def edgar_filings(gaps: list[Claim], asset: str, asset_kind: AssetKind) -> list[str]:
    """Odkrywacz: ostatnie raporty spolki z EDGAR-a. Tylko dla akcji.

    `gaps` jest tu nieuzywane - EDGAR adresujemy tickerem, nie trescia luki.
    Interfejs przyjmuje luki, bo kolejne odkrywacze (API wyszukiwania) beda
    budowac z nich zapytania.
    """
    if asset_kind != "equity":
        # Mapa tickerow EDGAR-a zawiera tez fundusze: "BTC" trafia w emitenta
        # ETF-u bitcoinowego, wiec agenci dostaliby raporty funduszu zamiast
        # danych o samym aktywie. Lepiej brak zrodla niz zrodlo o kims innym.
        print(
            f"[discover] EDGAR pominiety dla {asset}: asset_kind={asset_kind}. "
            "Ticker tokena trafia w mapie EDGAR-a w emitenta ETF-u, nie w samo "
            "aktywo - to byly raporty funduszu, nie tokena."
        )
        return []

    cik = _cik_for_ticker(asset)
    if cik is None:
        return []

    data = _sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if data is None:
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    documents = recent.get("primaryDocument", [])

    urls: list[str] = []
    for form, accession, document in zip(forms, accessions, documents):
        if form not in _EDGAR_FORMS or not document:
            continue
        urls.append(
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/{document}"
        )
        if len(urls) >= _EDGAR_MAX_FILINGS:
            break

    print(f"[discover] EDGAR {asset} (CIK {cik}): {len(urls)} raportow")
    return urls


def defillama_protocol(gaps: list[Claim], asset: str, asset_kind: AssetKind) -> list[SourceDoc]:
    """Odkrywacz on-chain: DOKUMENTY (nie URL-e) z DefiLlama, tylko dla tokenow.

    Token bierze protokol (defillama_slug) albo lancuch (defillama_chain) z
    assets.json. Bez zadnego z tych pol DefiLlama jest pomijana i zostaja
    seed_urls. `gaps` nieuzywane - endpoint adresujemy tickerem, nie trescia luki.
    """
    if asset_kind != "token":
        return []

    meta = asset_meta(asset)
    slug = meta.get("defillama_slug")
    chain = meta.get("defillama_chain")
    if not slug and not chain:
        print(
            f"[llama] {asset}: brak defillama_slug i defillama_chain w assets.json "
            "- zostaja same seed_urls"
        )
        return []

    docs: list[SourceDoc] = []
    tvl_doc = fetch_protocol_doc(slug) if slug else fetch_chain_doc(chain)
    if tvl_doc is not None:
        docs.append(tvl_doc)

    # Oplaty i przychod istnieja tylko dla protokolow (nie dla calych lancuchow)
    # i tylko gdy protokol je raportuje. Brak danych pomijamy cicho.
    if slug:
        fees_doc = fetch_fees_doc(slug, meta.get("market_cap"))
        if fees_doc is not None:
            docs.append(fees_doc)

    for doc in docs:
        print(f"[llama] {asset}: dokument on-chain ({len(doc.text)} zn.) z {doc.url}")
    return docs


# Odkrywacze URL-i (pobierane przez fetch_sources) i odkrywacze gotowych
# dokumentow (dane z API, ktorych nie ma sensu przepuszczac przez trafilature).
DISCOVERERS: list[Discoverer] = [edgar_filings]
DOC_DISCOVERERS = [defillama_protocol]


def discover_docs(
    gaps: list[Claim],
    asset: str,
    asset_kind: AssetKind = "equity",
) -> list[SourceDoc]:
    """Dokumenty z API. Blad jednego odkrywacza nie przerywa reszty."""
    docs: list[SourceDoc] = []
    for discoverer in DOC_DISCOVERERS:
        try:
            docs += discoverer(gaps, asset, asset_kind)
        except Exception as err:          # obce API nie moze wysadzic debaty
            print(f"[discover] {discoverer.__name__} padl ({type(err).__name__}: {err})")
    return docs


def discover_urls(
    gaps: list[Claim],
    asset: str,
    asset_kind: AssetKind = "equity",
    discoverers: Optional[list[Discoverer]] = None,
) -> list[str]:
    """Zbiera URL-e ze wszystkich odkrywaczy. Blad jednego nie przerywa reszty."""
    if not gaps:
        return []

    print(f"[discover] {len(gaps)} luk informacyjnych do zasypania:")
    for gap in gaps:
        print(f"[discover]   - [{gap.id}] {gap.agent}: {gap.text}")

    urls: list[str] = []
    for discoverer in discoverers or DISCOVERERS:
        try:
            found = discoverer(gaps, asset, asset_kind)
        except Exception as err:  # odkrywacz to obce API - nie moze wysadzic debaty
            print(f"[discover] {discoverer.__name__} padl ({type(err).__name__}: {err})")
            continue
        urls += [u for u in found if u not in urls]
    return urls


def _doc_event(doc: SourceDoc) -> dict:
    """Zrodlo dla frontendu - bez calej tresci, ktora ma i 15000 znakow."""
    return {
        "url": doc.url,
        "title": doc.title,
        "source_type": doc.source_type,
        "published_at": doc.published_at.isoformat() if doc.published_at else None,
        "chars": len(doc.text),
    }


def collect_gaps(state: DebateState, round_no: int) -> list[Claim]:
    """Luki informacyjne zgloszone w danej rundzie."""
    return [
        c
        for c in state.claims
        if c.round == round_no and c.claim_type == "information_gap"
    ]


def _serve_objections(
    state: DebateState,
    docs: list[SourceDoc],
    objections: "Optional[queue.Queue[str]]",
    emit: EventSink,
    round_no: int,
) -> None:
    """Odpowiada na pytania z widowni zebrane w trakcie rundy.

    Kolejke oprozniamy do konca: pytanie zadane w rundzie 1 nie ma po co czekac
    do konca debaty. Jedno pytanie to jedno wywolanie JUDGE_MODEL - odpowiedz
    nie jest twierdzeniem, nie wchodzi do stanu i nie rusza budzetow.
    """
    if objections is None:
        return

    while True:
        try:
            question = objections.get_nowait()
        except queue.Empty:
            return

        emit({"type": "objection", "round": round_no, "question": question})
        print(f"[debate] wtracenie z widowni (runda {round_no}): {question[:90]}")
        try:
            answer = answer_question(state, question, docs)
        except Exception as err:   # pytanie widza nie moze wysadzic debaty
            print(f"[debate] sedzia nie odpowiedzial ({type(err).__name__}: {err})")
            answer = "The court cannot take that question right now."
        emit({
            "type": "objection_answer",
            "round": round_no,
            "question": question,
            "answer": answer,
        })


def run_debate(
    asset: str,
    asset_kind: AssetKind = "equity",
    rounds: int = ROUNDS,
    seed_urls: Optional[list[str]] = None,
    tiers: Optional[list[str]] = None,
    verbose: bool = True,
    on_event: Optional[EventSink] = None,
    objections: "Optional[queue.Queue[str]]" = None,
    pack: Optional[list[SourceDoc]] = None,
) -> tuple[DebateState, Verdict]:
    """Pelna petla: rundy agentow -> sedzia -> uzupelnienie zrodel po lukach.

    on_event dostaje kolejne zdarzenia (round_start, claim, ruling,
    sources_added, objection, objection_answer, verdict) w momencie ich
    wystapienia. objections to kolejka pytan z widowni - sprawdzana po kazdej
    rundzie. pack, jesli podany, dostaje aktualny pakiet zrodel (ta sama lista
    dokumentow, ktora widza agenci) - dzieki temu wywolujacy moze po debacie
    odpowiadac na pytania z tym samym materialem. Zwraca (stan, werdykt).
    """
    def emit(event: dict) -> None:
        if on_event is not None:
            on_event(event)
    allowlist = allowlist_for(tiers or DEFAULT_TIERS)

    # Jeden kontener na cala debate - kolejne wywolania kwanta trafiaja do tego
    # samego srodowiska, wiec moze budowac na tym, co juz policzyl.
    container_id = f"orbio-{asset.lower()}-{uuid4().hex[:8]}"
    if verbose:
        print(f"[debate] kontener sandboxa: {container_id}")

    state = DebateState(
        asset=asset,
        asset_kind=asset_kind,
        agents={
            name: AgentState(agent=name, budget=CREDIBILITY_BUDGET)
            for name in ("bull", "bear", "quant")
        },
    )

    meta = asset_meta(asset)
    # Seedy z assets.json sa wpisane recznie, wiec traktujemy je jak zaufane i
    # nie przepuszczamy przez allowliste tierow (dokumentacja projektu rzadko
    # siedzi na sec.gov).
    curated = [u for u in (meta.get("seed_urls") or []) if u not in (seed_urls or [])]

    # Raporty spolki wchodza JUZ do pakietu startowego, obok seed_urls.
    # Dociaganie po zgloszonej luce zostaje, ale w krotkiej debacie luka czesto
    # nie pada wcale - i agenci caly czas opieraja sie na Wikipedii.
    try:
        edgar_urls = edgar_filings([], asset, asset_kind)
    except Exception as err:      # EDGAR to obce API - nie moze wysadzic debaty
        print(f"[debate] EDGAR pominiety ({type(err).__name__}: {err})")
        edgar_urls = []

    # dict.fromkeys zachowuje kolejnosc i usuwa duplikaty (seed moze wskazywac
    # ten sam raport, ktory znalazl EDGAR).
    start_urls = list(dict.fromkeys([*(seed_urls or []), *edgar_urls]))

    # Wspolny pakiet zrodel - rosnie po kazdej rundzie.
    docs: list[SourceDoc] = fetch_sources(start_urls, allowlist=allowlist)
    docs += fetch_sources(curated, allowlist=None)          # zaufane, recznie wpisane
    docs += discover_docs([], asset, asset_kind)            # on-chain dla tokenow

    # Dane referencyjne z assets.json wchodza do pakietu dla KAZDEGO aktywa -
    # akcji i tokena. Do tej pory widzial je tylko widz w panelu na scenie, a to
    # jedyne miejsce, gdzie kapitalizacja, FDV i przychod 30d stoja obok siebie.
    reference = asset_reference_doc(asset, meta)
    if reference is not None:
        docs.append(reference)
    elif meta:
        print(f"[debate] {asset}: wpis w assets.json nie dal sie zrenderowac")
    else:
        print(f"[debate] {asset}: brak wpisu w assets.json - bez danych referencyjnych")

    known_urls = {d.url for d in docs}
    if pack is not None:
        pack[:] = docs                    # wglad w pakiet dla wywolujacego

    z_edgara = sum(1 for d in docs if d.url in set(edgar_urls))
    onchain = sum(1 for d in docs if d.source_type == "onchain")
    print(
        f"[debate] pakiet startowy: {len(docs)} zrodel "
        f"(EDGAR: {z_edgara} z {len(edgar_urls)}, on-chain: {onchain}, "
        f"referencyjne: {1 if reference is not None else 0}, "
        f"seedy recznie: {len(curated)}, seedy z zapytania: {len(seed_urls or [])})"
    )
    if not docs:
        # Bez zrodel kazde twierdzenie faktyczne skonczy jako unsourced, a kara za
        # runde bez dowodow sie nie naliczy (pusty pakiet ja wylacza). Lepiej
        # wiedziec o tym przed debata niz po niej.
        print(
            f"[debate] UWAGA: pakiet zrodel dla {asset} ({asset_kind}) jest PUSTY. "
            "Agenci nie maja czego cytowac - spodziewaj sie samych unsourced. "
            "Sprawdz defillama_slug / defillama_chain / seed_urls w assets.json."
        )

    if docs:
        emit(
            {
                "type": "sources_added",
                "round": 0,
                "docs": [_doc_event(d) for d in docs],
                "pack_size": len(docs),
            }
        )

    for round_no in range(1, rounds + 1):
        state.current_round = round_no
        emit({"type": "round_start", "round": round_no, "rounds": rounds})
        if verbose:
            print(f"\n========== RUNDA {round_no}/{rounds} ==========")

        # Trzy wywolania naraz. Wolno, bo w OBREBIE rundy agenci sie nie widza:
        # kazdy dostaje ten sam pakiet zrodel i te sama historie z rund
        # poprzednich, a nowe twierdzenia wchodza do stanu dopiero po tym, jak
        # wszyscy skoncza. Kolejnosc wywolan nie zmienia wiec tego, co widza.
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="agent") as pool:
            # Sandbox dostaje tylko quant - bull i bear nie licza, tylko twierdza.
            futures = {
                name: pool.submit(
                    runner,
                    state,
                    docs=docs,
                    **({"container_id": container_id} if runner is run_quant else {}),
                )
                for name, runner in (
                    ("bull", run_bull),
                    ("bear", run_bear),
                    ("quant", run_quant),
                )
            }

        # Zbieramy w KOLEJNOSCI AGENTOW, nie w kolejnosci ukonczenia - arena ma
        # zawsze ten sam porzadek: byk, niedzwiedz, kwant.
        for name, future in futures.items():
            try:
                claims = future.result()
            except Exception as err:
                # Padniety agent nie zabiera reszcie rundy: dwa pozostale modele
                # juz odpowiedzialy (i juz kosztowaly), wiec ich twierdzenia
                # wchodza do debaty, a sedzia oceni to, co jest.
                print(f"[debate] {name} padl w rundzie {round_no} ({type(err).__name__}: {err})")
                continue
            for claim in claims:
                state.add_claim(claim)
                emit({"type": "claim", "round": round_no, "claim": claim.model_dump(mode="json")})
                if verbose:
                    stake = "-" if claim.stake is None else claim.stake
                    print(
                        f"  [{claim.id}] {claim.agent:5} {claim.claim_type:15} "
                        f"stake={stake} conf={claim.confidence}: {claim.text[:90]}"
                    )

        rulings = judge_round(state, round_no)
        # pack_size decyduje, czy naliczyc kare za runde bez dowodow.
        apply_rulings(state, rulings, pack_size=len(docs))

        # Budzety wysylamy PO rozliczeniu - pasek ma pokazac stan faktyczny.
        for ruling in rulings:
            oceniony = state.get_claim(ruling.claim_id)
            emit({
                "type": "ruling",
                "round": round_no,
                "ruling": ruling.model_dump(mode="json"),
                "status": oceniony.status if oceniony else ruling.new_status,
                "judge_note": oceniony.judge_note if oceniony else ruling.note,
                "budgets": {n: st.budget for n, st in state.agents.items()},
            })
        if verbose:
            for ruling in rulings:
                print(
                    f"  SEDZIA [{ruling.claim_id}] -> {ruling.new_status} "
                    f"(kara {ruling.calibration_penalty}): {ruling.note[:80]}"
                )
            print(f"  Budzety: {[(n, s.budget) for n, s in state.agents.items()]}")

        # Luki -> nowe zrodla do wspolnego pakietu (ostatnia runda nie potrzebuje).
        if round_no < rounds:
            gaps = collect_gaps(state, round_no)
            new_urls = [
                u
                for u in discover_urls(gaps, asset, asset_kind)
                if u not in known_urls
            ]
            fresh_docs = [
                d for d in discover_docs(gaps, asset, asset_kind) if d.url not in known_urls
            ]
            if new_urls or fresh_docs:
                fresh = fetch_sources(new_urls, allowlist=allowlist) + fresh_docs
                docs += fresh
                known_urls |= {d.url for d in fresh}
                if fresh:
                    emit({
                        "type": "sources_added",
                        "round": round_no,
                        "docs": [_doc_event(d) for d in fresh],
                        "pack_size": len(docs),
                    })
                if pack is not None:
                    pack[:] = docs
                if verbose:
                    print(f"  Pakiet zrodel: {len(docs)} dokumentow (+{len(fresh)})")

        # Pytania z widowni obsluzone PO rozliczeniu rundy, przed nastepna -
        # sedzia odpowiada wiedzac, co juz zostalo zweryfikowane.
        _serve_objections(state, docs, objections, emit, round_no)

    verdict = final_verdict(state)
    emit({"type": "verdict", "verdict": verdict.model_dump(mode="json")})
    return state, verdict


if __name__ == "__main__":
    import sys

    # Uzycie: python debate.py [TICKER] [RUNDY] [equity|token]
    asset = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    asset_kind = sys.argv[3] if len(sys.argv) > 3 else "equity"
    if asset_kind not in ("equity", "token"):
        sys.exit(f"Nieznany asset_kind: {asset_kind}. Wybierz: equity, token")

    state, verdict = run_debate(
        asset=asset,
        asset_kind=asset_kind,
        rounds=rounds,
        seed_urls=["https://en.wikipedia.org/wiki/Nvidia"],
    )

    print("\n========== PODSUMOWANIE ==========")
    print(f"Aktywo: {state.asset}, rund: {rounds}, twierdzen: {len(state.claims)}")
    for name, agent_state in state.agents.items():
        print(f"  {name:5} budzet={agent_state.budget} twierdzen={len(agent_state.claim_ids)}")
    by_status: dict[str, int] = {}
    for claim in state.claims:
        by_status[claim.status] = by_status.get(claim.status, 0) + 1
    print(f"  statusy: {by_status}")
    print(f"\nTEZA: {verdict.thesis}")
    print(f"PEWNOSC: {verdict.confidence}")
    for f in verdict.falsifiers:
        print(f"  falsyfikator: {f}")
