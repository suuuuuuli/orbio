"""Agenci debaty: bull, bear, quant.

Kazdy chodzi na innym modelu (stale z config.py) i zwraca partie twierdzen
przez llm.py (kazdy ma wyszukiwarke, quant dodatkowo sandbox).
Sedzia siedzi osobno, w judge.py.
"""

import re

from pydantic import BaseModel, Field

from config import (
    BEAR_MODEL,
    BULL_MODEL,
    EXEC_TIMEOUT,
    MAX_GAPS_PER_ROUND,
    MAX_QUOTE_CHARS,
    MAX_TOKENS,
    QUANT_MAX_TOKENS,
    QUANT_MODEL,
)
from executor import run_code
from llm import call_structured_with_tools
from models import Claim, DebateState, new_id
from sources import SourceDoc, normalize_text, render_for_prompt

MAX_CLAIMS_PER_ROUND = 3
MAX_STAKE_PER_ROUND = 20


class ClaimBatch(BaseModel):
    """Odpowiedz agenta: kilka twierdzen naraz (max MAX_CLAIMS_PER_ROUND)."""

    claims: list[Claim] = Field(default_factory=list)


# Zasady wspolne dla wszystkich agentow stawiajacych twierdzenia.
_RULES = f"""Rules:
- One claim is one specific proposition, not a paragraph. At most {MAX_CLAIMS_PER_ROUND} claims per round.
- You receive a SOURCE PACK (blocks marked [SOURCE id=S1 ...]). Ground every factual
  claim in it. You also have web search - use it before making a factual claim if
  the pack is not enough.
- source_url must be one of the URLs from the pack (the url= field in a block
  header), never from memory. A URL recalled from memory is rejected automatically,
  before the judge ever sees it.
- source_quote is ONE sentence copied VERBATIM from the source text - not a
  paraphrase, not a summary, not a longer passage (max {MAX_QUOTE_CHARS} characters).
  A quote that does not appear in the source invalidates the whole claim: the
  automation clears both the quote and the URL.
- If you have no source, set source_url to null and DO NOT invent a URL.
  An invented link is punished harder than no source at all.
- stake (1-20) reflects how much you are genuinely willing to risk. Your budget is
  finite, and the stake is forfeited when a claim is refuted or turns out unsourced.
- The SUM of stakes across your claims this round must not exceed {MAX_STAKE_PER_ROUND}.
  You have to choose what you actually back - spreading the same mid-range stake
  across every claim wastes the round.
- confidence (0-1) is your honest estimate, even when it is low. The judge punishes
  any gap between stated confidence and the stake you put behind it.
- You may attack an opponent's claim - put its id in the targets field.
- Preferably (not mandatory) contribute at least one claim about something your
  opponent has not raised at all. A real disagreement is about WHICH facts matter,
  not only about how to read the same ones - mirroring the same topics with the
  opposite sign is a weak debate. But do not invent a topic just to make it new:
  staying with what actually matters beats adding a pointless argument.
- DO NOT make claims about events that are absent from the source pack - not even
  without citing a URL. If you know something from memory but the pack does not
  contain it, report it as an information_gap, not as a claim.
- If a missing data point would settle the dispute, report it as a claim with
  claim_type="information_gap": in text describe EXACTLY what data is missing
  (e.g. "gross margin for the latest quarter from the 10-Q"). A gap carries no
  stake and no source_url. After the round, the arena tries to fetch a source that
  fills it. Limits: at most {MAX_GAPS_PER_ROUND} gap per round, and a gap CANNOT be
  your only contribution - a round made of gaps alone is rejected entirely.
- Leave code and result empty (null) unless you are the quant.
- WRITE EVERYTHING IN ENGLISH: claim text, quotes, code and anything the code
  prints. The sources may be in any language; your output is always English."""

_BULL_SYSTEM = f"""You are the BULL agent in a structured investment debate. You build the positive
thesis for the asset: you look for reasons its value should rise.
You do not produce a "buy" recommendation - you produce verifiable claims.

{_RULES}"""

_BEAR_SYSTEM = f"""You are the BEAR agent in a structured investment debate. You build the negative
thesis for the asset: you look for risks, weaknesses and reasons it should fall.
You do not produce a "sell" recommendation - you produce verifiable claims.

{_RULES}"""

_QUANT_SYSTEM = f"""You are the QUANT agent in a structured investment debate. You do NOT speculate and
you do not grade narratives. Your job: identify what in this debate can actually be
computed from real data, and write the code that computes it.

For each claim:
- set claim_type to "quantitative"
- text states exactly what you compute and how the result settles the dispute
- code is a self-contained Python script that prints its result, and it must be
  TERSE: up to 30 lines, no comments, no diagnostics - only the result matters.
  Labels inside print() are ENGLISH, e.g. print("yoy growth: 50.1%")
- the code is EXECUTED automatically after your answer, with a {EXEC_TIMEOUT}s
  limit, so it must be self-sufficient and run without network access. Compute on
  data from the source pack and from the debate; do not try to download anything.
  Use the standard library only - no pandas, no numpy, no yfinance
- result is filled in by the automation from the execution output; leave it null
  and leave status as "pending"
- if you attack another claim with numbers, put its id in targets

What you may compute:
- the computation must concern an ECONOMIC QUANTITY taken from the source pack or
  from claims made in the debate (revenue, margin, market share, valuation, growth
  rate, price, debt and so on). The numbers in code must come from there.
- ON-CHAIN data is valid material: a [SOURCE ... typ=onchain] block carries total
  value locked, per-chain breakdowns, a dated TVL series and - for protocols -
  fees and revenue over 24h / 7d / 30d / all time. Percentage changes, averages,
  trends and ratios computed from those figures are exactly the kind of work
  expected here - copy the numbers from the blocks, do not invent them.
- For a TOKEN, work through these named ratios - do not stop at market cap over
  TVL, which says the least of them:
  * market cap to annualised revenue: market_cap / (revenue_30d * 12). The
    earnings multiple of the token. Say which revenue window you annualised.
  * holders share of revenue: holders_revenue_30d / revenue_30d. How much of what
    the protocol earns actually reaches the token rather than the operator.
  * supply overhang: fdv / market_cap. How much of the supply is still to come.
    Above roughly 2 the circulating price is carrying a large unvested claim.
  * revenue to TVL: (revenue_30d * 12) / tvl. What the deposited capital earns -
    a protocol with huge TVL and no revenue is warehousing money, not selling
    anything.
  * when earnings_annualized is NEGATIVE: incentives_1y / (revenue_30d * 12).
    How many units of emission the protocol pays for one unit of revenue. Above 1
    it is buying its own activity.
  The reference-data source carries these fields for the asset under debate. For
  the remaining ratios: fees to TVL (how hard the locked capital works), revenue
  share of fees (how much the protocol keeps), TVL per unit of market cap. Say
  which two figures you divided and where each came from.
- NO claims about the state of the pack itself: how many sources there are, how many
  claims were made, how complete the data coverage is, how many gaps were reported.
  That is arena metadata, not analysis of the asset - such a claim is rejected
  automatically.
- if there is nothing in the pack worth computing, report ONE information_gap
  (what data is missing) and NOTHING ELSE - no substitute claims.

{_RULES}"""


def _normalize_url(url: str) -> str:
    """Do porownywania URL-i: bez bialych znakow, bez konca '/', bez rozroznienia wielkosci."""
    return url.strip().rstrip("/").lower()


def _build_context(state: DebateState, agent: str, docs: list[SourceDoc]) -> str:
    """Sklada pakiet zrodel + stan debaty w tekst dla promptu danego agenta."""
    my_state = state.agents.get(agent)
    budget = my_state.budget if my_state else 0

    lines = [
        "SOURCE PACK:",
        render_for_prompt(docs),
        "",
    ]
    lines += [
        f"Asset: {state.asset}",
        f"Mode: {state.mode}"
        + (f" (data frozen at {state.frozen_at})" if state.frozen_at else ""),
        f"Round: {state.current_round}",
        f"Your remaining credibility budget: {budget}",
        "",
    ]

    others = [c for c in state.claims if c.agent != agent]
    if others:
        lines.append("Claims by the other agents (you may attack them via targets):")
        for c in others:
            src = c.source_url or "no source"
            lines.append(
                f"- [{c.id}] {c.agent} (r{c.round}, {c.claim_type}, status={c.status}, "
                f"stake={c.stake}, confidence={c.confidence}, {src}): {c.text}"
            )
    else:
        lines.append("Nobody has made a claim yet - you open the debate.")

    mine = [c for c in state.claims if c.agent == agent]
    if mine:
        lines.append("")
        lines.append("Your earlier claims (do not repeat them):")
        for c in mine:
            lines.append(f"- [{c.id}] (r{c.round}, status={c.status}): {c.text}")

    return "\n".join(lines)


# Liczba w kodzie/tekscie: 215.9, 1,234.5, 26_914_000_000, 70%
_NUMBER = re.compile(r"\d[\d_.,]*")

# Ile znaczacych cyfr musi miec liczba, zeby jej dopasowanie cos znaczylo.
# Jedna cyfra ("2", "10", "100") trafia w dowolny tekst przez przypadek.
_MIN_SIG_DIGITS = 2


def _number_keys(raw: str) -> set[str]:
    """Klucze dopasowania jednej liczby: cyfry i cyfry znaczace.

    - cyfry ("70%" -> "70", "100" -> "100") lapia liczby okragle, ktore po
      obcieciu zer koncowych zrobilyby sie jednocyfrowe
    - cyfry znaczace ("215.9" i "215_900_000_000" -> oba "2159") lapia te sama
      wielkosc podana w innej skali, np. mld w zrodle vs jednostki w kodzie
    Kazdy klucz musi miec >= _MIN_SIG_DIGITS znakow - jedna cyfra trafia
    w dowolny tekst przez przypadek.
    """
    digits = re.sub(r"[^0-9]", "", raw).lstrip("0")
    keys = {digits, digits.rstrip("0")}
    return {k for k in keys if len(k) >= _MIN_SIG_DIGITS}


def _numbers_in(text: str) -> set[str]:
    """Klucze wszystkich liczb w tekscie."""
    found: set[str] = set()
    for match in _NUMBER.findall(text or ""):
        found |= _number_keys(match)
    return found


def _reject_groundless_code(
    claims: list[Claim],
    docs: list[SourceDoc],
    state: DebateState,
) -> list[Claim]:
    """Odrzuca wyliczenia, ktorych liczby nie pochodza z pakietu ani z debaty.

    Bez tego quant liczy cokolwiek - wlasne zalozenia, metadane areny albo
    liczby z pamieci modelu - i podaje to jako dowod. Zrodlem liczb moze byc
    tresc zrodel albo twierdzenia (wraz z ich wynikami) postawione w debacie.
    """
    grunt: set[str] = set()
    for doc in docs:
        grunt |= _numbers_in(doc.text)
        grunt |= _numbers_in(doc.title)
    for claim in state.claims:
        grunt |= _numbers_in(claim.text)
        grunt |= _numbers_in(claim.result or "")

    zostaje: list[Claim] = []
    for claim in claims:
        if claim.claim_type == "information_gap":
            zostaje.append(claim)
            continue

        w_kodzie = _numbers_in(claim.code or "")
        wspolne = w_kodzie & grunt
        if wspolne:
            zostaje.append(claim)
            continue

        print(
            f"[quant] odrzucam [{claim.id}]: zadna liczba z code nie pochodzi "
            f"z pakietu ani z debaty (w kodzie: "
            f"{sorted(w_kodzie)[:5] or 'brak liczb'})"
        )
    return zostaje


def _enforce_gap_limits(agent: str, claims: list[Claim]) -> list[Claim]:
    """Limituje luki informacyjne - to najtansza furtka do wypelnienia rundy.

    information_gap nie ma stawki i nie jest oceniany przez sedziego, wiec bez
    limitu agent moglby "brac udzial" w rundzie, nie ryzykujac nic.
    """
    gaps = [c for c in claims if c.claim_type == "information_gap"]
    if not gaps:
        return claims

    real = [c for c in claims if c.claim_type != "information_gap"]

    # Same luki: agent nie wniosl nic, za co odpowiada - odrzucamy wszystko.
    if not real:
        print(
            f"[{agent}] odrzucam cala runde: {len(gaps)} luk i zadnego twierdzenia "
            "(luka nie moze byc jedynym wkladem)"
        )
        return []

    if len(gaps) > MAX_GAPS_PER_ROUND:
        print(
            f"[{agent}] odsiewam {len(gaps) - MAX_GAPS_PER_ROUND} luk ponad limit "
            f"{MAX_GAPS_PER_ROUND}/runde"
        )
        gaps = gaps[:MAX_GAPS_PER_ROUND]

    # Kolejnosc jak w odpowiedzi modelu, ale bez nadwyzki luk.
    keep = {id(c) for c in real} | {id(c) for c in gaps}
    return [c for c in claims if id(c) in keep]


def _run_agent(
    agent: str,
    model: str,
    system: str,
    state: DebateState,
    context: str,
    docs: list[SourceDoc],
    max_tokens: int = MAX_TOKENS,
    container_id: str | None = None,
) -> tuple[list[Claim], list[str]]:
    """Wspolna sciezka: prompt -> narzedzia -> normalizacja twierdzen.

    Zwraca (twierdzenia, wyjscia z sandboxa). Sandbox dostaje tylko ten agent,
    ktory poda container_id (w praktyce: quant).
    """
    user = _build_context(state, agent, docs)
    if context.strip():
        user += f"\n\nAdditional context / data:\n{context.strip()}"
    user += f"\n\nMake between 1 and {MAX_CLAIMS_PER_ROUND} claims as the '{agent}' agent."

    batch, search_urls, shell_outputs = call_structured_with_tools(
        model=model,
        system=system,
        user=user,
        schema=ClaimBatch,
        max_tokens=max_tokens,
        container_id=container_id,
    )
    # Tekst zrodel trzymamy pod znormalizowanym URL-em - po nim sprawdzamy cytaty.
    pack = {_normalize_url(d.url): normalize_text(d.text) for d in docs}
    from_search = {_normalize_url(u) for u in search_urls}

    claims = batch.claims[:MAX_CLAIMS_PER_ROUND]
    for claim in claims:
        # Id, autora i runde ustawiamy my - model nie ma prawa sie tu pomylic
        # (potrafi wymyslac wlasne id typu "bull-r1-1", co grozi kolizjami).
        claim.id = new_id()
        claim.agent = agent
        claim.round = state.current_round
        # targets musi wskazywac na istniejace twierdzenie, inaczej graf sie sypie.
        if claim.targets and state.get_claim(claim.targets) is None:
            claim.targets = None
        # Twarda walidacja zrodla. Prompt nie wystarcza - model potrafi podac
        # link z pamieci i dopasowac do niego cytat, ktorego tam nie ma.
        if claim.source_quote and len(claim.source_quote) > MAX_QUOTE_CHARS:
            print(
                f"[{agent}] cytat za dlugi ({len(claim.source_quote)} zn.), przycinam "
                f"do {MAX_QUOTE_CHARS}"
            )
            claim.source_quote = claim.source_quote[:MAX_QUOTE_CHARS]

        if claim.source_url:
            url_key = _normalize_url(claim.source_url)
            if url_key in pack:
                # Zrodlo z pakietu - mamy tekst, wiec cytat da sie sprawdzic doslownie.
                quote = normalize_text(claim.source_quote or "")
                if quote and quote not in pack[url_key]:
                    print(
                        f"[{agent}] cytat nie wystepuje w zrodle - kasuje URL i cytat: "
                        f"{claim.source_url}"
                    )
                    claim.source_url = None
                    claim.source_quote = None
            elif url_key in from_search:
                # URL z wyszukiwania: pochodzenie pewne, ale tresci nie mamy,
                # wiec cytatu nie da sie zweryfikowac - zostaje sam URL.
                if claim.source_quote:
                    print(f"[{agent}] cytat niesprawdzalny (URL z wyszukiwania), kasuje cytat")
                    claim.source_quote = None
            else:
                print(f"[{agent}] odrzucam URL spoza pakietu i wyszukiwania: {claim.source_url}")
                claim.source_url = None
                claim.source_quote = None

    claims = _enforce_gap_limits(agent, claims)

    total_stake = sum(c.stake or 0 for c in claims)
    if total_stake > MAX_STAKE_PER_ROUND:
        # Nie przeskalowujemy - stawka to sygnal konwikcji, a nie liczba do naciagania.
        print(
            f"[{agent}] suma stawek {total_stake} > {MAX_STAKE_PER_ROUND} "
            f"(model zignorowal limit)"
        )
    return claims, shell_outputs


def run_bull(
    state: DebateState, context: str = "", docs: list[SourceDoc] | None = None
) -> list[Claim]:
    """Buduje teze pozytywna."""
    claims, _ = _run_agent("bull", BULL_MODEL, _BULL_SYSTEM, state, context, docs or [])
    return claims


def run_bear(
    state: DebateState, context: str = "", docs: list[SourceDoc] | None = None
) -> list[Claim]:
    """Buduje teze negatywna."""
    claims, _ = _run_agent("bear", BEAR_MODEL, _BEAR_SYSTEM, state, context, docs or [])
    return claims


def run_quant(
    state: DebateState,
    context: str = "",
    docs: list[SourceDoc] | None = None,
    container_id: str | None = None,
) -> list[Claim]:
    """Wskazuje, co da sie policzyc, liczy w sandboxie i wpisuje wynik.

    container_id to kontener debaty (jeden na cala debate, patrz debate.py).
    Bez niego quant dostaje tylko wyszukiwarke i `result` zostaje pusty.
    """
    # Quant zwraca kod, wiec potrzebuje wiekszego limitu - inaczej JSON sie urywa.
    claims, shell_outputs = _run_agent(
        "quant",
        QUANT_MODEL,
        _QUANT_SYSTEM,
        state,
        context,
        docs or [],
        max_tokens=QUANT_MAX_TOKENS,
        container_id=container_id,
    )

    claims = _reject_groundless_code(claims, docs or [], state)

    computed = [c for c in claims if c.claim_type != "information_gap"]
    for claim in computed:
        claim.claim_type = "quantitative"
        claim.status = "pending"   # status nadaje sedzia, nie agent

    if shell_outputs:
        # Bramka przepuscila sandbox. Nie uzywamy tych zrzutow, bo nie wiadomo,
        # ktore wywolanie nalezy do ktorego twierdzenia - liczymy kazde osobno.
        print(
            f"[quant] bramka zwrocila {len(shell_outputs)} wyjsc z sandboxa "
            "- ignoruje je, wykonuje kod twierdzen osobno"
        )

    # Kazde twierdzenie wykonywane OSOBNO - dlatego result nalezy do niego,
    # a nie do wspolnego zrzutu. To rozwiazuje problem przypisania 1:1.
    for claim in computed:
        if not (claim.code or "").strip():
            claim.result = None
            print(f"[quant] [{claim.id}] bez kodu - nie ma czego wykonac")
            continue
        claim.result = run_code(claim.code, timeout=EXEC_TIMEOUT)
        pierwsza = claim.result.splitlines()[0] if claim.result else ""
        print(f"[quant] [{claim.id}] wykonano: {pierwsza[:80]}")
    return claims


if __name__ == "__main__":
    # Uzycie: python agents.py [bull|bear|quant|both|all]   (domyslnie bull)
    #   both -> bull, potem bear widzacy twierdzenia byka
    #   all  -> bull, bear, quant
    import sys

    from config import CREDIBILITY_BUDGET
    from models import AgentState
    from sources import fetch_sources

    RUNNERS = {"bull": run_bull, "bear": run_bear, "quant": run_quant}
    SEQUENCES = {"both": ["bull", "bear"], "all": ["bull", "bear", "quant"]}

    arg = (sys.argv[1] if len(sys.argv) > 1 else "bull").lower()
    sequence = SEQUENCES.get(arg, [arg])
    unknown = [name for name in sequence if name not in RUNNERS]
    if unknown:
        sys.exit(f"Nieznany agent: {', '.join(unknown)}. Wybierz: bull, bear, quant, both, all")

    state = DebateState(
        asset="NVDA",
        current_round=1,
        agents={
            name: AgentState(agent=name, budget=CREDIBILITY_BUDGET)
            for name in ("bull", "bear", "quant")
        },
    )

    for name in sequence:
        print(f"\n===== {name.upper()} =====")
        for claim in RUNNERS[name](state, context=""):
            state.add_claim(claim)
            target = f"  -> atakuje [{claim.targets}]" if claim.targets else ""
            print(
                f"[{claim.id}] {claim.claim_type} stake={claim.stake} "
                f"confidence={claim.confidence}{target}"
            )
            print(f"    {claim.text}")
            print(f"    zrodlo: {claim.source_url or 'brak'}")
            print(f"    cytat:  {claim.source_quote or 'brak'}")
            if claim.code:
                print(f"    kod:\n{claim.code}")
