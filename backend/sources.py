"""Warstwa dostarczania zrodel: pobranie, ekstrakcja tekstu, cache, render.

Zasada: lepiej brak dokumentu niz dokument okrojony. Okrojony artykul (paywall)
wyglada jak zrodlo, a nim nie jest - agent postawi na nim twierdzenie, ktorego
nie da sie obronic. Dlatego takie dokumenty odrzucamy z jawnym logiem.
"""

import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import httpx
import trafilatura
from pydantic import BaseModel

from config import MAX_SOURCE_CHARS, MIN_SOURCE_CHARS, SOURCE_TIERS

# "onchain" to dane z API (DefiLlama), "reference" to wpis z data/assets.json -
# ani jedno, ani drugie nie jest dokumentem do pobrania.
SourceType = Literal["filing", "article", "pdf", "onchain", "reference"]

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

# Opisowy UA Z ADRESEM KONTAKTOWYM - oba warunki sprawdzone na zywo:
# podszywanie sie pod Chrome dostaje 403 z Wikipedii, opisowy UA bez kontaktu
# tez dostaje 403, dopiero UA z "+https://..." dostaje 200.
_USER_AGENT = "ArenaDebateBot/0.1 (+https://github.com/suuuli/orbio)"

# SEC wymaga UA w formacie "nazwa kontakt@domena" - UA z "+https://..." dostaje
# 403, UA przegladarkowy tez. Sprawdzone na www.sec.gov i data.sec.gov.
_UA_OVERRIDES = {"sec.gov": "Orbio Arena Research arena@orbio.local"}

_TIMEOUT = httpx.Timeout(20.0, connect=10.0)

# Frazy typowe dla sciany paywalla / zgody na cookies zamiast tresci.
_PAYWALL_PHRASES = (
    "subscribe to continue",
    "subscription required",
    "already a subscriber",
    "sign in to read",
    "sign in to continue",
    "create a free account",
    "this article is for subscribers",
    "become a member to read",
    "to continue reading",
    "wykup dostep",
    "wykup dostęp",
    "zaloguj sie, aby czytac",
    "zaloguj się, aby czytać",
    "tresc dostepna w prenumeracie",
    "treść dostępna w prenumeracie",
    "dalsza czesc artykulu",
    "dalsza część artykułu",
)

# Domeny raportow i komunikatow spolek - inny typ zrodla niz zwykly artykul.
_FILING_HINTS = ("sec.gov", "investor.", "ir.", "/investor", "press-release", "annualreport")


def _cache_path(url: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"


def _guess_type(url: str) -> SourceType:
    low = url.lower()
    if low.endswith(".pdf"):
        return "pdf"
    if any(hint in low for hint in _FILING_HINTS):
        return "filing"
    return "article"


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def user_agent_for(url: str) -> str:
    """UA dla hosta - niektore serwisy maja wlasne wymagania (patrz _UA_OVERRIDES)."""
    host = _host(url)
    for domain, ua in _UA_OVERRIDES.items():
        if host == domain or host.endswith(f".{domain}"):
            return ua
    return _USER_AGENT


def allowlist_for(tiers: list[str]) -> list[str]:
    """Sklada allowliste domen z wybranych tierow z SOURCE_TIERS."""
    domains: list[str] = []
    for tier in tiers:
        if tier not in SOURCE_TIERS:
            raise KeyError(f"nieznany tier zrodel: {tier} (mam: {list(SOURCE_TIERS)})")
        domains += [d for d in SOURCE_TIERS[tier] if d not in domains]
    return domains


def tier_of(url: str) -> Optional[str]:
    """Do ktorego tieru nalezy URL (None, jesli do zadnego)."""
    host = _host(url)
    for tier, domains in SOURCE_TIERS.items():
        for domain in domains:
            if host == domain or host.endswith(f".{domain}"):
                return tier
    return None


def _domain_allowed(url: str, allowlist: Optional[list[str]]) -> bool:
    """Pusta/None allowlist = wszystko wolno. Inaczej: domena lub jej poddomena."""
    if not allowlist:
        return True
    host = _host(url)
    return any(
        host == entry or host.endswith(f".{entry}")
        for entry in (e.lower().removeprefix("www.") for e in allowlist)
    )


def _looks_paywalled(text: str) -> bool:
    if len(text) < MIN_SOURCE_CHARS:
        return True
    low = text.lower()
    return any(phrase in low for phrase in _PAYWALL_PHRASES)


def _parse_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw[:10]).date()
    except ValueError:
        return None


class SourceDoc(BaseModel):
    """Jeden pobrany i wyekstrahowany dokument zrodlowy."""

    url: str
    title: str
    published_at: Optional[date] = None
    source_type: SourceType
    text: str


def _load_cached(url: str) -> Optional[SourceDoc]:
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        doc = SourceDoc.model_validate(payload["doc"])
        print(f"[sources] z cache ({payload.get('fetched_at', '?')}): {url}")
        return doc
    except (ValueError, KeyError) as err:
        print(f"[sources] cache uszkodzony, pobieram ponownie: {url} ({err})")
        return None


def _store_cached(doc: SourceDoc) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": doc.url,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "doc": json.loads(doc.model_dump_json()),
    }
    _cache_path(doc.url).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def fetch_sources(
    urls: list[str],
    allowlist: Optional[list[str]] = None,
) -> list[SourceDoc]:
    """Pobiera i ekstrahuje dokumenty. Blad pojedynczego URL-a nie przerywa calosci.

    allowlist: lista dopuszczonych domen (poddomeny tez przechodza), zwykle
    zlozona z tierow przez allowlist_for(["filing", "official"]). None/pusta
    oznacza brak ograniczen. Kazde odrzucenie leci na stdout z powodem:
    poza allowlista / 403 / timeout / paywall / brak tresci.
    """
    docs: list[SourceDoc] = []

    with httpx.Client(
        headers={
            "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.8,*/*;q=0.5",
            "Accept-Language": "pl,en;q=0.8",
        },
        timeout=_TIMEOUT,
        follow_redirects=True,
    ) as client:
        for url in urls:
            if not _domain_allowed(url, allowlist):
                print(f"[sources] pominieto (poza allowlista, tier={tier_of(url)}): {url}")
                continue

            cached = _load_cached(url)
            if cached is not None:
                docs.append(cached)
                continue

            try:
                response = client.get(url, headers={"User-Agent": user_agent_for(url)})
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                code = err.response.status_code
                powod = (
                    "403 (blokada antybotowa)"
                    if code == 403
                    else f"{code} (odpowiedz serwera)"
                )
                print(f"[sources] pominieto ({powod}): {url}")
                continue
            except httpx.TimeoutException:
                print(f"[sources] pominieto (timeout): {url}")
                continue
            except httpx.HTTPError as err:
                print(f"[sources] pominieto (blad pobrania: {type(err).__name__}): {url}")
                continue

            extracted = trafilatura.extract(
                response.text,
                output_format="json",
                with_metadata=True,
                favor_precision=True,
            )
            if not extracted:
                print(f"[sources] pominieto (brak tresci do ekstrakcji): {url}")
                continue

            meta = json.loads(extracted)
            text = (meta.get("text") or "").strip()

            if _looks_paywalled(text):
                powod = (
                    f"paywall? tekst {len(text)} zn. < {MIN_SOURCE_CHARS}"
                    if len(text) < MIN_SOURCE_CHARS
                    else "paywall? fraza o subskrypcji w tresci"
                )
                print(f"[sources] pominieto ({powod}): {url}")
                continue

            doc = SourceDoc(
                url=url,
                title=(meta.get("title") or _host(url)).strip(),
                published_at=_parse_date(meta.get("date")),
                source_type=_guess_type(url),
                text=text[:MAX_SOURCE_CHARS],
            )
            _store_cached(doc)
            print(f"[sources] pobrano ({len(doc.text)} zn.): {url}")
            docs.append(doc)

    return docs


# ---------------------------------------------------------------------------
# Dane referencyjne z data/assets.json
#
# Panel na scenie pokazywal je tylko widzowi - agenci ich nie widzieli. A to
# jedyne miejsce, gdzie kapitalizacja, FDV czy przychod 30d stoja obok siebie
# dla tego samego aktywa, wiec bez nich kwant nie ma z czego zlozyc mnoznikow.
#
# Wartosci przepisujemy DOSLOWNIE z assets.json, bez przeformatowania: cytat
# musi wystapic w tekscie zrodla znak w znak (patrz walidacja w agents.py), a
# "20.551B" przeformatowane na "$20,551,000,000" zlamaloby kazdy cytat.
# ---------------------------------------------------------------------------

# Kolejnosc = kolejnosc w dokumencie. Klucze nieznane trafiaja na koniec z
# etykieta zrobiona z nazwy pola, wiec nowe pole w assets.json dziala od razu.
_REFERENCE_LABELS: dict[str, str] = {
    "price": "Price",
    "market_cap": "Market cap",
    "fdv": "Fully diluted valuation",
    "outstanding_fdv": "Outstanding fully diluted valuation",
    "pe_forward": "Forward price to earnings",
    "revenue_ttm": "Revenue, trailing twelve months",
    "profit_margin": "Profit margin",
    "tvl": "Total value locked",
    "fees_30d": "Fees, 30 days",
    "revenue_30d": "Revenue, 30 days",
    "holders_revenue_30d": "Holders revenue, 30 days",
    "earnings_annualized": "Annualised earnings",
    "incentives_1y": "Token incentives, 1 year",
    "staked": "Staked supply",
    "perp_volume_30d": "Perpetuals volume, 30 days",
    "dex_volume_30d": "DEX volume, 30 days",
    "dex_volume_24h": "DEX volume, 24 hours",
    "open_interest": "Open interest",
    "stablecoins_mcap": "Stablecoins market cap",
    "chain_fees_24h": "Chain fees, 24 hours",
    "chain_revenue_24h": "Chain revenue, 24 hours",
    "app_fees_24h": "Application fees, 24 hours",
    "app_revenue_24h": "Application revenue, 24 hours",
    "active_addresses_24h": "Active addresses, 24 hours",
    "total_raised": "Total raised",
}

# Kwoty dostaja "$". Reszta (mnozniki, marze, liczba adresow, ilosc tokenow)
# nie - dolar przy "3.04M adresow" bylby klamstwem w dokumencie zrodlowym.
_REFERENCE_PLAIN = {
    "pe_forward",
    "profit_margin",
    "active_addresses_24h",
    "staked",
}

# Pola opisowe i techniczne - nie sa metryka.
_REFERENCE_SKIP = {
    "name",
    "kind",
    "as_of",
    "blurb",
    "seed_urls",
    "defillama_slug",
    "defillama_chain",
}

# Recznie wpisane "brak danych" w assets.json.
_REFERENCE_EMPTY = {"", "-", "\u2014", "n/a", "na", "none", "null", "?"}


def _reference_line(key: str, value: str) -> str:
    label = _REFERENCE_LABELS.get(key) or key.replace("_", " ").capitalize()
    raw = str(value).strip()
    prefix = "" if key in _REFERENCE_PLAIN or raw.startswith("$") else "$"
    return f"{label}: {prefix}{raw}"


def asset_reference_doc(ticker: str, entry: dict) -> Optional[SourceDoc]:
    """Wpis z data/assets.json jako dokument zrodlowy. Pusty wpis -> None."""
    if not entry:
        return None

    ticker = ticker.strip().upper()
    as_of = entry.get("as_of") or "not reported"
    name = entry.get("name") or ticker

    # Najpierw pola ze znanej kolejnosci, potem wszystko, czego nie znamy.
    keys = [k for k in _REFERENCE_LABELS if k in entry]
    keys += [k for k in entry if k not in _REFERENCE_SKIP and k not in _REFERENCE_LABELS]

    lines = [
        f"REFERENCE DATA — {name}, as of {as_of}",
        f"Asset class: {entry.get('kind', 'equity')}",
        "",
    ]
    metrics = 0
    for key in keys:
        value = entry.get(key)
        if value is None or str(value).strip().lower() in _REFERENCE_EMPTY:
            continue
        if isinstance(value, (list, dict)):
            continue
        lines.append(_reference_line(key, value))
        metrics += 1

    if not metrics:
        print(f"[sources] {ticker}: wpis w assets.json bez metryk - pomijam")
        return None

    if entry.get("blurb"):
        lines += ["", f"Description: {entry['blurb']}"]
    lines += [
        "",
        "These figures are static reference data recorded by hand on the date above. "
        "They are a snapshot, not a live feed, and they do not update during the debate.",
    ]

    return SourceDoc(
        url=f"assets://{ticker}",
        title=f"{name} reference data (as of {as_of})",
        published_at=_parse_date(as_of),
        source_type="reference",
        text="\n".join(lines)[:MAX_SOURCE_CHARS],
    )


def render_for_prompt(docs: list[SourceDoc]) -> str:
    """Pakiet zrodel dla promptu. Krotkie id S1, S2... do odwolywania sie."""
    if not docs:
        return "Brak dostepnych zrodel - nie masz na czym oprzec twierdzen faktycznych."

    blocks = []
    for i, doc in enumerate(docs, start=1):
        published = doc.published_at.isoformat() if doc.published_at else "brak"
        blocks.append(
            f"[ZRODLO id=S{i} url={doc.url} data={published} typ={doc.source_type}]\n"
            f"{doc.title}\n\n{doc.text}\n"
            f"[/ZRODLO]"
        )
    return "\n\n".join(blocks)


def normalize_text(text: str) -> str:
    """Normalizacja do porownywania cytatow: biale znaki, cudzyslowy, wielkosc liter."""
    text = text.replace("“", '"').replace("”", '"').replace("„", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


if __name__ == "__main__":
    examples = [
        "https://en.wikipedia.org/wiki/Nvidia",
        "https://www.ft.com/content/0f0b5a4a-0000-0000-0000-000000000000",  # paywall/404
        "https://investor.nvidia.com/financial-info/financial-reports/default.aspx",
    ]

    allowlist = allowlist_for(["filing", "official", "news_open"])
    print(f"allowlist z 3 tierow: {len(allowlist)} domen\n")
    docs = fetch_sources(examples, allowlist=allowlist)

    print(f"\nPobrano {len(docs)} z {len(examples)} dokumentow:")
    for doc in docs:
        print(
            f"- {doc.url}\n"
            f"  typ={doc.source_type} data={doc.published_at} "
            f"dlugosc={len(doc.text)} zn.\n"
            f"  tytul: {doc.title}"
        )

    odrzucone = len(examples) - len(docs)
    print(f"Odrzucono: {odrzucone} (powody wyzej w logach [sources])")

    print("\n--- render_for_prompt (pierwsze 400 znakow) ---")
    print(render_for_prompt(docs)[:400])
