"""Zrodla on-chain z DefiLlama - dla tokenow, ktore nie skladaja raportow do SEC.

Trzy wejscia, wszystkie bez klucza API:
- fetch_protocol_doc(slug)  -> /protocol/{slug}                  (TVL protokolu)
- fetch_chain_doc(chain)    -> /v2/historicalChainTvl/{chain}    (TVL lancucha)
- fetch_fees_doc(slug, ...) -> /summary/fees/{slug}?dataType=... (oplaty i przychod)

Oba zwracaja SourceDoc z source_type="onchain" i TEKSTEM, nie JSON-em: agenci
maja cytowac zdania, a kwant ma dopasowywac liczby przez _reject_groundless_code,
wiec kazda wartosc jest wypisana w pelni (1234567890), bez skrotow typu 1.23B.
Blad sieci albo brak protokolu konczy sie None z logiem - nigdy wyjatkiem.
"""

from datetime import date, datetime, timezone
from typing import Optional

import httpx

from config import MAX_SOURCE_CHARS
from sources import SourceDoc

_BASE = "https://api.llama.fi"
_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
_UA = "ArenaDebateBot/0.1 (+https://github.com/suuuli/orbio)"

# Ile punktow szeregu TVL wchodzi do dokumentu (dzienne, wiec ~miesiac).
SERIES_POINTS = 30


def _get(url: str) -> Optional[object]:
    """GET zwracajacy sparsowany JSON albo None. Nie rzuca."""
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as err:
        print(f"[llama] {err.response.status_code} dla {url} - brak takiego zasobu?")
        return None
    except (httpx.HTTPError, ValueError) as err:
        print(f"[llama] nie udalo sie pobrac {url} ({type(err).__name__})")
        return None


def _money(value: float) -> str:
    """Pelna kwota w dolarach, bez skrotow - zeby dala sie zacytowac i dopasowac."""
    return f"${round(float(value)):,}"


def _day(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).date().isoformat()


def _pct_change(series: list[tuple[float, float]], days_back: int) -> Optional[float]:
    """Zmiana procentowa miedzy ostatnim punktem a punktem days_back wczesniej."""
    if len(series) <= days_back:
        return None
    latest = series[-1][1]
    earlier = series[-1 - days_back][1]
    if not earlier:
        return None
    return (latest - earlier) / earlier * 100


def _series_table(series: list[tuple[float, float]], points: int = SERIES_POINTS) -> list[str]:
    lines = ["", "date        tvl_usd"]
    for stamp, value in series[-points:]:
        lines.append(f"{_day(stamp)}  {round(float(value)):,}")
    return lines


def _parse_cap(raw: Optional[str]) -> Optional[float]:
    """'20.15B' / '636M' / '1.27T' -> liczba. Kreska albo bzdura -> None."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip().upper().replace("$", "").replace(",", "")
    scale = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(text[-1:])
    try:
        return float(text[:-1]) * scale if scale else float(text)
    except ValueError:
        return None


def _pct_line(label: str, change: Optional[float]) -> Optional[str]:
    if change is None:
        return None
    return f"{label}: {change:+.2f}%"


def fetch_protocol_doc(slug: str) -> Optional[SourceDoc]:
    """Dokument on-chain dla protokolu DeFi (TVL, rozbicie po lancuchach, szereg)."""
    url = f"{_BASE}/protocol/{slug}"
    data = _get(url)
    if not isinstance(data, dict) or not data.get("name"):
        print(f"[llama] protokol '{slug}' nie zwrocil danych - pomijam")
        return None

    name = data.get("name", slug)
    symbol = data.get("symbol") or "?"
    category = data.get("category") or "not reported"
    # Protokoly-parenty maja puste "chains"; wtedy lancuchy czytamy z chainTvls.
    # Odsiewamy klucze pochodne ("X-borrowed", "X-staking", "pool2").
    chains: list[str] = data.get("chains") or [
        chain for chain in (data.get("chainTvls") or {}) if "-" not in chain
    ]

    # Szereg globalny: [{date, totalLiquidityUSD}, ...]
    series = [
        (point["date"], point.get("totalLiquidityUSD") or 0.0)
        for point in data.get("tvl") or []
        if isinstance(point, dict) and point.get("date")
    ]
    if not series:
        print(f"[llama] protokol '{slug}' bez szeregu TVL - pomijam")
        return None

    current = series[-1][1]
    as_of = _day(series[-1][0])

    lines = [
        f"{name} ({symbol}) - on-chain metrics from DefiLlama",
        f"Category: {category}",
        f"Chains: {', '.join(chains) if chains else 'not reported'}",
        "",
        f"Total value locked: {_money(current)} as of {as_of}",
    ]
    if data.get("mcap"):
        lines.append(f"Market capitalisation: {_money(data['mcap'])} as of {as_of}")

    # Rozbicie po lancuchach - ostatni punkt szeregu kazdego lancucha.
    for chain, payload in (data.get("chainTvls") or {}).items():
        if "-" in chain:                      # pochodne serie, nie osobny lancuch
            continue
        points = payload.get("tvl") if isinstance(payload, dict) else None
        if not points:
            continue
        last = points[-1]
        if isinstance(last, dict) and last.get("totalLiquidityUSD") is not None:
            lines.append(f"TVL on {chain}: {_money(last['totalLiquidityUSD'])} as of {_day(last['date'])}")

    for line in (
        _pct_line("1-day change", _pct_change(series, 1)),
        _pct_line("7-day change", _pct_change(series, 7)),
        _pct_line("30-day change", _pct_change(series, 30)),
    ):
        if line:
            lines.append(line)

    lines.append("")
    lines.append(f"Daily total value locked, last {min(SERIES_POINTS, len(series))} points:")
    lines += _series_table(series)

    return SourceDoc(
        url=url,
        title=f"{name} on-chain metrics (DefiLlama)",
        published_at=date.today(),
        source_type="onchain",
        text="\n".join(lines)[:MAX_SOURCE_CHARS],
    )


def fetch_chain_doc(chain: str) -> Optional[SourceDoc]:
    """Dokument on-chain dla calego lancucha (historyczne TVL lancucha)."""
    url = f"{_BASE}/v2/historicalChainTvl/{chain}"
    data = _get(url)
    if not isinstance(data, list) or not data:
        print(f"[llama] lancuch '{chain}' nie zwrocil szeregu - pomijam")
        return None

    series = [
        (point["date"], point.get("tvl") or 0.0)
        for point in data
        if isinstance(point, dict) and point.get("date")
    ]
    if not series:
        print(f"[llama] lancuch '{chain}' bez punktow TVL - pomijam")
        return None

    current = series[-1][1]
    as_of = _day(series[-1][0])

    lines = [
        f"{chain} - chain-level on-chain metrics from DefiLlama",
        "Category: blockchain",
        "",
        f"Total value locked on {chain}: {_money(current)} as of {as_of}",
    ]
    for line in (
        _pct_line("7-day change", _pct_change(series, 7)),
        _pct_line("30-day change", _pct_change(series, 30)),
    ):
        if line:
            lines.append(line)

    lines.append("")
    lines.append(f"Daily total value locked, last {min(SERIES_POINTS, len(series))} points:")
    lines += _series_table(series)

    return SourceDoc(
        url=url,
        title=f"{chain} chain TVL (DefiLlama)",
        published_at=date.today(),
        source_type="onchain",
        text="\n".join(lines)[:MAX_SOURCE_CHARS],
    )


def fetch_fees_doc(slug: str, market_cap: Optional[str] = None) -> Optional[SourceDoc]:
    """Oplaty i przychod protokolu - podstawa do sporu o UZYTECZNOSC, nie tylko TVL.

    Dwa strzaly w ten sam endpoint (dataType=dailyFees i dailyRevenue). Protokol
    bez danych o oplatach jest pomijany cicho - to normalne, nie kazdy je raportuje.
    market_cap (napis z assets.json, np. "20.15B") sluzy tylko do policzenia
    mnoznika przychodowego.
    """
    base = f"{_BASE}/summary/fees/{slug}"
    fees = _get(f"{base}?dataType=dailyFees")
    revenue = _get(f"{base}?dataType=dailyRevenue")
    if not isinstance(fees, dict) and not isinstance(revenue, dict):
        return None
    fees = fees if isinstance(fees, dict) else {}
    revenue = revenue if isinstance(revenue, dict) else {}

    name = fees.get("name") or revenue.get("name") or slug
    lines = [
        f"{name} - fees and revenue from DefiLlama",
        "Category: protocol economics",
        "",
    ]

    def block(label: str, payload: dict) -> list[str]:
        rows = []
        for window, key in (("24 hours", "total24h"), ("7 days", "total7d"),
                            ("30 days", "total30d"), ("all time", "totalAllTime")):
            value = payload.get(key)
            if value is not None:
                rows.append(f"{label} over {window}: {_money(value)}")
        change = payload.get("change_1d")
        if change is not None:
            rows.append(f"{label} 1-day change: {float(change):+.2f}%")
        return rows

    lines += block("Fees", fees)
    if fees and revenue:
        lines.append("")
    lines += block("Revenue", revenue)

    # Mnoznik przychodowy: kapitalizacja / przychod roczny (30d x 12).
    cap = _parse_cap(market_cap)
    monthly = revenue.get("total30d")
    if cap and monthly:
        annualised = float(monthly) * 12
        lines += [
            "",
            f"Annualised revenue (30-day revenue x 12): {_money(annualised)}",
            f"Market capitalisation from reference data: {_money(cap)}",
            f"Market cap to annualised revenue: {cap / annualised:.2f}x",
        ]

    if len(lines) <= 4:                       # same naglowki, zero danych
        print(f"[llama] protokol '{slug}' nie raportuje oplat - pomijam")
        return None

    return SourceDoc(
        url=f"{base}?dataType=dailyFees",
        title=f"{name} fees and revenue (DefiLlama)",
        published_at=date.today(),
        source_type="onchain",
        text="\n".join(lines)[:MAX_SOURCE_CHARS],
    )


if __name__ == "__main__":
    import sys

    # Uzycie: python defillama.py [protocol <slug> | chain <name> | fees <slug>]
    kind = sys.argv[1] if len(sys.argv) > 1 else "protocol"
    name = sys.argv[2] if len(sys.argv) > 2 else "hyperliquid"

    if kind == "chain":
        doc = fetch_chain_doc(name)
    elif kind == "fees":
        doc = fetch_fees_doc(name, sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        doc = fetch_protocol_doc(name)
    if doc is None:
        sys.exit(f"brak dokumentu dla {kind} {name}")
    print(f"url: {doc.url}\ntyp: {doc.source_type}\ndata: {doc.published_at}\n")
    print(doc.text[:1400])
    print(f"\n[...] razem {len(doc.text)} znakow")
