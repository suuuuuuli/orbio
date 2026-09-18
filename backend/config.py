import os
from dotenv import load_dotenv

load_dotenv()

# ---- SECRETS (z .env) ----
ORBIO_BASE_URL = os.getenv("ORBIO_BASE_URL")              # bramka w formacie Anthropic
ORBIO_OPENAI_BASE_URL = os.getenv("ORBIO_OPENAI_BASE_URL")  # ta sama bramka, API OpenAI
ORBIO_KEY = os.getenv("ORBIO_KEY")

# Kod dostepu do uruchamiania debat na zywo. Pusty = bez kodu (tryb lokalny).
# NIGDY nie loguj wartosci - do diagnostyki sluzy sam fakt ustawienia.
ACCESS_CODE = (os.getenv("ACCESS_CODE") or "").strip()

BULL_MODEL  = "google/gemini-3.1-pro-preview"
BEAR_MODEL  = "openai/gpt-5.5"
QUANT_MODEL = "openai/gpt-5.3-codex"
JUDGE_MODEL = "anthropic/claude-fable-5"

MODEL_PROVIDERS = {
    BULL_MODEL:  "openai",
    BEAR_MODEL:  "openai",
    QUANT_MODEL: "openai",
    JUDGE_MODEL: "openai",
}


def provider_for(model: str) -> str:
    """Dostawca dla modelu. Nieznany model konczy sie bledem, nie domyslem."""
    try:
        return MODEL_PROVIDERS[model]
    except KeyError:
        raise KeyError(
            f"model {model!r} nie jest w MODEL_PROVIDERS "
            f"(mam: {sorted(MODEL_PROVIDERS)})"
        ) from None

# ---- PARAMETRY DEBATY ----
ROUNDS = 3
CREDIBILITY_BUDGET = 100
# 2000 bylo za malo: modele o dluzszym stylu (Gemini) ucinaly sie przy trzech
# twierdzeniach, a ponowienie z tym samym limitem powtarzalo to samo obciecie.
MAX_TOKENS = 6000
# Quant zwraca w polu `code` caly skrypt, wiec ma wlasny, wyzszy limit.
QUANT_MAX_TOKENS = 6000

# ---- ZRODLA ----
MAX_SOURCE_CHARS = 15000   # przyciecie tekstu jednego dokumentu
MIN_SOURCE_CHARS = 500     # ponizej tego traktujemy dokument jako okrojony (paywall?)
MAX_QUOTE_CHARS = 300      # maksymalna dlugosc source_quote w twierdzeniu

# ---- WYKONYWANIE KODU KWANTA ----
EXEC_TIMEOUT = 10          # sekundy na jeden skrypt
MAX_RESULT_CHARS = 2000    # przyciecie tego, co wraca do pola result

# ---- ROZLICZENIE WIARYGODNOSCI ----
# Kara za runde bez ani jednego twierdzenia ze zrodlem - nalicza sie TYLKO gdy
# pakiet zrodel w tej rundzie byl niepusty (inaczej karalibysmy agenta za
# ograniczenie systemu, nie za jego wybor).
NO_EVIDENCE_PENALTY = 15
# Premia za twierdzenie factual ze zweryfikowanym cytatem uznane za verified.
# Budzet musi moc rosnac, ale tylko lekko: porazka ma bolec bardziej niz cieszy sukces.
EVIDENCE_BONUS = 1
# Ile luk informacyjnych wolno zglosic jednemu agentowi w jednej rundzie.
MAX_GAPS_PER_ROUND = 1

# ---- LIMITY UZYCIA ----
# Jedna debata to kilkanascie platnych wywolan bramki, wiec liczba uruchomien na
# zywo jest ograniczona na dobe (reset o polnocy UTC). Odtwarzanie zapisow,
# pytania i wtracenia limitu nie maja - nie wolaja modelu albo wolaja go raz.
MAX_LIVE_DEBATES_PER_DAY = 20

# ---- TIERY ZRODEL ----
# Od najbardziej wiarygodnych (raporty skladane pod rygorem prawnym) do
# najmniej dostepnych (media za paywallem - pobranie zwykle sie nie udaje).
# fetch_sources dostaje allowliste zlozona z wybranych tierow:
#   allowlist_for(["filing", "official"])
SOURCE_TIERS = {
    "filing": [
        "sec.gov",
        "data.sec.gov",
        "annualreports.com",
    ],
    "official": [
        "federalreserve.gov",
        "bls.gov",
        "bea.gov",
        "ecb.europa.eu",
        "eurostat.ec.europa.eu",
        "nbp.pl",
        "stat.gov.pl",
        "knf.gov.pl",
        "gpw.pl",
        "nasdaq.com",
        "nyse.com",
    ],
    "news_open": [
        "apnews.com",
        "cnbc.com",
        "marketwatch.com",
        "finance.yahoo.com",
        "investing.com",
        "wikipedia.org",
    ],
    "news_paywalled": [
        "ft.com",
        "wsj.com",
        "bloomberg.com",
        "economist.com",
        "barrons.com",
        "reuters.com",
        "nytimes.com",
        "pb.pl",
        "parkiet.com",
    ],
}

# Domyslny zestaw tierow dla debaty: paywall odpada, bo i tak nie da sie pobrac.
DEFAULT_TIERS = ["filing", "official", "news_open"]
