import os
from dotenv import load_dotenv

load_dotenv()

# ---- SECRETS (z .env) ----
ORBIO_BASE_URL = os.getenv("ORBIO_BASE_URL")
ORBIO_KEY = os.getenv("ORBIO_KEY")

# ---- MODELE AGENTÓW ----
BULL_MODEL  = "claude-sonnet-4-6"
BEAR_MODEL  = "claude-sonnet-4-6"   # docelowo inny dostawca
QUANT_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-sonnet-4-6"

# ---- PARAMETRY DEBATY ----
ROUNDS = 3
CREDIBILITY_BUDGET = 100
MAX_TOKENS = 2000