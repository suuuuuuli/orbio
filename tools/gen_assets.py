"""Jednorazowe generowanie assetow do areny. Nie czesc aplikacji."""

import base64
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---- CONFIG ----
BASE_URL = "https://api.orbio.so/api/v1/images"
KEY = os.getenv("ORBIO_KEY")
OUT_DIR = ROOT / "frontend" / "assets"
SCENE_MODEL = "black-forest-labs/flux.2-pro"
CHAR_MODEL = "openai/gpt-image-2"
# Parametry wysylane tylko wtedy, gdy sa w danym zadaniu - modele odrzucaja
# nieobslugiwane klucze calym zapytaniem.
OPTIONAL = ("aspect_ratio", "background", "resolution", "quality",
            "output_format", "seed")
# ----------------

STYLE = (
    "16-bit pixel art sprite, limited 16-color palette, crisp pixels, "
    "no anti-aliasing, side-facing fighting game character portrait, "
    "dark background"
)

# Postacie na plaskim szarym tle - przezroczystosc wycinamy lokalnie (rembg),
# bo gpt-image-2 nie wspiera background=transparent.
FLAT_BG = ", plain flat dark grey background, no scenery, no props"
PIXEL = (
    "16-bit pixel art sprite, limited 16-color palette, crisp pixels, "
    "no anti-aliasing, fighting game character portrait, "
    "plain flat dark grey background"
    )

JOBS = {
    
    "courtroom": dict(
    model=CHAR_MODEL,
    aspect_ratio="16:9",
    resolution="1K",
    prompt=(
        "16-bit pixel art background, empty courtroom interior, dark wood "
        "panelling, tall arched windows with dim evening light, elevated "
        "judge's bench in the centre, two lecterns facing each other, "
        "wooden floor, limited 16-color palette, crisp pixels, "
        "no anti-aliasing, side-scrolling game background, no characters, "
        "no people, no text, no signage"
    ),
),
    "bull": dict(
        model=CHAR_MODEL,
        aspect_ratio="1:1",
        prompt=(
            "Anthropomorphic bull in a tailored charcoal three-piece suit, "
            "confident posture, chest-up portrait, facing slightly right, "
            + STYLE + FLAT_BG
        ),
    ),
    "bull-win": dict(model=CHAR_MODEL, aspect_ratio="1:1", resolution="1K",
    prompt="Anthropomorphic bull in a tailored charcoal three-piece suit, "
           "triumphant confident grin, chest raised, head high, chest-up "
           "portrait, " + PIXEL),
           "bull-lose": dict(model=CHAR_MODEL, aspect_ratio="1:1", resolution="1K",
    prompt="Anthropomorphic bull in a rumpled charcoal suit, loosened tie, "
           "dejected slumped shoulders, head lowered, defeated expression, "
           "chest-up portrait, " + PIXEL),
"bear-win": dict(model=CHAR_MODEL, aspect_ratio="1:1", resolution="1K",
    prompt="Anthropomorphic brown bear in a dark green suit, smug satisfied "
           "smirk, chest raised, head high, chest-up portrait, " + PIXEL),
"bear-lose": dict(model=CHAR_MODEL, aspect_ratio="1:1", resolution="1K",
    prompt="Anthropomorphic brown bear in a rumpled dark green suit, "
           "loosened tie, dejected slumped shoulders, head lowered, "
           "defeated expression, chest-up portrait, " + PIXEL),
    "bear": dict(
        model=CHAR_MODEL,
        aspect_ratio="1:1",
        prompt=(
            "Anthropomorphic brown bear in a dark green suit, skeptical "
            "expression, chest-up portrait, facing slightly left, "
            + STYLE + FLAT_BG
        ),
    ),
    "judge": dict(
        model=CHAR_MODEL,
        aspect_ratio="1:1",
        prompt=(
            "Anthropomorphic owl in black judicial robes with white collar, "
            "small round spectacles, stern impartial expression, chest-up "
            "portrait, facing viewer, " + STYLE + FLAT_BG
        ),
    ),
    
}


def generate(name: str, variant: int = 1) -> None:
    job = JOBS[name]
    body = {"model": job["model"], "prompt": job["prompt"]}
    for key in OPTIONAL:
        if key in job:
            body[key] = job[key]

    print(f"[{name}] generuje wariant {variant} ({job['model']})...")
    try:
        r = httpx.post(
            BASE_URL, json=body,
            headers={"Authorization": f"Bearer {KEY}"}, timeout=300,
        )
    except httpx.HTTPError as err:
        print(f"[{name}] blad sieci: {err}")
        return

    if r.status_code != 200:
        print(f"[{name}] {r.status_code}: {r.text[:400]}")
        return

    payload = r.json()
    items = payload.get("data") or []
    if not items or not items[0].get("b64_json"):
        print(f"[{name}] odpowiedz bez obrazu: {str(payload)[:300]}")
        return

    media = items[0].get("media_type", "image/png")
    ext = {"image/png": "png", "image/jpeg": "jpg",
           "image/webp": "webp", "image/svg+xml": "svg"}.get(media, "png")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}-v{variant}.{ext}"
    path.write_bytes(base64.b64decode(items[0]["b64_json"]))
    cost = payload.get("usage", {}).get("cost")
    print(f"[{name}] zapisano {path.name} (koszt: {cost})")


if __name__ == "__main__":
    if not KEY:
        sys.exit("ORBIO_KEY nie jest w .env")

    args = sys.argv[1:]
    variant = 1
    if args and args[-1].isdigit():
        variant = int(args.pop())
    names = args or list(JOBS)

    unknown = [n for n in names if n not in JOBS]
    if unknown:
        sys.exit(f"nieznane zadania: {unknown} (mam: {list(JOBS)})")

    for n in names:
        generate(n, variant)