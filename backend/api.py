"""FastAPI: strumien przebiegu debaty (SSE) + odtwarzanie zapisanych przebiegow.

POST /debate       - uruchamia debate i STREAMUJE zdarzenia w trakcie
POST /objection    - wtracenie z widowni do TRWAJACEJ debaty
POST /ask          - pytanie do zakonczonego (albo odtworzonego) przebiegu
GET  /assets       - dane referencyjne aktywow z data/assets.json
GET  /replay/{name}- odtwarza zapisany przebieg z data/replays/ ze opoznieniem
GET  /replays      - lista dostepnych przebiegow
GET  /             - frontend (frontend/index.html)

Debata jest synchroniczna i dluga (kilkadziesiat sekund na runde), wiec leci
w osobnym watku, a zdarzenia wracaja przez kolejke. Kazdy przebieg na zywo jest
zapisywany do data/replays/, zeby dalo sie go potem odtworzyc bez bramki -
demo na zywo nie moze zalezec od tego, czy bramka odpowie.
"""

import asyncio
import json
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import CREDIBILITY_BUDGET, ROUNDS
from debate import run_debate
from judge import MAX_QUESTION_CHARS, answer_question
from models import AgentState, AssetKind, Claim, DebateState
from sources import SourceDoc

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
FRONTEND = FRONTEND_DIR / "index.html"
ASSETS = FRONTEND_DIR / "assets"
REPLAY_DIR = ROOT / "data" / "replays"
ASSET_FILE = ROOT / "data" / "assets.json"   # hand-entered reference data

# Opoznienie przy odtwarzaniu - inaczej cala debata wyskakuje w jednej klatce.
# Claims get 2s: the viewer has to be able to read one before the next lands.
# Live debates are paced by the gateway, not by this table.
REPLAY_DELAYS = {
    "round_start": 0.6,
    "claim": 2.0,
    "ruling": 0.9,
    "sources_added": 0.5,
    "objection": 1.5,          # tyle trwa przerywnik OBJECTION! we frontendzie
    "objection_answer": 1.0,
    "verdict": 1.2,
    "done": 0.0,
}
DEFAULT_REPLAY_DELAY = 0.4

# Trwajaca debata (jedna na proces - to demo, nie hosting) i jej kolejka pytan.
_ACTIVE: dict = {}

# Przebiegi, z ktorymi mozna jeszcze rozmawiac po werdykcie: id -> (stan, pakiet).
# Trzymane w pamieci, wiec restart serwera je gubi - pytania po werdykcie sa
# rozmowa o przebiegu, nie danymi do przechowania.
_FINISHED: dict[str, tuple[DebateState, list[SourceDoc]]] = {}
_FINISHED_LIMIT = 12

app = FastAPI(title="Arena - debata inwestycyjna")

# Grafika sali i portrety agentow. Bez tego /assets/*.png zwraca 404.
if ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
else:
    print(f"[api] UWAGA: brak katalogu {ASSETS} - frontend poleci bez grafiki")


class QuestionRequest(BaseModel):
    """Pytanie z widowni - i tylko pytanie, nigdy dowod (patrz judge.py)."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    run: Optional[str] = None       # id przebiegu; brak = ostatni zakonczony


class DebateRequest(BaseModel):
    asset: str = Field(min_length=1, max_length=40)
    # "token" wylacza EDGAR - jego mapa tickerow trafia wtedy w emitenta ETF-u.
    asset_kind: AssetKind = "equity"
    rounds: int = Field(default=ROUNDS, ge=1, le=6)
    seed_urls: list[str] = Field(default_factory=list)


def _remember(run_id: str, state: DebateState, pack: list[SourceDoc]) -> None:
    """Zapamietuje przebieg do pytan po werdykcie, trzymajac tylko ostatnie."""
    _FINISHED[run_id] = (state, list(pack))
    for old in list(_FINISHED)[:-_FINISHED_LIMIT]:
        _FINISHED.pop(old, None)


def _state_from_events(asset: str, events: list[dict]) -> DebateState:
    """Odtwarza stan debaty z zapisanych zdarzen - na potrzeby pytan do replaya.

    Z zapisu wracaja twierdzenia, ich statusy po werdykcie sedziego i salda.
    Tresci zrodel w zapisie nie ma (celowo - to megabajty), wiec sedzia
    odpowiada na pytania z samego rekordu debaty.
    """
    state = DebateState(
        asset=asset,
        agents={
            name: AgentState(agent=name, budget=CREDIBILITY_BUDGET)
            for name in ("bull", "bear", "quant")
        },
    )
    for event in events:
        kind = event.get("type")
        if kind == "claim":
            try:
                state.add_claim(Claim.model_validate(event["claim"]))
            except Exception:            # stary zapis z innym schematem
                continue
        elif kind == "ruling":
            claim = state.get_claim(event.get("ruling", {}).get("claim_id", ""))
            if claim is not None:
                claim.status = event.get("status", claim.status)
                claim.judge_note = event.get("judge_note") or claim.judge_note
            for name, budget in (event.get("budgets") or {}).items():
                if name in state.agents:
                    state.agents[name].budget = budget
        elif kind == "round_start":
            state.current_round = event.get("round", state.current_round)
    return state


def _sse(event: dict) -> str:
    """Jedno zdarzenie w formacie Server-Sent Events."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _save_replay(asset: str, events: list[dict], run_id: Optional[str] = None) -> Path:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Nazwa pliku = id przebiegu, wiec POST /ask po odtworzeniu trafia w ten sam
    # identyfikator, ktory frontend dostal w zdarzeniu "done".
    name = run_id or f"{asset.lower()}-{stamp}"
    path = REPLAY_DIR / f"{name}.json"
    path.write_text(
        json.dumps({"asset": asset, "recorded_at": stamp, "events": events},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


@app.get("/")
def index() -> FileResponse:
    if not FRONTEND.exists():
        raise HTTPException(status_code=404, detail="brak frontend/index.html")
    return FileResponse(FRONTEND)


@app.get("/assets")
def assets() -> dict:
    """Reference data for the ticker picker and the asset panel.

    Hand-maintained in data/assets.json. A missing or broken file is not an
    error - the frontend then only offers the custom ticker field.
    """
    if not ASSET_FILE.exists():
        print(f"[api] brak {ASSET_FILE.name} - frontend poleci bez danych referencyjnych")
        return {}
    try:
        return json.loads(ASSET_FILE.read_text(encoding="utf-8"))
    except ValueError as err:
        print(f"[api] {ASSET_FILE.name} jest uszkodzony ({err}) - oddaje pusty obiekt")
        return {}


@app.get("/replays")
def replays() -> dict:
    """Lista zapisanych przebiegow - nazwa bez .json trafia do /replay/{name}."""
    if not REPLAY_DIR.exists():
        return {"replays": []}
    return {
        "replays": sorted(p.stem for p in REPLAY_DIR.glob("*.json"))
    }


@app.post("/debate")
async def debate(request: DebateRequest) -> StreamingResponse:
    """Uruchamia debate i streamuje zdarzenia W TRAKCIE jej trwania."""

    run_id = f"{request.asset.lower()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    async def stream() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        zapis: list[dict] = []

        # Pytania z widowni wrzuca tu POST /objection, a run_debate oprznia je
        # po kazdej rundzie. Kolejka z modulu queue, bo czyta ja watek debaty.
        objections: "queue.Queue[str]" = queue.Queue()
        pack: list[SourceDoc] = []
        _ACTIVE.clear()
        _ACTIVE.update({"run": run_id, "objections": objections, "asset": request.asset})

        def on_event(event: dict) -> None:
            # Wolane z watku debaty - do petli asyncio wchodzimy bezpiecznie.
            zapis.append(event)
            loop.call_soon_threadsafe(events.put_nowait, event)

        def worker() -> None:
            try:
                state, _ = run_debate(
                    asset=request.asset,
                    asset_kind=request.asset_kind,
                    rounds=request.rounds,
                    seed_urls=request.seed_urls,
                    verbose=True,
                    on_event=on_event,
                    objections=objections,
                    pack=pack,
                )
                _remember(run_id, state, pack)
                koniec = {"type": "done", "asset": request.asset, "run": run_id}
            except Exception as err:      # debata nie moze wysadzic serwera
                koniec = {
                    "type": "done",
                    "asset": request.asset,
                    "run": run_id,
                    "error": f"{type(err).__name__}: {err}",
                }
            zapis.append(koniec)
            loop.call_soon_threadsafe(events.put_nowait, koniec)

        threading.Thread(target=worker, name="debate", daemon=True).start()

        try:
            while True:
                event = await events.get()
                yield _sse(event)
                if event.get("type") == "done":
                    break
        finally:
            # Rozlaczenie widza zamyka ksiege wtracen - nie ma komu pokazac
            # odpowiedzi, a kazde pytanie to osobne, platne wywolanie modelu.
            if _ACTIVE.get("run") == run_id:
                _ACTIVE.clear()

        try:
            path = _save_replay(request.asset, zapis, run_id)
            print(f"[api] przebieg zapisany: {path.name}")
        except OSError as err:
            print(f"[api] nie udalo sie zapisac przebiegu: {err}")

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/objection")
def objection(request: QuestionRequest) -> dict:
    """Wtracenie z widowni do TRWAJACEJ debaty.

    Pytanie laduje w kolejce; sedzia odpowie po biezacej rundzie. Odpowiedz
    wraca strumieniem debaty (zdarzenia objection i objection_answer), nie tym
    endpointem - inaczej pytanie blokowaloby zapytanie HTTP na kilkanascie sekund.
    """
    objections = _ACTIVE.get("objections")
    if objections is None:
        raise HTTPException(status_code=409, detail="no debate is running")

    question = " ".join(request.question.split())
    objections.put(question)
    print(f"[api] wtracenie w kolejce ({objections.qsize()}): {question[:80]}")
    return {"queued": objections.qsize(), "run": _ACTIVE.get("run")}


@app.post("/ask")
def ask(request: QuestionRequest) -> dict:
    """Pytanie do zakonczonego przebiegu. Odpowiada ten sam sedzia, tym samym
    promptem co wtracenia (judge.answer_question).

    run to id z zdarzenia "done" albo nazwa zapisanego przebiegu. Brak run =
    ostatni zakonczony przebieg w tym procesie.
    """
    run_id = request.run or (next(reversed(_FINISHED)) if _FINISHED else None)
    if run_id is None:
        raise HTTPException(status_code=409, detail="no finished run to ask about")

    entry = _FINISHED.get(run_id)
    if entry is None:
        # Przebieg nie jest w pamieci (np. odtworzony zapis albo restart
        # serwera) - odtwarzamy stan z zapisu na dysku.
        path = REPLAY_DIR / f"{run_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as err:
            raise HTTPException(status_code=500, detail=f"broken replay: {err}") from None
        state = _state_from_events(payload.get("asset", run_id), payload.get("events", []))
        entry = (state, [])
        _remember(run_id, *entry)

    state, pack = entry
    try:
        answer = answer_question(state, request.question, pack)
    except Exception as err:
        raise HTTPException(status_code=502, detail=f"{type(err).__name__}: {err}") from None
    return {"run": run_id, "question": request.question, "answer": answer}


@app.get("/replay/{name}")
async def replay(name: str, speed: float = 1.0) -> StreamingResponse:
    """Odtwarza zapisany przebieg w tym samym formacie zdarzen.

    speed > 1 przyspiesza (speed=2 to dwa razy szybciej). Nazwa jest walidowana
    - do katalogu replayow wchodzimy tylko po nazwie pliku, bez sciezek.
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="niedozwolona nazwa przebiegu")

    path = REPLAY_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"nie ma przebiegu {name}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as err:
        raise HTTPException(status_code=500, detail=f"uszkodzony przebieg: {err}") from None

    events: list[dict] = payload.get("events", [])
    mnoznik = 1.0 / max(speed, 0.1)

    async def stream() -> AsyncIterator[str]:
        for event in events:
            await asyncio.sleep(REPLAY_DELAYS.get(event.get("type"), DEFAULT_REPLAY_DELAY) * mnoznik)
            yield _sse(event)
        if not events or events[-1].get("type") != "done":
            yield _sse({"type": "done", "asset": payload.get("asset", "?")})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
