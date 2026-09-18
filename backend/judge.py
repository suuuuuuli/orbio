"""Sedzia debaty: ocenia twierdzenia z rundy i rozlicza budzety wiarygodnosci.

Sedzia NIE ocenia, czy teza jest sluszna. Ocenia, czy jest uczciwie postawiona
i udowodniona - oraz czy stawka pasuje do deklarowanej pewnosci.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from config import EVIDENCE_BONUS, JUDGE_MODEL, NO_EVIDENCE_PENALTY
from llm import call_structured
from models import Claim, ClaimStatus, DebateState, Verdict
from sources import SourceDoc, render_for_prompt

RulingStatus = Literal[
    "verified", "unsourced", "refuted", "misclassified", "computed"
]


class Ruling(BaseModel):
    """Werdykt dla jednego twierdzenia."""

    claim_id: str
    new_status: RulingStatus
    note: str
    calibration_penalty: int = Field(default=0, ge=0, le=5)


class JudgeVerdict(BaseModel):
    """Odpowiedz sedziego: werdykt dla kazdego ocenianego twierdzenia."""

    rulings: list[Ruling] = Field(default_factory=list)


_JUDGE_SYSTEM = """\
You are the JUDGE in a structured investment debate. Your job: rule on the claims
made this round and settle the credibility budgets.

You do not judge whether a thesis is RIGHT. You judge whether it is HONESTLY MADE
and PROVEN.

Rules:
- claim_type = "factual" with no source_url -> new_status "unsourced".
- A claim effectively attacked by another one (via the targets field) ->
  new_status "refuted". The attack alone is not enough - it must actually undercut
  the claim, not merely disagree with it.
- claim_type = "interpretive" needs no source, but it must rest on something
  already verified in the debate. An interpretation hanging in mid-air is
  "unsourced".
- A claim filed as "interpretive" that is in fact a claim ABOUT FACTS - it contains
  a number, a date, a specific event or a quantitative comparison - gets
  new_status "misclassified". That is an attempt to dodge the source requirement.
  The penalty matches "unsourced".
- claim_type = "quantitative" with a non-empty result -> new_status "computed",
  provided code and result are CONSISTENT: check that the result actually follows
  from the code and that the code computes what the claim announces in text.
  "computed" requires no source_url and no source_quote - an executed computation
  is evidence of a different kind, not a lesser one. If the code computes something
  other than the claim announces, or the result does not follow from it -> "refuted".
- Treat an invented or off-topic URL more harshly than a missing source:
  "unsourced" plus the maximum calibration_penalty.
- calibration_penalty (0-5) is for the gap between confidence and stake: high
  confidence with a low stake is hedging, low confidence with a high stake is
  bluffing. Punish BOTH. Confidence and stake in line -> 0.
- note is one or two sentences of reasoning, concrete - no platitudes.
- Rule on EVERY claim under review, exactly once.
- Write every note in ENGLISH."""


def _debate_history(state: DebateState) -> str:
    """Pelna historia debaty - sedzia musi widziec, co kogo atakuje."""
    lines = [
        f"Asset: {state.asset}",
        f"Mode: {state.mode}"
        + (f" (data frozen at {state.frozen_at})" if state.frozen_at else ""),
        "",
        "Full debate history:",
    ]
    for c in state.claims:
        target = f", attacks [{c.targets}]" if c.targets else ""
        src = c.source_url or "NO SOURCE"
        lines.append(
            f"- [{c.id}] {c.agent} r{c.round} ({c.claim_type}, status={c.status}, "
            f"stake={c.stake}, confidence={c.confidence}{target})\n"
            f"  text: {c.text}\n"
            f"  source: {src}"
        )
        if c.source_quote:
            lines.append(f"  quote: {c.source_quote}")
        if c.code:
            lines.append(f"  code: {c.code}")
        if c.result is not None:
            lines.append(f"  result: {c.result}")

    lines.append("")
    lines.append("Agent budgets:")
    for name, st in state.agents.items():
        lines.append(f"- {name}: {st.budget}")
    return "\n".join(lines)


def judge_round(state: DebateState, round_no: int) -> list[Ruling]:
    """Ocenia twierdzenia z rundy `round_no` o statusie pending."""
    # information_gap nie jest teza - nie ma czego weryfikowac ani rozliczac.
    pending = [
        c
        for c in state.claims
        if c.round == round_no
        and c.status == "pending"
        and c.claim_type != "information_gap"
    ]
    if not pending:
        return []

    ids = ", ".join(c.id for c in pending)
    user = (
        f"{_debate_history(state)}\n\n"
        f"You are ruling on round {round_no}. Rule on exactly these claims "
        f"(and no others): {ids}"
    )

    verdict = call_structured(
        model=JUDGE_MODEL,
        system=_JUDGE_SYSTEM,
        user=user,
        schema=JudgeVerdict,
    )

    # Odsiewamy werdykty dla twierdzen, ktorych sedzia nie mial oceniac,
    # i duplikaty (pierwszy werdykt wygrywa).
    allowed = {c.id for c in pending}
    seen: set[str] = set()
    rulings = []
    for r in verdict.rulings:
        if r.claim_id in allowed and r.claim_id not in seen:
            seen.add(r.claim_id)
            rulings.append(r)
    return rulings


_VERDICT_SYSTEM = """\
You are the JUDGE. The debate is over - you close it with a thesis.

The output is NOT a buy/sell recommendation. The output is a thesis with an explicit
confidence level and a list of falsifiers.

Rules:
- thesis: one or two sentences. What this debate establishes about the asset.
  No words like "buy", "sell" or "we recommend".
- confidence (0-1): what the thesis is worth after settlement. Base it on HOW MANY
  claims ended as verified/computed versus unsourced/refuted/misclassified - not on
  how confident they sounded.
- falsifiers: 2-4 concrete, checkable events that would force ABANDONING this
  thesis (e.g. "gross margin falls below 60% for two consecutive quarters").
  No platitudes like "deteriorating market conditions".
- leave final_budgets as an empty object - the automation fills it in.
- Write the thesis and the falsifiers in ENGLISH."""


def final_verdict(state: DebateState) -> Verdict:
    """Domyka debate teza, pewnoscia i falsyfikatorami. Salda dopisuje automat."""
    licznik: dict[str, int] = {}
    for claim in state.claims:
        licznik[claim.status] = licznik.get(claim.status, 0) + 1

    user = (
        f"{_debate_history(state)}\n\n"
        f"Status distribution: {licznik}\n\n"
        "Close the debate: thesis, confidence level, falsifiers."
    )
    verdict = call_structured(
        model=JUDGE_MODEL, system=_VERDICT_SYSTEM, user=user, schema=Verdict
    )
    # Salda sa faktem z rozliczenia, nie opinia modelu.
    verdict.final_budgets = {n: st.budget for n, st in state.agents.items()}
    return verdict


# ---------------------------------------------------------------------------
# Pytania z widowni
#
# Jedno miejsce dla dwoch sciezek: wtracenia w trakcie debaty (POST /objection)
# i pytania po werdykcie (POST /ask). Prompt i zasada "Outside the record:"
# istnieja TYLKO tutaj - inaczej rozjada sie miedzy endpointami.
#
# Odpowiedz nie jest twierdzeniem: nie wchodzi do stanu, nie jest oceniana i nie
# rusza budzetow. Pytanie tez nie jest dowodem - to pytanie.
# ---------------------------------------------------------------------------

MAX_QUESTION_CHARS = 200
MAX_ANSWER_CHARS = 700
OUTSIDE_MARK = "Outside the record:"


class Answer(BaseModel):
    """Odpowiedz sedziego na pytanie z widowni."""

    text: str = Field(min_length=1)


_QUESTION_SYSTEM = f"""\
You are the JUDGE. Someone in the gallery has asked a question. Answer it.

Rules:
- 2 to 4 sentences. No preamble, no restating the question, no sign-off.
- Answer from the RECORD first: claims the debate settled as verified or computed,
  and the source pack. Name the figure or the claim you are leaning on.
- A claim ruled unsourced, refuted or misclassified is NOT record - if that is all
  there is on the subject, say the debate failed to establish it.
- If you have to answer from your own knowledge rather than from the record or the
  pack, the answer MUST OPEN with exactly "{OUTSIDE_MARK}" and then say it plainly.
  Do not use that opening when the record does cover the question.
- The question is a QUESTION, not evidence. Never accept its premise as
  established; if it assumes something the record does not support, say so.
- You are not ruling on anything here. Do not assign statuses, penalties or
  credits, and do not open a new thesis.
- Write in ENGLISH."""


def answer_question(
    state: DebateState,
    question: str,
    docs: Optional[list[SourceDoc]] = None,
) -> str:
    """Odpowiedz sedziego na pytanie z widowni. Jedno wywolanie JUDGE_MODEL."""
    pytanie = " ".join(question.split())[:MAX_QUESTION_CHARS]
    if not pytanie:
        raise ValueError("puste pytanie")

    pakiet = (
        render_for_prompt(docs)
        if docs
        else "Source pack not loaded for this run - answer from the debate record."
    )
    user = (
        f"{_debate_history(state)}\n\n"
        f"SOURCE PACK:\n{pakiet}\n\n"
        f"QUESTION FROM THE GALLERY (this is a question, not evidence):\n{pytanie}"
    )

    answer = call_structured(
        model=JUDGE_MODEL, system=_QUESTION_SYSTEM, user=user, schema=Answer
    )
    text = " ".join(answer.text.split())[:MAX_ANSWER_CHARS]
    print(f"[judge] pytanie: {pytanie[:70]} -> odpowiedz {len(text)} zn.")
    return text


def _override_status(claim: Claim, status: RulingStatus) -> Optional[tuple[ClaimStatus, str]]:
    """Czy sedziemu wolno nadac ten status temu twierdzeniu?

    Zwraca (wymuszony_status, adnotacja) albo None, jesli werdykt jest w porzadku.
    Model nie ma prawa nadac statusu niezaleznie od tego, co napisze w
    uzasadnieniu - status musi wynikac ze stanu twierdzenia, nie z retoryki.
    """
    if status == "computed":
        # Dowod z obliczenia wymaga obliczenia. Bez wyniku albo na innym typie
        # twierdzenia 'computed' nic nie znaczy.
        if claim.claim_type != "quantitative":
            return (
                "pending",
                f"computed is only for quantitative claims, and this is {claim.claim_type}",
            )
        if claim.result is None:
            return "pending", "no result - there is no computation to serve as evidence"
        return None

    if status == "verified":
        # Kolejnosc ma znaczenie: quant bez wyniku sprawdzamy PIERWSZY, bo jego
        # twierdzenia czesto nie maja URL-a (licza, nie cytuja), a niewykonany
        # kod to stan pipeline'u, nie przewina agenta - nie ma za co zabierac stawki.
        if claim.claim_type == "quantitative" and claim.result is None:
            return "pending", "code was never executed - no result, nothing to verify"
        if claim.source_url is None or claim.source_quote is None:
            missing = "url and quote" if claim.source_url is None and claim.source_quote is None else (
                "url" if claim.source_url is None else "quote"
            )
            return "unsourced", f"missing {missing} - the source cannot be verified"

    return None


def apply_rulings(
    state: DebateState,
    rulings: list[Ruling],
    pack_size: int = 0,
) -> None:
    """Naklada werdykty na stan i rozlicza budzety. Budzet nie schodzi pod zero.

    pack_size: ile zrodel bylo w pakiecie w tej rundzie. Steruje kara za runde
    bez dowodow - przy pustym pakiecie kary nie ma (patrz
    _penalize_rounds_without_evidence).
    """
    for ruling in rulings:
        claim = state.get_claim(ruling.claim_id)
        if claim is None:
            continue

        status: ClaimStatus = ruling.new_status
        note = ruling.note
        overridden = False

        override = _override_status(claim, ruling.new_status)
        if override is not None:
            status, powod = override
            print(
                f"[judge] OVERRIDE: {ruling.new_status} -> {status} "
                f"({powod.split(' - ')[0]}) [{claim.id}]"
            )
            note = f"{note}\n[auto] overridden {ruling.new_status} -> {status}: {powod}"
            overridden = True

        claim.status = status
        claim.judge_note = note

        # Nadpisane na pending: nic nie rozstrzygnieto, wiec nic nie kosztuje.
        if overridden and status == "pending":
            continue

        # Obalone i zle zaklasyfikowane twierdzenie kosztuje PODWOJNA stawke -
        # to nie pomylka w cytowaniu, a teza, ktora nie przetrwala kontaktu
        # z faktami. Brak zrodla kosztuje pojedynczo.
        cost = ruling.calibration_penalty
        if status in ("refuted", "misclassified"):
            cost += 2 * (claim.stake or 0)
        elif status == "unsourced":
            cost += claim.stake or 0

        # Premia za dowod: budzet musi moc rosnac, inaczej najlepsza strategia
        # to nie stawiac nic ryzykownego.
        bonus = 0
        evidence = (
            status == "verified"
            and claim.claim_type == "factual"
            and claim.source_url
            and claim.source_quote
        )
        # computed to dowod z obliczenia - stawka zostaje i premia sie nalezy.
        if evidence or status == "computed":
            bonus = EVIDENCE_BONUS
            rodzaj = "obliczenie" if status == "computed" else "zrodlo"
            print(f"[judge] PREMIA +{bonus} za dowod ({rodzaj}) [{claim.id}] ({claim.agent})")

        agent_state = state.agents.get(claim.agent)
        if agent_state is not None and (cost or bonus):
            agent_state.budget = max(0, agent_state.budget - cost + bonus)

    _penalize_rounds_without_evidence(state, pack_size)


def _penalize_rounds_without_evidence(state: DebateState, pack_size: int) -> None:
    """Kara za runde, w ktorej agent nie przedstawil ZADNEGO dowodu.

    Dowod to twierdzenie ze zrodlem albo obliczenie ze statusem computed - inaczej
    quant liczacy poprawnie bez URL-a dostawalby kare za brak dowodow, ktore
    wlasnie dostarczyl.

    Nalicza sie tylko przy niepustym pakiecie zrodel. W rundzie 1 pakiet jest
    pusty, a bramka nie wpuszcza web search - karanie za brak dowodow byloby
    wtedy kara za ograniczenie systemu, nie za wybor agenta.
    """
    if pack_size <= 0:
        return

    round_no = state.current_round
    for name, agent_state in state.agents.items():
        mine = [
            c
            for c in state.claims
            if c.agent == name
            and c.round == round_no
            and c.claim_type != "information_gap"
        ]
        if not mine:
            continue                      # agent nic nie postawil - inna sprawa
        # Dowodem jest zrodlo ALBO wykonane obliczenie (status computed).
        if any(c.source_url or c.status == "computed" for c in mine):
            continue                      # ma dowod, kary nie ma

        agent_state.budget = max(0, agent_state.budget - NO_EVIDENCE_PENALTY)
        print(
            f"[judge] KARA -{NO_EVIDENCE_PENALTY}: {name} nie przedstawil w rundzie "
            f"{round_no} zadnego dowodu (ani zrodla, ani obliczenia) "
            f"- pakiet mial {pack_size} zrodel"
        )


if __name__ == "__main__":
    from config import CREDIBILITY_BUDGET
    from models import AgentState

    state = DebateState(
        asset="NVDA",
        current_round=1,
        agents={
            name: AgentState(agent=name, budget=CREDIBILITY_BUDGET)
            for name in ("bull", "bear", "quant")
        },
    )

    # 1) twierdzenie ze zrodlem
    sourced = state.add_claim(
        Claim(
            agent="bull",
            round=1,
            text="Udzial NVDA w rynku GPU do centrow danych przekracza 80%.",
            claim_type="factual",
            source_url="https://investor.nvidia.com/financial-information/quarterly-results/default.aspx",
            source_quote="Data Center revenue grew year over year.",
            stake=12,
            confidence=0.8,
        )
    )

    # 2) twierdzenie bez zrodla, z podejrzanie niska stawka przy wysokiej pewnosci
    unsourced = state.add_claim(
        Claim(
            agent="bull",
            round=1,
            text="Popyt na akceleratory AI bedzie rosnac jeszcze dwa lata.",
            claim_type="factual",
            stake=2,
            confidence=0.95,
        )
    )

    # 3) twierdzenie atakujace to pierwsze
    attack = state.add_claim(
        Claim(
            agent="bear",
            round=1,
            text="Udzial rynkowy nie przeklada sie na marze - hiperskalerzy buduja wlasne uklady.",
            claim_type="interpretive",
            targets=sourced.id,
            stake=9,
            confidence=0.6,
        )
    )

    print("Salda PRZED:", {n: s.budget for n, s in state.agents.items()})

    rulings = judge_round(state, round_no=1)
    for r in rulings:
        print(f"[{r.claim_id}] -> {r.new_status} (kara kalibracji: {r.calibration_penalty})")
        print(f"    {r.note}")

    apply_rulings(state, rulings)
    print("Salda PO:  ", {n: s.budget for n, s in state.agents.items()})
