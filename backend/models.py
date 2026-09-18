"""Modele danych areny debaty.

Zrodlo prawdy dla schematu twierdzenia (Claim). Reszta backendu (agents, judge,
debate) oraz frontend polegaja na tych polach - nie zmieniaj nazw bez powodu.
"""

from datetime import date
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

# ---- TYPY DOMENOWE ----
AgentName = Literal["bull", "bear", "quant", "judge"]
# information_gap to nie teza, a zgloszenie brakujacej danej: nic nie twierdzi,
# wiec nie ma stake ani source_url. Debata zbiera te luki po rundzie i probuje
# dociagnac zrodla, ktore je zasypia (patrz debate.py).
ClaimType = Literal["factual", "interpretive", "quantitative", "information_gap"]
# misclassified: zgloszone jako interpretive, a w istocie jest twierdzeniem
# o faktach (liczba, data, zdarzenie, porownanie ilosciowe). Karane jak unsourced.
ClaimStatus = Literal[
    "pending", "verified", "unsourced", "refuted", "computed", "misclassified"
]
DebateMode = Literal["live", "historical"]
# Rodzaj aktywa decyduje, ktore zrodla maja sens: raporty do SEC sklada emitent
# akcji, a nie token (patrz edgar_filings w debate.py).
AssetKind = Literal["equity", "token"]


def new_id() -> str:
    """Krotkie id (8 znakow) - czytelne w logach i w UI areny."""
    return uuid4().hex[:8]


class Claim(BaseModel):
    """Jedno twierdzenie postawione przez agenta w danej rundzie."""

    id: str = Field(default_factory=new_id)
    agent: AgentName
    round: int = Field(ge=0)
    text: str

    claim_type: ClaimType
    source_url: Optional[str] = None
    source_quote: Optional[str] = None

    # stake jest celowo oddzielone od confidence - sedzia karze rozjazd miedzy
    # deklarowana pewnoscia a gotowoscia postawienia punktow.
    # Dla information_gap stake jest None - nie ma czego obstawiac.
    stake: Optional[int] = Field(default=None, ge=1, le=20)
    confidence: float = Field(ge=0.0, le=1.0)

    # id twierdzenia, ktore to twierdzenie atakuje (None = teza samodzielna).
    # To pole zamienia plaska liste w graf, ktory rysuje arena.
    targets: Optional[str] = None

    status: ClaimStatus = "pending"
    judge_note: Optional[str] = None

    # Tylko quant: kod wykonany w sandboxie i jego wynik.
    code: Optional[str] = None
    result: Optional[str] = None

    @model_validator(mode="after")
    def _check_stake(self) -> "Claim":
        """information_gap: bez stawki i bez zrodla. Pozostale typy: stawka wymagana."""
        if self.claim_type == "information_gap":
            self.stake = None
            self.source_url = None
            self.source_quote = None
        elif self.stake is None:
            raise ValueError(
                f"claim_type={self.claim_type} wymaga stake (1-20); "
                "bez stawki jest tylko information_gap"
            )
        return self


class AgentState(BaseModel):
    """Stan budzetu wiarygodnosci jednego agenta."""

    agent: AgentName
    budget: int
    claim_ids: list[str] = Field(default_factory=list)


class DebateState(BaseModel):
    """Pelny stan debaty - to jest serializowane do frontendu."""

    asset: str                      # ticker giełdowy albo token, np. "NVDA"
    asset_kind: AssetKind = "equity"
    mode: DebateMode = "live"
    frozen_at: Optional[date] = None  # data zamrozenia dla trybu historycznego
    current_round: int = 0
    claims: list[Claim] = Field(default_factory=list)
    agents: dict[str, AgentState] = Field(default_factory=dict)

    # ---- pomocnicze ----
    def get_claim(self, claim_id: str) -> Optional[Claim]:
        for claim in self.claims:
            if claim.id == claim_id:
                return claim
        return None

    def add_claim(self, claim: Claim) -> Claim:
        """Dopisuje twierdzenie i rejestruje je w stanie autora."""
        self.claims.append(claim)
        state = self.agents.get(claim.agent)
        if state is not None:
            state.claim_ids.append(claim.id)
        return claim

    def refute(self, claim_id: str, by_claim_id: str) -> Optional[Claim]:
        """Oznacza twierdzenie jako obalone i zabiera stawke jego autorowi.

        Zwraca obalone twierdzenie albo None, jesli id nie istnieje.
        Twierdzenie juz obalone nie jest karane po raz drugi.
        """
        claim = self.get_claim(claim_id)
        if claim is None:
            return None
        if claim.status == "refuted":
            return claim

        claim.status = "refuted"
        claim.judge_note = f"obalone przez {by_claim_id}"

        state = self.agents.get(claim.agent)
        if state is not None:
            state.budget = max(0, state.budget - claim.stake)
        return claim


class Verdict(BaseModel):
    """Wyjscie debaty: teza + pewnosc + falsyfikatory. NIE rekomendacja."""

    thesis: str
    confidence: float = Field(ge=0.0, le=1.0)
    falsifiers: list[str] = Field(default_factory=list)
    final_budgets: dict[str, int] = Field(default_factory=dict)


if __name__ == "__main__":
    state = DebateState(
        asset="NVDA",
        mode="historical",
        frozen_at=date(2025, 6, 30),
        current_round=1,
        agents={
            "bull": AgentState(agent="bull", budget=100),
            "bear": AgentState(agent="bear", budget=100),
        },
    )

    bull_claim = state.add_claim(
        Claim(
            agent="bull",
            round=1,
            text="Popyt na akceleratory AI rosnie szybciej niz podaz.",
            claim_type="factual",
            source_url="https://example.com/raport",
            source_quote="Zamowienia przekraczaja moce produkcyjne.",
            stake=8,
            confidence=0.7,
        )
    )

    bear_claim = state.add_claim(
        Claim(
            agent="bear",
            round=1,
            text="Kontrakty nie sa potwierdzone w raporcie kwartalnym.",
            claim_type="interpretive",
            targets=bull_claim.id,
            stake=5,
            confidence=0.55,
        )
    )

    state.refute(bull_claim.id, by_claim_id=bear_claim.id)

    verdict = Verdict(
        thesis="Teza wzrostowa trzyma sie tylko przy utrzymaniu marzy.",
        confidence=0.45,
        falsifiers=["Marza brutto spada pod 60%", "Odwolane zamowienia w kolejnym kwartale"],
        final_budgets={name: st.budget for name, st in state.agents.items()},
    )

    print(state.model_dump_json(indent=2))
    print(verdict.model_dump_json(indent=2))
