"""Pydantic models for the observation kernel.

Two models live here:

- `ModelOutput` — the small JSON object we ask Claude to return for a single
  action. This is what we parse + validate straight off the model's text.
- `Record` — one logged line per agent action per round. Every `Record` becomes
  one JSON object in the run's .jsonl log; analyze.py reads them back.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Roles and kinds are closed sets — Literal gives us free validation.
Role = Literal["artist", "critic"]
Kind = Literal["concept", "evaluation"]


class ModelOutput(BaseModel):
    """What Claude returns for one action, as a small JSON object.

    Artists return `content` (the artwork *concept*) + `reasoning`.
    Critics additionally return `score`. We keep `score` optional so the same
    model validates both roles; run.py is responsible for the role contract.
    """

    content: str
    reasoning: str
    score: Optional[float] = None


class Record(BaseModel):
    """One agent action in one round — the unit we log and later analyze."""

    round: int
    agent: str
    role: Role
    model: str
    kind: Kind
    # Set only on concepts (kind == "concept"); None on evaluations.
    concept_id: Optional[str] = None
    content: str
    # Set only by critics; None on concepts.
    score: Optional[float] = None
    reasoning: str = Field(default="")
