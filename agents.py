"""The Agent and the prompt machinery around it.

Read this file top-to-bottom: the shared system prompt, then the feed
formatter, then the JSON extractor, then the Agent itself. `Agent.act` is the
whole flow for one agent in one round — build prompt, call Claude, parse, return
a Record.
"""

import json
import re
from typing import Optional

from anthropic import AsyncAnthropic

from schema import ModelOutput, Record

# The single model used by every agent. Multi-model is a later version; for now
# this is the one knob to change the model everywhere.
MODEL = "claude-sonnet-4-6"


# --- The shared system prompt -------------------------------------------------
# Edit this to change the framing given to *every* agent. Each agent's own
# disposition (from agents.yaml) is appended after this, then the feed.
SHARED_SYSTEM_PROMPT = """\
You are one agent in a small studio of artists and critics who work only in
language. Artists invent short *concepts* for artworks — vivid text prompts,
never images. Critics evaluate those concepts in writing.

Always reply with a single JSON object and nothing else. Use these keys:
  - "content":   your contribution as a string (an artwork concept, or an evaluation).
  - "reasoning": one or two sentences on why, as a string.
  - "score":     critics only — a number from 0.0 to 1.0 rating the concept(s). Omit if you are an artist.

Do not wrap the JSON in markdown fences. Do not add commentary outside the JSON.
"""


# --- Feed formatting ----------------------------------------------------------
def format_feed(feed: list[Record]) -> str:
    """Render the shared feed (all prior records) into prompt text.

    Kept deliberately simple and separate so it's easy to edit how much history
    agents see and how it reads.
    """
    if not feed:
        return "(The studio is empty. Nothing has been made or said yet.)"

    lines: list[str] = []
    for r in feed:
        if r.kind == "concept":
            lines.append(f"[round {r.round}] ARTIST {r.agent} proposed a concept:\n  {r.content}")
        else:  # evaluation
            score = "n/a" if r.score is None else f"{r.score:.2f}"
            lines.append(f"[round {r.round}] CRITIC {r.agent} (score {score}) wrote:\n  {r.content}")
    return "\n".join(lines)


def _format_new_concepts(new_concepts: list[Record]) -> str:
    """Render this round's concepts that a critic must evaluate now."""
    lines = []
    for r in new_concepts:
        lines.append(f"- (by {r.agent}) {r.content}")
    return "\n".join(lines)


# --- Robust JSON extraction ---------------------------------------------------
def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of Claude's text and parse it.

    We ask for bare JSON, but stray prose or code fences happen. Try a direct
    parse first; if that fails, grab the outermost {...} span and parse that.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text!r}")
    return json.loads(match.group(0))


# --- The Agent ----------------------------------------------------------------
class Agent:
    """An artist or critic with a fixed disposition, backed by one Claude model."""

    def __init__(self, name: str, role: str, disposition: str, client: AsyncAnthropic):
        self.name = name
        self.role = role  # "artist" | "critic"
        self.disposition = disposition
        self.client = client

    def _task_instruction(self, new_concepts: list[Record]) -> str:
        """The role-specific ask, appended to the user prompt after the feed."""
        if self.role == "artist":
            return (
                "Your turn: invent ONE new artwork concept. Make it specific and "
                "evocative. Return your JSON object."
            )
        # critic
        return (
            "Your turn: evaluate the concept(s) proposed THIS round, in light of "
            "the studio's history above. Write a short critique in 'content', give "
            "a 'score' from 0.0 to 1.0, and a 'reasoning'.\n\n"
            "Concepts to evaluate this round:\n"
            f"{_format_new_concepts(new_concepts)}\n\n"
            "Return your JSON object."
        )

    async def act(
        self,
        round_idx: int,
        feed: list[Record],
        new_concepts: list[Record],
    ) -> Record:
        """Build the prompt, call Claude, parse the JSON, return a Record.

        `feed` is the full history before this agent acts. `new_concepts` is this
        round's concepts (passed to critics so they can evaluate them; empty for
        artists).
        """
        system_prompt = f"{SHARED_SYSTEM_PROMPT}\nYour disposition: {self.disposition}"
        user_prompt = (
            "Studio so far:\n"
            f"{format_feed(feed)}\n\n"
            f"{self._task_instruction(new_concepts)}"
        )

        response = await self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        output = ModelOutput.model_validate(_extract_json(text))

        # Concepts get an id (so critics/analysis can reference them); evals don't.
        concept_id: Optional[str] = (
            f"r{round_idx}-{self.name}" if self.role == "artist" else None
        )
        return Record(
            round=round_idx,
            agent=self.name,
            role=self.role,
            model=MODEL,
            kind="concept" if self.role == "artist" else "evaluation",
            concept_id=concept_id,
            content=output.content,
            score=output.score if self.role == "critic" else None,
            reasoning=output.reasoning,
        )
