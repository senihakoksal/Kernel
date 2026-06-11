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
from pydantic import ValidationError

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
  - "title":     artists only — a short title for the artwork, at most six words. Omit if you are a critic.
  - "content":   your contribution as a string (an artwork concept, or an evaluation).
  - "reasoning": one or two sentences on why, as a string.
  - "score":     critics only — a number from 0.0 to 1.0 rating the concept(s). Omit if you are an artist.

Keep "content" under 200 words — concepts and critiques alike are short by design.
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
            title = f" “{r.title}”" if r.title else " a concept"
            lines.append(f"[round {r.round}] ARTIST {r.agent} proposed{title}:\n  {r.content}")
        else:  # evaluation
            score = "n/a" if r.score is None else f"{r.score:.2f}"
            lines.append(f"[round {r.round}] CRITIC {r.agent} on {r.concept_id} "
                         f"(score {score}) wrote:\n  {r.content}")
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

    def _task_instruction(self, target: Optional[Record]) -> str:
        """The role-specific ask, appended to the user prompt after the feed."""
        if self.role == "artist":
            return (
                "Your turn: invent ONE new artwork concept with a short title. "
                "Make it specific and evocative. Return your JSON object."
            )
        # critic
        return (
            "Your turn: evaluate ONE concept proposed this round, in light of "
            "the studio's history above. Write a short critique in 'content', "
            "give a 'score' from 0.0 to 1.0, and a 'reasoning'.\n\n"
            f"The concept to evaluate (by {target.agent}):\n{target.content}\n\n"
            "Return your JSON object."
        )

    async def act(
        self,
        round_idx: int,
        feed: list[Record],
        target: Optional[Record] = None,
    ) -> Record:
        """Build the prompt, call Claude, parse the JSON, return a Record.

        `feed` is the full history before this agent acts (for critics it
        already includes this round's concepts). `target` is the single concept
        a critic must evaluate; None for artists.
        """
        # Prompt layout is built for cache reuse: the shared system prompt and
        # the feed are identical for every agent in a phase, so both are marked
        # as cache breakpoints; only the small final block (disposition + task)
        # differs per agent. Cache reads cost ~10% and crucially do NOT count
        # toward the input-tokens-per-minute rate limit — without this, the
        # growing feed re-billed in full per call kills long runs with 429s.
        # (This is why the disposition sits in the user message, after the
        # feed, instead of in the system prompt: any per-agent text before the
        # feed would break the shared cache prefix.)
        feed_block = f"Studio so far:\n{format_feed(feed)}"
        agent_block = (
            f"Your disposition: {self.disposition}\n\n"
            f"{self._task_instruction(target)}"
        )

        # max_tokens is kept modest: content is capped at ~200 words by the
        # system prompt, and the API's output-tokens-per-minute bucket counts
        # this cap, not actual usage. A response that still overruns it gets
        # truncated mid-JSON and fails to parse — so retry a couple of times
        # rather than let one bad response kill a long run.
        output: Optional[ModelOutput] = None
        for attempt in range(3):
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=800,
                system=[{"type": "text", "text": SHARED_SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": feed_block,
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": agent_block},
                ]}],
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            try:
                output = ModelOutput.model_validate(_extract_json(text))
                break
            except (ValueError, ValidationError):
                if attempt == 2:
                    raise

        # Concepts get a fresh id; an evaluation carries its target's id so
        # every critique is linked to the work it judges.
        concept_id: Optional[str] = (
            f"r{round_idx}-{self.name}" if self.role == "artist"
            else target.concept_id
        )
        return Record(
            round=round_idx,
            agent=self.name,
            role=self.role,
            model=MODEL,
            kind="concept" if self.role == "artist" else "evaluation",
            concept_id=concept_id,
            title=output.title if self.role == "artist" else None,
            content=output.content,
            score=output.score if self.role == "critic" else None,
            reasoning=output.reasoning,
        )
