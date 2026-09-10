"""The Agent and the prompt machinery around it.

Read this file top-to-bottom: the shared system prompt, then the feed
formatter, then the JSON extractor, then the Agent itself. `Agent.act` is the
whole flow for one agent in one round — build prompt, call Claude, parse, return
a Record.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from schema import ModelOutput, Record

# The single model used by every agent. Multi-model is a later version; for now
# this is the one knob to change the model everywhere.
MODEL = "claude-sonnet-4-6"

# How many completed rounds of critiques an agent can still see. Concepts are
# never windowed (see windowed_feed); this bounds evaluations only.
#
# An unbounded log does not give agents perfect memory — it gives them
# position-dependent attention with a decay we cannot characterise or measure.
# A term coined 300 records back is present in the context and functionally
# invisible. Bounding the window makes availability a property of the system
# instead of something we hope the attention mechanism approximates: a term
# stays in view while it is being repeated and falls out when nobody picks it
# up, which is the transmission dynamic this project exists to observe.
FEED_WINDOW = 3


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

Keep "content" between 80 and 180 words — concepts and critiques alike are short by design.
Do not wrap the JSON in markdown fences. Do not add commentary outside the JSON.
"""


# --- Feed windowing -----------------------------------------------------------
def evaluation_round_bounds(round_idx: int, window: int = FEED_WINDOW) -> tuple[int, int]:
    """The [lo, hi) round range of evaluations an agent may see at `round_idx`.

    THE single definition of the window rule. Both arms of the experiment read
    their bound from here — run.py through windowed_feed(), control.py inside
    isolated_feed()'s own filter — so there is one rule and two assemblers
    rather than two rules. Two rules drift: someone fixes a bug in one arm, not
    the other, and the treatment/control comparison is silently invalid.

    `hi` is exclusive and equals round_idx, so the current round's critiques are
    never visible. Critics already act in parallel and could not see them
    anyway; stating it here makes the invariant testable rather than incidental.

    window=0 yields an empty range — concepts only. That is the
    artists-blind-to-critique configuration; it is representable but not the
    default.
    """
    return max(0, round_idx - window), round_idx


def windowed_feed(feed: list[Record], round_idx: int,
                  window: int = FEED_WINDOW) -> list[Record]:
    """All concepts, plus evaluations from the last `window` completed rounds.

    Concepts are NEVER windowed. The control condition replays the treatment's
    artworks, and that replay is only valid if the stimulus is identical and
    complete in both arms — so the window bounds critique visibility only.

    Order is preserved: the feed is append-ordered and format_feed renders it
    as-is, so filtering must not reshuffle it.
    """
    lo, hi = evaluation_round_bounds(round_idx, window)
    return [rec for rec in feed
            if (rec.kind == "concept" and rec.round <= round_idx)
            or (rec.kind != "concept" and lo <= rec.round < hi)]


# --- Feed formatting ----------------------------------------------------------
def format_feed(feed: list[Record]) -> str:
    """Render the records an agent is to see into prompt text.

    Renders exactly the list it is handed, in order, and filters nothing. The
    caller decides what is visible — see windowed_feed — so this stays a pure
    formatter and the question of how much history an agent sees lives in one
    place instead of two.
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


# --- The cached prefix --------------------------------------------------------
# Every agent in a phase sends an identical system block and feed block, and
# both carry a cache breakpoint. These two builders are the single definition of
# that prefix: Agent.act sends it, prewarm_cache() warms it, and if they were
# built separately a one-character drift would silently stop every cache hit
# without failing anything. tests/test_prewarm.py pins them byte-for-byte.
def system_blocks() -> list[dict]:
    return [{"type": "text", "text": SHARED_SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}}]


def feed_block(feed: list[Record]) -> dict:
    return {"type": "text", "text": f"Studio so far:\n{format_feed(feed)}",
            "cache_control": {"type": "ephemeral"}}


async def prewarm_cache(client: AsyncAnthropic, feed: list[Record], round_idx: int,
                        tally: "Optional[UsageTally]" = None) -> bool:
    """Write the shared feed block into the prompt cache with one call.

    Without this, a phase's calls all fire at once, all miss the cache, and all
    WRITE it — so the layout that was supposed to cost one write plus N-1 reads
    costs min(N, MAX_CONCURRENT) writes instead. Measured on the 8-round 6x6 run
    of 2026-09-10: with MAX_CONCURRENT equal to the artist count, the artist
    phase got no caching at all, cache writes came to 71% of the bill, and about
    55% of the run's cost was redundant writes.

    One max_tokens=0 request sends the prefix and generates nothing, so the
    block is cached once and the real calls read it — which decouples cache
    efficiency from concurrency instead of trading one against the other.

    Never raises. A failed pre-warm costs cache efficiency, not the run, and a
    run that dies because an optimisation failed is a worse outcome than a run
    that is billed badly. Returns whether the call went through.
    """
    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=0,                       # verified accepted; generates nothing
            system=system_blocks(),
            messages=[{"role": "user", "content": [feed_block(feed)]}],
        )
    except Exception as exc:                    # noqa: BLE001 - deliberately broad
        print(f"    (cache pre-warm failed, continuing: {type(exc).__name__})")
        return False
    if tally is not None:
        tally.add(round_idx, response.usage)
    return True


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


# --- Token accounting ---------------------------------------------------------
USAGE_FIELDS = ("input_tokens", "output_tokens",
                "cache_creation_input_tokens", "cache_read_input_tokens")


@dataclass
class UsageTally:
    """Per-round token totals, accumulated across every agent call in a run.

    Exists to measure the prompt cache rather than assume it. Within a round
    every agent sends an identical feed block, so the expected pattern is one
    cache write plus N-1 reads; windowing changes the size of both, not the
    shape. Reading it back off the responses is the only way to know.

    Every attempt is counted, including retries after a malformed-JSON
    response — a retry costs real tokens, so hiding it would understate the run.
    """

    per_round: dict[int, dict[str, int]] = field(default_factory=dict)
    calls: dict[int, int] = field(default_factory=dict)

    def add(self, round_idx: int, usage) -> None:
        bucket = self.per_round.setdefault(round_idx, {k: 0 for k in USAGE_FIELDS})
        for name in USAGE_FIELDS:
            # Older SDKs omit the cache fields entirely, and the API returns
            # null rather than 0 when a request neither wrote nor read cache.
            bucket[name] += getattr(usage, name, 0) or 0
        self.calls[round_idx] = self.calls.get(round_idx, 0) + 1

    def totals(self) -> dict[str, int]:
        out = {k: 0 for k in USAGE_FIELDS}
        for bucket in self.per_round.values():
            for name in USAGE_FIELDS:
                out[name] += bucket[name]
        return out

    def report(self) -> str:
        """A per-round table plus totals, for printing at the end of a run."""
        if not self.per_round:
            return "No token usage recorded."
        head = (f"{'round':>6} {'calls':>6} {'input':>9} {'cache wr':>9} "
                f"{'cache rd':>9} {'output':>8}")
        lines = ["Token usage (uncached input, cache writes, cache reads):", head]
        for rnd in sorted(self.per_round):
            b = self.per_round[rnd]
            lines.append(f"{rnd:>6} {self.calls[rnd]:>6} {b['input_tokens']:>9,} "
                         f"{b['cache_creation_input_tokens']:>9,} "
                         f"{b['cache_read_input_tokens']:>9,} "
                         f"{b['output_tokens']:>8,}")
        t = self.totals()
        lines.append(f"{'total':>6} {sum(self.calls.values()):>6} {t['input_tokens']:>9,} "
                     f"{t['cache_creation_input_tokens']:>9,} "
                     f"{t['cache_read_input_tokens']:>9,} {t['output_tokens']:>8,}")
        return "\n".join(lines)


# --- The Agent ----------------------------------------------------------------
class Agent:
    """An artist or critic with a fixed disposition, backed by one Claude model."""

    def __init__(self, name: str, role: str, disposition: str,
                 client: AsyncAnthropic, tally: Optional[UsageTally] = None):
        self.name = name
        self.role = role  # "artist" | "critic"
        self.disposition = disposition
        self.client = client
        self.tally = tally  # None = don't account for tokens

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

        `feed` is exactly what this agent should see, already assembled and
        windowed by the caller (run.py's loop, control.py's isolated_feed) — for
        critics it already includes this round's concepts. act() renders it
        as-is and applies no window of its own: keeping the filtering out of
        here is what leaves the prompt blocks and their cache breakpoints
        untouched. `target` is the single concept a critic must evaluate; None
        for artists.
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
                system=system_blocks(),
                messages=[{"role": "user", "content": [
                    feed_block(feed),
                    {"type": "text", "text": agent_block},
                ]}],
            )
            if self.tally is not None:
                self.tally.add(round_idx, response.usage)
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
