"""Verify that a control run's critics received byte-identical prompts to the
treatment run's critics.

The artworks are identical by construction — control.py replays them. What this
checks is the thing that could accidentally differ: the exact prompt string each
critic receives, i.e. system prompt + feed + disposition + task instruction.

Prompts are not logged. Nothing in the .jsonl holds them, so they have to be
reconstructed. Two rules keep that reconstruction honest:

  1. The payload is captured from the REAL code path. Each Agent is handed a
     fake client whose messages.create() records its kwargs and aborts, then
     Agent.act() is called for real. What we compare is the literal API request
     agents.py builds — not a re-implementation of it that could drift.

  2. The whole request is compared, not concatenated text. Commit 940a61a moved
     the disposition out of the system prompt and after the feed (for prompt-
     cache reuse). Same three pieces, same bytes end-to-end, different prompt.
     So the hash covers system blocks, message-block boundaries, cache_control,
     model and max_tokens too.

The feeds are built the way each condition built them: for the treatment, the
log is append-ordered, so a round's critic feed is the log prefix ending just
before that round's first evaluation (actual recorded order, not re-derived);
for the control, isolated_feed()'s rule — artworks so far plus only this
critic's own critiques from earlier rounds.

VERSION DRIFT is the real hazard and the reason --treatment-rev exists. A run
uses whatever agents.py and agents.yaml were on disk at the time, which is not
necessarily what is committed, nor what is in the working tree today. Pin each
side to a git revision to test what a run actually used; the default (working
tree for both) assumes nothing changed since, and says so loudly.

Usage:
    uv run python prompt_check.py logs/control_run_X_Y.jsonl
    uv run python prompt_check.py logs/control_run_X_Y.jsonl --round 0
    uv run python prompt_check.py logs/control_run_X_Y.jsonl --all-rounds
    uv run python prompt_check.py logs/control_run_X_Y.jsonl \
        --treatment-rev 940a61a --control-rev c0a68c5
    uv run python prompt_check.py logs/control.jsonl --treatment logs/run.jsonl --show
"""

import argparse
import asyncio
import contextlib
import difflib
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

from schema import Record

# Files a pinned revision needs on disk to import agents.py standalone.
PINNED_FILES = ("agents.py", "schema.py")


# --- Capturing the real prompt ------------------------------------------------
class _Captured(BaseException):
    """Carries the captured request out of Agent.act().

    Deliberately a BaseException: Agent.act() catches (ValueError,
    ValidationError) around its parse step, and we must never be swallowed by a
    retry loop if that try-block ever widens.
    """

    def __init__(self, kwargs: dict):
        self.kwargs = kwargs


class _CapturingMessages:
    async def create(self, **kwargs):
        raise _Captured(kwargs)


class CapturingClient:
    """Stands in for AsyncAnthropic. Records one request, makes no network call."""

    def __init__(self):
        self.messages = _CapturingMessages()


def capture_prompt(agent, round_idx: int, feed: list, target) -> dict:
    """Run agent.act() far enough to build its request, and return that request."""
    try:
        asyncio.run(agent.act(round_idx, feed, target))
    except _Captured as c:
        return c.kwargs
    raise RuntimeError(
        f"{agent.name}: Agent.act() returned without calling messages.create() — "
        "agents.py no longer builds its prompt where this check expects it."
    )


# --- Loading a pinned agents.py / agents.yaml ---------------------------------
def _git_show(rev: str, path: str) -> str:
    try:
        return subprocess.run(["git", "show", f"{rev}:{path}"],
                              capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        sys.exit(f"Cannot read {path} at revision {rev}: {e.stderr.strip()}")


@contextlib.contextmanager
def agents_at(rev):
    """Yield (agents_module, agents_yaml_text) for a git revision, or the tree.

    The pinned agents.py is imported under its own module name, with the pinned
    schema.py and the temp directory visible ONLY for the duration of that
    import. Both are torn down before yielding: leaving the temp dir on sys.path
    would make a later plain `import agents` resolve to the pinned copy, silently
    reconstructing both sides from the same code and reporting a false match.
    """
    if rev is None:
        import agents
        yield agents, Path("agents.yaml").read_text()
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for name in PINNED_FILES:
            (tmpdir / name).write_text(_git_show(rev, name))
        yaml_text = _git_show(rev, "agents.yaml")

        saved = sys.modules.pop("schema", None)
        sys.path.insert(0, str(tmpdir))
        try:
            for name in ("schema", f"agents_{rev}"):
                src = tmpdir / ("schema.py" if name == "schema" else "agents.py")
                spec = importlib.util.spec_from_file_location(name, src)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
            pinned = sys.modules.pop(f"agents_{rev}")
        finally:
            # Tear down before yielding, not after: see docstring.
            sys.path.remove(str(tmpdir))
            sys.modules.pop("schema", None)
            if saved is not None:
                sys.modules["schema"] = saved
        yield pinned, yaml_text


def source_id(agents_module) -> str:
    """Short hash of the agents.py a side is actually using.

    Printed for both sides so that a silent fallback to the same source — the
    exact bug the sys.path teardown above prevents — is visible in the output
    instead of masquerading as a match.
    """
    src = Path(agents_module.__file__).read_text()
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def build_critics(agents_module, yaml_text: str) -> dict:
    """{name: Agent} for every critic in a roster, wired to a capturing client."""
    client = CapturingClient()
    spec = yaml.safe_load(yaml_text)
    return {
        a["name"]: agents_module.Agent(a["name"], a["role"], a["disposition"], client)
        for a in spec["agents"] if a["role"] == "critic"
    }


# --- Reading logs -------------------------------------------------------------
def read_log(path: Path) -> list[Record]:
    """Records in file order. Order matters: it IS the feed order."""
    if not path.exists():
        sys.exit(f"Log not found: {path}")
    return [Record(**json.loads(line)) for line in path.read_text().splitlines()
            if line.strip()]


def infer_treatment(control_path: Path) -> Path:
    """control_<treatment-stem>_<timestamp>.jsonl -> logs/<treatment-stem>.jsonl."""
    stem = control_path.stem.removeprefix("control_")
    return control_path.parent / f"{'_'.join(stem.split('_')[:-2])}.jsonl"


def treatment_feed(records: list[Record], round_idx: int) -> list[Record]:
    """The feed as it stood when round `round_idx`'s critics acted.

    run.py appends concepts, then critiques, so that moment is exactly the log
    prefix ending before this round's first evaluation. Read off recorded order
    rather than re-derived, so a reordering bug in the run would show up here
    instead of being reproduced by the check.
    """
    for i, r in enumerate(records):
        if r.kind == "evaluation" and r.round == round_idx:
            return records[:i]
    sys.exit(f"Treatment log has no evaluations in round {round_idx}.")


def control_feed(concepts_by_round: dict, own: dict, rounds: list[int],
                 upto: int, critic: str) -> list[Record]:
    """isolated_feed()'s rule: artworks so far + only this critic's own earlier
    critiques. Mirrors control.py; kept here so the check does not import the
    module it is auditing."""
    feed: list[Record] = []
    for r in rounds:
        if r > upto:
            break
        feed.extend(concepts_by_round[r])
        feed.extend(rec for rec in own[critic] if rec.round == r)
    return feed


# --- Rendering and comparing --------------------------------------------------
def canonical(payload: dict) -> str:
    """Structure-sensitive canonical form. Block boundaries and cache_control are
    part of the prompt, so they are part of the hash."""
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


def _blocks(value) -> list[tuple[str, str]]:
    """Normalize a plain string or a list of content blocks to [(label, text)].

    Handles both prompt layouts: pre-940a61a passed bare strings, post-940a61a
    passes lists of cache-controlled blocks.
    """
    if isinstance(value, str):
        return [("", value)]
    out = []
    for i, b in enumerate(value):
        cc = " cache_control=ephemeral" if b.get("cache_control") else ""
        out.append((f"[{i}]{cc}", b.get("text", "")))
    return out


def render(payload: dict) -> str:
    """Readable line-oriented view of a request, for diffing."""
    lines = [f"### model: {payload.get('model')}",
             f"### max_tokens: {payload.get('max_tokens')}"]
    for label, text in _blocks(payload.get("system", "")):
        lines.append(f"### system{label}")
        lines.extend(text.splitlines())
    for msg in payload.get("messages", []):
        for label, text in _blocks(msg.get("content", "")):
            lines.append(f"### {msg.get('role')}{label}")
            lines.extend(text.splitlines())
    return "\n".join(lines)


def digest(payload: dict) -> str:
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def compare_round(round_idx, t_records, c_records, t_critics, c_critics,
                  concepts_by_round, rounds, show, context):
    """Compare every (critic, concept) prompt pair in one round. Returns
    (n_pairs, n_identical, [failure descriptions])."""
    concepts = concepts_by_round[round_idx]
    t_feed = treatment_feed(t_records, round_idx)

    # Each critic's own critiques from earlier rounds, in control-log order.
    own = defaultdict(list)
    for rec in c_records:
        if rec.kind == "evaluation" and rec.round < round_idx:
            own[rec.agent].append(rec)

    names = sorted(set(t_critics) & set(c_critics))
    missing = sorted((set(t_critics) | set(c_critics)) - set(names))
    problems = [f"critic {n} exists in only one roster — cannot pair" for n in missing]

    identical = 0
    pairs = 0
    for name in names:
        c_feed = control_feed(concepts_by_round, own, rounds, round_idx, name)
        for concept in concepts:
            pairs += 1
            t_payload = capture_prompt(t_critics[name], round_idx, t_feed, concept)
            c_payload = capture_prompt(c_critics[name], round_idx, c_feed, concept)
            t_hash, c_hash = digest(t_payload), digest(c_payload)
            tag = f"round {round_idx}  {name}  {concept.concept_id}"
            if t_hash == c_hash:
                identical += 1
                print(f"  MATCH  {tag}  sha256 {t_hash[:12]}")
                if show:
                    print(_indent(render(t_payload)))
                continue
            print(f"  DIFFER {tag}")
            print(f"         treatment sha256 {t_hash[:12]} | control sha256 {c_hash[:12]}")
            diff = list(difflib.unified_diff(
                render(t_payload).splitlines(), render(c_payload).splitlines(),
                fromfile="treatment", tofile="control", lineterm="", n=context))
            print(_indent("\n".join(diff)))
            problems.append(f"{tag}: prompts differ ({len(diff)} diff lines)")
    return pairs, identical, problems


def _indent(text: str) -> str:
    return "\n".join("         " + line for line in text.splitlines())


def main() -> None:
    p = argparse.ArgumentParser(
        description="Are the control's critic prompts identical to the treatment's?")
    p.add_argument("control", type=Path, help="control_*.jsonl")
    p.add_argument("--treatment", type=Path, help="treatment log (default: inferred)")
    p.add_argument("--round", type=int, default=0, help="round to check (default 0)")
    p.add_argument("--all-rounds", action="store_true", help="check every replayed round")
    p.add_argument("--treatment-rev", help="git rev of agents.py/agents.yaml the "
                                           "treatment ran with (default: working tree)")
    p.add_argument("--control-rev", help="git rev the control ran with "
                                         "(default: working tree)")
    p.add_argument("--show", action="store_true", help="print matching prompts too")
    p.add_argument("--context", type=int, default=3, help="diff context lines")
    args = p.parse_args()

    treatment_path = args.treatment or infer_treatment(args.control)
    t_records = read_log(treatment_path)
    c_records = read_log(args.control)

    # Concepts come from the CONTROL log: control.py logs the artworks it
    # actually replayed, so this compares against what the control really fed
    # its critics, not what we assume it copied.
    concepts_by_round = defaultdict(list)
    for rec in c_records:
        if rec.kind == "concept":
            concepts_by_round[rec.round].append(rec)
    critiqued = {rec.round for rec in c_records if rec.kind == "evaluation"}
    rounds = sorted(r for r in concepts_by_round if r in critiqued)
    if not rounds:
        sys.exit(f"{args.control} has no critiqued rounds.")

    targets = rounds if args.all_rounds else [args.round]
    unknown = [r for r in targets if r not in rounds]
    if unknown:
        sys.exit(f"Round(s) {unknown} not replayed in the control. Available: {rounds}")

    print(f"Treatment: {treatment_path.name}  ({args.treatment_rev or 'working tree'})")
    print(f"Control:   {args.control.name}  ({args.control_rev or 'working tree'})")
    if args.treatment_rev is None or args.control_rev is None:
        print("NOTE: sides without an explicit --*-rev are reconstructed from the "
              "CURRENT agents.py/agents.yaml. That assumes neither file has changed "
              "since the run — verify before trusting a match.")
    print(f"Checking round(s) {targets}\n")

    # Verify the replayed artworks really are identical before comparing prompts.
    # If they are not, a prompt diff would be explained by that and nothing else.
    t_concepts = defaultdict(list)
    for rec in t_records:
        if rec.kind == "concept":
            t_concepts[rec.round].append(rec)
    replay_problems = []
    for r in targets:
        t_side = [(c.concept_id, c.title, c.content) for c in t_concepts[r]]
        c_side = [(c.concept_id, c.title, c.content) for c in concepts_by_round[r]]
        if t_side != c_side:
            replay_problems.append(f"round {r}: replayed artworks differ from the "
                                   "treatment's (order or content)")
    for msg in replay_problems:
        print(f"  ARTWORK MISMATCH  {msg}")

    total = identical = 0
    problems = list(replay_problems)
    with agents_at(args.treatment_rev) as (t_mod, t_yaml), \
         agents_at(args.control_rev) as (c_mod, c_yaml):
        t_critics = build_critics(t_mod, t_yaml)
        c_critics = build_critics(c_mod, c_yaml)

        # Provenance: which agents.py each side actually loaded, and whether the
        # rosters differ. Printed so a match can be read as evidence rather than
        # taken on faith.
        t_src, c_src = source_id(t_mod), source_id(c_mod)
        print(f"  agents.py  treatment {t_src} | control {c_src}"
              f"{'  (same source)' if t_src == c_src else '  (DIFFERENT source)'}")
        y_tag = "same roster text" if t_yaml == c_yaml else "DIFFERENT roster text"
        print(f"  agents.yaml {y_tag}\n")

        for r in targets:
            n, ok, probs = compare_round(r, t_records, c_records, t_critics,
                                         c_critics, concepts_by_round, rounds,
                                         args.show, args.context)
            total += n
            identical += ok
            problems.extend(probs)

    print(f"\n{identical}/{total} prompt pairs byte-identical.")
    if problems:
        print("\nProblems:")
        for msg in problems:
            print(f"  - {msg}")
    verdict = "YES" if problems == [] and identical == total and total else "NO"
    print(f"\nAre the round-{targets[0]} prompts identical? {verdict}"
          if len(targets) == 1 else f"\nAll checked rounds identical? {verdict}")
    sys.exit(0 if verdict == "YES" else 1)


if __name__ == "__main__":
    main()
