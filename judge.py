"""Tier 2 — the judged checks, and the hand labels that make the judge usable.

eval_agents.py (Tier 1) is free: deterministic rules and a local embedding
model. It deliberately leaves out the two checks that need a reader —
disposition adherence, and whether a score agrees with the prose attached to
it. Tier 1's embedding proxy for the second came back at rho = 0.06, too weak
to read either way, which is the whole reason this file exists.

WHY HAND LABELS, AND WHY THEY ARE NOT JUST JUDGE CALIBRATION
The agents are Sonnet 4.6. Any Claude judging them shares priors with them, so
a judge that identifies dispositions brilliantly may be recognising its own
family's habits rather than reading character off the page. The human labels
are the family-independent reference point: if the judge scores far above the
human, that gap IS the shared prior, and the judged numbers cannot be read as a
property of the work. Validating the judge is the second use of the labels, not
the first.

THE TWO TASKS, AND WHY THEY TAKE THIS SHAPE
Both have ground truth in the log, and both are answerable by a human and a
model from the same prompt:

  wrote  — six critic dispositions, lettered. Which one wrote this critique?
           Truth from the log; chance is 1/6.
  score  — the critique with its number removed. What score was attached?
           Truth from the log; the null is predicting the global mean.

The obvious alternative — asking a judge to rate adherence 1-5 — has no ground
truth and no human-comparable scale, so a judge disagreeing with a human would
tell you nothing about which of them is right. A forced choice and a number can
each be scored against truth independently, and against each other.

Nothing here runs without being asked. `sheet` and `estimate` touch no network;
`run` is the only command that spends money and it requires --confirm.

Usage:
    uv run python judge.py sheet                  # build the labelling sheet
    uv run python judge.py estimate               # tokens and cost, no calls
    uv run python judge.py run --confirm          # the only command that pays
    uv run python judge.py score                  # human vs judge vs truth
"""

import argparse
import asyncio
import base64
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from schema import Record

LOG_DIR = Path("logs")
# Hand labels and judge answers are expensive to produce — a person's attention
# in one case, money in the other — so they are committed, not written to the
# gitignored figures/ directory beside the derived plots.
LABEL_DIR = Path("labels")

JUDGE_MODEL = "claude-opus-5"
DEFAULT_N = 25
SAMPLE_SEED = 20260910
MAX_CONCURRENT = 4          # starter tier is 50 req/min; 2 calls per item

LETTERS = "ABCDEF"

# Per million tokens, input/output. Only rates that have been confirmed against
# the pricing page belong here — a dollar figure the reader cannot trust is
# worse than a token count, so an absent rate prints tokens and asks for
# --price-in/--price-out rather than guessing.
PRICING = {
    "claude-sonnet-4-6": (3.0, 15.0),
}

JUDGE_THRESHOLDS = {
    # Is the judge usable as a stand-in for the human reader at all?
    "judge_agrees_wrote": {"pass": 0.60, "warn": 0.40},   # Cohen's kappa
    "judge_agrees_score": {"pass": 0.60, "warn": 0.40},   # Spearman, judge vs human
    # The findings themselves. Only meaningful once the two rows above pass.
    "disposition_adherence": {"pass": 0.50, "warn": 0.33},  # accuracy, chance 1/6
    "score_legible": {"pass": 0.40, "warn": 0.20},          # Spearman vs the real score
}


# --- the sample ---------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    """One critique put to a reader, with the answers it is hiding.

    `critic`, `score`, `arm` and `source` are truth: recorded here, and never
    rendered into the sheet or the judge prompt.
    """
    item_id: str
    source: str
    arm: str
    round: int
    critic: str
    score: float
    work_title: str
    work_content: str
    critique: str            # redacted; the raw text is in the log
    signposted: bool         # did it name its own influence before redaction
    redactions: int


def load(path: Path) -> list[Record]:
    return [Record(**json.loads(line)) for line in path.read_text().splitlines()
            if line.strip()]


def load_dispositions(logs: list[Path],
                      roster: Path = Path("agents.yaml")) -> dict[str, str]:
    """The critic dispositions THOSE runs used, not whatever the file says now.

    Same trap as the word band: reading the live roster would show a rater
    dispositions the critics never had, and silently invalidate every label
    collected against them. The sidecar records a hash of the roster; if it
    disagrees with the file on disk, that is a stop, not a warning — the
    labelling effort would be spent on the wrong text.
    """
    current = hashlib.sha256(roster.read_bytes()).hexdigest()
    for log in logs:
        meta_path = log.with_suffix(".meta.json")
        if not meta_path.exists():
            sys.exit(f"{log.name} has no sidecar — cannot confirm which roster it used.")
        recorded = json.loads(meta_path.read_text()).get("agents_file_sha256")
        if recorded and recorded != current:
            sys.exit(
                f"agents.yaml has changed since {log.name} ran\n"
                f"  recorded: {recorded[:16]}\n  on disk:  {current[:16]}\n"
                "The dispositions on disk are not the ones those critics were given.\n"
                "Check out the roster at that revision before building a sheet.")
    agents = yaml.safe_load(roster.read_text())["agents"]
    return {a["name"]: " ".join(a["disposition"].split())
            for a in agents if a["role"] == "critic"}


def influence_names(dispositions: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Each critic's named influence, read out of its own disposition."""
    out = {}
    for name, text in dispositions.items():
        m = re.search(r"inspired by ([A-Z][a-z]+) ([A-Z][a-z]+)", text)
        if m:
            out[name] = (m.group(1), m.group(2))
    return out


REDACTION = "[name removed]"


def redact(text: str, influences: dict[str, tuple[str, str]]) -> tuple[str, int]:
    """Remove every critic's named influence from a critique.

    44% of the critiques in the September run name their own influence by
    surname — "But Ruskin would ask: where is the craftsman's restraint?" —
    and almost none name anyone else's. Left in, the attribution task measures
    whether a critic signposts, which is a lookup, not a reading: a rater would
    score near-perfectly on half the set without attending to the prose at all.

    Every critic's influence goes, not just the author's own, so that a name
    surviving in someone else's critique cannot act as a cue by elimination.
    Whether a critique signposted is kept as a field, because "adherence is
    mostly signposting" is a finding and not an inconvenience.
    """
    removed = 0
    for first, last in influences.values():
        for form in (f"{first} {last}", last, first):
            pattern = re.compile(rf"\b{re.escape(form)}\b")
            text, k = pattern.subn(REDACTION, text)
            removed += k
    # Collapse the runs the full-name-then-surname passes can leave behind.
    text = re.sub(rf"(?:{re.escape(REDACTION)}[\s,'’]*){{2,}}",
                  REDACTION + " ", text)
    return text, removed


def sample_items(logs: dict[str, Path], dispositions: dict[str, str],
                 n: int = DEFAULT_N, seed: int = SAMPLE_SEED,
                 exclude: set[str] = frozenset()) -> list[Item]:
    """A balanced, deterministic sample of critiques across arms and critics.

    Balanced across (arm, critic) so the judge is not validated on feed-exposed
    prose alone, and so the labels stay usable if the set is ever grown into an
    arm comparison. At this size they cannot settle one: 25 items over two arms
    and six critics is two per cell.
    """
    rng = random.Random(seed)
    works, buckets = {}, {}
    for arm, path in logs.items():
        recs = load(path)
        for r in recs:
            if r.kind == "concept":
                works[(arm, r.concept_id)] = (r.title or "", r.content)
        for r in recs:
            if r.kind != "evaluation" or r.score is None:
                continue
            work = works.get((arm, r.concept_id))
            if work is None:        # a critique of a work outside this log
                continue
            if fingerprint(redact(r.content, influence_names(dispositions))[0]) in exclude:
                continue            # already shown to a rater in an earlier set
            buckets.setdefault((arm, r.agent), []).append((r, work))

    for b in buckets.values():
        rng.shuffle(b)
    order = sorted(buckets)
    rng.shuffle(order)

    picked = []
    while len(picked) < n and any(buckets[k] for k in order):
        for key in order:
            if buckets[key] and len(picked) < n:
                picked.append((key[0], *buckets[key].pop()))
    rng.shuffle(picked)          # so arm and critic do not run in streaks

    influences = influence_names(dispositions)
    items = []
    for i, (arm, rec, (title, content)) in enumerate(picked, 1):
        own = influences.get(rec.agent)
        signposted = bool(own and re.search(rf"\b{re.escape(own[1])}\b", rec.content))
        critique, removed = redact(rec.content, influences)
        items.append(Item(item_id=f"i{i:02d}", source=logs[arm].stem, arm=arm,
                          round=rec.round, critic=rec.agent, score=float(rec.score),
                          work_title=title, work_content=content,
                          critique=critique, signposted=signposted,
                          redactions=removed))
    return items


def signposting_rate(logs: dict[str, Path], dispositions: dict[str, str]) -> dict:
    """How often a critique names its own influence, over the whole pair.

    Deterministic and free, and the reason the sheet is redacted — recorded
    with the sample so the result can state it without reopening the logs.
    """
    influences = influence_names(dispositions)
    out = {}
    for arm, path in logs.items():
        ev = [r for r in load(path) if r.kind == "evaluation"]
        own = sum(1 for r in ev if influences.get(r.agent)
                  and re.search(rf"\b{re.escape(influences[r.agent][1])}\b", r.content))
        out[arm] = {"n": len(ev), "named_own_influence": own,
                    "rate": round(own / len(ev), 3) if ev else None}
    return out


def key_for(critics: list[str], seed: int = SAMPLE_SEED) -> dict[str, str]:
    """Letter -> critic name. Seeded, so it is not roster order."""
    shuffled = list(critics)
    random.Random(seed + 1).shuffle(shuffled)
    return dict(zip(LETTERS, shuffled))


# --- the sheet ----------------------------------------------------------------
SCORE_RULE = ('"score": critics only — a number from 0.0 to 1.0 rating the '
              'concept(s).')

SHEET_HEAD = """# Hand labels — {n} critiques

Two questions per item. Fill in the `answer` line under each one; leave a `?`
where you would rather not guess. Nothing else in this file is read back, so
write notes anywhere you like.

**wrote** — which of the six critics below wrote this critique? Give a letter.

**score** — the critic attached a number from 0.0 to 1.0 to this critique, and
it has been removed. What was it? Give a number to two decimals. This was the
whole of the instruction the critics were given about the scale:

> {score_rule}

There was no rubric beyond that line, so the question is not whether a score is
*correct* — it is whether the critic's own scale is legible from its own prose.

Where a critique named one of the six critics' own influences by name, the name
has been replaced with `[name removed]`. Nearly half of them did, and left in,
the first question would be a lookup rather than a reading. The marker tells you
only that a name was there, never which.

Answer each item on its own. Do not try to spread your answers evenly across
the six critics, and do not revise earlier items to balance the set: an even
spread is the sampling's job, not yours, and a rater who balances turns their
own accuracy into an artefact of their bookkeeping.

## The six critics

{key_block}

---
"""

ITEM_BLOCK = """
### {item_id}

**The work** — *{work_title}*

{work_content}

**The critique**

{critique}

- answer: wrote=?  score=?

---
"""


def sheet(items: list[Item], key: dict[str, str],
          dispositions: dict[str, str]) -> str:
    """The labelling sheet. Deliberately says nothing about the distribution.

    No mean, no spread, no count per critic: telling a rater that scores average
    0.74 would pull every estimate toward 0.74 and the recovered correlation
    would be measuring the briefing rather than the prose.
    """
    key_block = "\n\n".join(
        f"**{letter}** — {dispositions[name]}" for letter, name in sorted(key.items()))
    out = SHEET_HEAD.format(n=len(items), score_rule=SCORE_RULE, key_block=key_block)
    for it in items:
        out += ITEM_BLOCK.format(item_id=it.item_id, work_title=it.work_title or "untitled",
                                 work_content=it.work_content, critique=it.critique)
    return out


ANSWER_RE = re.compile(r"^-\s*answer:\s*wrote\s*=\s*(\S+)\s+score\s*=\s*(\S+)\s*$",
                       re.IGNORECASE)


def parse_sheet(text: str) -> tuple[dict[str, dict], list[str]]:
    """Read a filled sheet back. Returns (labels, problems).

    Strict on the answer line and loud about what it could not read — a label
    silently dropped is worse than one reported as unreadable, because the
    agreement figure would be computed over a set nobody checked.
    """
    labels, problems = {}, []
    current = None
    for lineno, line in enumerate(text.splitlines(), 1):
        head = re.match(r"^###\s+(i\d+)\s*$", line.strip())
        if head:
            current = head.group(1)
            continue
        m = ANSWER_RE.match(line.strip())
        if not m:
            if line.strip().lower().startswith("- answer:"):
                problems.append(f"line {lineno}: cannot read {line.strip()!r}")
            continue
        if current is None:
            problems.append(f"line {lineno}: answer with no item heading above it")
            continue
        wrote_raw, score_raw = m.group(1).strip(), m.group(2).strip()
        wrote = wrote_raw.upper()
        if wrote in ("?", ""):
            wrote = None
        elif wrote not in LETTERS:
            problems.append(f"line {lineno} ({current}): wrote={wrote_raw!r} is not A-F")
            wrote = None
        if score_raw in ("?", ""):
            score = None
        else:
            try:
                score = float(score_raw)
            except ValueError:
                problems.append(f"line {lineno} ({current}): score={score_raw!r} is not a number")
                score = None
            else:
                if not 0.0 <= score <= 1.0:
                    problems.append(f"line {lineno} ({current}): score={score} outside 0.0-1.0")
                    score = None
        labels[current] = {"wrote": wrote, "score": score}
        current = None
    return labels, problems


# --- the judge ----------------------------------------------------------------
WROTE_SYSTEM = """You are reading criticism written by one of six critics in a
small studio. Each critic works from a fixed disposition, given below. Decide
which of them wrote the critique you are shown.

{key_block}

Reply with a single JSON object and nothing else:
{{"wrote": "<one letter A-F>", "why": "<one sentence>"}}"""

SCORE_SYSTEM = """You are reading criticism written by a critic in a small
studio. Each critique carried a number from 0.0 to 1.0, which has been removed.
Recover it. This was the whole of the instruction the critic was given about the
scale:

{score_rule}

There was no rubric beyond that line, so you are not judging whether a score
would be correct — you are recovering the number this critic attached to this
prose.

Reply with a single JSON object and nothing else:
{{"score": <number between 0.0 and 1.0>, "why": "<one sentence>"}}"""

USER_BLOCK = """The work being criticised — *{work_title}*

{work_content}

The critique

{critique}"""


def judge_request(item: Item, task: str, dispositions: dict[str, str],
                  seed: int = SAMPLE_SEED) -> tuple[str, str, dict[str, str]]:
    """(system, user, letter->critic) for one item and one task.

    The letters are shuffled per item for the judge, where the human's sheet
    keeps one fixed key. The letter is only a label and both answers are mapped
    back to critic names before anything is compared, so the shuffle costs
    nothing and removes the model's option of favouring a position. It forecloses
    caching the disposition block, which at this volume is a rounding error.
    """
    names = sorted(dispositions)
    random.Random(f"{seed}-{item.item_id}").shuffle(names)
    key = dict(zip(LETTERS, names))
    user = USER_BLOCK.format(work_title=item.work_title or "untitled",
                             work_content=item.work_content, critique=item.critique)
    if task == "wrote":
        key_block = "\n\n".join(f"{letter} — {dispositions[name]}"
                                for letter, name in sorted(key.items()))
        return WROTE_SYSTEM.format(key_block=key_block), user, key
    return SCORE_SYSTEM.format(score_rule=SCORE_RULE), user, key


def parse_judge(text: str, task: str, key: dict[str, str]) -> dict:
    """The judge's answer, mapped back to a critic name or a number."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"error": f"no JSON in reply: {text[:120]!r}"}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return {"error": f"bad JSON: {exc}"}
    out = {"why": str(obj.get("why", ""))[:300]}
    if task == "wrote":
        letter = str(obj.get("wrote", "")).strip().upper()[:1]
        if letter not in key:
            return {"error": f"answer {obj.get('wrote')!r} is not one of the letters offered"}
        out["wrote"] = key[letter]
    else:
        try:
            out["score"] = float(obj.get("score"))
        except (TypeError, ValueError):
            return {"error": f"score {obj.get('score')!r} is not a number"}
    return out


async def run_judge(items: list[Item], dispositions: dict[str, str], model: str,
                    client=None) -> dict[str, dict]:
    """Two calls per item. Returns {item_id: {wrote: ..., score: ...}}."""
    if client is None:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(max_retries=5)
    gate = asyncio.Semaphore(MAX_CONCURRENT)
    answers: dict[str, dict] = {it.item_id: {} for it in items}

    async def one(item: Item, task: str) -> None:
        system, user, key = judge_request(item, task, dispositions)
        async with gate:
            try:
                response = await client.messages.create(
                    model=model, max_tokens=300, system=system,
                    messages=[{"role": "user", "content": user}])
            except Exception as exc:
                answers[item.item_id][task] = {"error": f"{type(exc).__name__}: {exc}"}
                return
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        answers[item.item_id][task] = parse_judge(text, task, key)
        done = sum(1 for a in answers.values() if len(a) == 2)
        print(f"\r    judged {done}/{len(items)}", end="", flush=True)

    await asyncio.gather(*(one(it, task) for it in items for task in ("wrote", "score")))
    print()
    return answers


# --- scoring ------------------------------------------------------------------
# Travels with every result, because a number read six months later will not
# remember who produced it.
AUTHOR_CAVEAT = (
    "The only labeller so far also wrote the dispositions being identified, so "
    "the human accuracy is an upper bound set by the roster's author rather than "
    "a reading by an arms-length critic. Replication by someone who did not "
    "write the dispositions is outstanding, and until then this is not evidence "
    "that a reader can tell these critics apart.")


def looks_copied(items: list[Item], scores: dict[str, float]) -> dict:
    """Do these score labels look like the answer key rather than a reading?

    The first draw kept the answers in a file beside the sheet and the labels
    came back at rho 0.95 — with 22 of 25 answers landing exactly on the truth
    rounded to a tenth and 4 rank inversions in 300 pairs. No reading of prose
    produces that, and a flat correlation would not have revealed it.

    So the signature is checked rather than trusted: answers sitting on the
    truth's own rounding grid, and an ordering too clean for an estimate. This
    is a diagnostic, not a verdict on anyone — a genuinely excellent reader
    would trip it, which is why it reports the numbers it is reasoning from.
    """
    if len(scores) < 10:
        return {}
    by_id = {it.item_id: it for it in items}
    ids = sorted(scores)
    truth = [by_id[i].score for i in ids]
    said = [scores[i] for i in ids]
    on_grid = np.mean([round(s, 2) in (round(round(t, 1), 2),
                                       round(np.ceil(t * 10) / 10, 2))
                       for s, t in zip(said, truth)])
    pairs = [(a, b) for a in range(len(ids)) for b in range(a + 1, len(ids))]
    inversions = sum(1 for a, b in pairs
                     if (truth[a] - truth[b]) * (said[a] - said[b]) < 0)
    rate = inversions / len(pairs) if pairs else None
    return {"on_rounding_grid": round(float(on_grid), 3),
            "inversion_rate": round(rate, 4) if rate is not None else None,
            "flag": bool(on_grid >= 0.6 and rate is not None and rate < 0.05)}


def verdict(value, key: str) -> str:
    t = JUDGE_THRESHOLDS[key]
    if value is None:
        return "insufficient"
    return "pass" if value >= t["pass"] else "warn" if value >= t["warn"] else "fail"


def score_labels(items: list[Item], human: dict, judge: dict,
                 key: dict[str, str]) -> dict:
    """Human, judge and truth, scored against each other on the same items.

    `key` maps the sheet's letters to critic names. The human answers in
    letters and the judge's answers are already mapped back to names by
    parse_judge, so without the key here a letter is compared against a name,
    which is never equal and pins the human's accuracy to exactly 0.
    """
    from sklearn.metrics import cohen_kappa_score
    by_id = {it.item_id: it for it in items}
    out: dict = {"n_items": len(items)}
    # Answers for items outside the sample are dropped rather than crashed on:
    # an edited sheet or a judge file from an older draw should narrow the set
    # the agreement is computed over, not take the tool down.
    human = {i: v for i, v in human.items() if i in by_id}
    judge = {i: v for i, v in judge.items() if i in by_id}

    # wrote — accuracy against truth, and agreement between the two readers
    truth = {i: it.critic for i, it in by_id.items()}
    hw = {i: key[v["wrote"]] for i, v in human.items()
          if v.get("wrote") and v["wrote"] in key}
    jw = {i: a["wrote"]["wrote"] for i, a in judge.items()
          if isinstance(a.get("wrote"), dict) and a["wrote"].get("wrote")}
    both = sorted(set(hw) & set(jw))
    out["wrote"] = {
        "n_human": len(hw),
        "n_judge": len(jw),
        "n_both": len(both),
        "chance": round(1 / len({it.critic for it in items}), 3),
        "human_accuracy": (round(float(np.mean([w == truth[i] for i, w in hw.items()])), 3)
                           if hw else None),
        "judge_accuracy": (round(float(np.mean([w == truth[i] for i, w in jw.items()])), 3)
                           if jw else None),
        "agree_rate": (round(float(np.mean([hw[i] == jw[i] for i in both])), 3)
                       if both else None),
        "kappa": (round(float(cohen_kappa_score([hw[i] for i in both],
                                                [jw[i] for i in both])), 3)
                  if len(both) >= 5 else None),
        "by_signposting": {
            kind: round(float(np.mean([jw[i] == truth[i] for i in jw
                                       if by_id[i].signposted is flag])), 3)
            for kind, flag in (("signposted", True), ("never_named", False))
            if any(by_id[i].signposted is flag for i in jw)},
        "disagreements": [
            {"item": i, "truth": truth[i], "human": hw[i], "judge": jw[i],
             "judge_why": judge[i]["wrote"].get("why", "")}
            for i in both if hw[i] != jw[i]],
    }

    # score — recovery against truth, with predict-the-mean as the null
    hs = {i: v["score"] for i, v in human.items() if v.get("score") is not None}
    js = {i: a["score"]["score"] for i, a in judge.items()
          if isinstance(a.get("score"), dict) and a["score"].get("score") is not None}
    real = np.array([by_id[i].score for i in by_id])
    baseline = float(np.abs(real - real.mean()).mean())

    def rho(a, b) -> tuple[float | None, float | None]:
        """Spearman, or None where it is undefined.

        A reader who gives every item the same score has a constant input, and
        scipy returns nan for it. nan is not valid JSON and would reach the page
        as a broken value, so an undefined correlation is reported as absent —
        which is also the honest reading of it.
        """
        if len(a) < 5 or len(set(a)) < 2 or len(set(b)) < 2:
            return None, None
        r = stats.spearmanr(a, b)
        if np.isnan(r.statistic):
            return None, None
        return round(float(r.statistic), 3), round(float(r.pvalue), 5)

    def against_truth(pred: dict) -> dict:
        if len(pred) < 5:
            return {"n": len(pred), "mae": None, "spearman": None, "p_value": None}
        ids = sorted(pred)
        p = np.array([pred[i] for i in ids])
        t = np.array([by_id[i].score for i in ids])
        r, pv = rho([pred[i] for i in ids], [by_id[i].score for i in ids])
        return {"n": len(ids), "mae": round(float(np.abs(p - t).mean()), 3),
                "spearman": r, "p_value": pv}

    shared = sorted(set(hs) & set(js))
    hj, _ = rho([hs[i] for i in shared], [js[i] for i in shared])
    out["score"] = {
        "mean_baseline_mae": round(baseline, 3),
        "human": against_truth(hs),
        "judge": against_truth(js),
        "n_both": len(shared),
        "human_judge_spearman": hj,
        "human_looks_copied": looks_copied(items, hs),
        "judge_looks_copied": looks_copied(items, js),
    }
    out["caveats"] = [AUTHOR_CAVEAT]
    if out["score"]["human_looks_copied"].get("flag"):
        out["caveats"].append(
            "These score labels are not usable. "
            f"{out['score']['human_looks_copied']['on_rounding_grid']:.0%} of them "
            "sit exactly on the real score's rounding grid with an inversion rate of "
            f"{out['score']['human_looks_copied']['inversion_rate']:.1%}, which is "
            "the signature of the answer key rather than a reading of the prose. "
            "The sample this set was drawn from kept the answers in a plaintext "
            "file beside the sheet; later sets separate them.")

    # The decision the labels exist to support, in the order it has to be made:
    # is the judge usable, and only then, what does it find.
    w, s = out["wrote"], out["score"]
    out["criteria"] = [
        {"check": "judge agrees with the human on who wrote it", "group": "judge",
         "value": w["kappa"], "key": "judge_agrees_wrote",
         "detail": (f"kappa {w['kappa']} over {w['n_both']} items"
                    if w["kappa"] is not None else f"only {w['n_both']} items labelled by both"),
         "verdict": verdict(w["kappa"], "judge_agrees_wrote")},
        {"check": "judge agrees with the human on the score", "group": "judge",
         "value": hj, "key": "judge_agrees_score",
         "detail": (f"rho {hj} over {s['n_both']} items" if hj is not None
                    else f"only {s['n_both']} items scored by both"),
         "verdict": verdict(hj, "judge_agrees_score")},
        {"check": "critics write their disposition", "group": "judged",
         "value": w["judge_accuracy"], "key": "disposition_adherence",
         "detail": (f"judge {w['judge_accuracy']}, human {w['human_accuracy']}, "
                    f"chance {w['chance']}"),
         "verdict": verdict(w["judge_accuracy"], "disposition_adherence")},
        {"check": "the score is legible from the prose", "group": "judged",
         "value": s["judge"]["spearman"], "key": "score_legible",
         "detail": ("the human labels for this set look copied from the answer "
                    "key — no verdict"
                    if s["human_looks_copied"].get("flag") else
                    f"judge rho {s['judge']['spearman']}, human rho "
                    f"{s['human']['spearman']}, MAE {s['judge']['mae']} vs "
                    f"{s['mean_baseline_mae']} for predicting the mean"),
         "verdict": ("insufficient" if s["human_looks_copied"].get("flag")
                     else verdict(s["judge"]["spearman"], "score_legible"))},
    ]
    return out


# --- cost ---------------------------------------------------------------------
def estimate(items: list[Item], dispositions: dict[str, str], model: str,
             price: tuple[float, float] | None) -> dict:
    """Token counts from the real payloads, offline.

    Four characters to the token, which is close enough for prose to decide
    whether a run is affordable and is not close enough to quote to the cent.
    """
    tokens_in = 0
    for item in items:
        for task in ("wrote", "score"):
            system, user, _ = judge_request(item, task, dispositions)
            tokens_in += (len(system) + len(user)) // 4
    calls = 2 * len(items)
    tokens_out = calls * 80          # a letter or a number, plus one sentence
    out = {"model": model, "items": len(items), "calls": calls,
           "input_tokens": tokens_in, "output_tokens": tokens_out}
    if price:
        out["usd"] = round(tokens_in / 1e6 * price[0] + tokens_out / 1e6 * price[1], 4)
    return out


# --- plumbing -----------------------------------------------------------------
def default_pair() -> dict[str, Path]:
    """The newest run that has a control, as {arm: log path}."""
    runs = sorted(LOG_DIR.glob("run_*.jsonl"), reverse=True)
    for run in runs:
        controls = sorted(LOG_DIR.glob(f"control_{run.stem}_*.jsonl"), reverse=True)
        for control in controls:
            if len(load(control)) > 10:
                return {"feed": run, "isolated": control}
    sys.exit("No paired run found in logs/.")


def paths(set_id: str) -> dict[str, Path]:
    return {"items": LABEL_DIR / f"items_{set_id}.json",
            "truth": LABEL_DIR / f"truth_{set_id}.b64",
            "sheet": LABEL_DIR / f"sheet_{set_id}.md",
            "judge": LABEL_DIR / f"judge_{set_id}.json",
            "result": LABEL_DIR / f"result_{set_id}.json"}


# Which fields a rater must not see. The first draw kept all of them in
# items_<id>.json, in the same directory as the sheet, which put the answer key
# next to the exam: the score labels came back at rho 0.95 with 22 of 25
# answers landing exactly on the truth rounded to a tenth, which is not a
# reading of prose. Splitting them out is the fix.
ANSWERS = ("critic", "score", "arm", "source", "round", "signposted", "redactions")


def fingerprint(critique: str) -> str:
    """Stable id for a critique, so a later draw can avoid reusing it."""
    return hashlib.sha256(critique.encode()).hexdigest()[:16]


def write_set(set_id: str, items: list[Item], key: dict[str, str],
              logs: dict[str, Path], signposting: dict) -> dict[str, Path]:
    """Questions in one file, answers in another, base64 over the answers.

    The encoding is a blindfold, not a secret — anyone can decode it in one
    line, and that is the point: it cannot be read by accident or by glancing
    at the directory, so looking becomes a deliberate act.
    """
    p = paths(set_id)
    questions = [{"item_id": i.item_id, "work_title": i.work_title,
                  "work_content": i.work_content, "critique": i.critique}
                 for i in items]
    p["items"].write_text(json.dumps(
        {"set_id": set_id, "seed": SAMPLE_SEED, "key": key,
         "logs": {k: str(v) for k, v in logs.items()},
         "note": "Questions only. The answers are in " + p["truth"].name,
         "items": questions}, indent=2) + "\n")
    answers = {i.item_id: {**{f: getattr(i, f) for f in ANSWERS},
                           "fingerprint": fingerprint(i.critique)} for i in items}
    payload = json.dumps({"set_id": set_id, "signposting": signposting,
                          "answers": answers}, indent=2)
    p["truth"].write_text(
        "# base64. A blindfold, not a secret — decoding it before you label\n"
        "# turns your labels into a copy of the answers, which is how the\n"
        "# first draw's score labels had to be thrown away.\n"
        + base64.b64encode(payload.encode()).decode() + "\n")
    return p


def read_truth(path: Path) -> dict:
    body = "".join(l for l in path.read_text().splitlines()
                   if not l.startswith("#"))
    return json.loads(base64.b64decode(body))


def read_items(set_id: str) -> tuple[list[Item], dict[str, str], dict]:
    p = paths(set_id)
    blob = json.loads(p["items"].read_text())
    truth = read_truth(p["truth"])
    items = [Item(**{**q, **{f: truth["answers"][q["item_id"]][f] for f in ANSWERS}})
             for q in blob["items"]]
    return items, blob["key"], truth.get("signposting", {})


def used_fingerprints() -> set[str]:
    """Every critique already shown to a rater, across all label sets.

    A rater who has seen an item's answer cannot be asked about it again, so a
    fresh draw skips them rather than quietly re-using them.
    """
    used = set()
    for path in sorted(LABEL_DIR.glob("truth_*.b64")):
        try:
            truth = read_truth(path)
        except Exception:
            continue
        for a in truth.get("answers", {}).values():
            if a.get("fingerprint"):
                used.add(a["fingerprint"])
    return used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("sheet", "estimate", "run", "score"))
    ap.add_argument("--set", dest="set_id", default=None,
                    help="label set id (default: the paired run's stem)")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--confirm", action="store_true",
                    help="required by `run`: yes, spend money on API calls")
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    args = ap.parse_args()

    LABEL_DIR.mkdir(exist_ok=True)
    logs = default_pair()
    set_id = args.set_id or logs["feed"].stem
    p = paths(set_id)

    if args.command == "sheet":
        dispositions = load_dispositions(list(logs.values()))
        if p["items"].exists() and p["truth"].exists():
            items, key, _ = read_items(set_id)
            print(f"  reusing the existing sample of {len(items)} "
                  f"({p['items']}) — delete it to draw a new one")
        else:
            used = used_fingerprints()
            items = sample_items(logs, dispositions, args.n, exclude=used)
            if used:
                print(f"  skipping {len(used)} critiques already shown to a rater")
            if len(items) < args.n:
                print(f"  only {len(items)} unseen critiques available")
            key = key_for(sorted(dispositions))
            write_set(set_id, items, key, logs, signposting_rate(logs, dispositions))
            print(f"  wrote {p['items']} ({len(items)} items)")
            print(f"  wrote {p['truth']} — the answers, base64. Do not decode it "
                  "before labelling.")
        if p["sheet"].exists():
            filled, _ = parse_sheet(p["sheet"].read_text())
            answered = sum(1 for v in filled.values() if v["wrote"] or v["score"] is not None)
            if answered:
                sys.exit(f"  {p['sheet']} already has {answered} answers in it — "
                         "not overwriting. Move it aside first.")
        p["sheet"].write_text(sheet(items, key, dispositions))
        print(f"  wrote {p['sheet']} — fill in the answer lines, then:")
        print(f"      uv run python judge.py estimate --set {set_id}")
        by_critic: dict[str, int] = {}
        for it in items:
            by_critic[f"{it.arm}/{it.critic}"] = by_critic.get(f"{it.arm}/{it.critic}", 0) + 1
        print(f"  sample: {len(set(i.arm for i in items))} arms, "
              f"rounds {min(i.round for i in items)}-{max(i.round for i in items)}, "
              f"{len(by_critic)} arm/critic cells, {min(by_critic.values())}-"
              f"{max(by_critic.values())} items each")
        return

    if not (p["items"].exists() and p["truth"].exists()):
        sys.exit(f"No sample yet. Run: uv run python judge.py sheet --set {set_id}")
    items, key, signposting = read_items(set_id)
    dispositions = load_dispositions(list(logs.values()))

    if args.command == "estimate":
        price = ((args.price_in, args.price_out)
                 if args.price_in is not None and args.price_out is not None
                 else PRICING.get(args.judge_model))
        est = estimate(items, dispositions, args.judge_model, price)
        print(f"\n  {est['calls']} calls to {est['model']} over {est['items']} items")
        print(f"  roughly {est['input_tokens']:,} input + {est['output_tokens']:,} "
              "output tokens")
        if "usd" in est:
            print(f"  about ${est['usd']:.2f} at ${price[0]}/${price[1]} per MTok")
        else:
            print(f"  no confirmed rate for {est['model']} — pass --price-in and "
                  "--price-out to get a figure")
        if p["sheet"].exists():
            filled, problems = parse_sheet(p["sheet"].read_text())
            done = sum(1 for v in filled.values() if v["wrote"] and v["score"] is not None)
            print(f"\n  hand labels: {done}/{len(items)} items fully answered"
                  + (f", {len(problems)} unreadable lines" if problems else ""))
            for msg in problems[:5]:
                print(f"      {msg}")
        return

    if args.command == "run":
        if not args.confirm:
            sys.exit("`run` makes paid API calls. Re-run with --confirm when you "
                     "mean it, or use `estimate` first.")
        print(f"  judging {len(items)} items with {args.judge_model}")
        answers = asyncio.run(run_judge(items, dispositions, args.judge_model))
        errors = [(i, t, a[t]["error"]) for i, a in answers.items()
                  for t in a if "error" in a[t]]
        p["judge"].write_text(json.dumps(
            {"set_id": set_id, "model": args.judge_model, "answers": answers},
            indent=2) + "\n")
        print(f"  wrote {p['judge']}"
              + (f" — {len(errors)} of {2*len(items)} calls failed" if errors else ""))
        for i, t, msg in errors[:5]:
            print(f"      {i}/{t}: {msg}")
        return

    # score
    if not p["sheet"].exists():
        sys.exit(f"No sheet at {p['sheet']}.")
    human, problems = parse_sheet(p["sheet"].read_text())
    for msg in problems:
        print(f"  sheet: {msg}")
    judge = (json.loads(p["judge"].read_text())["answers"]
             if p["judge"].exists() else {})
    if not judge:
        print(f"  no judge answers yet ({p['judge']}) — human labels only")
    result = score_labels(items, human, judge, key)
    result["signposting"] = signposting
    p["result"].write_text(json.dumps(result, indent=2) + "\n")

    w, s = result["wrote"], result["score"]
    print(f"\n  WHO WROTE IT — chance {w['chance']}")
    print(f"    human {w['human_accuracy']} (n={w['n_human']})   "
          f"judge {w['judge_accuracy']} (n={w['n_judge']})")
    print(f"    the two agree on {w['agree_rate']} of {w['n_both']} shared items, "
          f"kappa {w['kappa']}")
    if w.get("by_signposting"):
        split = "   ".join(f"{k} {v}" for k, v in sorted(w["by_signposting"].items()))
        print(f"    judge, split by whether the critique had named its influence: {split}")
    if result.get("signposting"):
        for arm, d in sorted(result["signposting"].items()):
            print(f"    ({arm}: {d['named_own_influence']}/{d['n']} critiques named "
                  f"their own influence, {d['rate']:.0%} — redacted before labelling)")
    print(f"\n  RECOVERING THE SCORE — predicting the mean gives "
          f"MAE {s['mean_baseline_mae']}")
    for who in ("human", "judge"):
        d = s[who]
        print(f"    {who:<6} MAE {d['mae']}  rho {d['spearman']} "
              f"(p={d['p_value']}, n={d['n']})")
    print(f"    human vs judge rho {s['human_judge_spearman']} over {s['n_both']} items")
    print()
    for row in result["criteria"]:
        print(f"    [{row['verdict']:>12}] {row['check']:<46} {row['detail']}")
    for note in result.get("caveats", []):
        print("\n  CAVEAT — " + "\n           ".join(
            __import__("textwrap").wrap(note, 74)))
    print(f"\n  wrote {p['result']}")


if __name__ == "__main__":
    main()
