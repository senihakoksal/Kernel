"""Do the agents behave the way their prompts ask?

Distinct from the other two checkers. check_run.py asks whether a log is usable
data; analyze.py asks what the run shows. This asks whether the agents did what
they were told — a property of the prompts and the model, not of the kernel.

Everything here is free: deterministic rules, or the sentence-transformer that
already ships with the analysis extra. No API calls, so it can be run over every
log as often as you like. The judged checks that need a model — disposition
adherence, score/reasoning coherence — are deliberately NOT here; those cost
money, need a validated judge, and belong in their own tool.

A note on what "good scoring" can mean. There is no ground truth for whether an
artwork deserves 0.72, so nothing below claims a score is correct. What is
measurable is whether scores DISCRIMINATE between works, whether critics AGREE
beyond chance, and whether a score matches the sentiment of the text attached to
it. A score can be all three and still be wrong; it cannot be good without them.

Usage:
    uv run python eval_agents.py                      # newest run_*.jsonl
    uv run python eval_agents.py logs/run_X.jsonl
    uv run python eval_agents.py logs/run_X.jsonl --json out.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from agents import SHARED_SYSTEM_PROMPT
from schema import Record

LOG_DIR = Path("logs")
FIGURE_DIR = Path("figures")
EMBED_MODEL = "all-MiniLM-L6-v2"

# Near-duplicate threshold for "invent ONE NEW concept". Cosine on normalised
# MiniLM embeddings; 0.85 is tight enough that only genuine restatements trip it.
NOVELTY_LIMIT = 0.85

# Anchors for the sentiment proxy. Not a sentiment model — the difference
# between a critique's similarity to these two poles, which is crude but costs
# nothing and is enough to catch a score that contradicts its own text.
POS_ANCHOR = ("This work is excellent. It succeeds, it is accomplished, "
              "original and moving, and I admire it.")
NEG_ANCHOR = ("This work fails. It is weak, derivative and confused, "
              "it does not earn its effect, and I do not admire it.")


def band_for(path: Path) -> tuple[tuple[int, int], str]:
    """The length rule THIS run was given, preferring its own sidecar.

    Reading the live prompt would re-judge every past run against whatever the
    band says today, so a prompt edit would silently turn old rows red for a
    rule those agents never received. Runs written before the band was recorded
    fall back to the current prompt, which is the best available guess and is
    labelled as such on the page.
    """
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        band = json.loads(meta_path.read_text()).get("content_word_band")
        if band:
            return (int(band[0]), int(band[1])), "recorded with the run"
    m = re.search(r"between (\d+) and (\d+) words", SHARED_SYSTEM_PROMPT)
    return ((int(m.group(1)), int(m.group(2))) if m else (0, 10**6),
            "from the current prompt — this run predates the record")


def load(path: Path) -> list[Record]:
    return [Record(**json.loads(line)) for line in path.read_text().splitlines()
            if line.strip()]


# --- Tier 0: the stated contract, per agent ----------------------------------
def conformance(records: list[Record], band: tuple[int, int],
                band_source: str) -> dict:
    """Compliance with each rule the system prompt states, agent by agent.

    Reported per agent rather than per run because the interesting failures are
    not uniform: on run_20260910_154044 the artists hold the word band perfectly
    and the critics miss it four times out of five.
    """
    lo, hi = band
    rows = {}
    for rec in records:
        r = rows.setdefault(rec.agent, {"role": rec.role, "n": 0, "in_band": 0,
                                        "title_ok": 0, "titled": 0, "reason_ok": 0,
                                        "score_ok": 0, "scored": 0, "field_ok": 0})
        words = len(rec.content.split())
        r["n"] += 1
        r["in_band"] += lo <= words <= hi
        # "one or two sentences" — count terminators, allowing a trailing one.
        r["reason_ok"] += len(re.findall(r"[.!?](?:\s|$)", rec.reasoning or "")) <= 2
        if rec.role == "artist":
            r["titled"] += rec.title is not None
            r["title_ok"] += bool(rec.title) and len(rec.title.split()) <= 6
            r["field_ok"] += rec.score is None           # artists omit score
        else:
            r["scored"] += rec.score is not None
            r["score_ok"] += rec.score is not None and 0.0 <= rec.score <= 1.0
            r["field_ok"] += rec.title is None           # critics omit title
    return {"band": (lo, hi), "band_source": band_source, "agents": rows}


# --- Tier 1: behaviour, measured with local embeddings -----------------------
def embed(model, texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def on_target(records: list[Record], model) -> dict:
    """Did each critic judge the work it was asked to judge?

    The task instruction names one concept. If a critique sits closer to a
    different work of the same round, the agent drifted off-task. Reported as
    top-1 accuracy and mean rank of the true target, both per round, since
    drift is most likely once the feed fills up.
    """
    by_round = defaultdict(lambda: {"concepts": [], "evals": []})
    for r in records:
        by_round[r.round]["concepts" if r.kind == "concept" else "evals"].append(r)

    per_round, hits, ranks = {}, 0, []
    for rnd in sorted(by_round):
        concepts = by_round[rnd]["concepts"]
        evals = [e for e in by_round[rnd]["evals"] if e.concept_id]
        if len(concepts) < 2 or not evals:
            continue
        cvec = embed(model, [c.content for c in concepts])
        evec = embed(model, [e.content for e in evals])
        ids = [c.concept_id for c in concepts]
        sims = evec @ cvec.T
        r_hits, r_ranks = 0, []
        for i, e in enumerate(evals):
            if e.concept_id not in ids:
                continue
            target = ids.index(e.concept_id)
            order = np.argsort(-sims[i])
            rank = int(np.where(order == target)[0][0]) + 1
            r_hits += rank == 1
            r_ranks.append(rank)
        per_round[rnd] = {"n": len(r_ranks), "top1": r_hits / len(r_ranks),
                          "mean_rank": float(np.mean(r_ranks)), "of": len(concepts)}
        hits += r_hits
        ranks += r_ranks
    return {"per_round": per_round, "top1": hits / len(ranks) if ranks else None,
            "mean_rank": float(np.mean(ranks)) if ranks else None,
            "chance_top1": 1 / np.mean([v["of"] for v in per_round.values()])
            if per_round else None}


def novelty(records: list[Record], model) -> dict:
    """Is each artwork actually new? The artist is told to invent ONE NEW concept."""
    concepts = sorted([r for r in records if r.kind == "concept"],
                      key=lambda r: (r.round,))
    if len(concepts) < 2:
        return {"n": len(concepts), "near_duplicates": [], "max_similarity": None}
    vecs = embed(model, [c.content for c in concepts])
    worst, dupes = [], []
    for i, c in enumerate(concepts):
        prior = [j for j, p in enumerate(concepts) if p.round < c.round]
        if not prior:
            continue
        sims = vecs[i] @ vecs[prior].T
        k = int(np.argmax(sims))
        worst.append(float(sims[k]))
        if sims[k] >= NOVELTY_LIMIT:
            dupes.append({"work": c.concept_id, "closest": concepts[prior[k]].concept_id,
                          "similarity": round(float(sims[k]), 3)})
    return {"n": len(worst), "near_duplicates": dupes,
            "max_similarity": round(max(worst), 3) if worst else None,
            "mean_closest": round(float(np.mean(worst)), 3) if worst else None}


def score_reliability(records: list[Record]) -> dict:
    """Do critics agree about which works are good, beyond chance?

    Two readings. Mean pairwise Spearman is how much any two critics rank the
    same works alike. The permutation test asks whether the spread of per-work
    mean scores is larger than you would get by shuffling the scores among the
    works of the same round — if critics were rating at random, between-work
    variance would collapse to that null.
    """
    ev = [r for r in records if r.kind == "evaluation" and r.score is not None
          and r.concept_id]
    if not ev:
        return {}
    table = defaultdict(dict)                 # critic -> work -> score
    for r in ev:
        table[r.agent][r.concept_id] = r.score

    critics = sorted(table)
    rhos = []
    for i in range(len(critics)):
        for j in range(i + 1, len(critics)):
            shared = sorted(set(table[critics[i]]) & set(table[critics[j]]))
            if len(shared) < 3:
                continue
            a = [table[critics[i]][w] for w in shared]
            b = [table[critics[j]][w] for w in shared]
            rho = stats.spearmanr(a, b).statistic
            if not np.isnan(rho):
                rhos.append(rho)

    by_work = defaultdict(list)
    work_round = {}
    for r in ev:
        by_work[r.concept_id].append(r.score)
        work_round[r.concept_id] = r.round
    observed = float(np.var([np.mean(v) for v in by_work.values()]))

    rng = np.random.default_rng(0)            # seeded: the test must be reproducible
    rounds = defaultdict(list)
    for r in ev:
        rounds[r.round].append(r)
    null = []
    for _ in range(2000):
        shuffled = defaultdict(list)
        for rnd, group in rounds.items():
            scores = rng.permutation([r.score for r in group])
            for r, s in zip(group, scores):
                shuffled[r.concept_id].append(s)
        null.append(np.var([np.mean(v) for v in shuffled.values()]))
    null = np.array(null)
    return {"n_critics": len(critics), "n_works": len(by_work),
            "mean_pairwise_spearman": round(float(np.mean(rhos)), 3) if rhos else None,
            "pairwise_range": [round(float(min(rhos)), 3), round(float(max(rhos)), 3)]
            if rhos else None,
            "between_work_variance": round(observed, 5),
            "null_mean": round(float(null.mean()), 5),
            "p_value": round(float((null >= observed).mean()), 4)}


def score_text_coherence(records: list[Record], model) -> dict:
    """Does the number agree with the prose it is attached to?

    A proxy, not a sentiment model: each critique's similarity to a positive
    anchor minus its similarity to a negative one, correlated against the score.
    Crude, and it costs nothing — enough to catch a critic whose numbers
    contradict its own writing, not enough to grade tone.
    """
    ev = [r for r in records if r.kind == "evaluation" and r.score is not None]
    if len(ev) < 5:
        return {}
    anchors = embed(model, [POS_ANCHOR, NEG_ANCHOR])
    vecs = embed(model, [r.content for r in ev])
    polarity = vecs @ anchors[0] - vecs @ anchors[1]
    scores = np.array([r.score for r in ev])
    overall = stats.spearmanr(polarity, scores)
    per_critic = {}
    by_critic = defaultdict(list)
    for r, p in zip(ev, polarity):
        by_critic[r.agent].append((p, r.score))
    for critic, pairs in sorted(by_critic.items()):
        if len(pairs) < 5:
            continue
        rho = stats.spearmanr([p for p, _ in pairs], [s for _, s in pairs]).statistic
        per_critic[critic] = round(float(rho), 3) if not np.isnan(rho) else None
    return {"n": len(ev), "spearman": round(float(overall.statistic), 3),
            "p_value": round(float(overall.pvalue), 5), "per_critic": per_critic}


def disposition_separability(records: list[Record], model) -> dict:
    """Can a critic be identified from its writing alone?

    Leave-one-out nearest-centroid over critique embeddings: for each critique,
    build each critic's centroid from their OTHER critiques and assign to the
    nearest. Above chance means the dispositions are doing something to the
    text. Tracked per round, a decline is convergence measured through a
    completely different instrument than the vocabulary overlap metric — which
    is worth having, since that metric has a lot of machinery behind it.
    """
    ev = [r for r in records if r.kind == "evaluation"]
    critics = sorted({r.agent for r in ev})
    if len(critics) < 2 or len(ev) < len(critics) * 3:
        return {}
    vecs = embed(model, [r.content for r in ev])
    idx = {c: np.array([i for i, r in enumerate(ev) if r.agent == c]) for c in critics}

    correct, per_round = 0, defaultdict(lambda: [0, 0])
    for i, r in enumerate(ev):
        best, best_sim = None, -2.0
        for c in critics:
            others = idx[c][idx[c] != i]
            if len(others) == 0:
                continue
            sim = float(vecs[i] @ vecs[others].mean(axis=0))
            if sim > best_sim:
                best, best_sim = c, sim
        hit = best == r.agent
        correct += hit
        per_round[r.round][0] += hit
        per_round[r.round][1] += 1
    rounds_acc = {int(k): round(v[0] / v[1], 3) for k, v in sorted(per_round.items())}
    trend = {}
    if len(rounds_acc) >= 4:
        xs, ys = list(rounds_acc), list(rounds_acc.values())
        fit = stats.linregress(xs, ys)
        rho = stats.spearmanr(xs, ys)
        trend = {"slope_pts_per_round": round(float(fit.slope) * 100, 2),
                 "slope_p": round(float(fit.pvalue), 4),
                 "spearman": round(float(rho.statistic), 3),
                 "spearman_p": round(float(rho.pvalue), 4)}
    return {"n": len(ev), "n_critics": len(critics),
            "accuracy": round(correct / len(ev), 3),
            "chance": round(1 / len(critics), 3),
            "per_round": rounds_acc, "trend": trend}


# --- Criteria ----------------------------------------------------------------
# Thresholds are conventions, not laws, so they are named, stated on the page,
# and kept in one place to be argued with. They cover only the checks that have
# a right answer: did the agents do what the prompt asked, and is the scoring
# usable. Convergence is NOT among them — critics growing alike is the thing
# the project is studying, so a red light on it would be a category error. Those
# appear on the page as measurements, with no verdict attached.
THRESHOLDS = {
    "length":        {"pass": 0.80, "warn": 0.50},
    "on_target":     {"pass": 0.90, "warn": 0.75},
    "reasoning":     {"pass": 0.90, "warn": 0.75},
    "fields":        {"pass": 1.00, "warn": 0.99},
    "title":         {"pass": 0.95, "warn": 0.85},
    "scores_valid":  {"pass": 1.00, "warn": 0.99},
    "novelty":       {"pass": 1.00, "warn": 0.98},   # share of works that are new
    "discrimination": {"pass": 0.05, "warn": 0.02},  # min per-critic score sd
}


WHAT = {
    "length": "The prompt asks for every artwork and critique to fall between {lo} "
              "and {hi} words. This counts how many did. Failing it does not make a "
              "run unusable — it means the length instruction is not landing, which "
              "matters because output length is what the rate limit meters.",
    "reasoning": "Alongside its contribution each agent returns a short rationale, "
                 "and the prompt asks for one or two sentences. This counts how many "
                 "stayed that short.",
    "fields": "Artists must leave out the score, critics must leave out the title. A "
              "record carrying the wrong field means the agent misread which role it "
              "was playing.",
    "title": "Artists are asked for a title of at most six words. This counts how "
             "many obliged.",
    "scores_valid": "Every critique has to carry a score between 0 and 1. A missing "
                    "or out-of-range score breaks every measurement downstream of it.",
    "on_target": "Each critic is handed one specific artwork to judge. This checks "
                 "the critique is about that work: it compares the critique's wording "
                 "against every artwork of that round, and counts it on target when "
                 "the assigned one is the closest. If a critique is nearer some other "
                 "work, the critic drifted off the task it was given.",
    "novelty": "Artists are told to invent a NEW concept each round. This compares "
               "every artwork against every earlier one and flags any that is close "
               "enough to count as a restatement rather than a new work.",
    "discrimination": "A critic that gives everything the same score is not "
                      "evaluating, and one such critic is invisible in an average. "
                      "This takes the least varying critic in the run and checks that "
                      "its scores actually spread.",
}


def verdict(value, key, higher_is_better=True):
    """pass / warn / fail against a named threshold, or None when inapplicable."""
    if value is None:
        return None
    t = THRESHOLDS[key]
    if higher_is_better:
        return "pass" if value >= t["pass"] else "warn" if value >= t["warn"] else "fail"
    return "pass" if value <= t["pass"] else "warn" if value <= t["warn"] else "fail"


def criteria_checks(result: dict) -> list[dict]:
    """The pass/fail view: one row per criterion, per role where it differs.

    Each row carries the measured value, the threshold it was judged against,
    the verdict and a group. ORDER MATTERS: the behavioural checks come first —
    whether a critic judged the work it was handed, whether an artwork is new,
    whether a critic discriminates at all — because those are about what the
    agents did. Format compliance follows. A reader scanning left to right
    should meet the substantive checks before the word counts, and the page
    builds its columns and its glossary from this list so both follow it.
    """
    conf = result["conformance"]
    lo, hi = conf["band"]
    rows = []          # behaviour: what the agents did
    contract = []      # format: how they packaged it

    by_role = {}
    for a in conf["agents"].values():
        r = by_role.setdefault(a["role"], {k: 0 for k in
                                           ("n", "in_band", "reason_ok", "field_ok",
                                            "title_ok", "titled", "score_ok", "scored")})
        for k in r:
            r[k] += a[k]

    t = result["on_target"]
    if t.get("top1") is not None:
        rows.append({"check": "critiques judge the assigned work", "group": "task",
                     "value": t["top1"], "detail": f"{t['top1']:.0%} (chance {t['chance_top1']:.0%})",
                     "rule": f">= {THRESHOLDS['on_target']['pass']:.0%} nearest to the right work",
                     "what": WHAT["on_target"],
                     "verdict": verdict(t["top1"], "on_target")})

    n = result["novelty"]
    if n.get("n"):
        share = (n["n"] - len(n["near_duplicates"])) / n["n"]
        rows.append({"check": "artworks are not restatements", "group": "task",
                     "value": share,
                     "detail": f"{n['n'] - len(n['near_duplicates'])}/{n['n']}",
                     "rule": f"no work >= {NOVELTY_LIMIT} similar to an earlier one",
                     "what": WHAT["novelty"],
                     "verdict": verdict(share, "novelty")})

    sd = result.get("discrimination")
    if sd:
        rows.append({"check": "every critic discriminates", "group": "scoring",
                     "value": sd["min_sd"],
                     "detail": f"lowest spread {sd['min_sd']:.3f} ({sd['min_critic']})",
                     "rule": f"every critic's score sd >= {THRESHOLDS['discrimination']['pass']}",
                     "what": WHAT["discrimination"],
                     "verdict": verdict(sd["min_sd"], "discrimination")})
    for role, v in sorted(by_role.items()):
        if not v["n"]:
            continue
        contract.append({"check": f"{role}s: content {lo}-{hi} words", "group": "contract",
                     "value": v["in_band"] / v["n"], "detail": f"{v['in_band']}/{v['n']}",
                     "rule": f">= {THRESHOLDS['length']['pass']:.0%} within the band",
                     "what": WHAT["length"].format(lo=lo, hi=hi),
                     "verdict": verdict(v["in_band"] / v["n"], "length")})
        contract.append({"check": f"{role}s: reasoning <= 2 sentences", "group": "contract",
                     "value": v["reason_ok"] / v["n"], "detail": f"{v['reason_ok']}/{v['n']}",
                     "rule": f">= {THRESHOLDS['reasoning']['pass']:.0%}",
                     "what": WHAT["reasoning"],
                     "verdict": verdict(v["reason_ok"] / v["n"], "reasoning")})
        contract.append({"check": f"{role}s: role fields correct", "group": "contract",
                     "value": v["field_ok"] / v["n"], "detail": f"{v['field_ok']}/{v['n']}",
                     "rule": "every record (artists omit score, critics omit title)",
                     "what": WHAT["fields"],
                     "verdict": verdict(v["field_ok"] / v["n"], "fields")})
        if role == "artist" and v["titled"]:
            contract.append({"check": "artists: title <= 6 words", "group": "contract",
                         "value": v["title_ok"] / v["titled"],
                         "detail": f"{v['title_ok']}/{v['titled']}",
                         "rule": f">= {THRESHOLDS['title']['pass']:.0%}",
                         "what": WHAT["title"],
                         "verdict": verdict(v["title_ok"] / v["titled"], "title")})
        if role == "critic" and v["scored"]:
            contract.append({"check": "critics: score present and in range", "group": "contract",
                         "value": v["score_ok"] / v["n"], "detail": f"{v['score_ok']}/{v['n']}",
                         "rule": "every critique, 0.0-1.0",
                         "what": WHAT["scores_valid"],
                         "verdict": verdict(v["score_ok"] / v["n"], "scores_valid")})

    return rows + contract


def discrimination(records: list[Record]) -> dict:
    """The least discriminating critic. One rubber-stamper is invisible in a mean."""
    per = defaultdict(list)
    for r in records:
        if r.kind == "evaluation" and r.score is not None:
            per[r.agent].append(r.score)
    sds = {c: float(np.std(v)) for c, v in per.items() if len(v) > 2}
    if not sds:
        return {}
    worst = min(sds, key=sds.get)
    return {"min_sd": round(sds[worst], 4), "min_critic": worst,
            "per_critic": {c: round(v, 4) for c, v in sorted(sds.items())}}


# --- Reporting ---------------------------------------------------------------
def report(path: Path, result: dict) -> None:
    lo, hi = result["conformance"]["band"]
    print(f"\n{path.name}")
    print(f"\n  CONTRACT — compliance with the prompt's stated rules")
    print(f"  {'agent':<18}{'role':<9}{'n':>4}{f'  {lo}-{hi} words':>14}"
          f"{'title<=6':>10}{'reason<=2':>11}{'fields':>8}")
    for name, a in sorted(result["conformance"]["agents"].items(),
                          key=lambda kv: (kv[1]["role"], kv[0])):
        n = a["n"]
        title = f"{a['title_ok']}/{a['titled']}" if a["role"] == "artist" else "—"
        print(f"  {name:<18}{a['role']:<9}{n:>4}{f'{a['in_band']}/{n}':>14}"
              f"{title:>10}{f'{a['reason_ok']}/{n}':>11}{f'{a['field_ok']}/{n}':>8}")

    t = result["on_target"]
    if t.get("top1") is not None:
        print(f"\n  ON TARGET — did the critic judge the work it was asked to judge")
        print(f"    top-1 {t['top1']*100:.1f}%  (chance {t['chance_top1']*100:.1f}%)   "
              f"mean rank {t['mean_rank']:.2f}")
        print("    per round: " + "  ".join(
            f"r{k} {v['top1']*100:.0f}%" for k, v in t["per_round"].items()))

    n = result["novelty"]
    if n.get("max_similarity") is not None:
        print(f"\n  NOVELTY — is each artwork new")
        print(f"    closest prior work: mean {n['mean_closest']}, max {n['max_similarity']}"
              f"   near-duplicates (>= {NOVELTY_LIMIT}): {len(n['near_duplicates'])}")
        for d in n["near_duplicates"][:3]:
            print(f"      {d['work']} ~ {d['closest']}  {d['similarity']}")

    s = result["reliability"]
    if s:
        print(f"\n  SCORE RELIABILITY — do critics agree beyond chance")
        print(f"    mean pairwise Spearman {s['mean_pairwise_spearman']} "
              f"(range {s['pairwise_range'][0]} to {s['pairwise_range'][1]})")
        print(f"    between-work variance {s['between_work_variance']} vs shuffled "
              f"{s['null_mean']}   p = {s['p_value']}")

    c = result["coherence"]
    if c:
        print(f"\n  SCORE vs TEXT — does the number match the prose")
        print(f"    Spearman {c['spearman']}  (p = {c['p_value']}, n = {c['n']})")
        worst = sorted(((v, k) for k, v in c["per_critic"].items() if v is not None))[:2]
        for v, k in worst:
            print(f"      weakest: {k} {v}")

    d = result["separability"]
    if d:
        print(f"\n  DISPOSITION SEPARABILITY — can a critic be identified by its writing")
        print(f"    accuracy {d['accuracy']*100:.1f}%  (chance {d['chance']*100:.1f}%, "
              f"{d['n_critics']} critics)")
        print("    per round: " + "  ".join(
            f"r{k} {v*100:.0f}%" for k, v in d["per_round"].items()))
        if d.get("trend"):
            t = d["trend"]
            print(f"    trend {t['slope_pts_per_round']:+.2f} pts/round "
                  f"(p = {t['slope_p']})")


def main() -> None:
    p = argparse.ArgumentParser(description="Do the agents behave as their prompts ask?")
    p.add_argument("log", nargs="?", type=Path, help="run log (default: newest)")
    p.add_argument("--json", type=Path,
                   help="write the numbers here instead of figures/eval_<run_id>.json")
    p.add_argument("--quiet", action="store_true", help="write the file, skip the report")
    args = p.parse_args()

    path = args.log
    if path is None:
        found = sorted(LOG_DIR.glob("run_*.jsonl"))
        if not found:
            sys.exit("No run_*.jsonl logs found.")
        path = found[-1]
    if not path.exists():
        sys.exit(f"Log not found: {path}")

    records = load(path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    result = {
        "run_id": path.stem,
        "conformance": conformance(records, *band_for(path)),
        "on_target": on_target(records, model),
        "novelty": novelty(records, model),
        "reliability": score_reliability(records),
        "coherence": score_text_coherence(records, model),
        "separability": disposition_separability(records, model),
        "discrimination": discrimination(records),
    }
    result["criteria"] = criteria_checks(result)
    if not args.quiet:
        report(path, result)
    # Written by default, like analyze.py's summaries: a result nobody records
    # is a result nobody can compare against the next run.
    FIGURE_DIR.mkdir(exist_ok=True)
    out = args.json or FIGURE_DIR / f"eval_{path.stem}.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
