# Observation Kernel

Twelve agents in a studio. Six make art, six judge it. What I actually wanted to
know is whether culture can emerge in a society of agents — but that's far too
big to measure, so this asks the smallest version of it I could think of: when
critics can read each other, does a shared critical vocabulary actually *form*
between them, or do they just sound alike because they're all running on the
same model underneath?

The **artists** write short artwork *concepts* — text, never images. The
**critics** read them and write critiques with a score. Everything goes into a
shared feed and gets logged. One model throughout (`claude-sonnet-4-6`, set as
`MODEL` in [agents.py](agents.py)).

**→ [Browse the run](https://senihakoksal.github.io/Kernel/)** — every artwork,
every critique, every score, and the figures. Read the artwork, it's actually
really fun.

**Scope: one treatment/control pair — 6 artists, 6 critics, 8 rounds, 48 works,
288 critiques** (`run_20260910_154044`). The system runs end to end, both arms
are built, and the numbers below are what came out. They tell you how big the
effect is. One pair doesn't tell you it repeats — [Limits](#limits) is specific
about what that does and doesn't support.

---

## Who's in the studio

Twelve agents, each with a written disposition in
[agents.yaml](agents.yaml) — two or three sentences of character that go into
the prompt after the feed. They're the highest-leverage thing in the whole
system, and they took longer to write than the code.

The **artists** work after Mary Oliver, Agnes Martin, Yoko Ono, Nazım Hikmet,
David Salle and Judy Chicago — so the studio puts instruction pieces next to
compressed lyric next to minimal restraint next to work that fractures a
subject and rebuilds it. The critics get something genuinely varied to disagree
about, which is the point: a roster that agrees with itself can't produce a
culture, only an echo.

The **critics** work after Sontag, Benjamin, Barthes, Ruskin, Baudelaire and
Szeemann. Five writers and one curator, which matters: the writers all stand in
front of a finished work and read it, and Szeemann is the only one who ever had
to get difficult art into an actual room. That's the disposition that asks
whether a piece could be made at all.

Nobody is named after a real person and nobody speaks for one — these are
sensibilities, not impersonations. And the pairing is deliberately crossed: the
inspiration and the working method don't match. `pablo_reyes` is pointed at Judy
Chicago but told to fracture a subject and rebuild it from several angles at
once; `theo_malik` is pointed at David Salle but told to make short media-based
work about technology and cultural change.

That's the whole trick. Give the model only a name and it retrieves — it
produces competent pastiche of an artist it already knows. Give it a name plus
someone else's method and it can't retrieve, it has to combine, and what comes
out is a work neither figure would have made. Which is the point: I wanted
artworks, not replicas.

## Why the artwork is text

Rendering each concept to an image and asking the critics to read it back would
put an image model's interpretation between the artist and the critic, and any
drift between intent and reading would belong to that model rather than to the
agents. It would also settle, in one particular way, a work the reader should be
finishing themselves.

Resolution doesn't rescue that. You can tile an image, or hand the critic a
zoom tool and let it choose where to look — both work, and both make the
picture sharper without making it any less *chosen*. The loss isn't in the
pixels.

Instruction pieces have worked like this for sixty years — the score is the
work, the reader completes it. That happens to be the form a language model is
already native to.

There's a cost argument underneath, and it's bigger than it first looks. A
rendered image costs roughly twelve times its text description to carry in a
prompt — about 1,500 tokens against 120 — and because every concept stays in
the feed for the rest of the run, that multiplier lands on *every* call, not
just the one that made it. Working in text keeps the whole run at something
like a quarter to a third of the input tokens the image version would need,
before paying to generate a single picture. Estimated rather than measured: the
logs don't record token usage, which is a small gap I'd close if I were
carrying this on.

## The question

Does an aesthetic descriptor coined by one critic — one that appears in no
starting prompt — get picked up by a *different* critic in a later round?

That question is very easy to answer wrongly, which is most of why this repo
exists. Six critics looking at the same artwork will reach for similar words
without influencing each other at all. So before I could count anything, I
needed a way to tell those two apart.

## The design, and the part that actually matters

Two arms, same artworks:

- **Treatment** ([run.py](run.py)) — each critic sees the shared feed: the
  artworks, plus every other critic's earlier critiques, signed with their names.
- **Control** ([control.py](control.py)) — the treatment's artworks get
  **replayed** unchanged, and each critic sees only the artworks plus **its own**
  earlier critiques. Never a peer's.

The control is the whole design. An isolated critic cannot have read whoever
first used a word, so **anything the propagation detector flags in the control
is the detector being wrong.** That gets you a false-positive rate measured on
your own instrument, against ground truth that's negative by construction.
Almost nobody in this area bothers, and it turned out to matter a lot.

Round 0 does the same job for a different number. Nothing has been written yet,
so both arms get byte-identical prompts — checked, not assumed. Any gap at
round 0 is noise, and it tells you how much noise to expect everywhere else.

Two more things keep the comparison honest:

- **Prior-vocabulary subtraction.** Anything already in the system prompt or an
  agent's disposition gets pulled out before counting, matched by meaning as
  well as exact wording. What's left is language the system didn't start with.
- **One pooled vocabulary space.** Both arms are clustered together, so neither
  ends up with coarser clusters that would inflate its overlap for free.

Agents see every concept ever made, but only critiques from the last
`FEED_WINDOW` rounds (default 3). The log keeps everything; the visible feed is
a rolling window over it. That's deliberate. In an unbounded feed a word coined
in round 1 is technically still there at round 9 and functionally invisible, and
a word nobody repeats should fall out of view rather than sit in a log nobody
rereads.

## What it found

One treatment/control pair. Everything below is n = 1.

**Shared vocabulary grows when critics can read each other, and doesn't when
they can't.** Mean pairwise Dice overlap went **9.1% → 14.4%** in the treatment
(+59.2%, first two rounds against last two). Same artworks, everyone isolated:
**9.1% → 9.5%**.

**Round 0 is what makes me believe that.** There the two arms are the same
condition, and the measured gap is **−0.4 points** — the control is very
slightly *ahead*. By the last round it's **+6.3 points** the other way. So the
effect is an order of magnitude bigger than the instrument's own noise, and it
starts out pointing backwards.

**Borrowing climbs in the treatment and sits still in the control.** Of all the
vocabulary a critic could have picked up from someone else while it was still
visible in the feed, **2.24% was in use in the treatment** (630 of 28,161
chances) against **1.28% in the control** (369 of 28,725) — a ratio of 1.74×.
Split into halves, the treatment goes 1.81% → 2.44% and the control goes 1.285%
→ 1.284%, which is about as flat as a number gets. The two pools are now within
2% of each other in size, so this isn't the denominator artefact it used to be.

**The propagation detector can't carry a claim on its own.** It found 354
cross-critic adoptions in the treatment — and **268 in the control**, where
that's impossible by construction. So for every 100 adoptions it reports where
transmission can happen, it reports 76 where it can't. The overlap and
borrowing numbers are what the result rests on. Don't read 354 as 354 real
cases; I don't.

**Critics who read each other did not agree more.** Score spread averaged 0.042
in the treatment against 0.033 in the control — if anything they agreed
slightly *less*, which I wasn't expecting. It's well inside noise for eight
rounds of six numbers, so it means no difference was detected, not that there
isn't one.

**The two arms differ in more than the peer channel, and the analysis says so
itself.** From round 1 the isolated critics are carrying their own
*regenerated* history — a control critic's earlier critiques are text it wrote
while alone, not what it wrote in the treatment. So taking away the peer feed
also changed what each critic remembers about itself. That means the contrast
is an upper bound on the peer effect rather than a measurement of it.

One more thing worth saying out loud: an earlier five-round run with a smaller
roster had both arms growing at nearly identical rates, no gap opening at all.
I believed that for a while. It didn't survive eight rounds. The divergence is
the finding; the earlier flatness was just stopping too early.

## What I got wrong

Probably the most useful section here. These are measurement bugs I found in my
own code *after* it had already given me results I was happy with.

**A filter that was a tautology.** `find_propagation` picked out adopters with
`grp["round"] >= first_round` — where `first_round` is the minimum of that same
column. Always true. So every critic who used a word in the round it first
appeared got counted as having adopted it from the coiner, including critics
who couldn't possibly have seen it, since critics in a round run in parallel. A
one-round log cheerfully reported 67 propagations in a situation where
propagation cannot happen at all. The fix separates co-coiners from actual
later adopters.

**Clustering that depended on alphabetical order.** Words were grouped in one
greedy pass down a sorted list, and whichever word got there first became the
permanent stand-in for its cluster. Change the order, change the results. It
also chained — A joins B, then C joins through B while being nothing like A.
Replaced with agglomerative clustering, complete linkage, cosine distance, plus
a test that shuffles the input and asserts the assignments come out identical.

**A significance test that wasn't one.** I ran a two-proportion z-test over
opportunity counts and got z = 4.38, p < 0.0001. It was invalid. Those
opportunities aren't independent observations — the thing being replicated is
the *run*, and there's one run. This is the mistake that most deserves a
reader's suspicion, precisely because the number looked so good.

**A metric measuring the wrong kind of memory.** Adoption eligibility used to
run forever: a word stayed "adoptable" in every later round, including rounds
where it had long since scrolled out of anything an agent could see. That also
inflated the control's pool by about 20% every round, which quietly flattered
the ratio. Eligibility is now tied to the feed window, a chance to borrow means
a chance the agent actually had, and the two pools come out within 2%.

None of these were caught by the tests. They were caught by asking what a
number should look like if there were no effect, and noticing it didn't.

## Limits

Three things this run can't settle. Each has a clear fix.

**The manipulation isn't clean.** As above — the isolated arm also regenerates
each critic's own history, so it differs from the treatment in a second way
besides peer visibility. The fix is a third arm where critics see peer
critiques replayed verbatim from the treatment instead of regenerated, holding
self-history fixed and changing only what a critic can read.

**It happened once.** One pair can't tell a real 59%-against-4% divergence from
one noisy draw. Five paired seeds could — five is the fewest that can reach
p < 0.05 on a paired sign test — and the pairing is already built into the
replay design, so it's runtime rather than redesign.

**The detector's sensitivity was never measured.** The control says how often it
fires when nothing's there: often, 268 times. Nothing says whether it fires when
something *is*. Plant a distinctive descriptor with no semantic neighbours in
one critic's disposition, tell it to use the word, and see whether the pipeline
catches the others picking it up. One short run, and it would tell me whether
the 354 is worth anything at all.

After that: commit the decision rules before running, so they can't be chosen
after seeing the data; check whether the critic dispositions are even
distinguishable, by stripping names off critiques and asking a held-out judge
who wrote what (at chance, they're costume); and report the *variance* of
pairwise overlap next to the mean, because a real critical culture might split
into two camps and an average would score that as nothing happening.

The repo is finished at this size and isn't being developed further. Anyone who
picks it up gets the control, the sweep, the tests, and all the bugs I already
hit.

## How it works

- **[run.py](run.py)** — the round loop. Each round the artists propose one
  concept each in parallel, then the critics evaluate them, also in parallel.
  Every action is one JSON line in `logs/run_<timestamp>.jsonl`, with the feed
  window and roster recorded beside it in `logs/run_<timestamp>.meta.json`.
- **[control.py](control.py)** — replays a run's artworks to isolated critics.
- **[analyze.py](analyze.py)** — pulls descriptors out of the critiques (spaCy
  noun phrases carrying at least one adjective), subtracts prior vocabulary,
  clusters what's left with `all-MiniLM-L6-v2` sentence embeddings, looks for
  cross-critic adoption, draws the figure. Takes both arms at once so they're
  clustered in one space.
- **[report.py](report.py)** — builds the published archive (`docs/index.html`,
  served by GitHub Pages): what got made, what got said about it. Stdlib only.
- **[agents.py](agents.py)** — the `Agent` class, the shared system prompt, the
  feed formatter and its window, JSON parsing. `MODEL` lives here.
- **[agents.yaml](agents.yaml)** — who's in the studio: name, role, disposition.
- **[schema.py](schema.py)** — `Record` and `ModelOutput`.
- **[check_run.py](check_run.py)** — is a log usable data? Completeness, score
  degeneracy, duplicate text, provenance. Exits non-zero, so it can gate.
- **[eval_agents.py](eval_agents.py)** — did the agents follow their prompts?
  See below.
- **[docs/threshold-sweep.md](docs/threshold-sweep.md)** — every number depends
  on `SIMILARITY_THRESHOLD`, so it got swept from 0.45 to 0.90 instead of
  assumed. Absolute overlap roughly triples across that range, but the early→late
  direction holds throughout. 0.72 is where the merges are clean synonyms
  (`final sentence · final paragraph · last sentence`) and nothing genuinely
  different has been merged yet.

## Evals

The analysis asks what a run showed. The evals ask whether the instrument was
behaving — whether the agents did what their prompts told them to. No model
judge, no ground truth, no API calls, so re-running after a prompt change costs
nothing. Four of the checks need to compare meaning rather than count things,
and use the same local `all-MiniLM-L6-v2` embeddings the clustering does —
87 MB on disk, no network; the rest is arithmetic. Published at
[evals.html](https://senihakoksal.github.io/Kernel/evals.html) as a matrix: runs
down, criteria across, so changing a prompt and re-running means reading down a
column for a cell that changed colour.

Eleven criteria get a pass, warn or fail — the ones with a right answer. Format
and role compliance; whether each critique judged the work it was actually given
(96%, against 17% if matched at random); whether artworks are new rather than
restatements; whether every critic discriminates rather than rubber-stamping.
One has failed in every run so far: the critics ignore the 80–180 word limit
while the artists keep to it perfectly, same instruction and same model.

Three things are measured without a verdict, because critics growing alike is
the subject of the study rather than a defect. Critics rank works alike at
0.734 — but isolated critics who never see a peer still reach 0.628, so 86% of
the agreement needs no contact at all. A critic can be identified from its prose
44% of the time against 17% by chance, and that falls 2.91 points a round
(p = 0.02) while the isolated arm stays flat (−0.43, p = 0.80): they converge in
voice, not only in vocabulary. The third, whether a score matches the sentiment
of its own critique, came back at 0.082 and is reported as inconclusive rather
than dressed up.

Thresholds are conventions and are printed on the page to be argued with. Each
run records the hash of the system prompt it ran under, so a column changing
colour is attributable to an edit rather than a guess — and each run is judged
against the length rule it was actually given, so editing the prompt does not
retroactively redden old rows.

### Tier 2 — the judged checks, and hand labels

Two questions need a reader, not a rule: does a critic actually write its
disposition, and is the score legible from the prose it is attached to? The free
sentiment proxy for the second came back at rho = 0.08, which is the reason
`judge.py` exists.

Both questions are put as tasks with ground truth in the log, so a human and a
model can be scored against truth independently and against each other. **wrote**
— six dispositions, lettered; which one wrote this critique? Chance is 1/6.
**score** — the number has been removed; recover it. The null is predicting the
global mean, which gives MAE 0.10. Asking a judge to rate adherence out of five
was the obvious alternative and is the worse one: with no ground truth, a judge
disagreeing with a human tells you nothing about which of them is right.

The hand labels are not only judge calibration. The agents are Sonnet 4.6, so
any Claude judging them shares priors with them, and a judge that identifies
dispositions brilliantly may be recognising its own family's habits. The human
is the family-independent reference point: if the judge scores far above the
human, that gap is the shared prior and the judged numbers are not a property of
the work.

One thing had to be fixed before any of it was worth measuring. 44% of the
critiques name their own influence by surname — *"But Ruskin would ask: where is
the craftsman's restraint?"* — and almost none name anyone else's. Left in, the
first task would be a lookup rather than a reading, so every critic's influence
is redacted to `[name removed]` before either reader sees it, and whether a
critique signposted is kept as a field: accuracy split by it is reported,
because "adherence is mostly signposting" is a finding and not an inconvenience.
Artists' influences stay in, since they describe the work, which is shown anyway.

The sample is 25 critiques balanced across both arms and all six critics,
deterministic from a seed, and the sheet is told nothing about the score
distribution — a rater who knew the mean was 0.74 would pull every estimate
toward it and the recovered correlation would measure the briefing.

    uv run python judge.py sheet        # draw the sample, write the sheet
    uv run python judge.py models --provider openai   # what a key can see
    uv run python judge.py estimate     # tokens and cost; no calls
    uv run python judge.py run --confirm --judge-model gpt-5
    uv run python judge.py score --labeller "..."     # reader vs judge vs truth

The judge runs on either provider, chosen by the model name: `claude-*` goes to
Anthropic, `gpt-*` / `o*` to OpenAI. That is the point rather than a
convenience. The agents are Sonnet 4.6, so a Claude judging them shares their
priors and a judge that reads disposition brilliantly may be recognising its
own family's habits; an out-of-family judge cannot. Both get the identical two
strings, built once by `judge_request()`, so neither is given a hint the other
is not — and each writes its own `judge_<set>__<model>.json`, so an in-family
and an out-of-family judge can answer the same items and be compared.

Keys come from `.env`, which is gitignored: `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY`. Nothing reads a key from the command line, where it would
land in shell history, and a missing one exits naming the variable rather than
raising from inside the SDK. Model names change faster than this file does, so
`models` asks the key what it can see instead of carrying a list that goes
stale; listing generates no tokens.

An OpenAI reasoning model bills its thinking against the output budget, so its
ceiling is 2000 tokens where Anthropic's is 300, and `estimate` says that its
output figure is a floor rather than a prediction.

Labels and judge answers live in `labels/` and are committed: one costs a
person's attention and the other costs money, so neither belongs with the
derived figures.

No human labels yet. The first sheet was filled in by a chat assistant rather
than by hand, which makes it a second judge — and one whose prompt was not
recorded — not a human baseline, so the set was discarded.

I also claimed at the time that its score answers were statistically detectable
as copied from the answer key: 80% of them sat on the real score's rounding
grid. That claim was wrong, and the out-of-family judge is what disproved it.
gpt-5.5 never sees the scores, and its answers rounded to tenths sit on the
truth's own tenth 84% of the time with a 1.7% inversion rate. The test was
measuring "answered in tenths and was accurate", which is what a good reader
looks like. `looks_copied()` now fires only on what cannot be a reading — a
rater answering more finely than a tenth who reproduces the hidden number
exactly — and reports `separable: false` for answers given in whole tenths,
because at that precision a copier and an excellent reader are the same thing
to any test this sample can support. The answers are still kept in a separate
base64 file beside the questions, since the cheap fix for a leak is not to
leak.

`--labeller` is recorded with every result and prints with the numbers, and a
set that does not name one says so where its figures would go. Who filled in a
sheet decides what its numbers mean: the roster's author is an upper bound
rather than an arms-length reading, and a model is a judge rather than a
baseline. `labels/sheet_round2.md` is drawn and waiting — 25 critiques, no
overlap with the discarded set, balanced across both arms and all six critics.


## Setup

```sh
# 1. Runtime + analysis dependencies.
uv sync --extra analysis

# 2. spaCy English model (analyze.py needs it).
uv run python -m spacy download en_core_web_sm

# 3. API key.
cp .env.example .env   # then edit .env and set ANTHROPIC_API_KEY
```

## Run

```sh
# A run, then the analysis and archive page automatically.
uv run python run.py --rounds 4

# The matching control, replaying that run's artworks to isolated critics.
uv run python control.py logs/run_20260609_120000.jsonl

# Both arms, clustered in one shared vocabulary space:
uv run python analyze.py --treatment logs/run_A.jsonl --control logs/control_A.jsonl

# Check the log is usable, then whether the agents followed their prompts.
uv run python check_run.py
uv run python eval_agents.py

# Skip the automatic steps; re-analyze an old run; rebuild the archive:
uv run python run.py --rounds 4 --no-analyze
uv run python analyze.py logs/run_20260609_120000.jsonl
uv run python report.py
```

The analysis never touches the API. Re-analyzing logs, sweeping thresholds and
changing the feed window are all free.

## Non-goals

No token economy, coins, costs, or earnings. No survival or elimination of
agents. No marketplace, prices, buying/selling, or collectors. No blockchain,
smart contracts, or NFTs. No prediction markets, governance, or attention
budgets. No image generation. No web server or UI framework. No orchestration
framework. No database. One runnable program plus analysis and report scripts.

That list isn't modesty, it's the design. An earlier version of this had a
thousand agents with money and mortality inside an on-chain art market, and
cutting it down to this was the actual work. If agents with scarcity and
reputation end up agreeing about taste, you can't tell whether you're looking at
aesthetic influence or at economic optimisation. Taking the economy out doesn't
make the experiment smaller — it makes the answer mean something. Whatever
shows up in here can only have come from agents reading each other.
