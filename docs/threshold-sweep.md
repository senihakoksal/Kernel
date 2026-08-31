# Similarity threshold sweep

`SIMILARITY_THRESHOLD` decides when two descriptors count as the same idea —
used both for clustering and for subtracting prior vocabulary. Every
downstream number depends on it, so it was swept rather than assumed.

Run: <log filename> · 2 artists, 5 critics, 5 rounds · 495 descriptors
Clustering: agglomerative, cosine, complete linkage

| threshold | clusters | singletons | largest | overlap early→late |
|---|---|---|---|---|
| 0.90 | 482 | 469 | 2 | 5.5% → 6.9% (+26%) |
| 0.82 | 461 | 429 | 3 | 6.3% → 8.1% (+27%) |
| 0.72 | 414 | 351 | 5 | 7.9% → 10.3% (+30%) |
| 0.65 | 378 | 292 | 7 | 9.2% → 12.8% (+39%) |
| 0.55 | 308 | 196 | 10 | 11.7% → 14.9% (+28%) |
| 0.45 | 242 | 116 | 10 | 13.3% → 19.0% (+43%) |

## What it shows

The absolute overlap level is very sensitive to the threshold — it roughly
triples from 0.90 to 0.45 — so no single overlap figure means anything on its
own. The early→late direction, by contrast, is stable across the whole range.

## Why 0.72

At 0.72 the merges are clean synonym groups:

- `entire weight · full weight · whole weight`
- `finest proposal · most accomplished proposal · strongest proposal`
- `final sentence · final paragraph · last sentence`

Nothing at this setting groups clearly distinct ideas. Below 0.65 merging
turns aggressive and the largest cluster reaches ten members.

## Known limitation

Only 63 of 414 clusters have more than one member at 0.72, so the clustering
step is close to inert: descriptors are matching largely by exact repetition
rather than by meaning. That is a limitation of the extraction step, not of
this threshold.