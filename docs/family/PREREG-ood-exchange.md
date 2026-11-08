# Pre-registration: stabilising german-xix-blb-v1 by its first convolution (RQ3 on out-of-domain material)

Written 08.11.2026, after practice and confirmation, **before any held-out line below was read** by any model
in this study. Committed before the held-out run; the commit hash is the time stamp.

## Subject and donor

- S = `kraken-german-xix-blb-v1` (file sha256 in `docs/family/ood-practice-german-xix-blb-v1.json`), fine-tuned from
- G = `kraken-german-xix-v1` (its parent; one lineage, body cosine > 0.6).
- A move sets a block to `S + β (G − S)`; everything else stays S. Rebuilt and checked by tensor digest
  (`digest` in `docs/family/ood-confirm-german-xix-blb-v1.json`).

## Candidates (two, nothing else is scored)

| id | moves | practice (own val / federal / Basel) | confirmation (own train / federal / Basel) |
|---|---|---|---|
| **A** | `C_0` half-way (β = 0.5) | +0.12 / −1.99 / −0.30 | −0.02 / −2.31 / −0.40 |
| **B** | `C_0` transplanted (β = 1) | +0.21 / −2.37 / −0.78 | +0.27 / −2.84 / −0.76 |

(Δ CER in points against S unedited.) Chosen because `C_0` is the only single move with a clear gain on the out-of-domain
sets and no cost on S's own validation lines; every combination that adds `C_3`, `C_6`, `C_9` or `O_18` adds own-material cost
(+0.28 to +0.53 on fresh lines) for at most −0.6 more on Basel. The first grid's suggestion (`C_6` half-way, plus `C_0`) was
tested: `C_0`+`C_6` half-way confirms (−2.43 / −0.95) but costs +0.28 (interval excluding 0) on S's own training lines.

## Held-out sets (built by data properties only; none read before)

| role | list | lines | clusters | sha256 |
|---|---|---:|---:|---|
| primary out-of-domain | `data/heldout/federal-minutes-heldout.jsonl` — Federal Council minutes (1848–1903): the 6 Transkribus containers drawn with `random.Random(20261108)` before any reading (`data/ood/federal-minutes.split.json`), minus every line whose *source* document (`docId_<n>` in the file name) also occurs in the practice pool | 545 | 30 source documents, ~535 pages | `ddb031e8a1e55c61f0fc39c449ff082392dbd08a3d9cd3596ddba927221f51de` |
| secondary out-of-domain | `data/heldout/basel-grundbuch-heldout.jsonl` — Basel land register: 7 containers by the same rule, minus every line whose volume (`HGB_<a>_<b>_<c>`) also occurs in the pool | 1,888 | 111 pages, 93 volumes | `ff85c86081f10b08109d8373c679ea15d7d93735e7a7b0fb5804fb023b9b9844` |
| own material | `data/heldout/blb-eval-hand-b.jsonl` — the family-2 held-out set for studies that practise on blb-val-300 | 1,718 | 70 pages | `078b208a4e104d4ce165e7d21fdcd963062f308faa6389911a78c09f87116a4f` |

Audit (`data/ood/audit.json`, `scripts/audit_ood.py`): no line, page, container or source document of the practice or
confirmation lists in the held-out lists (the first audit found 4 shared pages / 6 source documents in the federal minutes and
10 shared volumes in Basel, because the Transkribus container ids used for the split are not the source documents; those
held-out lines were removed by the rule above, unread); neither source in the training data of S or G (the federal minutes were G's training *benchmark*, never trained on).

## Statistic

Paired Δ CER (candidate − S) on each set; 95 % intervals from 2,000 resamples, seed 20261106, line level and two-stage
page-clustered (`wsp.metrics.bootstrap_delta`; page = `page_<id>` in the key for the federal minutes, `project/page` otherwise).
**Decisions use the page-clustered interval.**

## Decision rule (fixed now)

A candidate **qualifies** if all of:

1. **gain where it matters, with a minimum effect:** on `federal-minutes-heldout`, Δ ≤ −1.0 point and the upper end of the
   page-clustered interval ≤ −0.5 points;
2. **no cost on own material (non-inferiority, margin 0.5):** on `blb-eval-hand-b`, the upper end of the page-clustered interval
   ≤ +0.5 points;
3. **no loss on the second out-of-domain source:** on `basel-grundbuch-heldout`, the lower end of the page-clustered interval < 0
   is not required, but the upper end must be ≤ +0.5 points.

If both qualify, **A** is preferred unless B's federal gain exceeds A's by at least 0.5 points *and* B also meets rule 2.
If neither qualifies, that is the result. Collapsed-line counts and per-Unicode-block errors are reported, not decisive.

Prediction: A qualifies (federal −2.0 ± 0.6, own ±0.2, Basel −0.3); B gains more on the federal minutes (about −2.6) but
fails or nearly fails rule 2 (own +0.2 to +0.4).

Qualifying makes a candidate *eligible*, nothing more: nothing is registered, uploaded or served without the user's OK.

## Bern lineage: no candidate

`kraken-bern-hertwig-v2` reads the Schlussredaktion hand (A V 1447, `schluss-300`) at 29.6 %; its parent cursive_inzigkofen at
65.5 % and catmus_medieval at 47.6 %. No conv move towards cursive_inzigkofen lowered CER on `schluss-300`; the suggested
`C_3`+`C_9` half-way costs +1.41 (page interval +0.72 to +2.11) there and +0.39 on own lines
(`docs/family/ood-practice-bern-hertwig-v2.json`). Towards the grandparent catmus_medieval (also a worse reader of A V 1447)
`C_9` half-way gained −1.00 on `schluss-300` but cost +0.42 (page interval +0.10 to +0.78) on own lines; on fresh lines the gain
fell to −0.40 (`schluss-conf-300`, page interval includes 0), and on `bern-val` it costs +0.82 on the Hertwig lines for −1.10 on
the Schlussredaktion lines (`docs/family/ood-confirm-bern-hertwig-v2-from-catmus_medieval.json`): a one-for-one trade, which fails
rule 2's margin. No candidate, so `schluss-heldout` and `bern-eval-hertwig` stay unread.
