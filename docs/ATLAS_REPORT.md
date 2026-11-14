# Activation atlas: report (epic thodel/weightspace#13)

Levels L0 (activation map) and L2 (dependence on the image) first, as the epic orders them; Gate 1 decides
whether the viewer may claim *where* in a line something happens. L1, L3 and L4 are not started.

## Gate 1 rule (fixed 13.11.2026, before any occlusion run)

Written and committed before `wsp.atlas` was run on any line, so that the result cannot shape the rule.

**Models and lines.** Practice lines the model was not trained on:

- `kraken-german-xix-blb-v1` on `blb-val-300` (no line seen by it);
- `kraken-bern-hertwig-v2` on the lines of `bern-val` whose `seen_by` does not name it (301 of 513).

From each set, lines of at least 15 characters, in the order of `sha256("20261113:" + key)`, the first 80
the transform can scale. No filter on CER: the sample is not chosen by how well the model reads.

**Occlusion.** A window 16 px wide (the full line height) is set to the line's median pixel value (the
background) and slid over the line with a step of 4 px. For each reference character *i* with forced CTC
run (*a*, *b*) (frames, from `wsp.layerview.forced_align`) and label *l*, its score is
`s_i = max over t in [a−2, b+2] of log p(t, l)`. Its occlusion profile is the drop `s_i(line) − s_i(occluded)`
per window position, and its *occlusion peak* is the window centre (px) with the largest drop.

**Hit.** A character is a hit when its occlusion peak lies nearer to the centre of its own CTC run
(`(a + b + 1) / 2 × px_per_frame`) than to the centre of any other character's run in the line.
A character whose largest drop is below 0.1 nat counts as a miss.

**Characters scored.** Non-space reference characters that the greedy reading gets right (`ok` in
`wsp.layerview.char_status`). Spaces (no ink to occlude), substituted and deleted characters are reported
separately and do not enter the gate.

**Pass.** Gate 1 passes for a model when the hit share is at least 0.80 and the lower end of its page-clustered
95 % bootstrap interval (`wsp.metrics.page_of`, 2,000 resamples) is at least 0.75. If it passes for both
models, the viewer may draw a character's location from the CTC alignment. If it fails for a model, that
model's viewer shows activation only and claims no location (the epic's rule).

**Reported alongside, not gating.** The signed offset (occlusion peak − CTC centre, px); the measured receptive
field of every block (relative change of the block's frame activation by the offset of the window), against
the receptive field computed from the VGSL spec and against a randomly initialised network of the same spec;
gradient × input of `s_i` summed per window, and how often its peak falls on the same character as the occlusion
peak.

## Result (13.11.2026): Gate 1 fails for both models

`python -m wsp.atlas run` for each model (80 lines; 40 lines with random weights), then
`python -m wsp.atlas report --out docs/atlas`. Caches in `~/atr-cache/atlas/`. Figures, summaries and the viewer
are in `docs/atlas/` (`index.html`). The line images are git-ignored.

| model | practice set | lines (pages) | letters scored | hit share [page-clustered 95 %] | median offset (abs.) | Gate 1 |
|---|---|---|---|---|---|---|
| `kraken-german-xix-blb-v1` | `blb-val-300` | 80 (21) | 2,709 | 0.61 [0.55, 0.67] | +4 px (8 px) | **fails** |
| `kraken-bern-hertwig-v2` | `bern-val`, parent-unseen | 80 (11) | 2,145 | 0.73 [0.69, 0.77] | 0 px (4 px) | **fails** |

Under the rule, the viewers show activation only. Each page says so in a banner. The character row is placed by
the CTC alignment so a reader can follow the line; the page does not claim that the character *is* there.

Not gating, reported as the rule says:

- **Offset.** Occlusion peaks sit on or just after the CTC centre: for `xix-blb-v1`, the middle half lies within
  0 to +12 px and 80 % within −4 to +20 px. For `hertwig-v2`, 80 % lie within −12 to +20 px.
  The misses are mostly off by one character, not far away.
- **Other characters.** Spaces are almost never hits (0.03 and 0.28), as expected: blanking a gap leaves the gap.
  Substituted letters score 0.40 and 0.50; deleted letters 0.29 and 0.35.
- **Gradient × input does not reproduce occlusion.** Its peak falls on the same character as the occlusion
  peak for only 23 % (`xix-blb-v1`) and 41 % (`hertwig-v2`) of letters. The median Spearman correlation of the
  two profiles is 0.17 and 0.07. In this network the first-order estimate is not a usable stand-in for
  occlusion. If the atlas needs locations, it needs occlusion.

**Post hoc, descriptive only (not part of the rule; looked at after the gate failed).**

- Characters are close together: the median distance between neighbouring CTC centres is 20 px
  (`xix-blb-v1`) and 24 px (`hertwig-v2`). The rule therefore asks for a peak within about 10–12 px.
  The window is 16 px wide and moves in 4 px steps.
- If a peak on the neighbouring character also counted, the share would be 0.97 and 0.98.
- Restricting to lines read at CER ≤ 20 % hardly changes the share (0.63 and 0.71), so the misses are not
  crop defects.

What follows from this:

- Occlusion locates a letter to within one neighbour. That is not the character-level location the gate asked for.
- A finer window, or a test at the level of words, would be a new rule fixed before a new run, not a
  re-reading of this one.

## Receptive fields (L2)

Shown in `rf-<model>__<set>.svg`. The quantity is the relative change of a block's frame activation, by the
distance between the occluding window and the frame.

**The 140 px of the epic is right.** The receptive field computed from the spec is 13, 38, 72 and 140 px for
`C_0`, `C_3`, `C_6` and `C_9`. The measured extent (the farthest window that changes the frame at all) is 16,
28, 44 and 76 px from the frame centre in both models, each just inside the bound that the spec, the 8 px frame
and the 16 px window give together (18, 30, 46, 78). A random network of the same spec has the same extents, as
it must.

**Within that field, training narrows the dependence.** Change of a frame's activation at a given distance of the window, relative to the block's peak:

| block | distance | `xix-blb-v1` trained | random | `hertwig-v2` trained | random |
|---|---|---|---|---|---|
| `C_9` | ±40 px | 0.15 | 0.62 | 0.21 | 0.61 |
| `L_12` | ±100 px | 0.06 | 0.38 | 0.04 | 0.31 |
| `L_16` | ±100 px | 0.20 | 0.88 | 0.22 | 0.86 |
| `L_16` | ±400 px | 0.04 | 0.38 | 0.03 | 0.37 |
| `O_18` | ±100 px | 0.12 | 0.88 | 0.15 | 0.86 |

- **The BiLSTMs could use the whole line but mostly look near the frame.** With random weights, a recurrent
  frame depends on the whole line almost evenly: at ±100 px, `L_16` still keeps 0.86–0.88 of its peak.
- **Trained, each recurrent block is sharply local.** Half of the change comes from within ±28–104 px, about
  one to five characters.
- **There is also a small, line-wide tail.** It still reaches 2–4 % of the peak 400 px away. Summed over a long
  line, that tail holds most of the mass (90 % within ±600–1,250 px), so the 90 % widths in the figure describe
  the tail, not the core.
- **The dependence widens a little from `L_12` to `L_16`.** At ±100 px it goes from 0.04–0.06 to 0.20–0.22 of
  the peak in both models. This is a block-level form of the context question for L1.

## What this means for the next levels

- **Gate 1 failed.** Unit cards (L1) may describe *when* in a line a unit is active, in frames. They may not
  claim *which character it looks at* from the CTC alignment.
- **L3 (ablation) does not depend on Gate 1** and can go ahead.
- **Gate 2 still needs the controls** of the epic: the random network used here, seed runs, and disjoint pages.

## Reproducing

```bash
python -m wsp.atlas run --model models/kraken/kraken-german-xix-blb-v1.mlmodel --practice data/practice/blb-val-300.jsonl --n 80
python -m wsp.atlas run --model models/kraken/kraken-bern-hertwig-v2.mlmodel --practice data/practice/bern-val.jsonl --n 80
python -m wsp.atlas run --model models/kraken/kraken-german-xix-blb-v1.mlmodel --practice data/practice/blb-val-300.jsonl --n 40 --random
python -m wsp.atlas run --model models/kraken/kraken-bern-hertwig-v2.mlmodel --practice data/practice/bern-val.jsonl --n 40 --random
python -m wsp.atlas report --out docs/atlas
```

The four runs take about 6 minutes on 48 CPU cores (12 workers × 4 threads). The random control is seed 20261113.

## L3 rule: ablation (fixed 14.11.2026, before any ablation run)

Written and committed before `wsp.atlas_ablate` was run on any line.

**Models and lines.** All lines the model was not trained on, with no length or CER filter:

- `kraken-german-xix-blb-v1` on `blb-val-300` (300 lines, 24 pages);
- `kraken-bern-hertwig-v2` on the `bern-val` lines whose `seen_by` does not name it (301 lines, 11 pages).

**Selection and evaluation.** The pages (`wsp.metrics.page_of`) are ordered by `sha256("20261114:" + page)` and
assigned alternately to *selection* and *evaluation*. Units, groups and circuits are chosen on selection pages
only. Every claim below is made on evaluation pages only. No page is on both sides.

**Units.** The 192 conv channels (`C_0` 32, `C_3` 32, `C_6` 64, `C_9` 64) and the 1,200 BiLSTM units
(`L_12`, `L_14`, `L_16`: 200 forward and 200 reverse each), 1,392 in all.

**Ablation.** *Mean ablation*: the unit's output, as the next layer sees it, is replaced on every frame (and, for
a conv channel, at every height) by one number, the unit's mean over all positions of the selection lines.
Whole-block ablation (mean and zero) is reported for reference only.

**Measures.** Corpus CER (points), with the paired page-clustered bootstrap of `wsp.metrics.bootstrap_delta`
(2,000 resamples). Per reference character *c*: its accuracy is the share of its occurrences that the greedy
reading gets right (`ok` in `wsp.layerview.char_status`). Its *loss* is the accuracy before ablation minus
the accuracy after, in points.

**Stage A (selection pages).** Every unit is ablated alone.

**Choices made on selection pages:**

- the 20 units with the largest ΔCER;
- per block, its 10 units with the largest ΔCER;
- per target character, its *circuit*: the up to 10 units, across all blocks, with the largest loss of *c*
  (only units whose loss is above 0).

The targets are the space and the 10 most frequent other characters of the selection lines.

**Controls.**

- For each block group: 50 random groups of 10 units from the same block.
- For each circuit: 50 random groups of the same size from all 1,392 units.
- Seed 20261114.

**Claims, on evaluation pages.**

- **U, a unit matters:** ΔCER ≥ 1.0 point and the lower end of its page-clustered 95 % interval > 0.
  Tested for the 20 units above.
- **G, a block's top group matters beyond chance:** ΔCER ≥ 1.0 point, the lower end of the interval > 0, and
  larger than at least 48 of the 50 random groups of that block.
- **C, a circuit is specific to its character.** All three must hold:
  (a) the loss of *c* is ≥ 5 points, with the lower end of its page-clustered 95 % interval > 0;
  (b) it is larger than the loss of *c* under at least 48 of the 50 random groups of the same size;
  (c) it is at least twice the loss of all other characters pooled.
- **Effects below 1 point of CER** (U, G) or below 5 points of a character's accuracy (C) are reported as
  inconclusive, not as absent.

**Predictions and descriptions (not claims).**

- **Prediction:** mean-ablating a dead `C_0` filter of `docs/family/functional.json` changes nothing.
  The dead filters are 9 for `xix-blb-v1` and 6 for `hertwig-v2`.
- **Description:** the Spearman correlation between a unit's selection ΔCER and the relative change of its
  incoming weights in the last fine-tune (`german-xix-v1` → `german-xix-blb-v1`, `bern-hertwig-v1` →
  `bern-hertwig-v2`).
- **Answer to the epic's question:** whether the deleted spaces have a circuit, and in which blocks it lies.
