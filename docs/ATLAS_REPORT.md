# Activation atlas: report (epic thodel/weightspace#13)

Levels L0 (activation map) and L2 (dependence on the image) first, as the epic orders them; Gate 1 decides
whether the viewer may claim *where* in a line something happens. L3 (ablation) and L1 (selectivity) followed on 14.11.2026; Gate 2 fails for both models; L4 is not started.
The [closing synthesis](#closing-synthesis-15112026) at the end answers the epic's five questions.

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

## L3 result (14.11.2026)

`python -m wsp.atlas_ablate run` for each model, then `python -m wsp.atlas_ablate report --out docs/atlas`.
Caches in `~/atr-cache/atlas/<model>/<set>--ablate/`. Summaries (`ablation-<model>__<set>.json`) and figures
(`ablation-units-*.svg`, `ablation-circuits-*.svg`) are in `docs/atlas/`. The runs take about 35 minutes (`xix-blb-v1`) and 70 minutes (`hertwig-v2`) on 48 CPU cores (12 workers × 4 threads).

| model | selection: lines (pages), CER | evaluation: lines (pages), CER |
|---|---|---|
| `kraken-german-xix-blb-v1` | 133 (12), 15.7 % | 167 (12), 12.2 % |
| `kraken-bern-hertwig-v2` | 173 (6), 29.3 % | 128 (5), 25.6 % |

`hertwig-v2` has only 11 pages, so its evaluation intervals rest on 5 clusters and are coarse.

### Stage A: no single unit matters much

Mean-ablating one unit on the selection pages changes the CER by at most 0.66 points (`xix-blb-v1`, `C_3 ch 8`)
and 1.20 points (`hertwig-v2`, `C_0 ch 3`). Only that one unit, of 2 × 1,392, reaches 1 point. BiLSTM units
change the CER by at most 0.24 and 0.47 points. Almost every live unit changes some reading, and many single ablations
lower the CER a little. In `xix-blb-v1` the 400 units of `L_16` sum to −14.5 points.

### Claim U (20 units per model, evaluation pages)

- **`xix-blb-v1`: no unit holds.** The largest effects are below 1 point. Three have an interval above 0 and
  are *inconclusive*: `C_0 ch 27` (+0.93 [+0.32, +1.85]), `C_0 ch 13` (+0.66 [+0.03, +1.51]) and
  `C_9 ch 2` (+0.44 [+0.05, +0.93]).
- **`hertwig-v2`: one unit holds.** `C_0 ch 3`: +1.26 points [+0.12, +2.43]. The other 19 have intervals that
  include 0.

### Claim G (top 10 units of a block, against 50 random groups of 10 from the same block)

| block | `xix-blb-v1` ΔCER [95 %] | beats | G | `hertwig-v2` ΔCER [95 %] | beats | G |
|---|---|---|---|---|---|---|
| `C_0` | +87.8 [+76.2, +94.2] | 50 | **holds** | +74.4 [+72.6, +76.1] | 50 | **holds** |
| `C_3` | +3.6 [+1.2, +7.5] | 46 | no (b) | +1.4 [+0.2, +2.8] | 34 | no (b) |
| `C_6` | +1.3 [+0.3, +2.5] | 37 | no (b) | +1.3 [+0.2, +2.6] | 50 | **holds** |
| `C_9` | +2.2 [+1.0, +4.1] | 47 | no (b) | +2.8 [+1.3, +4.5] | 50 | **holds** |
| `L_12` | +0.4 [+0.0, +1.0] | 45 | inconclusive | −0.2 [−0.7, +0.4] | 13 | inconclusive |
| `L_14` | +0.3 [−0.2, +0.7] | 46 | inconclusive | +0.9 [+0.4, +1.4] | 50 | inconclusive |
| `L_16` | +0.2 [−0.3, +0.7] | 33 | inconclusive | +0.5 [+0.0, +1.1] | 50 | inconclusive |

"no (b)" means the group passes the 1-point bar but beats fewer than 48 random groups.

- **The `C_0` groups silence the model.** With the top 10 `C_0` channels mean-ablated, every evaluation line is
  read as empty. The CER rise equals the rise under whole-block ablation. Random groups of 10 `C_0` channels
  degrade the reading gradually (median +2.6 and +3.3 points). One of the 50 random groups in `hertwig-v2` also
  silences most lines.
- **The units that matter most alone, taken together, cut the first layer's signal.** No single one of them
  reaches 1 point (except `C_0 ch 3` in `hertwig-v2`). This is a strongly non-additive effect.
- **In `hertwig-v2`, the top groups of `L_14` and `L_16` beat all 50 random groups** and their intervals lie
  above 0. They stay below the 1-point bar, so they are inconclusive, not absent.

### Claim C (circuits of 10 units per character)

Loss = drop in the character's accuracy, in points. "Others" is the pooled loss of all other characters.

| character | `xix-blb-v1` loss [95 %] | beats | others | C | `hertwig-v2` loss [95 %] | beats | others | C |
|---|---|---|---|---|---|---|---|---|
| space | 4.4 [1.3, 9.0] | 50 | 2.3 | inconclusive | 73.1 [64.5, 79.6] | 50 | 55.7 | no (c) |
| `e` | 0.6 [−0.4, 2.0] | 48 | 0.8 | inconclusive | 6.0 [3.3, 9.2] | 50 | 0.1 | **holds** |
| `n` | 3.7 [0.0, 11.9] | 50 | 4.9 | inconclusive | 8.3 [2.6, 13.8] | 50 | 12.8 | no (c) |
| `i` | 9.9 [3.7, 21.2] | 50 | 6.1 | no (c) | 0.3 [−5.8, 7.1] | 44 | 2.6 | inconclusive |
| `r` | 3.4 [0.0, 8.9] | 50 | 1.7 | inconclusive | 12.5 [6.7, 18.2] | 50 | 2.8 | **holds** |
| `a` | 1.4 [0.0, 4.5] | 50 | 1.1 | inconclusive | 20.3 [13.2, 28.9] | 50 | 1.0 | **holds** |
| `s` | 3.7 [−1.6, 11.4] | 50 | 4.6 | inconclusive | 5.9 [1.4, 10.8] | 50 | 1.6 | **holds** |
| `t` | 5.5 [1.8, 10.1] | 50 | 1.6 | **holds** | 17.1 [11.9, 23.1] | 50 | 1.2 | **holds** |
| `h` | 2.2 [0.0, 6.9] | 50 | 2.4 | inconclusive | 13.9 [6.2, 22.5] | 50 | 0.9 | **holds** |
| `d` | 2.3 [−0.4, 8.1] | 50 | 3.0 | inconclusive | 14.0 [8.0, 21.1] | 50 | 10.1 | no (c) |
| `l` | 5.1 [0.0, 14.1] | 50 | 4.9 | no (a, c) | 23.5 [15.2, 32.5] | 50 | 1.1 | **holds** |

The targets are the same 11 characters in both models, in a slightly different order. `hertwig-v2` reads `h`
and `l` poorly even before ablation (58 % and 55 %).

- **`xix-blb-v1`: one specific circuit, `t`.** Most circuits move their character by less than 5 points.
- **`hertwig-v2`: seven of eleven hold**: `e`, `r`, `s`, `t`, `a`, `h`, `l`. Each costs its character 6–23
  points of accuracy and the rest of the alphabet at most 3 points.
- **Circuits beat random groups almost without exception** (50 of 50 in 20 of 22 cases). Ten units chosen for a
  character are not like ten random units. Where C fails, it fails on specificity (c) or on size (a).
- **The circuits lie in the conv blocks.** Of 110 circuit units per model, 11 (`xix-blb-v1`) and 8
  (`hertwig-v2`) are BiLSTM units, although these are 86 % of all units. Of the eight circuits that hold, none
  has more than two BiLSTM units.

### The deleted spaces: the epic's question

**Neither model has a space circuit under the rule.**

- **`xix-blb-v1`.** The best 10 units for the space cost it 4.4 points, inconclusive under the 5-point bar. The
  cost to the other characters is 2.3 points, so the circuit would fail (c) at that size as well. The circuit
  lies entirely in the conv blocks: 2 `C_0`, 3 `C_3`, 5 `C_9`.
- **`hertwig-v2`.** The space circuit contains four `C_0` channels, including `C_0 ch 3`. It costs the space 73
  points, but every other character 56. It degrades the whole reading; it is not about spaces.

Deleted spaces are therefore not the work of a small set of units that a 10-unit ablation finds. If they have a
mechanism, it is spread more widely, or it lies in the BiLSTMs' interaction rather than in single units.

### Prediction and descriptions

- **The dead-filter prediction holds.** Mean-ablating each dead `C_0` filter (9 in `xix-blb-v1`, 6 in
  `hertwig-v2`) leaves every selection reading unchanged.
- **The weight change in the last fine-tune does not predict importance.** The Spearman correlation between a
  unit's selection ΔCER and the relative change of its incoming weights is +0.01 (`xix-blb-v1`) and −0.02
  (`hertwig-v2`) over all units. Per block it lies within ±0.19, except `C_0`, which has +0.47 and −0.32 (32
  units each, opposite signs).
- **Whole-block ablation.** For reference only: mean or zero ablation of any block silences both models. The
  exceptions are three settings that leave a few characters: `xix-blb-v1` `C_3` mean (+86.0 instead of +87.8),
  `C_6` zero (+86.7), and `hertwig-v2` `L_12` zero (+68.0 instead of +74.4).

**Post hoc, descriptive only.** The readings show the `C_0` collapse directly: the top-10 group gives empty
strings on all evaluation lines in both models. So does whole-block ablation. A finer rule (groups of 3 or 5,
or a cumulative curve) would be a new rule fixed before a new run.

### What follows for the atlas

- **Unit cards (L1) should not rank BiLSTM units by ablation.** A single BiLSTM unit never moves the CER by more
  than half a point. Its card can show activation and, at most, an inconclusive group effect.
- **Character circuits are a claim the atlas can make for `hertwig-v2`** (seven characters) **and, for
  `xix-blb-v1`, only for `t`.** Each sits mainly in `C_3`–`C_9`.
- **`C_0` is fragile.** Ten of its 32 channels, chosen by single effect, carry the whole reading.

```bash
python -m wsp.atlas_ablate run --model models/kraken/kraken-german-xix-blb-v1.mlmodel --practice data/practice/blb-val-300.jsonl
python -m wsp.atlas_ablate run --model models/kraken/kraken-bern-hertwig-v2.mlmodel --practice data/practice/bern-val.jsonl
python -m wsp.atlas_ablate report --out docs/atlas
```

## L1 rule: selectivity (fixed 14.11.2026, before any selectivity run)

Written and committed before `wsp.atlas_select` was run on any line.

**What L1 may say.** Gate 1 failed, so L1 does not claim *where in the image* a unit looks. Frames are labelled by the forced CTC
alignment of the reference text (`wsp.layerview.forced_align`). A selectivity claim reads: "the unit is active (or silent)
on the frames that the alignment gives to *c*", not "the unit responds to *c* in the image". Because the labels come from
the model's own output, selectivity in `L_16` is partly circular. The controls below test content, not that circularity.

**Models and lines.** Practice lines only; no held-out data.

- `kraken-german-xix-blb-v1` on `blb-val-300` (300 lines, 24 pages).
- `kraken-bern-hertwig-v2` on the 846 lines of the Schlussredaktion project, from three sets (106 pages):
  - the 301 lines of `bern-val` that `hertwig-v2` was not trained on;
  - `schluss-300`;
  - `schluss-conf-300`;
  - duplicate keys counted once.

  No Hertwig model and no seed run was trained on this project. The hand differs from the one `hertwig-v2` was trained on.
- The five seed runs of `docs/SEED_NULL.md` (`~/atr-cache/wsp/seed-null/seed1`–`seed5`, `best_*.mlmodel`) and their parent
  `cursive_inzigkofen`, on the same 846 lines. The architecture and line transform are identical to those of `hertwig-v2`.

**Frames and units.** The 1,392 units of L3 (`O_18` excluded):

- a conv channel's frame value is its output averaged over the height;
- a BiLSTM unit's is its output (forward units 0–199, reverse 200–399).

Labels come from the trained model's alignment:

- `xix-blb-v1` for its own lines;
- `hertwig-v2` for every Bern network (seeds, parent, random), so that all networks of the family are scored on the same frames.

Lines without an alignment are dropped.

**Labels of a frame.**

- *Character* *c*: frames in a CTC run of *c*. Characters with at least 50 runs on the selection pages and on the evaluation pages.
- *Class* (by letter identity, not glyph shape; in Kurrent the shapes differ from print):
  - space;
  - punctuation (Unicode P*);
  - digit;
  - uppercase;
  - ascender letters `b d f h k l t ß ſ`;
  - descender letters `g j p q y`;
  - round letters `a c e o`.
- *Position*:
  - word-initial and word-final (the first and last letter of a space-separated word; a one-letter word is both);
  - word-internal;
  - line-start margin (frames before the first run) and line-end margin (after the last run).

For characters and classes, AUROC compares a label's frames with all other run frames. For positions, it compares them with all
run and margin frames. Blank frames between runs are not labelled.

**Selectivity.** For a unit and a label, *s* = 2 · |AUROC − 0.5|: 0 is none and 1 is perfect. "On" when AUROC > 0.5, "off" otherwise.
A unit's *best label* is the label with the largest *s* on the selection pages.

**Selection and evaluation.** The pages are split as in L3: by `sha256("20261114:" + page)`, assigned alternately. Best labels,
tested units and probe regularisation are chosen on selection pages. Every claim is made on evaluation pages.

**Controls.**

- *Random network:* `wsp.atlas.random_net` (seed 20261113, the L2 control), on the trained model's labels.
- *Shuffled characters:* within each line, the non-space characters are permuted among the non-space runs (seed 20261114).
  Spaces and positions stay.
- *Blank crop:* the line image filled with its median pixel value (the L2 occlusion fill), with the real line's labels.
  This shows what position in the sequence gives without ink.

**Claim S, a unit is selective.** Tested for the 10 units per block with the largest selection *s*. On evaluation pages, for the
unit's best label, all of these must hold:

- (a) *s* ≥ 0.5 (AUROC ≥ 0.75 or ≤ 0.25);
- (b) the lower end of its page-clustered 95 % interval (2,000 resamples) is above the largest evaluation *s* of the random
  network's 10 tested units in the same block (chosen the same way);
- (c) that lower end is above the same unit's *s* for the same label on blank crops, and, for character and class labels, also
  above its *s* with shuffled characters.

If (b) and (c) hold but (a) does not, the result is inconclusive.

**Claim P, a block codes characters (sparse probe).**

- *Probe:* L1-regularised multinomial logistic regression from standardised frame values to the character of the frame.
  It covers the characters above, uses at most 20,000 training frames from the selection pages (seed 20261114), and has
  C ∈ {0.01, 0.1, 1}. C is chosen on alternate selection pages, then the probe is refit on all of them.
- *Blocks:* each of the 7 blocks, and each direction of the three BiLSTMs.
- *Measure:* balanced accuracy on evaluation pages, with a page-clustered 95 % interval.
- *Holds when* it is at least 0.10 above the random network's probe of the same block, and the lower end of its interval is
  above both the random-network probe and the probe trained and tested on shuffled characters.

**Prediction D, the two directions carry different context.** In `L_12`, `L_14` and `L_16`, on the frames of a character run,
the same probe predicts the previous and the next character (characters above, space included).

- The forward half predicts the previous character better than the reverse half.
- The reverse half predicts the next character better than the forward half.

It holds per block and target when the page-clustered interval of the paired difference in balanced accuracy is above 0.

**Stability (`hertwig-v2` only).**

- *Profile:* a unit's signed AUROC − 0.5 over all labels, on evaluation pages.
- *Stability of a block against a seed run:* the median over its units of the Pearson correlation between that unit's profile
  in `hertwig-v2` and in the seed run.
- *Null:* the same median with the seed run's units permuted within the block (1,000 permutations, seed 20261114).
- *Holds for a block* when it is above the null's 99th percentile for every one of the five seed runs.

The seed runs and `hertwig-v2` descend from `cursive_inzigkofen`. The same statistic against the parent is reported, so that
inheritance can be told from convergence.

**Gate 2.** It passes for a model when P holds in at least one block, S holds for at least one unit, and stability holds in at least
one block in which S holds.

- `xix-blb-v1` has no seed runs, so condition 3 cannot be met. Its atlas stays descriptive, and its L3 circuit (`t`) is
  shown as a description, not as a circuit claim.
- If Gate 2 fails for `hertwig-v2`, the same applies to its seven L3 circuits.

**Unit cards (description).** One card per unit. It shows:

- its best labels with AUROC on evaluation pages (with the interval if it was tested) and the three controls;
- its mean activation per label;
- where in the line, in frames, it is active, as a profile over the line, not as a character location;
- its L3 role, only where an L3 claim holds: claim U (`hertwig-v2` `C_0 ch 3`), or membership of a circuit for which C holds.
  Otherwise the card says that no single-unit effect of 1 point was found (inconclusive);
- dead `C_0` filters, marked;
- for `hertwig-v2`, the unit's profile correlation with each seed run and with the parent.

Every card carries the banner of L0: activation, not location.

**Descriptions reported alongside, not claims.**

- The Spearman correlation between a unit's evaluation *s* and its L3 selection ΔCER.
- Whether circuit units are selective for their own character.
- Block by block: where character selectivity first exceeds the random network (the epic's "stroke or character" question).

## L1 result (14.11.2026): Gate 2 fails for both models

`python -m wsp.atlas_select run` for each model, then `python -m wsp.atlas_select report --out docs/atlas`.
Caches in `~/atr-cache/atlas/<model>/<set>--select/`. Summaries (`selectivity-<model>__<set>.json`), figures
(`selectivity-units-*`, `selectivity-probes-*`, `selectivity-seeds-*`) and the unit cards (`units-<model>__<set>.html`, one
card per unit) are in `docs/atlas/`.

**Timing.**

- The rule was committed at 19:17 (cfc76c5).
- The first activation was extracted at 19:22, in a one-line smoke test; the full extraction started at 19:22.
- The first summary was written at 19:27.

**A bug in the first analysis, and the re-run.** The first analysis (19:22–19:42) had a bug in the probe. When a
model's selection pages held more than 20,000 character frames, the training frames and their labels were drawn as two
separate random samples, so they did not match.

- This affected only `hertwig-v2`. Its probes for claim P and prediction D all sat at exactly chance (1/28).
- The bug was fixed (`probe` draws one sample) and a test was added.
- Both models were analysed again. The `xix-blb-v1` summary came out byte-identical, since its selection pages hold fewer
  frames than the limit.
- The rule, the lines and the activations did not change.

| model | lines | selection: lines (pages) | evaluation: lines (pages) | frames | labels |
|---|---|---|---|---|---|
| `kraken-german-xix-blb-v1` | `blb-val-300` | 133 (12) | 167 (12) | 42,743 | 33 |
| `kraken-bern-hertwig-v2` | Schlussredaktion, 845 of 846 aligned | 419 (53) | 426 (53) | 120,510 | 38 |

### Claim S: no unit is selective beyond the random network

None of the 70 tested units per model holds; all 140 fail condition (b).

- **The bar.** The random network's tested units reach s = 0.70–0.95 in the conv blocks and 0.98–0.99 in the BiLSTMs. No
  trained unit's interval clears that bar.
- **Conditions (a) and (c) are met by most.** 46 (`xix-blb-v1`) and 50 (`hertwig-v2`) of the 70 tested units reach
  s ≥ 0.5 and beat the blank-crop and shuffle controls. They fail only on (b).
- **The best tested units are strongly selective.** For example, `L_14 rev 173` in `xix-blb-v1` has s = 0.97 [0.92, 0.99]
  for `g`, against 0.05 with shuffled characters and 0.02 on blank crops. The random network's bar in `L_14` is 0.98.

**Post hoc, descriptive only (not part of the rule).**

- **The random network's bar comes from the line margins.** A randomly initialised BiLSTM separates the frames before the
  first character and after the last almost perfectly (s = 0.98). For character labels, its best units reach only
  0.40–0.67 in `L_12`–`L_16`, against 0.84–0.97 for the trained units.
- **Positions are reachable without ink.** Many trained line-start units score as high on blank crops as on the line
  (for example `L_12 fwd 133`: 0.97 on the line, 0.94 on a blank crop), and condition (c) correctly rules them out.
- **The rule mixed label kinds.** It compares a unit's best label with the random network's best label of *any* kind. A
  bar matched to the kind of label would have let character units pass in the BiLSTMs.
- **The shuffle control cannot test spaces.** It leaves spaces in place by design, so a space-selective unit (such as
  `L_16 fwd 84`, s = 0.96) has the same s under shuffling and fails condition (c) by construction.

Both points are lessons for a new rule, fixed before a new run, not a re-reading of this one.

### Claim P: the blocks code characters

| block | `xix-blb-v1` bal. acc. [95 %] | random | shuffled | P | `hertwig-v2` bal. acc. [95 %] | random | shuffled | P |
|---|---|---|---|---|---|---|---|---|
| `C_0` | 0.10 [0.09, 0.11] | 0.09 | 0.06 | no | 0.12 [0.11, 0.13] | 0.16 | 0.05 | no |
| `C_3` | 0.15 [0.12, 0.18] | 0.10 | 0.07 | no (< 0.10 above random) | 0.29 [0.27, 0.32] | 0.19 | 0.06 | **holds** |
| `C_6` | 0.32 [0.24, 0.37] | 0.13 | 0.08 | **holds** | 0.48 [0.45, 0.51] | 0.23 | 0.07 | **holds** |
| `C_9` | 0.46 [0.36, 0.52] | 0.12 | 0.08 | **holds** | 0.57 [0.53, 0.60] | 0.16 | 0.07 | **holds** |
| `L_12` | 0.66 [0.55, 0.72] | 0.16 | 0.09 | **holds** | 0.59 [0.55, 0.62] | 0.30 | 0.07 | **holds** |
| `L_14` | 0.83 [0.71, 0.88] | 0.10 | 0.09 | **holds** | 0.74 [0.70, 0.77] | 0.16 | 0.07 | **holds** |
| `L_16` | 0.91 [0.80, 0.95] | 0.07 | 0.09 | **holds** | 0.76 [0.72, 0.79] | 0.08 | 0.07 | **holds** |

There are 22 classes (`xix-blb-v1`) and 28 (`hertwig-v2`), so chance is 0.05 and 0.04.

- **Character identity becomes readable in the later conv blocks.** It appears above the random network in `C_6`
  (`xix-blb-v1`) and in `C_3` (`hertwig-v2`), and grows through every BiLSTM.
- **`C_0` is at the level of a random first layer** in both models. This is the epic's "stroke or character" question:
  the first convolution is not character-selective; selectivity appears from the second or third conv block on.
- **The probe uses most units** (`units_used` in the summaries: 31 to all of a block's units from `C_3` on) even with L1 regularisation. The code is
  distributed, not carried by a few units.
- **One direction alone reads the current character a little worse than both together.** The reverse half is a little
  better than the forward half in all six cases (for example `L_14`: 0.77 against 0.70 in `xix-blb-v1`; 0.68 against
  0.60 in `hertwig-v2`).

### Prediction D: the two directions carry different context; holds in all 12 cases

| block | `xix-blb-v1` previous: fwd − rev | next: rev − fwd | `hertwig-v2` previous: fwd − rev | next: rev − fwd |
|---|---|---|---|---|
| `L_12` | +0.17 [+0.13, +0.19] | +0.15 [+0.09, +0.17] | +0.07 [+0.06, +0.09] | +0.08 [+0.07, +0.09] |
| `L_14` | +0.45 [+0.37, +0.49] | +0.41 [+0.33, +0.44] | +0.30 [+0.27, +0.32] | +0.35 [+0.32, +0.36] |
| `L_16` | +0.18 [+0.14, +0.21] | +0.22 [+0.17, +0.24] | +0.19 [+0.18, +0.20] | +0.20 [+0.18, +0.21] |

- **On a character's frames, the forward units know the character before it and the reverse units the one after it.**
- **The asymmetry is largest in `L_14`.** There the right direction reaches 0.54–0.69, the other 0.21–0.25.

### Stability (`hertwig-v2`): holds in every block, mostly inherited

Median per-unit correlation of the label profile with the same unit in another network:

| block | five seed runs | null, 99th percentile (largest) | parent `cursive_inzigkofen` | random network |
|---|---|---|---|---|
| `C_0` | 0.98–0.99 | 0.52 | 0.95 | 0.13 |
| `C_3` | 0.97–0.98 | 0.44 | 0.90 | 0.08 |
| `C_6` | 0.93–0.97 | 0.42 | 0.80 | 0.07 |
| `C_9` | 0.99 | 0.26 | 0.92 | 0.11 |
| `L_12` | 0.96 | 0.07 | 0.87 | −0.02 |
| `L_14` | 0.96–0.97 | 0.06 | 0.84 | 0.03 |
| `L_16` | 0.96–0.97 | 0.04 | 0.77 | 0.01 |

- **The seed runs reproduce `hertwig-v2` unit by unit, in every block.**
- **Most of this is inherited.** The seed runs and `hertwig-v2` were fine-tuned from the same parent, which already
  correlates 0.77–0.95.
- **Fine-tuning moves every unit the same way.** Each seed run is closer to `hertwig-v2` than the parent is, most in
  `L_16` (0.97 against 0.77).
- **It does not show that independent training finds the same units.** No network here was trained from scratch.

### Gate 2: fails for both models

| model | P (≥ 1 block) | S (≥ 1 unit) | stability in a block where S holds | Gate 2 |
|---|---|---|---|---|
| `xix-blb-v1` | yes (`C_6`–`L_16`) | no | not testable (no seed runs) | **fails** |
| `hertwig-v2` | yes (`C_3`–`L_16`) | no | no (S holds in no block) | **fails** |

- **Both atlases stay descriptive.** The unit cards show selectivity with its controls, but no card claims a unit "is" a
  character detector.
- **The L3 circuits become descriptions, not circuit claims.** That is the seven of `hertwig-v2` (`e r s t a h l`) and
  `t` of `xix-blb-v1`. The L3 ablation results themselves stand; Gate 2 governs only whether the atlas may call them circuits.

### Descriptions

- **Selectivity and ablation are unrelated across units.** The Spearman correlation between a unit's evaluation s and
  its L3 selection ΔCER is −0.18 (`xix-blb-v1`) and +0.08 (`hertwig-v2`). It is near 0 in every BiLSTM. `C_0` of
  `xix-blb-v1` is the exception (+0.77, 32 units; the nine dead filters, with s = 0 and ΔCER = 0, contribute to it).
- **Circuit units are somewhat selective for their own character.** In `hertwig-v2`'s seven holding circuits, the
  median s of the circuit's units for its own character is 0.21–0.43. Over all units it is 0.19–0.22. The `t` circuit of
  `xix-blb-v1` has 0.19 against 0.22.
- **The space circuits are not space-selective units.** Their units have s = 0.42 and 0.50 for the space, against 0.27
  and 0.31 for all units, while their ablation is non-specific (L3).
- **Where the BiLSTM units' best labels fall.** Most are characters, but `L_12` and `L_14` hold many position units
  (136 and 108 in `xix-blb-v1`) that the random network also has.

### What follows

- **The atlas can say, with controls,** that character identity is linearly readable from the third conv block on and
  that the two BiLSTM directions carry the preceding and following character.
- **It can also say** that this unit-level picture is reproduced by seed runs from the same parent.
- **It cannot name single character units under this rule.**
- **A rule with a label-matched random bar and a space-aware shuffle** would be the next test, fixed before its run.
- **L4 (Gate 3) needs a model for which Gate 2 passes.** None does yet.

```bash
python -m wsp.atlas_select run --model models/kraken/kraken-german-xix-blb-v1.mlmodel
python -m wsp.atlas_select run --model models/kraken/kraken-bern-hertwig-v2.mlmodel
python -m wsp.atlas_select report --out docs/atlas
```

Extraction takes about 2 minutes for both models (12 workers × 4 threads). The analysis takes about 20 minutes, most of it
for the probes (24 jobs).

## Closing synthesis (15.11.2026)

This section draws no new numbers. It collects the results of L0–L3 above, answers the questions of the epic, and states
what the atlas cannot say. Every level ran under a rule committed before its first run, on practice lines only and on CPU.

### The gates

| gate | rule | `xix-blb-v1` | `hertwig-v2` | consequence |
|---|---|---|---|---|
| Gate 1 (L0, L2) | occlusion peak on the character the CTC alignment gives, share ≥ 0.80 | fails, 0.61 | fails, 0.73 | the viewers show activation, not location |
| Gate 2 (L1, L3) | P in a block, S for a unit, stability where S holds | fails (S; no seed runs) | fails (S) | the atlas is descriptive; no circuit claims |
| Gate 3 (L4) | L4 only for models that pass Gate 2 | not run | not run | no comparison across models |

Gate 1 fails narrowly: occlusion finds a letter to within one neighbour (0.97 and 0.98), not on the letter itself.
Gate 2 fails on claim S only. Its bar, the random network's best unit of any label, came from the line margins (see
the L1 post hoc notes).

### What each part of the network responds to along a line

| block | depends on (L2) | codes characters (L1, P) | single units (L1 S, L3 U) | groups (L3 G, C) | stable across seeds (L1) |
|---|---|---|---|---|---|
| `C_0` | 16 px around the frame | no: at the random network's level | one unit matters in `hertwig-v2` (`C_0 ch 3`, +1.3 points); none is selective beyond the bar; dead filters do nothing | the top 10 silence both models | yes, 0.98–0.99 (parent 0.95) |
| `C_3` | 28 px | `hertwig-v2` only (0.29) | none | no | yes (parent 0.90) |
| `C_6`, `C_9` | 44 and 76 px | yes (0.32–0.57) | none | `hertwig-v2` yes; most circuit units lie in `C_3`–`C_9` | yes (parent 0.80–0.92) |
| `L_12` | the whole line in principle; half of the change within ±28–104 px | yes (0.59–0.66) | none; best units often mark positions, as in the random network | inconclusive (< 1 point) | yes, 0.96 (parent 0.87) |
| `L_14` | as `L_12` | yes (0.74–0.83) | none | inconclusive; in `hertwig-v2` above all random groups | yes (parent 0.84) |
| `L_16` | slightly wider than `L_12` | yes (0.76–0.91) | none | inconclusive | yes (parent 0.77) |

"Stable" is measured for `hertwig-v2` only, against seed runs fine-tuned from the same parent.

### The epic's questions

1. **Is the convolutional front end character-selective or only stroke-selective, and where does selectivity first
   appear?** The first convolution is not character-selective: a probe on `C_0` reads characters no better than on a
   random first layer. Character identity becomes readable above the random network in `C_3` (`hertwig-v2`) or `C_6`
   (`xix-blb-v1`) and grows through every later block. The atlas did not test stroke features, so it cannot say that
   `C_0` is stroke-selective, only that it is not character-selective. It is nevertheless where the reading is most
   fragile: ten of its channels together carry the whole reading.
2. **Do the BiLSTM units code characters, positions, word boundaries or context, and does one direction carry what the
   other does not?** Each BiLSTM layer codes the current character (probe up to 0.91), and the code is spread over most
   units, not carried by a few. Many units in `L_12` and `L_14` mark positions in the line, as units of a random network do.
   Context is clear: on a character's frames the forward units carry the previous character and the reverse units the
   next, in all 12 cases, most in `L_14`. Word boundaries were not shown separately. Space-selective units exist (for
   example `L_16 fwd 84`, s = 0.96), but the shuffle control cannot test them.
3. **Do the unit cards of one seed run reappear in another, and more than in a random network?** In `hertwig-v2`, yes,
   unit by unit in every block (median correlation 0.93–0.99 against ≈ 0 for the random network). Most of this is
   inherited from the shared parent (0.77–0.95). The atlas does not show that independent training finds the same
   units, since no network here was trained from scratch. `xix-blb-v1` has no seed runs.
4. **Do the units that matter in ablation coincide with selective units, with large weight changes in fine-tuning, or
   with the dead filters?** Not with selective units (Spearman −0.18 and +0.08) and not with weight change (+0.01 and
   −0.02). Circuit units are only somewhat selective for their own character (median s 0.21–0.43 against 0.19–0.22
   overall). The dead filters do nothing, as predicted: ablating them changes no reading, and their s is 0.
5. **Does the atlas explain a known failure, such as the deleted spaces and `e` decided upstream of the output layer?**
   No. Neither model has a space circuit: in `xix-blb-v1`, the Kurrent model of that failure, the best ten units cost
   the space, `e` and `n` 4.4, 0.6 and 3.7 points, all inconclusive; in `hertwig-v2` the space circuit degrades the whole
   reading. The deletions are consistent with a mechanism spread over many units or in the BiLSTMs' interaction, which a
   ten-unit ablation and a single-unit selectivity test do not find.

### What the atlas shows, and where it shows nothing

It can say, with its controls:

- character identity is linearly readable from the second or third conv block on;
- the BiLSTM directions carry the preceding and following character;
- trained recurrent blocks look mainly within one to five characters, although they could use the whole line;
- seed runs from the same parent reproduce the unit-level picture.

It shows nothing for:

- **location:** where in the image a unit looks, at character resolution (Gate 1);
- **single units:** single character units under the L1 rule (claim S), and single units that matter (one unit in
  2 × 1,392);
- **circuits:** circuits as claims (Gate 2); the eight L3 circuits remain descriptions;
- **spaces:** a mechanism for the deleted spaces;
- **other models:** anything across models (L4).

### What would change these answers

- **A new selectivity rule:** a random-network bar matched to the kind of label, and a control that moves spaces.
  It must be fixed before its run, on pages that L1 did not use, or be reported as exploratory, because it is written
  after the L1 results were seen.
- **Location at word level:** a finer occlusion window, or a test at the level of words, under a new Gate 1 rule.
- **Seed runs for `xix-blb-v1`:** this needs training on a GPU, outside the scope of this epic.
- **Seed runs from scratch:** these would separate convergence from inheritance.

### Deliverables against the epic

| epic deliverable | where |
|---|---|
| `wsp/atlas.py`, with selectivity, occlusion and ablation | split into `wsp/atlas.py` (extraction, occlusion, viewer), `wsp/atlas_ablate.py`, `wsp/atlas_select.py`; tests in `tests/` |
| a static viewer | `docs/atlas/index.html` and one page per model; unit cards `units-*.html`. Built by the `report` commands of `wsp.atlas` and `wsp.atlas_select`, not by a separate `scripts/atlas_build.py` |
| tables and figures for `xix-blb-v1` and `hertwig-v2`, with five seed runs as the null | `docs/atlas/` and this report |
| this report, and a short section for the article | this report; Section 6.6 of `paper/manuscript.md` |
