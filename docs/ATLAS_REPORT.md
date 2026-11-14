# Activation atlas: report (epic thodel/weightspace#13)

Levels L0 (activation map) and L2 (dependence on the image) first, as the epic orders them; Gate 1 decides
whether the viewer may claim *where* in a line something happens. L3 (ablation) followed on 14.11.2026; L1 and L4 are not started.

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
