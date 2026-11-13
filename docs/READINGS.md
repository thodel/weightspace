# Readings — passes = 0 (phase 0, §8 of the brief)

Written 06.11.2026 on asteraix from `wsp.stats` (`docs/readings/*.json` and
`*.md` hold the full tables). No forward pass was spent on any of this; the
confusion pairing comes from the C0 numbers in `docs/BASELINES.md`. Numbers are
relative L2 of the child minus parent per tensor (`‖Δ‖/‖parent‖`), cosine
between parent and child, and for the output layer per-row values with rows
**aligned by character** (`wsp.align.codec_row_map`), never by index — every
fine-tune here grew its codec (`resize: union`).

Architecture of the in-house models (all seven): `[1,120,0,1 Cr3,13,32 Do Mp2,2
Cr3,13,32 Do Mp2,2 Cr3,9,64 Do Mp2,2 Cr3,9,64 Do S1(1x0)1,3 Lbx200 Do Lbx200
Do Lbx200 Do O1c<N>]`, 4.11 M parameters, 34 tensors; the output layer is
`O_18.lin.weight` (N × 400) + bias. kraken's LSTMs carry an all-zero
`bias_ih` in every layer (only `bias_hh` is trained), so those six tensors are
unchanged in every pair.

## What is common to every pair

1. **Fine-tuning inflates the rows of rare characters.** In every parent→child
   pair the rows that moved most (relative L2 0.4–1.3) belong to characters the
   child saw rarely or never — `ü ö ä ſ · § ꝫ ⁊ ♀ ♂ ̅ ̶ ͠ ͛` — and they moved in
   one direction: the row norm grows by 30–110 % and the bias falls by 0.2–1.0.
   That is what CTC training does to a class that only ever receives negative
   gradient: the row is pushed away from every input it sees. The frequent
   characters (`e n i a r t s`) have the *smallest* norms in every model
   (4–5 against a mean of 11–15) and barely move. For the sessions this means
   the obvious "revert rare rows to the parent" edit is unlikely to change CER
   on practice text that does not contain them — and may be exactly right where
   a rare glyph is wrongly emitted.
2. **The blank row moves and its bias rises in every fine-tune**: relative
   change 0.18–0.25, bias Δ +0.06 (hertwig-v1), +0.002 (v1→v2), **+0.118
   (combined-v1)**, +0.056 (schlussredaktion), **+0.113 (xix-blb)**. Together
   with the dominant confusion in every C0 table (` →∅`, `e→∅`, `n→∅`:
   deletions), a blank-bias edit (`scale`/`add` on `O_18.lin.bias[0]`) is the
   first cheap hypothesis for every target — predicted effect: fewer deletions,
   more insertions, net CER −0.5 to −1 point if the deletion share is as large
   as the practice tables say; collapsed lines must not rise.
3. **The conv stack moves less than the recurrent stack and the output layer**,
   except `C_0` which barely moves at all (0.07–0.13): the first 3×13 filters
   are generic stroke detectors. `C_6`/`C_9` move 0.12–0.23, the three BiLSTMs
   0.06–0.14 mean (max 0.26–0.33 on single gate blocks), the output layer
   0.15–0.24.

## Corrections of 13.11.2026

Written after `docs/ZOO_SURVEY.md` (37 public models, 47 read in all), `docs/SEED_NULL.md` and the measured-frequency test of ZOO_SURVEY §4b. The text above stays as it was written on 06.11.2026.

- **1 (rare rows).** With counts instead of the proxy, on the six in-house pairs: rows of characters that never occur in the child's practice lines lose bias (median −0.32 to −0.75) and gain norm in almost every row; rows of characters that occur once or more
  move with the rest of the network (median bias change −0.03 to +0.05, norm ratio 0.97–1.04). There is a step at *absent*, not a gradient by frequency, and "the rows that moved most are the rare ones" holds only for the first adaptation from the public parent (Spearman of log count with relative move −0.6), not for
  continuations (+0.35) or the Kurrent pair (+0.02). On the 14 documented pairs of the zoo survey the proxy version holds in direction (norm up in 13, bias down in 14) but not in selectivity (all four criteria in 5; 1 of 7 public pairs). The "30–110 %" growth concerns the rows that moved most; the median over special rows is 0.85–1.56.
- **2 (blank bias).** Holds and is larger relative to the other biases: +0.16 to +0.57 in the in-house pairs against +0.06 to +0.12 raw, because the other biases fall; 14 of 14 documented pairs rise relative to the other biases (13 raw).
- **3 (conv versus recurrent).** Not supported as written. The numbers above (deeper conv blocks 0.12–0.23 against LSTM blocks 0.06–0.14) were means over tensors including the conv biases and the all-zero `bias_ih` of the LSTMs; over weights only, the conv stack is below the BiLSTMs in 11 of 14 pairs and below the output rows in 4 of 14.
  What holds: the first conv block moves least (12 of 14 pairs) and movement grows with depth (10 of 14), and the seed null shows this pattern between two runs that differ only by seed.

## cursive_inzigkofen → bern-hertwig-v1 (`diff_cursive_inzigkofen__bern-hertwig-v1`)

Codec 261 → 266 (adds `" ̅ „ ♀ ♂`, all with small norms 1.9–4.1: barely
trained). Conv blocks 0.07–0.12, LSTMs 0.06–0.09, output rows mean relative
change 0.26 — the largest output-layer move of the family, as expected for the
first adaptation of a 15th-century model to the 1528 hand. The rows that moved
most are abbreviation glyphs of the parent's domain (`ꝫ ⁊ ꝰ ꝝ ħ`) and umlauts
(`ü ä`): the parent's medieval abbreviations were suppressed, the umlauts
re-learnt. The new `"` row has norm 1.9: the model cannot emit it.

## bern-hertwig-v1 → bern-hertwig-v2 (`diff_bern-hertwig-v1__bern-hertwig-v2`)

Same codec, same data, 5× the learning rate and 91 instead of 60 epochs. The
move is uniform: every block 0.10–0.17, LSTM max 0.26, output layer 0.19
(bias 0.24). **No single tensor or row stands out** — the top rows (`ꝓ ü ͮ ö ä
⁊ ¬ ꝫ`) move by 0.17–0.33 with cosines ≥ 0.97; the blank bias is unchanged
(+0.002). This is a healthy continuation of the same training, not a repair of
something in v1: v2's gain over v1 on the practice set (CER 0.177 → 0.083,
train side) is spread over the whole network. Nothing here suggests a
localised edit that would turn v1 into v2.

## bern-hertwig-v2 → bern-combined-v1 (`diff_bern-hertwig-v2__bern-combined-v1`)

Codec 266 → 269 (`½ ƒ ̊`). The pair that the plan picked as target 1 because
the child is worse than the parent on held-out (28.8 % vs 16.7 % for the
sibling). The weights say: conv 0.08–0.18, LSTMs 0.12–0.14 (max 0.27), output
rows 0.19, **blank bias +0.118** — the largest blank shift of the family — and
the rare rows `· ſ ̶ ͠ ͛ ö ü ä ̅ § ꝝ ꝯ` grew by 40–100 % with biases down by
0.6–1.0. That profile (blank up, everything rare pushed down hard) is what a
model learns on **label noise**: and the training material of this model *was*
noisy — it was trained from the 25.10 re-export of the dataset, in which 44 of
the 50 Hertwig training pages carry the wrong page image for their XML
(docs/PHASE0_REPORT.md §3). On correctly paired lines (export 22.10) the
practice C0 numbers are hertwig-v2 0.083, combined-v1 0.185: the child lost
10 points on its parent's own training lines. So the "defect" of target 1 is
a data defect first; the weight-space question becomes whether the damage is
localised enough to undo (hypotheses for phase 1: (a) blank bias back to the
parent's value, predicted −1 to −2 points on practice; (b) output layer
transplanted from hertwig-v2 with the three new rows kept, predicted most of
the 10 points back if the recurrent stack is still sound; (c) per-layer α
along `hertwig-v2 + α(combined − hertwig-v2)` to find where the damage sits).
The retrained comparator (C4) on the corrected data is the honest baseline.

## bern-hertwig-v2 → bern-schlussredaktion-v1 (`diff_bern-hertwig-v2__bern-schlussredaktion-v1`)

Codec 266 → 268 (`½ ̊`). Conv 0.07–0.16, LSTMs 0.11–0.12, output rows 0.17,
blank bias +0.056. The rows that moved most are `♀ ♂ ̅` (relative 1.1–1.3,
from norms 4–5 to 9–10: these were the untrained new rows of hertwig-v1, now
pushed away hard) and the same rare set as above. On the Hertwig practice
lines this model reads at 0.214 (vs 0.083 for its parent): it has moved to
the Schlussredaktion hand, and its top confusions are `/→∅`, `/→,`, `d→D`,
`u→v` — the punctuation and case conventions of the second project, not
recognition failures. A practice set from the Schlussredaktion train project
is what a session on this model needs; this one was not built in phase 0.

## german-xix-v1 → german-xix-blb-v1 (`diff_german-xix-v1__german-xix-blb-v1`)

Codec 224 → 296: 72 new rows (Greek, combining marks, `ꝑ ꝓ ꝗ …`, `⁊`), all
with norms 8–13 and biases −0.5 to −0.8 — the BLB `resize: union` added the
ALTO corpus's whole character inventory, mostly characters the model will
never see; they are inert but they are 72 rows of the output softmax. The
largest move of all five pairs is in the **second conv block's bias**
(`C_3.co.bias` relative 1.09, cosine 0.68) and the deeper convs (`C_6`, `C_9`
0.26–0.33): the 19th-century Kurrent of the Laßberg letters re-tuned the
feature extractor, not only the classifier. Output rows mean 0.19, blank bias
+0.113; the rows that moved most are typographic (`« ¬ _ » … ° ‘`) with bias
drops of 2–2.7, consistent with an ALTO-derived GT that does not use them.
Practice C0 (docs/BASELINES.md): xix-v1 0.304 → xix-blb-v1 0.065 on the BLB
train lines; the top confusions of the child are ` →∅`, `n→∅`, `e→∅`.

## kraken+ `Cr255,1,85` — the padding rows (`padding_collapse-probe-131-cr255`)

`C_12.co.weight` is (85, 512, 255, 1). The kept row 127 has L2 12.38; the
other 254 rows have L2 12.045 ± 0.027 (min 11.94, max 12.12). The padding
rows are at their initialisation norm, as they must be (they never receive a
gradient); the kept row has grown by only 3 % because this checkpoint is the
collapsed probe run (val accuracy 0.0083). On a trained kraken+ model the gap
would be the size of the training signal. The rewrite that drops them
(`wsp.rewrite`, 15,207,871 → 4,153,791 parameters) gives identical text on
300 lines and a maximum softmax difference of 1.1e-6 (docs/RESULTS.md).

## Output layers at a glance (`describe_*`)

| model | rows | blank norm / bias | other rows norm | weakest rows |
|---|---:|---|---|---|
| cursive_inzigkofen | 262 | 3.07 / +0.82 | 10.8 ± 3.0 | `̄ n e ͠ ͛ · i a ü` |
| bern-hertwig-v2 | 267 | 2.97 / +0.88 | 12.8 ± 3.8 | `" „ ♀ n e i ♂ a t` |
| bern-schlussredaktion-v1 | 269 | 2.78 / +0.94 | 13.6 ± 4.0 | `" e n i ̊ a ½ r t` |
| bern-combined-v1 | 270 | 2.78 / +1.00 | 13.7 ± 4.2 | `" n e i a r t u o` |
| german-xix-blb-v1 | 297 | 4.64 / +1.23 | 14.8 ± 4.1 | `e ̓ ν r s t i n a` |
| kraken-plus-shard00-cr1 | 102 | 1.53 / +0.03 | 2.9 ± 1.1 | `e t a d o s g l h` |

The pattern is the same everywhere: the blank is the weakest row with the
highest bias; the frequent letters are the next weakest. The 85-channel
kraken+ head is a different regime (norms 1–5, biases within ±0.6): the
bottleneck keeps the classifier small, which is one more reason the
`Cr255 → Cr1` rewrite changes nothing.

## german-xix-v1 → german-xix-blb-v1: per-block α-sweep (C3, 16 passes, 06.11.2026)

`wsp.alpha_sweep`, `docs/baselines/alpha-xix-blb-v1.json`. One block at a time
set to `parent + α(child − parent)` with the rest of the network left as the
child; the output layer with rows aligned by character (225 shared rows
interpolated, the 72 new rows kept as the child has them). Practice set
`blb-train-300` only, CPU, 20 s per pass, 321 s in all; the held-out set was
not touched. Δ is against the C0 reading of the unedited child (CER 0.06458,
the parent reads 0.30388 on the same lines), paired per-line bootstrap, 2,000
resamples, seed 20261106. A control, outside any session budget.

| block | rel L2 child−parent | α | CER | Δ vs C0 (pt) | 95 % interval | WER | collapsed |
|---|---:|---:|---:|---:|---|---:|---:|
| C_0 | 0.119 | 0.5 | 0.06443 | −0.016 | [−0.125, +0.102] | 0.1699 | 10 |
| C_0 | 0.119 | 1.5 | 0.06668 | +0.210 | [+0.055, +0.370] | 0.1775 | 10 |
| C_3 | 0.167 | 0.5 | 0.06614 | +0.155 | [+0.008, +0.302] | 0.1755 | 10 |
| C_3 | 0.167 | 1.5 | 0.06730 | +0.272 | [+0.142, +0.430] | 0.1795 | 12 |
| C_6 | 0.256 | 0.5 | 0.06785 | +0.326 | [+0.139, +0.547] | 0.1791 | 9 |
| C_6 | 0.256 | 1.5 | 0.06699 | +0.241 | [+0.064, +0.424] | 0.1783 | 11 |
| C_9 | 0.295 | 0.5 | 0.06614 | +0.155 | [−0.008, +0.316] | 0.1775 | 10 |
| C_9 | 0.295 | 1.5 | 0.06629 | +0.171 | [+0.000, +0.355] | 0.1727 | 10 |
| L_12 | 0.274 | 0.5 | 0.08362 | +1.904 | [+1.318, +2.589] | 0.2107 | 13 |
| L_12 | 0.274 | 1.5 | 0.06730 | +0.272 | [+0.008, +0.553] | 0.1799 | 9 |
| L_14 | 0.260 | 0.5 | 0.07640 | +1.181 | [+0.872, +1.513] | 0.2071 | 11 |
| L_14 | 0.260 | 1.5 | 0.06831 | +0.373 | [+0.085, +0.664] | 0.1803 | 11 |
| L_16 | 0.266 | 0.5 | 0.08767 | +2.308 | [+1.971, +2.670] | 0.3191 | 10 |
| L_16 | 0.266 | 1.5 | 0.06505 | +0.047 | [−0.181, +0.278] | 0.1735 | 11 |
| O_18 | 0.195 | 0.5 | 0.06684 | +0.225 | [+0.101, +0.363] | 0.1827 | 10 |
| O_18 | 0.195 | 1.5 | 0.06575 | +0.117 | [+0.000, +0.245] | 0.1727 | 10 |

What it says:

1. **The 24-point gain is not in any one block.** Halving the fine-tune's move
   on a single block costs at most 2.3 points (L_16), and the eight halvings
   together add up to about 6.5 points against a 24-point gap: the gain is
   held jointly, every block depends on the others having moved with it. This
   is the profile of a healthy fine-tune, as the v1→v2 reading above described
   for the Bern family, not of a localised repair.
2. **Of what is localisable, the recurrent stack holds it**: L_16 2.3 points,
   L_12 1.9, L_14 1.2 at α = 0.5; the four conv blocks and the output layer
   0.15–0.33 each. The half-moved L_16 reads with the same letters but loses
   word boundaries (` `→∅ 103 → 263, WER 0.171 → 0.319, length ratio 0.969 →
   0.953): the last BiLSTM is where the child learnt the Kurrent spacing. The
   `C_3.co.bias` move that stood out in the weight reading (rel. L2 1.09) is
   worth 0.16 points, i.e. nothing: a large relative change in a 32-element
   bias is not a large function change.
3. **Only L_16 tolerates extrapolation.** At α = 1.5 every other block is
   worse than C0 with an interval that excludes 0 (C_0 +0.21, C_3 +0.27,
   C_6 +0.24, L_12 +0.27, L_14 +0.37) or touches it (C_9, O_18); L_16 is
   +0.05 with interval [−0.18, +0.28], indistinguishable from the child.
   `C_0` at α = 0.5 is the other indistinguishable point (−0.02): the first
   conv block hardly matters in either direction, as its small move said.
4. **For the next session (preregistered hypothesis, budget 10):** target
   `L_16` (its eight tensors as one block, or `weight_hh`/`weight_ih` forward
   and reverse separately). The sweep makes α = 1.5 a null and α = 0.5 a
   2.3-point loss; the preregistered prediction for α ∈ {1.1, 1.25} is
   therefore a change inside [−0.2, +0.2] points, and under the keep rule of
   PROTOCOL §3 the session is expected to keep nothing unless a sub-tensor of
   L_16 extrapolates where the whole block does not. The honest reading is
   that extrapolating the fine-tune is not where a gain lies on this model;
   the collapsed lines (phase-1 report §5.3) are the better next question.

## german-xix-v1 → german-xix-blb-v1: layer probes, output-bias curve and the per-character α overlay (07.11.2026)

Three readings from the visualisation epic (thodel/weightspace#3). The method, the figures and the
caveats are in [docs/viz/README.md](viz/README.md). All three use the practice set
`blb-train-300` only, run on CPU, and spend no session budget; the held-out set was not touched.

- `wsp.layerview` (`docs/viz/kraken-german-xix-blb-v1__blb-train-300.json`) — one-frame linear
  probes per layer. An *identity* probe says which character, an *emission* probe says character
  or blank. Reference characters are placed on the frames by a forced CTC alignment. The probes are
  cross-fitted over 5 document folds.
- `wsp.bias_sweep` (`docs/baselines/bias-xix-blb-v1.json`) — the space-row and blank biases over a
  grid of δ. The text is re-decoded exactly from the captured logits; this equals real edited passes
  on 300/300 lines and reproduces phase 1's CER at blank −1.0.
- `wsp.alpha_overlay` (`docs/viz/alpha-overlay-kraken-german-xix-blb-v1__blb-train-300.json`) —
  the C3 sweep again, adding α = 0. It keeps every reference character's fate: emitted, replaced
  or deleted. Its CERs equal the committed sweep's (16 of 16).

**What happens where.** The child, on these 300 lines: CER 6.46 %, 354 letters and 103 spaces
deleted, 316 substitutions, 58 insertions, 12,867 reference characters.

| | conv `C_0`–`C_9` | `L_12` | `L_14` | `L_16` | `O_18` |
|---|---|---|---|---|---|
| identity probe, correctly read letters | 0.10 → 0.36 | 0.59 | 0.82 | **0.94** | 0.92 |
| emission probe, kept spaces | 0.12 → 0.64 | **0.86** | 0.96 | 0.98 | 0.98 |
| block back to the parent (α = 0), ΔCER (points) | +0.27 to +0.80 each | +6.37 | +4.05 | +5.70 | +0.68 |
| … what it adds | substitutions (+30 to +52) | **+366 letters deleted**, +86 inserted | **+342 substitutions** | **+362 spaces deleted** | +37 letters / +38 spaces deleted |

1. **Letters are identified in the BiLSTMs.** From one frame of the conv stack a linear readout gets
   about a third of the letters; each BiLSTM adds a large step, and the output layer adds nothing.
2. **The fine-tune's work is divided by block:**
   - `L_12` learnt to *find* the letters of the BLB hands: without its share, 366 more letters are
     dropped;
   - `L_14` learnt to *tell them apart*: +342 substitutions;
   - `L_16` learnt the *word gaps*: +362 deleted spaces, 3.5 times the child's own count, with
     almost no insertions.

   The conv stack's share is shape detail (0.3–0.8 points, nearly all substitutions). This sharpens
   item 2 of the α-sweep reading above, and it is the answer so far to "which part of the
   recognition process happens where" on this model.
3. **Every α = 1.5 point is an exchange, not a null.** Counted against the child:

   | block at α = 1.5 | deleted | substituted | inserted | ΔCER (points) |
   |---|---:|---:|---:|---:|
   | `L_16` | +29 | −20 | −3 | +0.05 |
   | `L_12` | −34 | +41 | +28 | +0.27 |
   | `L_14` | +46 | −7 | +9 | +0.37 |

   Extrapolating a block moves it further along its own specialisation, at the cost of the kind of
   error it does not specialise in.
4. **Not targets:**
   - **Letter deletions.** 72 % of all deletions (259 of 361) are decodable at no layer: the
     network never forms these characters. That points to the ink or the reference, as with the
     phase-4 collapsed lines.
   - **The output biases.** The space-row bias lowers CER at no δ and recovers at most 5 of 31
     candidate spaces. The blank bias trades deleted spaces for inserted ones about one for one
     (−30 / +13 at δ −1.0, ΔCER −0.02). That is phase 1's null, now with its whole curve.
   - **Most "deleted spaces"** are attributed to the reference's spaced-out punctuation
     (`konnte .`) or to a dropped neighbour: only 18 of 77 lie between two correctly read
     characters.

**Hypotheses for a preregistered session on `german-xix-blb-v1`** (budget 10, practice
`blb-train-300`, held-out `blb-eval-hand`; keep rule of PROTOCOL §3 with `min_effect` 0). Item 3
suggests that a block's two opposite effects may sit in different parts of the block. The edit
class is *part-block extrapolation*: `parent + α(child − parent)` on a subset of a block's tensors,
the rest of the network unchanged. Each part is one direction of the BiLSTM (forward or reverse),
with its `weight_ih`, `weight_hh` and `bias_hh`.

| id | edit | idea | predicted ΔCER on practice (points) | passes |
|---|---|---|---|---:|
| H-L16 | `L_16` forward, then reverse, at α = 1.5 | one direction carries the −20 substitutions, the other the +28 letter deletions | best part −0.15, range [−0.25, +0.10]; spaces deleted ±5 | 2 |
| H-L12 | `L_12` forward, then reverse, at α = 1.5 | one direction carries the −34 deletions, the other the +41 substitutions and +28 insertions | best part −0.25, range [−0.35, +0.20] | 2 |
| H-L16+L12 | the better part of each, together | the two gains add if they are independent | sum of the two kept parts, ±0.1 | 1 |
| H-α | the kept part at α = 1.25 and 1.75 | the optimum is not at 1.5 | within ±0.1 of the α = 1.5 part | 2 |

Expectation, stated in advance:
- The keep rule needs the interval to exclude 0. On 300 lines the α-sweep intervals are about
  ±0.23 points wide, so an effect of −0.15 is **expected to be refused**. Only H-L12's upper
  estimate (−0.25 or better) can pass.
- The likeliest session outcome is again *no kept edit*. Its value is diagnostic: it shows whether
  the specialisations in item 3 split by direction.
- If that is the only question, the same four part-block points can run as an extension of
  control C3 (no budget, like the sweep) instead of as a session. The two must not both be run on
  the same practice set: the control would hand the session its answers.
- Collapsed lines must not rise: 9 under phase 5's detector (10 under the legacy test, which the committed bias and overlay JSON still use).
- Held-out once, after freeze, on `blb-eval-hand` (child 9.55 %).

> **Note (08.11.2026, after merging `main`).** Two later results bear on the session design above,
> not on the readings. They should be settled in the preregistration before any pass.
> (1) Line-level intervals are too narrow, because lines within a page are correlated. The
> two-stage page-clustered bootstrap (`wsp.metrics.bootstrap_delta(..., clusters=)`,
> docs/CONTROLS_REPORT.md §9) is the honest one. It widens the intervals, so the "expected to be
> refused" above holds more strongly.
> (2) The proposed amendment PROTOCOL §2 item 8 (not yet adopted) asks for practice lines the
> subject was not trained on. `blb-train-300` holds the child's training lines. On the trainer's
> validation pages, four attempts on this model kept nothing (docs/CONTROLS_REPORT.md §3).
> If the amendment is adopted, the part-block hypotheses move to that practice set, and the
> block-by-block readings here, which were taken on training lines, should be repeated there
> before they are relied on. The output-bias null agrees with the controls report's margins: a
> deleted character sits a median 3.2 nats below the blank.

## The readings of 07.11 on lines the models were not trained on (12.11.2026)

The probes and the α overlay were repeated on `blb-val-300` and `bern-val`, the practice sets of
the controls. Their lines come from objects and pages the subjects were not trained on. Tables,
commands and figures are in [docs/viz/README.md](viz/README.md) under "On lines the models were not
trained on".

**What holds:**
- Letters become linearly identifiable over the three BiLSTMs, with the same levels within 0.03,
  for both `xix-blb-v1` and `combined-v1`.
- About two thirds of all deletions (66 % for both models) are decodable at no layer.
- Deleted spaces keep a weaker emission signal than kept spaces.
- In `xix-blb-v1` the fine-tune's `L_12` move makes the letters appear (+820 letters deleted
  without it), and `L_16`'s move does most of the word separation (+264 spaces).

**What does not hold:**
- *`L_14` tells letters apart.* On unseen lines its share also prevents deletions (+294) rather than
  substitutions (−18). On unseen material, removing a block's share turns misreadings into dropped
  letters, so substitution counts are not a measure of identity.
- *Every α = 1.5 point is an exchange.* On unseen lines the moves are plain losses. The exception is
  `C_9`, at −0.07 points.

**Corrected: the Bern block shares.** They were measured on `bern-train-300` and confounded by the
parent `hertwig-v2`, which was trained on its Hertwig lines. On `bern-val` the same artefact shows
directly:
- on the 212 lines the parent trained on, every block set back to the parent "improves" CER (by up
  to 5.5 points);
- on the 301 lines it never saw, every block costs, with page-clustered intervals: `L_16` +2.45
  [+1.72, +3.24], mostly letters; `L_14` +1.42; the output layer +0.48 [+0.15, +0.81].

**The part-block hypotheses of 07.11 (H-L16, H-L12, H-L16+L12, H-α) are withdrawn.** They rested on
the α = 1.5 exchanges, which do not replicate on unseen lines. A session built on them would test a
training-line effect.

**What remains for a session on these models** is what the controls report already found:
interpolation toward a relative of the same lineage, judged on lines that relative never saw, with
page-clustered intervals.
