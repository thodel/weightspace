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
