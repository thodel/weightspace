# Hypotheses — planted-f1 / s14 / task (b), budget 0 (blind programmer)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort (no extended-thinking override),
tools used: cat/ls/grep over the three pack files and the allowed `wsp/` API files, `py_compile` of `program.py`.
No pass, no model loaded, no program run.

## Decision: ship no edit

Nothing in the pack is a reading I am *fairly sure* is a defect rather than ordinary fine-tuning or ordinary Kurrent
error behaviour. At budget 0 a wrong edit is scored unchecked, so the program applies nothing and freezes the model as is.

## What I looked at, and why each candidate did not reach "fairly sure"

| # | candidate (reading) | edit I would have made | predicted effect if it *were* a defect | why not shipped |
|---|---|---|---|---|
| H1 | Deletion-dominated confusions (` `→∅ 78, `e` 34, `n` 28, `i` 21, `r` 19, `s` 16, `.` 13, `o` 11, `c` 10 vs ∅→` ` 25), length ratio 0.979, 8 collapsed lines → blank over-predicted? | `add` −0.25…−0.5 to `O_18.lin.bias[0]` (blank) | CER −0.2 to −0.5 pt if the blank is inflated; +0.2 to +1 pt (insertions) if not | Blank bias is +1.23 in the child vs +1.12 in the parent (Δ +0.11): not planted. The deleted letters e/n/i/r/c/s/o are exactly the Kurrent minim letters (and spaces), which a healthy model under-counts; a/t/d/h/u, which have distinct shapes, are absent from the list. That is the natural error profile of this hand, not a blank defect. Coin flip at best → not applied. |
| H2 | Blank row moved rel L2 1.27 vs parent (more than the mean row 1.08), norm only 9.27 | `interpolate` blank row towards the parent's blank row (α 0.3–0.5) | CER −0.3 pt if the row was perturbed; +0.5…+2 pt if the parent's blank row no longer fits the retrained features | The parent is far worse on this hand and its L_16 features differ (rel 0.30, cos 0.95); any output row legitimately rotates. The whole aligned output layer moved rel ≈1.0, so 1.27 is not an outlier. |
| H3 | `O_18.lin.weight` rows `á ′ Ä š é ý ͞ ž ř ô` have norm 44–48 = ~2.0× the parent (cos 0.998, rel ≈1.02), bias Δ −0.7…−2.6 | `scale` those rows by 0.5 (or transplant parent rows) | ≈0 on CER either way (these characters do not occur in the practice text) | Rows of classes that never occur in the fine-tuning data grow along their own direction under the CTC gradient (gradient ∝ −h pushes an already anti-aligned row further out) and get a more negative bias; this is the textbook signature of natural fine-tuning, not a planted scale. No insertion of any of these characters appears in the confusions. |
| H4 | `L_16.layer.weight_ih_l0` abs max 4.70 (≈19 × rms) vs 6–10 × rms in every sibling tensor; rms 0.242 vs 0.226 for its reverse direction | clip the tensor to ±2.5, or `set` the outlier elements to 0 | ≤ ±0.2 pt | Back-solving parent norms from (rel L2, cosine) shows the forward/reverse asymmetry already exists in the parent (fwd ≈136/151 vs rev ≈128/144): inherited. A single 4.7 element carries 22 of 18,728 units of squared norm; rel L2 vs parent (0.296) is in line with the other LSTMs. L_16 is heavier-tailed in general (hh abs max 2.84 too). Likely a learned weight. |
| H5 | Space bias −7.85 is the most negative of 297 rows (≈3 σ below the mean of −1.92 ± 2.01); space is the top deletion | `add` +1…+2 to `O_18.lin.bias[cm[" "]]` | CER −0.3 pt if planted; +0.3…+1 pt (space insertions, WER up) if not | 78 deletions out of roughly 2,000 reference spaces is ~4 % — a planted shift of several logits would lose far more spaces and wreck the WER (0.175 is moderate). A strongly negative space bias with a strong gap-feature weight row is normal for CTC line models. |
| H6 | `C_3.co.bias` rel L2 1.09, cos 0.68 (all other conv biases rel 0.15–0.33) | transplant parent `C_3.co.bias` | ≈0 | The tensor is tiny (L2 0.10, rms 0.018; parent ≈0.067): a small absolute drift reads as a large relative one. Abs max 0.082 — nothing was set to a large value. |
| H7 | No substitution pair in the top-10 confusions; weakest rows are exactly the frequent letters (e r s t i n a 15–17), new rows at 15–29, no zero or runaway row | row swap / zeroed row repair | — | No evidence of a swapped, zeroed or exploded output row. |

Summary of the readings: 4.11 M parameters, all 34 tensors present with plausible statistics; the child-vs-parent diff is a
uniform fine-tuning picture (convs 0.12–0.33, LSTMs 0.20–0.33 rel L2, output layer re-learned); CER 0.065 / WER 0.175 with
minim-letter and space deletions as the top confusions. If a defect is planted, it is below what these three readings resolve
(e.g. a modest bias shift on a mid-frequency class, or a small perturbation inside one LSTM gate block), and I cannot locate it
without a pass. The 8 collapsed lines are the one reading I would have spent a pass on; blind, I do not know which tensor to touch.

Expected outcome: final CER = C0 CER (0.0650 on the practice set); no change on the held-out set.
