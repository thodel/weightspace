# Pre-registration — 20261111-planted-k5-output-weight-x2-b-f1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-k5-output-weight-x2-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s14/b/program.py`, copied unchanged.

1. **Subject:** `k5-output-weight-x2.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output layer weight matrix x2 (bias unchanged); `runs/planted/plant.json`) — tensor digest `2985c3e3f813d9b43bc0dfc951852ac29bb071c03729a7ce67a3a03235ab5fbe`, file sha256 `0e7be185c0f5730a826662ff3087b16b829eba160c7d6bb82170133d0957060b`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

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


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s14/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
