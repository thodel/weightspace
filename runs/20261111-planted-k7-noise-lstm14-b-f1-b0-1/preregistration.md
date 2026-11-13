# Pre-registration — 20261111-planted-k7-noise-lstm14-b-f1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-k7-noise-lstm14-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s02/b/program.py`, copied unchanged.

1. **Subject:** `k7-noise-lstm14.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: Gaussian noise on the weights of BiLSTM L_14, norm 25 % of the block's; `runs/planted/plant.json`) — tensor digest `8caee20d9b623c22c56b655dafd5d95e8aa89e7703c7220073c79443973801bc`, file sha256 `4fe5cd0f681847910dbae3ee44dbe2ab2a2f538e6c93be363731499a65040bf8`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — planted-f1 / s02 / task (b), blind programmer (budget 0)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: reasoning effort medium-low (default of the harness, no override);
tools used: Read/Bash (cat, ls) on the three pack files, the brief and `wsp/{ops,session,run,kraken_subject,align}.py`;
`py_compile` of `program.py`. No model loaded, no pass, nothing run. Written 11.11.2026.

## Reading of the pack

- **Practice (C0):** CER 0.0726, WER 0.192, **10 collapsed lines of 300**, length ratio 0.971, 934 edits. The top-10 confusions
  are all deletions (` `→∅ 103, `e` 38, `n` 31, `s` 23, `i` 22, `r` 21, `.` 18, `t` 14, `o` 14) plus 22 space insertions.
  Ten fully collapsed lines of ~43 characters are ~430 deleted characters in roughly the character frequencies of German text
  (~73 spaces, ~50 `e`, ~30 `n`, ...). That is about what the deletion counts show. On the remaining 290 lines the residual is
  ~500 edits on ~12,400 characters (CER ≈ 4 %) with space deletions (~30) roughly balanced by space insertions (22): a
  healthy child on train-side lines. So the picture is "healthy model + 10 lines that produce (nearly) nothing", not a
  global shift of the blank or of the space against the letters.
- **Output layer:** blank row norm 4.64, bias +1.23 (parent +1.12, rel change 0.225 — in line with the mean row change 0.193).
  Weakest rows are the frequent letters (7.6-8.3), strongest the rare accented ones (22-24): the usual CTC regime, not a
  plant signature. Space bias −7.85 is the most negative bias of all 297 classes; suspicious at first sight, but the
  balanced space insertions/deletions on the non-collapsed lines argue against a planted shift, and the parent's value
  is not in the pack. The "moved most" rows (`«`, `¬`, `_`, `»`, `…`, …) are rare punctuation pushed down by −2.4 to −2.7 in
  bias with cos 0.86-0.95: ordinary suppression of characters absent from the child's data.
- **Diff to the parent:** every weight tensor moved rel 0.12-0.41 at cos 0.92-0.99 (all elements changed, except the two
  first convs where the smallest drifts leave ~25 % of elements below the "changed" tolerance). The three LSTM `bias_hh`
  pairs are at rel 0.03-0.05 / cos 0.999: no bias plant. `C_3.co.bias` cos 0.68 is a tiny vector (norm 0.10 vs 0.07) that
  turned under training; harmless either way. L_14 moved more than L_12/L_16 in both directions and both matrices — a
  block property, consistent with training, not with a one-sided plant. No whole-tensor cosine is 1.0, so no whole-tensor
  rescale.
- **Unexplained:** `L_16.layer.weight_ih_l0` has abs max 4.70 (every other recurrent tensor ≤ 2.43) and a 7 % higher norm
  than its reverse twin; its `weight_hh_l0` max 2.84 is also the highest. Compatible with a few rows scaled by ~2 as well as
  with a natural heavy tail; its rel L2 / cos to the parent (0.296 / 0.956) are *less* moved than the reverse direction,
  which does not support a plant. A zeroed or rescaled conv filter in C_6/C_9 (one of 64 is 12 % of the norm) would hide
  inside their rel L2 of 0.26-0.29 as well.

**Conclusion from the pack alone (H0):** nothing is localised well enough for an unconditional blind edit. The symptoms
could be ten unreadable sliver lines in `blb-train-300` (the forward pass returns empty text for images the transform
cannot scale) with no defect at all. Any unconditional edit (lowering the blank bias, raising the space bias, clipping
L_16, interpolating a block toward the parent) has an expected effect of zero or worse. **Unconditionally: no edit.**

## What the program does instead

The program loads the parent as a donor (allowed; no forward pass) and applies a repair **only** where a weight-only
comparison finds a signature that continued training does not produce. Every check skips its tensor when it flags more
than 6 % of the slices (then my model of "natural" is wrong and nothing is touched). If nothing fires, the model is
frozen unchanged — which, given the pack, is the outcome I expect.

| # | hypothesis (signature) | edit | predicted effect if it fires |
|---|---|---|---|
| H0 | the pack shows no localisable defect | none | 0 |
| H1 | an aligned output row (by character) has cos < 0.3 to its parent row while the others keep > 0.86: swapped with another character (mutual best match > 0.8), negated (cos < −0.5), or replaced | `set_rows` on `O_18.lin.weight` (+ `bias` for a swap whose biases went with it): swap back / negate / parent row | CER −0.1 to −2 pt; small because a swap of frequent letters would show as a top confusion, which the practice set does not show |
| H2 | an output row scaled far outside its peers (ratio child/parent > 2.2× or < 0.35× the median, ≥ 6 robust σ), or an untrained row (cos > 0.999) off the common norm ratio by > 1.3× | `set_rows`: rescale to the median ratio (direction kept) | CER −0.1 to −2 pt (a shrunk frequent row deletes, a boosted rare row inserts its glyph) |
| H3 | the bias of a character with ≥ 50 practice occurrences is shifted > 4 (and > 6 robust σ) from the parent beyond the shift shared by all rows | `set_rows` on `O_18.lin.bias`: parent + shared shift | CER −0.5 to −3 pt (a space-bias plant of that size would delete a good share of the spaces) |
| H4 | a row / column / conv filter / in-channel of a conv or LSTM weight is dead (ratio < 0.02), negated (cos < −0.5), unrelated to the parent slice (cos < 0.3), or scaled > 2.2× / < 0.35× the tensor's median ratio | `set_rows` / `set_cols`: negate back, parent slice (dead or unrelated), or rescale to the median ratio | CER −0.1 to −3 pt depending on what the slice carries |
| H5 | one gate block (200 rows) of an LSTM `weight_ih`/`weight_hh` is at > 1.5× or < 0.67× the child/parent norm ratio of the other three gates with its direction intact (cos > 0.8) | `set_rows` on the block: rescale to the other gates' ratio | CER −1 to −4 pt (a gate out of scale saturates or silences the layer) |
| H6 | a whole tensor is a pure rescale of the parent (cos > 0.999, norm off by > 1.2×) or negated (cos < −0.5) | `scale` | CER −1 to −5 pt |

If none fires: no edit, the model leaves as it came. If a check fires wrongly, the model gets worse by the size of the
slice it touched; the thresholds are set so that this needs a trained slice to sit where I have never seen one sit
(cos < 0.3, a 2.2× norm jump, a 4-logit bias move of a frequent character). Nothing is measured, nothing can be
concluded about any of these at budget 0; the conclusions belong to the scored held-out comparison.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s02/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
