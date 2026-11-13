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
