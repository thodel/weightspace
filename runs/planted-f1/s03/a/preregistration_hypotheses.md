# Pre-registered hypotheses — planted-f1 / s03 / task (a), budget 10

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort, tools Read / Bash (`ls`, `py_compile`) / Write only;
nothing run, no model loaded. Read: the three pack files, the brief, `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`,
`wsp/kraken_subject.py`, `wsp/align.py`. Written before the first pass; `program.py` is fixed from here on.

## What the pack says

- C0: CER 0.0789, WER 0.2135, 14 collapsed lines, length ratio 0.941. Every top-10 confusion is a deletion (` `→∅ 177, `e` 81,
  `n` 53, `i` 38, `r` 37 …), no substitution among them. Deletions per unit of letter frequency are nearly uniform (~5 per % for
  e/n/i/r/s/t/u) — a global over-prediction of the blank (or collapsing lines), not a per-character defect — except the space,
  deleted ~2.5× more often than `e` relative to its frequency.
- Output layer: blank bias +3.23, Δ vs parent **+2.11** (the blank row's weights moved 0.225, barely above the mean 0.193 — a bias
  edit, not a row edit). Space bias **−7.85**, the most negative of 297 rows, sitting among `ú`, `Ⅲ`, `Ž` while frequent letters have
  biases near 0 (`s` +0.40, `d` +0.29).
- `C_3.co.bias`: rel L2 1.09, cosine 0.68 against the parent; its neighbours' biases moved 0.15–0.33 and `C_3`'s own weights only 0.17
  (cos 0.986). Numerically this is what "add noise of the tensor's own norm" looks like (norm 0.068 → 0.10, cos ≈ 0.7). The tensor is
  tiny (rms 0.018), so the CER effect is uncertain.
- `L_16.layer.weight_ih_l0`: abs max 4.70 (every other LSTM weight 1.3–2.8) with an ordinary rel L2 (0.296) — a hint of a localized
  edit that a tensor-level diff cannot resolve. Nothing else in the hidden tensors stands out; all LSTM biases moved < 0.05.
- The "moved most" rows (`«`, `¬`, `_`, `»`, `…`, …) are rare characters with bias Δ ≈ −2.5: continued training on data without them.
  No CER effect on this practice set; not touched.

`drift` below = median over the 40 most frequent practice characters of (child bias − parent bias): the natural bias drift of
well-trained rows, so a shift is judged relative to it (softmax is shift-invariant).

## Hypotheses (one per line: edit class — predicted CER change on the 300 practice lines — conclusion if refused)

- **H1 blank bias planted +2.1.** `set_rows` `O_18.lin.bias[0] = parent_blank + drift`, only if the blank's shift relative to `drift` is > 0.5. Predicted: CER **down 0.020–0.035** (0.079 → 0.045–0.06), collapsed lines 14 → ≤ 5, length ratio → 0.98–1.00, deletions of e/n/i/r halve. If refused: the +2.1 is the child's own calibration (the LSTM output scale compensates it); the deletions come from elsewhere; blank not planted.
- **H2 space row bias planted (−7.85).** `set_rows` on `O_18.lin.bias` for every frequent character whose bias sits > 2.5 below `parent + drift` (I expect exactly the space) → `parent + drift`. Predicted: CER **down 0.005–0.012** (space deletions 177 → < 60), WER down ≥ 0.05. If refused (space insertions appear): −7.85 is the model's own space calibration on this hand (parent similar), not a plant.
- **H3 output weight rows of frequent characters replaced/scrambled.** `set_rows` from the parent (weights + bias) for frequent rows whose rel L2 vs parent > max(0.6, 3× median). The pack predicts **none** (no substitutions in the top-10); if any are flagged and kept: CER down 0.002–0.010. If refused: a parent row read out in the child's drifted feature space is worse than the child's row; rows not planted.
- **H4 localized edit in a hidden tensor** (rows/units or columns/features of the convs and LSTMs; the `L_16.weight_ih_l0` outlier). Detector: per-slice rel L2 and norm ratio vs the parent (flag rel > max(0.75, median + 8·MAD) or ratio outside [0.4, 2.5]; 1-D biases by |Δ| > max(0.3, median + 10·MAD)); repair by `set_rows` / `set_cols` from the parent, one edit per axis, at most 40 slices; a tensor with > 30 % of its slices flagged is transplanted whole instead. Predicted if a plant is found: CER **down 0.003–0.03**; collapsed lines down. If nothing is flagged: the abs-max outlier is natural, no hidden plant detectable by a parent diff. If flagged and refused: the slices are natural drift (false positives of the detector).
- **H5 `C_3.co.bias` carries added noise.** `transplant` the whole tensor from the parent. Predicted: CER **down 0.000–0.005**, likely within the interval → refused; collapsed lines unchanged. If refused I conclude: natural drift of a tiny tensor, cosine 0.68 is not evidence at this norm. Kept only if the harness's interval is below 0.
- **H6 residual blank calibration after H1.** `add` −0.75 to `O_18.lin.bias[0]` while the last kept state is still deletion-heavy (length ratio < 0.985 or deletions > 1.5× insertions in the confusions); +0.5 if it became insertion-heavy; up to two steps. Predicted: CER **down 0.002–0.008** per step, then refused. If refused on the first step: H1's reset already sits at the optimum.

Own check on every edit, beyond the harness rule: an edit that raises the collapsed-line count by more than 3 is reverted even if its
interval passed. Order of passes: C0, H1, H2, H3, H4 (rows), H4 (cols), H4 (whole tensors), H6 step 1, H5, H6 step 2; steps whose
condition is not met are skipped and noted, unspent passes stay unspent. Overall prediction: final CER 0.045–0.060 if H1 and H2 hold;
if neither holds, the subject is healthy for this hand and ends unchanged.
