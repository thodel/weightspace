# Pre-registration — 20261111-planted-k4-blank-bias-plus2-a-f1-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a), **Fable 5.1 rerun** of `runs/20261107-planted-k4-blank-bias-plus2-a-b10-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s03/a/program.py`, copied unchanged.

1. **Subject:** `k4-blank-bias-plus2.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: blank bias +2.0; `runs/planted/plant.json`) — tensor digest `339161afc6ffa62ee51b2d9e5ce3dd0bbd0c3f3cd7939c6300bfa46500dc993b`, file sha256 `2727c5266842c02e67e61b779bf0d900f981af6137e12447ed7f279c833c268c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

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


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s03/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Wrote its program in advance, did not run it. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
