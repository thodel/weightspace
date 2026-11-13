# Preregistration — planted-f1 / s08 / task (a), budget 10

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort as set by the harness; tools used: Read and
Bash (`cat`, `ls`, `grep`, `sed -n`, `python -m py_compile`) on the pack, this brief and the five allowed `wsp/` files only.
No model was loaded and the program was not run before this file and `program.py` were written. Donor: `kraken-bern-hertwig-v2`
(unrelated lineage, Bern 1528), used only as a control.

## Reading of the pack

- `describe.md`: `O_18.lin.weight` row `e` has norm **0.00** (every other row 7.7–23.9; the common lower-case letters r, s, t, i, n, a sit at 7.8–8.3).
  No other row is anomalous. The `e` bias is not among the extremes (between −7.0 and +0.28).
- `c0_practice.md`: CER 0.1623, 2088 edits; `e`→∅ 910, `e`→a 131, `e`→i 90, `e`→o 52, `e`→r 36, `e`→c 35, `e`→u 32 — about 1290 of the 2088 edits
  are `e` errors. A zeroed row gives `e` a constant logit (its bias), so the most frequent German letter can never win a frame: deletion when the blank
  wins, a neighbouring vowel/consonant when a letter does.
- `diff_donor_to_subject.md`: the donor's tensors have cosine ≈ 0 with the subject's everywhere (different lineage); output rows aligned by
  character also have cos ≈ 0. The donor's `e` row therefore lives in a different 400-dim feature space and is not expected to transfer.
- Second, weak anomaly: `L_16.layer.weight_ih_l0` abs max 4.70 (all other LSTM weights 1.3–2.8), rms 0.2419 vs 0.2264 for its `_reverse` twin
  (the other two layers' forward/reverse pairs agree within 1 %). The excess sum of squares (~2300) is far more than one spiked element; it is consistent
  with a scaled block of columns or rows, or natural variation. This cannot be decided from the pack; the program inspects the tensor at run time and
  edits only if a block stands out against the twin.
- 9 collapsed lines and ` `→∅ 103 are read as consequences of the `e` defect / normal behaviour; no separate edit is planned for them.

## Hypotheses (one per line; edit class; predicted CER change on the 300 practice lines; conclusion if it fails)

- **H0 (baseline, pass 1)**: `s.measure("C0")` reproduces CER ≈ 0.162 with `e`→∅ as the top confusion. Program check: `‖W[e]‖ < 1e-3`. If the row is
  *not* zero the `e` branch is skipped entirely (the pack would then be stale) and only the L_16 scan is run.
- **H1 (control, pass 2) — donor `e` row transplant** (`set_rows` on `O_18.lin.weight` + `O_18.lin.bias`, donor row/bias aligned by character via
  `codec_map()`). Prediction: **no improvement or worse, CER +0.00 to +0.06** (random-direction row of norm ~16 → spurious `e` insertions). Expect the
  keep rule to refuse. If it is kept (CER drops significantly), I conclude the output feature spaces of the two lineages are more aligned than the
  cos ≈ 0 of the output rows suggests; the result is still accepted since the rule decides.
- **H2 (main repair, pass 3) — amplified-consensus row**: the new `e` row is `k · Σ w_c W[c]` and bias `k · Σ w_c b[c]` over the letters `c` the
  baseline confuses `e` into (from `m0.confusions`; fallback a, i, o, r, c, u), weights `w_c ∝ sqrt(count)` normalised to sum 1, and `k` chosen so
  that `‖W[e]‖` equals the median norm of the component rows (~8). Mechanism: `logit_e = k · (weighted mean of the component letters' logits)`; on a
  frame where one letter is clearly dominant the mean is low (e loses), on a frame where several of these letters are moderately active (the pattern of
  a true `e`, which is exactly what the confusion profile shows) the amplified mean beats them and the blank; between-letter blank frames have a negative
  mean and are amplified downwards. Prediction: **large drop, CER 0.162 → 0.08–0.11 (−0.05 to −0.08)**, `e`→∅ falls by at least half, few `e`
  insertions. If refused: the `e` feature direction is not a positive mixture of its confusion partners; fall to H2b.
- **H2b (fallback, only if H2 is refused)**: same construction over the uniform mean of all lower-case a–z rows present in the codec. Prediction:
  smaller drop, −0.02 to −0.05. If also refused I conclude the row cannot be rebuilt from mixtures of the remaining rows without activations, and
  leave `e` zeroed (note it; no further `e` edits).
- **H3 (tuning of the kept H2 row, up to 3 passes, each `set_rows`)**: decided from the latest top-10 confusions. If `e` insertions (`∅`→`e` plus
  `x`→`e`) exceed `e` deletions and are > 30: scale the row and bias by 0.8 (lower `k`); prediction −0.003 to −0.015. Else if `e`→∅ is still > 60 and
  more than twice the insertions: bias +1.0; prediction −0.005 to −0.02. Else stop tuning. Each step is kept only by the rule; a refused step ends the
  tuning (conclusion: the row is at its mixture optimum; residual `e` errors need the true feature direction).
- **H4 (L_16 block scan, ≤ 2 passes)**: for each BiLSTM `weight_ih_l0` / `_reverse` pair (the two directions read the same 400 input features, so
  column norms are comparable column by column) flag columns whose norm ratio to the twin exceeds `max(1.5 · median, median + 5 robust σ)` (at most
  100 columns), and rows whose norm exceeds both `median + 5 robust σ` of the tensor and 1.3 × the twin's largest row norm. If flagged: `set_cols`
  (or `set_rows`) rescaling the flagged columns to `twin_norm · median ratio` (rows to the median row norm). Prediction: **small drop, −0.002 to −0.01**
  if the block is a planted scaling; the rule will refuse if it is natural variation. If nothing is flagged no edit is made and I conclude the L_16
  statistics are natural (expected for the elevated abs max alone).
- **H5 (spike clip, ≤ 1 pass, only if H4 found no block in `L_16.layer.weight_ih_l0`)**: if 1–200 elements of `L_16.layer.weight_ih_l0` exceed 3.5 in
  magnitude (twin abs max 2.16), clip them to ±(twin abs max). Prediction: **≈ 0 (−0.003 to +0.001)**; the rule will almost certainly refuse. Failure
  means the spike is harmless or natural.

## Pass plan (≤ 10 measures)

1 baseline · 2 donor control · 3 H2 (or 4 H2b) · up to 3 tuning passes · up to 2 L_16 block passes · up to 1 spike pass. Every edit is preceded by
`s.hypothesis`, measured once, and decided by `s.keep` (refused edits are undone by the harness). The program never calls `measure` with
`budget_left == 0`, and ends with `s.freeze()`.

## What I would conclude overall

If H2 (or H2b) is kept and lands in the predicted range, the planted defect is the zeroed `e` output row and it is repairable to a large extent from
the model's own rows. If every `e` edit is refused, the defect is still the zeroed row but the repair needs activation statistics, which this task does
not allow. H1 kept would be surprising and would be noted as such. H4/H5 refused means the L_16 anomaly is not a defect worth editing (or is
undetectable at this budget).
