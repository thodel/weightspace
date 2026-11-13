# s05 / task (a), budget 10 — pre-registered hypotheses (written before the first pass)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: reasoning effort as set by the harness (not shown to me; default),
tools used: Read + Bash (`cat`, `ls`, `grep` on the pack, this brief and the five allowed `wsp/` files), `py_compile` on
`program.py`. No model loaded, no pass run, no other folder read.

## Reading of the pack

- C0: CER 0.1993, length ratio 0.828, 24 collapsed lines, every top-10 confusion is a deletion (` `, `e`, `n`, `i`, `r`, …).
  Output too short and blank-dominated: the character logits are too weak relative to the blank, not swapped or mis-mapped.
- `L_16` (third BiLSTM) is the outlier block: rel L2 to the parent 0.51–0.55 (L_12/L_14: 0.21–0.30 for the same tensors),
  while the cosines (0.947–0.981) equal those of L_12/L_14 — i.e. the extra distance is a **scale**, not a direction change.
  `L_16.bias_hh` is the smoking gun: cos 0.999 at rel L2 0.505 (a pure ×0.5 would give exactly cos 1, rel 0.5).
  The rms values agree: L_16 weight_ih 0.121 vs L_14 0.217, weight_hh 0.154 vs 0.310, bias_hh 0.42 vs 0.93 — all ≈ ×0.5.
  Solving rel² = (α−1)² + α²δ² with δ from the cosine gives α ≈ 0.47–0.50 for every L_16 tensor. Halving the last LSTM
  halves the O_18 input, so the logits shrink toward the biases; the blank has the largest bias (+1.23) → deletions.
- `C_3.co.bias`: rel L2 1.087, cos 0.68 — large *relative* change, but the tensor is tiny (rms 0.018) and its absolute
  drift (≈0.013 rms) matches C_6/C_9 (0.008–0.012). Probably natural; probed last, cheaply.
- `O_18`: blank row and bias ordinary (bias Δ +0.11 vs parent, rel 0.23 ≈ mean 0.19). The most-moved rows are rare
  punctuation with bias Δ ≈ −2.5 (natural). ` ` has the most negative bias of all (−7.85) — suspicious only if the parent's
  is much higher; checked in-program against the parent (aligned by character) with a robust outlier test.

## Hypotheses (one line each: edit class, predicted direction + magnitude, conclusion on failure)

1. **H1 — L_16 scaled ×0.5.** Edit: `scale` factor 2.0 over all 8 `L_16.layer.*` tensors (bias_ih is zero, no-op).
   Prediction: CER 0.199 → 0.04–0.10 (absolute drop ≥ 0.09), collapsed 24 → ≤ 3, length ratio 0.83 → 0.97–1.02,
   deletions leave the top of the confusion table. If refused: L_16's half-size statistics are natural for this child and
   the deletions have another cause; fall back to H1b.
2. **H1b (only if H1 refused) — milder scale.** `scale` factor 1.5 on the same tensors. Prediction: CER drop 0.02–0.08.
   If refused too: the L_16 block is not scaled; note and move on.
3. **H2 (data-conditional) — swapped output rows.** If a frequent character's `O_18` row (rel L2 to the parent > 0.7)
   matches a *different* parent character's row with cos > 0.85 reciprocally, `set_rows` swaps the two rows (weight+bias).
   Prediction: CER drop ≈ count of that confusion / 12,867 (0.005–0.03). If refused / nothing detected: no row swap.
4. **H3 (data-conditional) — planted bias on a frequent output row.** For aligned characters with ≥ 0.3 % frequency in the
   practice references, bias Δ to the parent with robust z > 4 and |Δ| > 2.5 is flagged; `set_rows` on `O_18.lin.bias`
   puts those biases at parent + median drift. Prediction: CER drop 0.005–0.03 via fewer deletions of those characters.
   If refused / nothing flagged: the biases (incl. ` ` −7.85) are natural.
5. **H4 (conditional) — blank over-biased.** Only if deletions still make ≥ 60 % of the top-10 confusion mass and length
   ratio < 0.97 after H1: `add` −0.6 to `O_18.lin.bias[0]`. Prediction: CER drop 0.002–0.01; at most one repeat if kept.
   If refused: blank calibration is fine once L_16 is repaired.
6. **H5 — C_3.co.bias.** `transplant` of the parent's `C_3.co.bias` (same shape). Prediction: |ΔCER| < 0.003 → the rule
   should refuse it; that refusal confirms the large relative change is natural drift of a tiny tensor, not a defect.
7. **H6 (only if H1 kept) — factor refinement.** `scale` ×1.1 (total 2.2) on L_16. Prediction: |ΔCER| ≤ 0.005, refused;
   refusal means the plant was ×0.5 and factor 2.0 is right. Kept would mean the true factor was ≈ 0.45.

Passes: 1 (C0) + H1 [+ H1b] + H2 [0–1] + H3 [0–1] + H4 [0–2] + H5 + H6 ≤ 10; every edit is guarded by `budget_left ≥ 1`
and decided by `s.keep`. Unspent passes are left unspent. Program ends with `s.freeze()` in all paths.

Expected final state: L_16 ×2 kept, everything else refused; CER ≈ 0.04–0.10.
