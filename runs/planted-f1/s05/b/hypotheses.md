# Hypotheses — planted-f1 / s05 / task (b), budget 0 (blind programmer)

- Model: claude-fable-5-1 (Claude Fable 5.1). Reasoning effort: default (medium-low). Tools: Read/Bash (cat, grep, ls)
  on the three pack files, the brief and `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`,
  `wsp/align.py`; `py_compile` of `program.py`. No model loaded, no pass run, sibling `a/` not read.

## What the readings say

- C0: CER 0.199, WER 0.491, 24 of 300 lines collapsed, length ratio 0.828. Every one of the top-10 confusions is a
  deletion (` `, `e`, `n`, `i`, `r`, `c`, `s`, `u`, `t`, `m` → ∅), i.e. the model under-emits across the board: the blank
  wins too often. Nothing character-specific.
- Diff against the parent: blocks C_0/C_6/C_9/L_12/L_14 moved like an ordinary fine-tune (rel L2 0.13–0.31, cosine
  0.95–0.99). **L_16 (third BiLSTM) stands out**: every non-zero tensor of it has rel L2 0.505–0.550, and the two
  `bias_hh` vectors have rel L2 0.5054 / 0.5059 with cosine 0.99897 / 0.99907. A vector that moved by half its length
  while keeping cosine 0.999 is the parent vector *times ≈0.5*, not a fine-tuned one (L_12/L_14 `bias_hh` moved by
  0.03–0.04 with the same cosine). Solving `child = k·parent + drift` from rel and cos gives k ≈ 0.495 for both biases.
  The weights fit the same story: a 0.5 scaling of a parent-plus-ordinary-drift tensor predicts rel ≈ 0.52 and
  cos ≈ 0.96, observed 0.51–0.55 / 0.947–0.981.
- Absolute norms agree: in the subject, L_16 is about half of L_14 (`weight_ih` rms 0.121 vs 0.217, `weight_hh` L2 61.8 vs
  123.9, `bias_hh` L2 12.0 vs 26.3) although the two layers have identical shapes and the first two LSTMs match each other.
- Mechanism → symptom: halving the gate pre-activations of the last LSTM squashes its output (gates toward 0.5,
  `tanh` candidates shrunk), so the logits `W·h + b` lose contrast and the output biases dominate; the blank bias is +1.23
  while letters sit around −1.9 → blank wins → deletions, short lines, whole lines collapsing. Exactly what C0 shows.

## H1 (applied, guarded): L_16 was scaled by ≈0.5 — undo it

- Edit class: `scale` of the six non-zero L_16 tensors (`weight_ih_l0`, `weight_hh_l0`, `bias_hh_l0`, forward and reverse)
  by one common factor 1/median(k_i), where k_i = ⟨child_i, parent_i⟩ / ⟨parent_i, parent_i⟩ is the least-squares scale of
  each tensor against the parent (expected ≈ 0.5, so the factor ≈ 2.0). `bias_ih` is all-zero and untouched.
- Guard (the program checks before editing, using only the parent as a donor; no pass): every k_i in [0.35, 0.65] and every
  cosine ≥ 0.9, while the same statistics for L_14 give k in [0.8, 1.25]. If the guard fails, nothing is applied and the
  session is frozen unchanged — a healthy child whose last LSTM is legitimately small must not be doubled.
- Predicted effect: CER falls from 0.199 to roughly 0.05–0.10 (at least −0.08 absolute); the deletion confusions shrink by
  most of their count, collapsed lines → ≈0, length ratio → ≈1.0. If it fails (CER unchanged or higher) the L_16 norm ratio
  is a property of this child's training rather than a defect, and the deletions have another cause (output layer /
  blank calibration).

## Considered and not applied

- Space bias −7.85 (most negative of all 297 rows) with 301 space deletions. But spaces are deleted at about the same rate
  as `e` (251), so this is the general under-emission, not a space-specific bias defect; space is not among the rows that
  moved most against the parent. Changing it blind would mis-calibrate a healthy row. Not touched.
- Blank row: bias Δ +0.11 vs parent, rel L2 0.23 (block mean 0.19) — ordinary. Not touched.
- `C_3.co.bias`: rel L2 1.09, cosine 0.68, but its norm is 0.10 (rms 0.018, parent's ≈ 0.07); a tiny bias of a ReLU conv
  whose filters have norm ≈ 7. Even if noise was added here its effect is negligible, and a transplant could undo a real
  fine-tune. Not touched.
- Rows that moved most (`«`, `¬`, `_`, `…`, `[`, …) are rare characters with bias Δ ≈ −2.5; consistent with fine-tuning
  on a hand where they do not occur. Not touched.
