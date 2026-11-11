# Pre-registered hypotheses — planted-f1 / s09 / task (a), budget 10

Programmer: claude-fable-5-1 (Claude Fable 5.1), run as a Claude Code subagent. Reasoning effort: the harness default
(not raised by me). Tools used: Read on the three pack files and the five allowed `wsp/` files, `ls`/`wc` on the pack
folder, Write for the two files, `py_compile` on `program.py`. No model loaded, no pass run, nothing else read.

## What the readings show

- `C_3.co.weight` and `C_3.co.bias` are **exactly zero** (L2 0.00, rms 0.0000, abs max 0.000; rel L2 to parent 1.0000,
  cosine nan, 39936/39936 and 32/32 entries changed). A zeroed conv followed by ReLU emits a constant 0 feature map, so
  C_6, C_9 and the LSTMs see a constant input and the output is the same frame everywhere -> CTC collapses to the empty
  string. This matches C0 exactly: CER 1.0000, WER 1.0000, 300/300 lines collapsed, length ratio 0.000, every top
  confusion is `x -> nothing`.
- Everything else looks like an ordinary fine-tuned child: C_0 rel 0.12–0.15, C_6 0.26, C_9 0.29–0.33, every LSTM
  weight 0.20–0.33 with cosine 0.95–0.98, `bias_ih` zero by construction, output rows mean rel change 0.19, blank row
  rel 0.23 and bias +1.23 (parent -> child Δ +0.11). The rows that moved most are rare glyphs (`«`, `¬`, `_`, `»`, `…`),
  as expected from a hand-specific fine-tune. The only mild oddity is `L_16.layer.weight_ih_l0` with abs max 4.702
  (the other LSTM input weights peak at 1.3–2.4) while its rel L2 (0.296) is in line with the other LSTM blocks; a few
  spiked entries would not move the block statistic, so this gets a data-driven check against the parent, not a
  blind edit.

## Hypotheses (edit class; predicted direction and magnitude of CER on the 300 practice lines; what failure means)

- **H1 — C_3 conv block was zeroed; the parent's C_3 is a working stand-in.** Edit class: *transplant*
  (`C_3.co.weight`, `C_3.co.bias` from `kraken-german-xix-v1`, same shapes). Prediction: CER falls from 1.000 to
  0.03–0.25 (a drop of >= 0.75), collapsed lines 300 -> fewer than 10. The parent's C_3 is not the child's lost C_3
  (the other conv blocks drifted 25–30 % rel L2 during fine-tuning), so the repaired model will be worse than a healthy
  child but far from CER 1. If it fails (CER stays >= 0.5 or the rule refuses): C_3 is not the only defect, or the
  parent's C_3 is incompatible with the child's neighbours -> fall back to H1b.
- **H1b (only if H1 fails) — the whole conv stack from the parent**, then if still failing **H1c** (all non-output
  tensors from the parent, then the output rows aligned by character). Edit class: transplant / set_rows. Prediction:
  CER < 0.5. If H1c also fails the defect is somewhere I cannot see from the pack; I stop with whatever was kept.
- **H2 — a few spiked entries in an LSTM/conv weight (data-driven check).** For every shared-shape weight tensor of
  C_6, C_9, L_12, L_14, L_16 I compute d = child - parent; entries with |d| > max(8 sigma_d, 1.0) are flagged. Edit class:
  *add* (delta that resets flagged entries to the parent's value), run only if 1–5000 entries are flagged. Prediction:
  if there is a planted spike, CER falls by 0.005–0.05; if the flagged set is empty the pass is not spent. If flagged
  but refused: the outliers are natural fine-tuning extremes and the spike reading was a red herring.
- **H4 — C_0 from the parent too (coherent C_0 -> C_3 pair).** Edit class: *transplant*. Prediction: |ΔCER| <= 0.01,
  most likely refused by the rule; kept only if the parent's front end is more coherent with the parent's C_3 than the
  child's lightly-drifted C_0 is. If refused: the child's C_0 is fine and stays.
- **H5 — C_6 half-way toward the parent (alpha 0.5).** Edit class: *interpolate*. Prediction: |ΔCER| <= 0.01, most
  likely refused; kept only if the child's C_6 (adapted to the lost C_3) matches the parent's C_3 worse than a blend.
  If refused: the child's C_6 stays.
- **H3 — blank balance after the repair.** From the confusions of the last kept state: if deletions (`x -> nothing`)
  exceed insertions by 1.5x, lower the blank bias by 0.5; if insertions dominate, raise it by 0.5; otherwise skip. Edit
  class: *add* on `O_18.lin.bias[0]`, at most two steps of 0.5 (second only if the first is kept). Prediction: CER
  -0.002 to -0.01 per kept step. If refused: the blank prior of the repaired model is already balanced; nothing more
  to do without gradient.

Pass plan (<= 10): 1 C0, 2 H1, [3–5 fallback H1b/H1c only on failure], 3 H2 (conditional), 4 H4, 5 H5, 6–7 H3.
Every edit is decided by `s.keep` (the rule) plus my own checks on `m.cer` / `m.collapsed` / `m.confusions`.
