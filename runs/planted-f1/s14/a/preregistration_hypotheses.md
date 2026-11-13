# s14, task (a), budget 10 — pre-registered hypotheses

- Programmer: claude-fable-5-1 (Claude Fable 5.1). Settings: reasoning effort as set by the harness (not chosen by me; default);
  tools used: Read/cat/ls on the three pack files and the five allowed `wsp/*.py` files and the brief, Write for the two files,
  `python -m py_compile` once. No model loaded, no program run, nothing else read.
- Readings used: `describe.md`, `diff_parent_to_subject.md`, `c0_practice.md` (CER 0.0650, 8 collapsed lines, length ratio 0.979,
  errors dominated by deletions: ` `→∅ 78, e→∅ 34, n→∅ 28, i→∅ 21, r→∅ 19, s→∅ 16; only 25 insertions of space).
- What stands out: (1) `L_16.layer.weight_ih_l0` abs max 4.702, twice the family's max/rms ratio (every other LSTM weight ≤ 2.8);
  (2) `C_3.co.bias` is the only tensor with cosine < 0.9 to the parent (0.679, rel L2 1.087) but its norm is 0.10 (parent ≈ 0.07);
  (3) the space row has the most negative bias of all 297 rows (−7.85) among otherwise rare characters (ú, Ⅲ, Ž, Ⅱ);
  (4) 8 collapsed lines and deletion-dominated errors point at blank dominance or a damaged feature path.
  The doubled norms of rare rows (á, Ä, š, é: cos 0.998, norm 23→47) and the ~1.0 mean relative change of output rows read as natural
  continued training, not as a defect; I do not edit them.
- Plan: pass 1 = C0 baseline. Then diagnostics against the parent inside the program (no passes), and the stages below in this order,
  each only if its trigger fires, each one edit → one measure → `s.keep` (harness rule) with an own veto if collapsed lines rise by > 3.
  At most 10 measures. Everything ends with `s.freeze()`.

## Hypotheses (edit class · trigger · prediction on the practice CER · conclusion if it fails)

- **D1 output rows swapped or destroyed** (`set_rows` on `O_18.lin.weight`+`bias`, rows swapped back / rebuilt from the parent row ×
  median norm ratio). Trigger: among characters with ≥ 30 occurrences in the practice references, a child row whose best-matching
  parent row is another character's (mutual, cosine margin > 0.25), or cos to its own parent row < 0.2, or norm ratio outside
  [0.3, 3]× the median. Prediction: CER −1 to −4 points if it fires (a swap/zeroed frequent row costs ≥ 1 point). If refused: the
  output rows are intact up to natural drift.
- **D2/D3 LSTM or conv rows/columns rescaled or zeroed** (`set_rows` / `set_cols`, rows rescaled to the median child/parent ratio,
  zero rows rebuilt from the parent). Trigger: per-row (axis 0) or per-column (axis 1) norm ratio child/parent outside
  [0.3, 2.5]× the tensor's median, on < 25 % of the rows. Prediction: −0.5 to −3 points. If refused: no structural damage to hidden
  units or input features; the flagged rows were natural drift.
- **D4 point spikes** (`add` of −(child−parent) on the flagged entries, all tensors in one edit). Trigger: |child−parent| > 14 × rms of
  the difference and > 0.5, or |child| > 1.6 × the parent's abs max; < 2 % of the tensor. Expected mainly in `L_16.layer.weight_ih_l0`
  (abs max 4.702). Prediction: −0.3 to −2 points, collapsed lines down. If refused or nothing flagged: the 4.7 entry is natural
  heavy tail, not a planted spike.
- **D5 whole-tensor scaling** (`scale` by 1/k). Trigger: least-squares scale k = ⟨child,parent⟩/⟨parent,parent⟩ with |k−1| > 0.2 on a
  weight tensor with parent norm ≥ 1. Prediction: −0.5 to −3 points. If refused: the tensor norms are natural.
- **D6 output bias shifted for a frequent character** (`set_rows` on `O_18.lin.bias` to parent bias + median bias drift). Trigger:
  bias Δ to the parent deviating by > 2.5 from the median Δ of frequent characters; prime suspect the space row (−7.85).
  Prediction: −0.3 to −0.8 points (space deletions 78 → ~30). If refused: the −7.85 is the model's own calibration of space.
- **D7 blank row rescaled** (`set_rows` of row 0 to the parent's norm, direction kept). Trigger: blank norm ratio child/parent
  outside [0.5, 2]. Prediction: −0.5 to −2 points, collapsed lines down. If refused: the blank row is natural.
- **P1 space bias probe** (`add` +1.5 to the space bias; only if D6 did not fix the space and space deletions ≥ 40).
  Prediction: −0.1 to −0.4 points, space deletions down, space insertions up. If refused: the space errors are feature-level, not
  calibration; no further space edit.
- **P2 blank bias probe** (`add` −0.5 to the blank bias; only if deletions ≥ 2 × insertions or collapsed ≥ 5 at baseline).
  Prediction: −0.1 to −0.4 points, deletions down, insertions up, collapsed lines down. If refused: the deletion pattern is not a blank
  calibration problem.
- **P3 `C_3.co.bias` transplant from the parent** (`transplant`). Trigger: always, if a pass is left. Prediction: |Δ| < 0.1 points,
  most likely refused by the rule (the tensor is tiny: norm 0.10). If refused: the cos 0.68 is natural drift of a near-zero tensor.
- **P4/P5 second steps** (another −0.5 on the blank bias / +1.0 on the space bias) only if the first step was kept and passes remain.
  Prediction: smaller gain than the first step or refusal; a refusal fixes the step size at the first value.

Overall: if none of D1–D7 fires and P1–P3 are refused, I conclude the model carries no planted defect visible in these readings
and the 6.5 % CER is its trained state; the frozen model then equals the start model.
