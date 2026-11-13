# Pre-registered hypotheses — planted-f1 / s11 / task (a), budget 10

Model: claude-fable-5-1 (Fable 5.1). Settings: reasoning effort as configured by the harness (not visible to me);
tools used: Read (pack, brief, the five allowed `wsp/` files), Bash (`ls` of my own folder, `py_compile`), Write.
No model loaded, program not run. Written 11.11.2026 before the first pass; `program.py` is fixed from here on.

## Readings that drive the hypotheses

- **L_14 block is displaced far more than its neighbours.** `weight_ih` (both directions) rel L2 0.75 / cos 0.79 and
  `weight_hh` 0.55 / cos 0.87 against the parent, where L_12 and L_16 sit at 0.30 / 0.96 (ih) and 0.21 / 0.98 (hh).
  L_14's biases moved *normally* (0.033, less than L_12/L_16). The child's L_14 norms are ~21 % (ih) and ~13 % (hh)
  above the parent's, while L_12/L_16 keep the parent's norm (natural drift is near-orthogonal and norm-preserving).
  Removing the natural-drift share (0.30 / 0.21) leaves an excess of 0.69 (ih) and 0.51 (hh) that corresponds to the
  *same* per-entry std of ≈ 0.15 on both shapes (800×400 and 800×200). That is the signature of i.i.d. noise added
  to the four L_14 weight matrices, not of training.
- **Space has the most negative output bias of all 297 classes (−7.85)**, below rare, never-seen classes (`ú`, `Ⅲ`, `Ž`
  −7.0 to −7.3), while frequent letters sit at ≈ 0 (`s` +0.40). ` ` → ∅ is the top confusion (156, 3× the next);
  ∅ → ` ` only 21. Length ratio 0.955, deletions dominate. A bias-only change on one row is invisible in the
  `moved most` table (ranked by weight distance), so the pack cannot confirm it; the program compares with the parent.
- `L_16.layer.weight_ih_l0` abs max 4.70, twice every other LSTM weight (1.4–2.4); tensor-level stats normal
  → if anything, a localized spike.
- 11 collapsed lines, deletions of e/n/r/i/s/t: consistent with an under-confident recurrent stack (noise in L_14).

Considered and **not** acted on: `C_3.co.bias` rel 1.09 / cos 0.68 is a tiny vector (parent L2 ≈ 0.07); its per-entry
drift (≈ 0.013) is in line with the other conv biases (0.008–0.027) — natural. Blank bias Δ +0.11, blank row rel 0.23
— natural. Rare punctuation rows (`«`, `¬`, `_`, `…`) pushed down by 2.4–2.7 with norms up — natural suppression of
classes absent from the child's data. 25 % of `C_0`/`C_3` weight entries unchanged from the parent — unexplained; only
actionable if they are exact zeros (checked in H4).

## Hypotheses (edit class; predicted CER change on the 300 practice lines; conclusion if it fails)

- **H1 — L_14 noise.** Interpolate the four `L_14.layer.weight_{ih,hh}_l0[_reverse]` toward the parent with a Wiener
  α = (var(Δ_L14) − var(Δ_L16)) / var(Δ_L14) computed in-program, clipped to [0.6, 0.95] (expected ≈ 0.85); biases untouched.
  Edit class: block interpolation to parent. Prediction: CER 0.120 → 0.06–0.09 (Δ −0.03 to −0.06), collapsed 11 → ≤ 5,
  deletions of e/n/r/i fall. If refused: one fallback at α = 0.4 (removes 40 % of a noise term while keeping most of the
  child's adaptation). If that is refused too: the L_14 displacement is the child's own training (or the parent's L_14 is
  incompatible with the child's L_12/L_16) and L_14 stays as it is; the noise reading was wrong.
- **H2 — planted output-row change, space first.** Scan all shared output rows against the parent (aligned by
  character): bias Δ beyond ±3.5 or weight rel L2 > 0.9 flags a row; also space alone if its bias Δ < −2. Repair flagged
  rows (1–10) with the parent's bias (and weights where the weight row is flagged). Edit class: output row
  transplant from parent. Prediction: Δ −0.005 to −0.012 (156 space deletions = 1.2 points at most). If the parent's
  space bias is also ≈ −7.85: nothing planted there; the −7.85 is natural (see H6). If refused: space deletions come
  from upstream, not from the bias; row stays.
- **H3 — localized LSTM damage in L_12 / L_16.** Entries with |w| > 3.0 and |w − parent| > 1.5 (spikes), or rows
  (gate units) with rel L2 > 1.2 or norm ratio outside [0.5, 2]: reset to the parent's values (one `add` edit, only if
  1–400 spikes / 1–40 rows). Edit class: sparse reset to parent. Prediction: Δ −0.001 to −0.01. If none found, no pass.
  If refused: the 4.70 is a natural outlier.
- **H4 — conv pruning / channel damage.** Conv weights with ≥ 5 % exact zeros where the parent is non-zero, or 1–8
  output channels with rel L2 > 0.9 or norm ratio outside [0.5, 2]: reset those entries/channels (+bias) to the parent.
  Edit class: sparse/channel reset to parent. Prediction: Δ −0.005 to −0.03 if triggered. If none found, no pass.
  If refused: the pattern is benign.
- **H5 — CTC calibration (not a defect repair).** If deletions still outnumber insertions > 1.5 : 1 after H1–H4:
  blank bias −0.5. Edit class: scalar bias shift. Prediction: Δ −0.002 to −0.008. A second −0.5 only if the first is
  kept and a pass remains. If refused: the deletion-heavy profile is the model's operating point, not a bias offset.
- **H6 — space calibration** (only if H2 found nothing planted on space): space bias +2.0. Edit class: scalar bias
  shift. Prediction: Δ −0.002 to −0.006. If refused: the −7.85 is natural and space suppression is upstream.

Pass plan: 1 baseline + H1 (1–2) + H2 (0–1) + H3 (0–1) + H4 (0–1) + H5 (0–2) + H6 (0–1) ≤ 10; every `measure` is
guarded by `budget_left`. Every edit is decided by `s.keep` (the harness rule) plus my own check that collapsed lines do
not rise by more than 5. Unspent passes stay unspent. Expected end state if H1 and H2 hold: CER ≈ 0.06–0.08.
