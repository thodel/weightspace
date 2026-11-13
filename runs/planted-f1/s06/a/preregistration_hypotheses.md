# Pre-registered hypotheses — planted-f1 / s06 / task (a), budget 10

Programmer: model `claude-fable-5-1` (Claude Fable 5.1), default reasoning effort, running as a Claude Code
subagent. Tools used: Read / Bash (`cat`, `ls`, `py_compile`) on the pack, the brief and the five allowed `wsp/`
files only. No model loaded, no pass run, nothing else read. Written before the first pass; `program.py` is
written in full in advance and branches only on `m.cer`, `m.collapsed`, `m.confusions`, `m.length_ratio`.

## What the readings show

- `describe.md`: output row `e` of `O_18.lin.weight` has norm **0.00** (parent: 7.64; the other frequent
  letters r/s/t/i/n/a sit at 7.8–8.3). `diff_parent_to_subject.md`: row `e` rel L2 1.000, cosine 0.000,
  bias Δ −0.142 — the weight row was zeroed, the bias was left alone.
- `c0_practice.md`: CER 0.1623, 2088 edits; `e→∅` 910, `e→a` 131, `e→i` 90, `e→o` 52, `e→r` 36,
  `e→c` 35, `e→u` 32 — about 1,290 of the 2,088 edits are the letter `e`. Length ratio 0.900 (the deleted e's).
- `C_3.co.bias`: rel L2 **1.087**, cosine 0.679 against the parent, whereas every other block's bias moved
  about as much as its weight (C_0 0.15/0.12, C_6 0.26/0.26, C_9 0.33/0.29) and `C_3.co.weight` moved only
  0.167. Consistent with noise of about the parent's own norm added to that 32-vector (child ≈ parent + noise),
  but the absolute size is tiny (rms 0.018), so the functional effect, if any, is small.
- Not treated: `L_16.layer.weight_ih_l0` abs max 4.70 (other LSTM tensors ≤ 2.8) with a normal rel L2 / cosine
  against the parent — no cheap, safe edit exists; rare-character rows (`«`, `¬`, `_`, …) pushed down by
  ~2.5 in bias with norms up 25 % — the usual fate of unseen characters in fine-tuning, not a defect.
  Space is the most negative bias (−7.85) and ` →∅` 103: plausibly ordinary, tested cheaply at the end.
- 9 collapsed lines: expected to be a by-product of the missing `e`; re-read after the repair.

## Hypotheses (edit class, predicted direction and size of the practice-set CER change, conclusion if it fails)

- **H1 — the `e` row was zeroed; the parent's `e` row restores it.** Edit class: donor row transplant
  (`set_rows` on `O_18.lin.weight`, row `cm['e']` ← parent row `pm['e']`, bias untouched).
  Prediction: CER falls by **0.07–0.11** (0.162 → 0.05–0.09), `e→∅` from 910 to < 150, length ratio → ≥ 0.96,
  collapsed lines fall. If refused or the fall is < 0.03: the child's L_16 feature space has drifted too far for
  the raw parent row; H2 is then the primary repair, and if that fails too the defect is not (only) in that row.
- **H2 — the donor row fits better after mapping into the child's feature space.** Edit class: donor row,
  adapted (`set_rows`; the row is `pe + pe·A`, with `A` a ridge-regularised least-squares map fitted on the 80
  weakest-norm shared rows so that `parent_row·(I+A) ≈ child_row`; fallback: parent row + mean row shift).
  Prediction: a **further** fall of 0.005–0.02 over H1 (CER 0.04–0.07). If refused: the raw parent row is
  already within the keep rule's resolution of the best obtainable; the mapping adds nothing measurable.
- **H3 — the recovered `e` competes wrongly with the blank; its bias is off.** Edit class: bias nudge
  (`add` ±0.5 on `O_18.lin.bias[e]`, direction from the residual: +0.5 if `e→∅` still exceeds `∅→e` by
  more than 2× + 10, −0.5 in the opposite case, skipped if balanced). Prediction: CER down **0.002–0.01**.
  A second ±0.5 step follows only if the first is kept and the residual still points the same way.
  If refused: the remaining `e` errors are directional (e↔a/i/o), not a threshold problem; stop touching `e`.
- **H4 — `C_3.co.bias` was noised; the parent's bias is the better one.** Edit class: donor transplant
  (`transplant` of the 32-vector). Prediction: CER down **0.002–0.015** if it was noised; otherwise no change or
  a tiny rise. If refused: the large relative move is ordinary fine-tuning drift of a near-zero vector (or the
  parent's bias no longer matches the child's C_3 weights) — not a defect, or not repairable from this donor.
- **H5 — spaces are under-predicted.** Edit class: bias nudge (`add` +0.75 on `O_18.lin.bias[' ']`), run only
  if ` →∅` still exceeds `∅→ ` by more than 20 after H1–H4. Prediction: CER down **≤ 0.005**, WER down more.
  If refused: the space bias is where fine-tuning put it; the 103 deletions are the model's ordinary error.
- **H6 — residual collapse is blank dominance.** Edit class: bias nudge (`add` −0.5 on `O_18.lin.bias[0]`),
  run only if `m.collapsed ≥ 4` after the `e` repair. Prediction: collapsed → ≤ 2, CER down ≤ 0.005.
  If refused or not triggered: the collapsed lines were the missing `e` (or unreadable slivers), not a blank defect.
- **H7 (leftover passes only) — the `e` row is right in direction but too short.** Edit class: row scale
  (`scale`-equivalent via `set_rows` ×1.2 of the current `e` row), run only if `e→∅` is still > 100 and ≥ 2
  passes remain. Prediction: CER down ≤ 0.005. If refused: the row's length is fine.

Passes: 1 (C0) + H1 + H2 + H3 (≤ 2) + H4 + H5 + H6 + H7 ≤ 10. Every edit is decided by `s.keep`; the program
never decides against the rule. Expected end state: CER 0.04–0.08 with only H1 (and perhaps H2) kept.
