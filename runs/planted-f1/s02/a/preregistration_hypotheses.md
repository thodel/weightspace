# Pre-registration — planted-f1 / s02 / task (a), budget 10

Programmer: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort, no extended tools; tools used: Read
and Bash (`cat`, `ls`) on the pack, the brief and the five allowed `wsp/` files, `py_compile` on the program. No model loaded,
no pass spent, no other file of the repository read. Written before the first pass; `program.py` is final as of this file.

## What the readings say

- C0: CER 0.0726, WER 0.1923, **10 collapsed lines** of 300, length ratio 0.971, 934 edits. The top ten confusions are all
  deletions (` `→∅ 103, `e`→∅ 38, `n`→∅ 31, `s`→∅ 23, `i`→∅ 22, `r`→∅ 21, `.`→∅ 18, `t`→∅ 14, `o`→∅ 14) plus 22 inserted
  spaces; **no substitution reaches the top ten**. The letter deletions are roughly in proportion to letter frequency, which is
  what 10 half-collapsed lines (≈260 characters) would produce; so the picture is "a healthy reader plus a few lines that
  collapse", not a confused character or a shifted row.
- Tensors: all parent→child relative changes are ordinary and forward/reverse pairs agree within 0.03, so no tensor-level
  scaling, zeroing or heavy noise. `C_3.co.bias` (rel 1.09, cos 0.68) is explained by its tiny norm: its absolute drift (0.07)
  equals that of `C_6`/`C_9` biases — not a plant. Oddities worth a zero-cost check in the program: `L_16.layer.weight_ih_l0`
  has abs max **4.70** (every other LSTM weight ≤ 2.84) and a norm 7 % above its reverse twin (other layers: < 1 %); the space
  row's bias **−7.85** is the most negative of all 297 rows, at the level of `ú`, `Ⅲ`, `Ž` (its drift against the parent is not
  in the pack). The blank is normal (bias Δ +0.11, row rel 0.23).
- Conclusion before the first pass: a plant, if any, is small in L2 and structured (spikes, a slice, a bias) or there is none.
  The program therefore spends 0 passes on a structured diff against the parent and only measures what that diff flags, then
  runs three calibration probes the keep rule can refuse.

## Hypotheses (one line each; edit class; predicted CER change on the practice set; what failure means)

- **H0 baseline** — pass 1 `measure("C0")`; predicted CER 0.0726 ± 0.001, collapsed 10. If not reproduced: note nondeterminism; the rule still applies.
- **H1 zeroed entries** — exact zeros where the parent is non-zero (training never writes exact zeros) → masked `add` restoring the parent's values. Predicted: if found, CER **−0.5 to −3 pp**, collapsed lines fall. Not found → no zeroing plant, no edit. Refused → the zeros are inert (dead units), not the defect.
- **H2/H4 replaced or scaled slices** — a row/column/channel whose norm ratio to the parent leaves [1/1.6, 1.6] or whose cosine to the parent is < 0.3 → rescale (pure scale, cos > 0.9) or transplant the parent's slice (`add`). Predicted **−0.3 to −4 pp**. Not found → no slice plant. Refused → the drift is genuine training.
- **H3 spikes** — entries with |w| > 8 rms of their tensor that moved > 4 rms from the parent (suspect: `L_16` forward ih, abs max 4.70) → set back to the parent's values (`add`). Predicted **−0.5 to −2.5 pp**, mainly through collapsed lines 10 → ≤ 3. Not found → the 4.70 is inherited or natural. Refused → the spikes are harmless and the collapses have another cause.
- **H4b scaled blocks** — an LSTM gate block, an input/column direction half or a whole weight tensor with norm ratio outside [1/1.25, 1.25] → rescale (cos > 0.9) or transplant (`add`). Predicted **−0.3 to −2 pp**. Refused → natural drift.
- **H5 bias outliers** — an `O_18` row whose bias drift vs the parent is > 3.0 from the median drift (> 2.0 from the frequent-character median for frequent letters, punctuation, blank and space), or an LSTM/conv bias entry > max(0.3, 8 MAD) from its median drift → recentre to parent + median drift (`add`). Predicted **−0.3 to −2 pp** for a frequent row, **−0.5 to −0.8 pp** if the space row is the outlier (103 space deletions). Refused → the shift is the model's own calibration.
- **P1 space bias +0.75** — calibration probe (`add` to one bias row), run only if the space row was not repaired under H5. Deletions 103 vs insertions 22 are out of balance; predicted **−0.15 to −0.35 pp**. Refused → the space asymmetry is the hand / the collapsed lines, not a bias defect.
- **P2 blank bias −0.4** — calibration probe (`add` to row 0). Predicted **−0.1 to −0.3 pp** if the blank over-fires. Refused → the blank is not over-firing; the deletions sit in the collapsed lines.
- **P3 new rows −1.5** — the 72 child-only output rows (except `⸗`) have biases ≈ −0.65, far above the parent's rare rows (−2 to −7); lower them by 1.5 (`add`). Predicted **−0.05 to −0.3 pp** from fewer spurious rare-character outputs; 0 if they never fire. Refused → harmless, no gain.

## Decision rules and budget

- Every edit: `s.hypothesis` → one `add` (per-tensor deltas computed from the parent) → `s.measure` → my own check (revert if
  collapsed lines rise by more than 2 against the kept state, whatever the CER) → `s.keep` (the harness's paired-interval rule).
- Each signature class (H1, H2/H4, H3, H4b, H5) is recomputed from the *current* weights just before its edit, so kept edits
  are not corrected twice; a class with nothing flagged costs no pass. At most 5 structural edits.
- Budget: 1 (C0) + ≤ 5 structural + ≤ 3 probes = ≤ 9 ≤ 10. Unspent passes are left unspent. No uncounted forward
  (`logits`/`activations`) is used; the practice references are not read by the program.
- If nothing is kept: no plant signature in the parent diff and no calibration gain → either the model is healthy at CER 0.0726
  (the 10 collapses being its own) or the defect is below the resolution of these readings; I report it as "no defect found".
