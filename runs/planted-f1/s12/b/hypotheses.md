# Hypotheses — planted-f1 s12, task (b), budget 0 (blind programmer)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort, no extended-thinking toggle
known; tools used: Read / Bash (`ls`, `wc`, `grep`, `cat` on the allowed files, `py_compile`). No pass, no model loaded.
Read: the three pack files, PROGRAMMER_BRIEF.md, `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`,
`wsp/align.py`, `wsp/subject.py` (only the `apply`/`set_tensor` signatures).

## What the readings show

- C0 on the 300 practice lines: CER 0.2443, 3144 edits. The two confusions `e → n` (1364) and `n → e` (837) alone are
  2201 of the 3144 edits (70 %). With `e ≈ 17 %` and `n ≈ 10 %` of German text (≈ 2200 `e`, ≈ 1300 `n` in 12,867
  characters) that is almost every `e` read as `n` and almost every `n` read as `e`; the 1.6 : 1 ratio is the
  frequency ratio of the two letters. The remaining ~940 edits (≈ 7.3 % CER) are the ordinary profile of a Kurrent
  model (space deletions 98, a few `r`/`i`/`.` deletions, 9 collapsed lines).
- Output layer against the parent (rows aligned by character): mean relative change 0.20; the two rows that moved
  most are exactly `e` (rel L2 1.19, cosine 0.356) and `n` (rel L2 1.10, cosine 0.361). Every other "moved" row is a
  rare glyph (`«`, `¬`, `_`, `»`, `…`) with cosine ≥ 0.88 and a bias pushed down by ≈ 2.4 — the usual drift of unseen
  classes in a fine-tuned child, not a defect.
- Norms: parent `e` 7.64 → child `e` 8.33; parent `n` 7.96 → child `n` 7.57. The child's `e` row has the size of a
  (slightly grown) `n` row and vice versa. A row that had merely drifted 20 % like its neighbours would keep cosine
  ≈ 0.95; a row replaced by noise would grow by √2; a row mixed with its neighbour would keep cosine ≈ 0.8.
  Cosine 0.36 with unchanged norm fits one thing: **the `e` and `n` rows of `O_18.lin.weight` have been exchanged.**
  (`e` and `n` are the classic Kurrent look-alikes, so the two healthy rows are moderately similar — cosine 0.36 is
  what a swap of them against the parent should look like.)
- Nothing else stands out as planted. `C_3.co.bias` has rel L2 1.09 / cosine 0.68 against the parent, but it is a
  32-element bias of rms 0.018 (abs max 0.08) behind weights of rms 0.2: noise-level drift of a negligible tensor,
  and neither a scale (cosine would be 1) nor a sign flip (−1). `L_16.layer.weight_ih_l0` has an abs max of 4.7
  (others ≈ 2) but the same rel L2 (0.296) and cosine (0.956) against the parent as every other LSTM input matrix,
  so the outlier is inherited, not planted. Blank row (bias +1.23, Δ +0.11), the space bias (−7.85, the parent's
  too: the row is not among the movers) and the new codec rows (norms 8–15, bias ≈ −0.65) look healthy.

## H1 — the `e` and `n` output rows are swapped (edit class: `set_rows` on the output layer)

- **Edit:** swap rows `cm['e']` and `cm['n']` of `O_18.lin.weight` and the same two entries of `O_18.lin.bias`
  (one `set_rows` edit, `per_tensor` values). The bias is swapped too: a planted class swap most naturally exchanges
  the whole row (weight and bias); the bias readings (Δ −0.04 and +0.18 against the parent) cannot distinguish the
  two cases, and the cost of being wrong on a scalar of that size is a fraction of a percent CER either way.
- **Gate inside the program (no pass needed):** load the parent as donor and compare the child's `e`/`n` rows to the
  parent's `e`/`n` rows by cosine. The swap is applied only if the child's `e` row is closer to the parent's `n` row
  than to the parent's `e` row **and** the child's `n` row is closer to the parent's `e` row than to the parent's
  `n` row (expected: cross cosines ≈ 0.9, straight cosines ≈ 0.36 as in the pack). If the gate fails the program
  edits nothing and notes why.
- **Predicted effect:** CER falls from 0.244 to about 0.07–0.09 (the 2201 `e↔n` substitutions disappear; most of the
  `e → ∅`, `n → ∅`, `∅ → e`, `∅ → n` edits are CTC side-effects of the same swap and should go too). WER from 0.63
  to roughly 0.3. Collapsed lines unchanged or fewer.
- **If it were wrong:** the only way this edit hurts is if the rows were never swapped, in which case the gate
  (cross cosines higher than straight ones) could not pass; the program then ships the model untouched.

## Not applied

- No edit to `C_3.co.bias`, `L_16`, the blank row, the space row or the rare-glyph rows: nothing in the readings
  points to a defect there, and a blind edit on a healthy tensor can only make the model worse.
