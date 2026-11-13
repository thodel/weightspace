# Hypotheses — planted-f1 / s13 / task (b), budget 0 (blind programmer)

- Model: claude-fable-5-1 (Claude Fable 5.1), running as a Claude Code subagent.
- Settings: default reasoning effort; tools used: Read, Bash (ls, wc, py_compile), Write. No model loaded, no pass run.
- Read: `pack/describe.md`, `pack/diff_parent_to_subject.md`, `pack/c0_practice.md`, `PROGRAMMER_BRIEF.md`,
  `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`, `wsp/align.py`.

## Readings that matter

1. `c0_practice.md`: CER 0.7456, length ratio 1.611, 79 collapsed lines. The top confusion is `∅ → n` with **7,667**
   insertions (the next is 136); also `e→n` 111, `i→n` 52, ` →n` 48, while real `n` is missed (`n→m` 85, `n→∅` 52, `n→r` 51).
   The model emits `n` on almost every frame and does not emit it where it belongs.
2. `diff_parent_to_subject.md`, output rows aligned by character: row `n` has rel L2 **2.042** and cosine **-0.991**
   against the parent, norm 7.96 → 8.33, bias Δ -0.142. Every other shared row has cosine 0.88–0.95+ and rel L2 ≤ 0.63.
   A cosine of -0.991 with a norm in the healthy range is a sign flip of the child's own trained row (a transplant of the
   negated parent row would give cosine -1.000 and norm 7.96).
3. Mechanism: the healthy `n` row gives a strongly negative logit on frames that are not `n`; negated, it gives a strongly
   positive logit on every such frame, so `n` beats the blank and every other class there and loses where `n` actually is.
   That is exactly the confusion table.
4. Everything else looks like ordinary fine-tuning drift: conv weights rel 0.12–0.30 with cosine 0.96–0.99; LSTM `weight_ih`/`weight_hh`
   rel 0.20–0.33, cosine 0.95–0.98 across all three layers in both directions (no single tensor stands out); `bias_hh` rel 0.03–0.05;
   `bias_ih` zero by construction; blank row rel 0.23, bias +1.23 (normal for CTC). `C_3.co.bias` has cosine 0.68, but it is a tiny
   vector (rms 0.018, abs max 0.08, parent norm ≈ 0.07 by the reported rel/cos) and the cosine of conv biases falls with their
   magnitude across C_0/C_6/C_9 — noise, not a plant, and its effect on the output is negligible either way. `L_16.layer.weight_ih_l0`
   has an abs max of 4.7, but its diff to the parent (rel 0.296, cos 0.956) is identical to its siblings, so the outlier is inherited.
   Rare-character rows with bias Δ ≈ -2.5 are the usual down-weighting of characters absent from the fine-tuning data.

## H1 (applied): the `O_18.lin.weight` row of `n` was negated — negate it back

- Edit class: `set_rows` on `O_18.lin.weight`, index `[cm['n']]`, values `-W[cm['n']]` (exact sign flip, restores the child's own row).
- Runtime guard: the parent is loaded as a donor, the cosine between the child's `n` row and the parent's `n` row is recomputed;
  the edit is applied only if that cosine is < -0.9 (it is -0.991 in the pack; the guard protects against a row-index mix-up).
  If the guard fails the program applies nothing and notes why.
- Predicted effect: the `∅→n` insertions (≈7,700 of 9,593 edits) disappear and the missed `n` return. CER falls from 0.746 to the
  range of a healthy child on this hand, roughly 0.05–0.15 (a drop of ≥ 0.55 absolute); length ratio 1.61 → ≈1.0; collapsed lines 79 → few.
- If this were wrong (the row was not flipped): the model would lose `n` entirely and CER would stay high — but a cosine of -0.991
  to the parent on a frequent character is not a state a trained model reaches by fine-tuning, so I am confident.

## H2 (applied only if the data decide it): the `O_18.lin.bias` entry of `n` may have been negated with the row

- The bias Δ of -0.142 is compatible with both "row flipped, bias untouched" (ordinary drift of -0.14) and "row and bias flipped"
  (healthy child and parent biases both ≈ +0.07). The pack cannot separate them, and the stakes are small (≤ ~0.3 logit).
- Decision rule at runtime: with `bc` the current child bias of `n` and `bp` the parent's, flip the bias (`set_rows` on `O_18.lin.bias`,
  value `-bc`) only if `|(-bc) - bp| < 0.25 * |bc - bp|`, i.e. the flipped value is far closer to the parent than the current one.
  Otherwise leave the bias as it is.
- Predicted effect: at most a small CER change (< 0.01 absolute) in either case; it is a tidy-up, not the repair.

## Not applied

- No transplant from the parent (its CER on this hand is much worse; the child's own row, un-negated, is the right one).
- No edit to `C_3.co.bias`, the LSTMs, the convolutions or the blank row: no reading points at them.
