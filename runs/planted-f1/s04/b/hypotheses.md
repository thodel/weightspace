# Hypotheses — planted-f1 / s04 / task (b), budget 0 (blind programmer)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: reasoning effort low-medium (default for this
subagent), tools used: Read/Bash (cat, ls, grep, py_compile) on the pack, the brief and the allowed `wsp/` API
files only. No pass, no model loaded, program not run.

## What the readings show

1. **Space bias is an extreme outlier.** `O_18.lin.bias[' '] = -7.85` is the most negative bias of all 297
   classes, below even the rarest glyphs (`ú` -7.33, `Ⅲ` -7.12, `Ⅱ` -7.01) and ~3 sd below the layer mean
   (-1.92 ± 2.01). Space is the most frequent class in the training text; frequent classes in this layer sit near
   zero or slightly positive (`s` +0.40, `E` +0.33, `d` +0.29) and have the smallest row norms (e, r, s, t, i, n, a
   at 7.6–8.3). A trained space row would never end at -7.85. Consistent with it, the top confusion of C0 is
   ` ` → ∅ (124 of 935 edits, 2.3× the next confusion `e` → ∅ 55), and the length ratio is 0.959 (hypothesis
   too short). The space row is not among the "moved most" rows (that table ranks by weight-row rel L2), so the
   weight row of the space is probably untouched and the bias alone was pushed down.
2. **Everything else looks like ordinary fine-tuning drift.** Conv and LSTM blocks have rel L2 0.13–0.33 and
   cosine 0.94–0.99 to the parent, uniformly across blocks and directions; `bias_ih` all zero by construction.
   `C_3.co.bias` has rel L2 1.09 / cos 0.68, but the tensor is tiny (L2 0.10, rms 0.018; parent ≈ 0.07) so it
   is relative noise, not a defect worth editing. `L_16.layer.weight_ih_l0` has an abs max of 4.23 (other
   `weight_ih` tensors max at 1.2–2.0) with otherwise normal L2/rel L2/cosine; a possible single-element spike,
   not visible in the aggregate statistics. Blank row (norm 4.64, bias +1.23, Δ +0.11 to parent) is normal.
   12 collapsed lines of 300 cannot be attributed to any tensor from these readings; I leave them alone.

## Hypotheses, edits, predictions

| # | hypothesis | edit | predicted effect on practice CER |
|---|---|---|---|
| H1 | The bias of the space row was pushed to -7.85 (planted). Restoring it to where the parent (plus the frequent-letter drift of this child) puts it removes most of the 124 space deletions. | `set_rows` on `O_18.lin.bias`, index `cm[' ']`, value = parent bias of `' '` + median(child − parent bias over frequent Latin letters). **Guarded:** applied only if the child-minus-parent bias Δ of the space is below -3 and is a >5-MAD outlier against the Δ of the frequent letters (so an inherited, naturally low value is not touched). Fallback without parent: median bias of the frequent letters, same child-only outlier guard. | CER 0.0727 → about 0.065–0.068 (−0.5 to −0.8 pp): roughly 80–110 fewer space deletions, perhaps 10–30 new space insertions. WER should fall more (−2 to −4 pp) since every missing space merges two words. |
| H2 | Isolated weight elements far from the parent (|child − parent| > 1.5, i.e. >20× the per-element drift rms of ~0.06) in conv/LSTM weight or bias tensors are planted spikes (candidate: the 4.23 in `L_16.layer.weight_ih_l0`). | `add` of (parent − child) on exactly those elements, one edit per affected tensor. **Guarded:** only tensors with 1–200 such elements; 0 → no edit, >200 → not a spike, skipped with a note. Output layer excluded (its elements are legitimately large and rows drift more). | If a spike exists: small improvement, −0.1 to −0.5 pp, possibly fewer collapsed lines. If none: no edit written. |

Not applied: anything on `C_3.co.bias`, the blank row, the LSTM gates, or the conv kernels as wholes — the
readings do not show a defect there, and a wrong edit at budget 0 cannot be caught.

## What the program does

Loads the parent as donor, aligns output rows by character, evaluates the guards above on the weights alone
(no forward pass), and for every guard that fires writes `s.hypothesis` → `s.edit` → `s.keep_unmeasured`;
every guard that does not fire is recorded with `s.note`. Ends with `s.freeze()`.
