# Blind programmer (b), budget 0
Model: claude-sonnet-5-5. Reasoning effort: low/default. Tools: Read, Bash (cat, ls, grep, py_compile only); no model loaded, no passes.

## H1 (apply): the output row of `e` was zeroed
Evidence: describe.md gives `e` row norm 0.00 (every other row >= 7.7); diff shows `e` rel L2 1.000, cosine 0.000, norm 7.64 -> 0.00;
practice CER 0.162 with `e` -> nothing 910x and `e` -> a/i/o/r/c/u as top confusions (the most frequent German letter is never emitted; with a zero row
only the bias, -0.14, competes with the other classes). Nothing else is anomalous: other moved rows are ordinary fine-tuning drift, length ratio 0.90 matches deleted `e`s.
Edit: set_rows on `O_18.lin.weight`, row cm['e'], values = parent's row pm['e'] (parent 7.64 norm, same scale as the child's common-letter rows 7.7-8.3). The child bias for `e` is kept (-0.142 vs parent, not anomalous).
Predicted effect: large improvement, CER 0.162 -> roughly 0.05-0.09 (the parent basis has drifted ~20 % in the LSTM, so not a perfect row).
If wrong (e.g. the child was always like this): the edit costs little, since a zero row cannot be better than a trained-ish row.
Guard in program: applied only if the current `e` row norm < 1e-6.

## Not applied
- C_3 bias rel L2 1.09 (cos 0.68) in the parent diff: C_3 bias is tiny (rms 0.018), likely ordinary fine-tuning; no evidence of fault.
- `L_16.weight_ih_l0` abs max 4.7 and bias extremes: no clear defect; not touching.
