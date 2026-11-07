# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5. Settings: reasoning effort low, tools: file read/shell only (no model loaded, no passes used before writing).

Evidence: C0 CER 0.7456, 79 collapsed lines, length ratio 1.61; 7667 insertions of `n` (of 9593 edits). Diff to parent: output row `n` moved rel L2 2.04, cosine -0.991 (all other rows cos > 0.88, blank 0.2). The row norm is unchanged (7.96 -> 8.33): sign flip of the row `n` in O_18.lin.weight. Blank row bias is fine; `n` bias moved only -0.142.

H1 (set_rows / sign flip): negate O_18.lin.weight[n]. Predicted CER 0.7456 -> about 0.25-0.40 (drop > 0.30, collapsed lines 79 -> < 10, `∅->n` insertions nearly gone). If it fails (CER drop < 0.05): the flip is not the cause of the insertions; revert, and next suspect the `n` bias and C_3 (H2, H3).
H2 (set_rows, bias, donor): set O_18.lin.bias[n] to the parent's value (small shift, 0.14). Predicted change -0.00 to -0.01 CER; probably not significant, so the keep rule will likely refuse it. If refused: the bias was untouched, fine.
H3 (interpolate, C_3.co.bias toward parent, alpha 0.5): C_3 bias has rel change 1.09 (cos 0.68) vs 0.15-0.33 elsewhere; this may be a second defect (or just fine-tuning drift). Predicted change: 0 to +0.02 (mostly harmful); run only if CER after H1 is still > 0.30. If it fails/worsens: it is training drift, leave it.
H4 (set_rows, donor transplant of row n from parent at alpha 1, instead of negation) only if H1 is kept but `n` confusions (e->n, i->n, n->m) remain large: predicted CER -0.01 to -0.03. If it fails: the child's row is fine, remaining error is the hand domain, stop.
No further edits: the rest of the child differs from the parent by ordinary fine-tuning.
