# Blind programmer (b), budget 0
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools used: Read/Bash (read pack files and brief only, no model loaded, no passes).

Readings: CER 0.551, 123/300 lines collapsed, length ratio 0.554, top confusions are all "character -> nothing" (deletions, spaces 1001, e 630, n 454...).
The model under-emits: the blank wins too often.

## H1: blank dominates (blank bias +1.23 vs others -1.9; blank row norm 5.0)
Possible edit: lower O_18.lin.bias[0] by a constant. Predicted CER down if true. Not applied: the blank's high bias and low row norm are typical of CTC
models, the parent comparison shows only +0.113 bias drift, and I cannot size the shift without a pass.

## H2: C_3.co.bias changed 109 % vs parent (cos 0.68) while its weights changed 19 %
Odd but tiny tensor (L2 0.10); no donor-vs-child evidence of which is right; a transplant could break the fine-tuned conv stack. Not applied.

## H3: broad diffuse change (every tensor moves 15-30 % from parent, no single locus)
Consistent with ordinary fine-tuning (72 new characters) or with diffuse noise; no tensor shows a localised anomaly (no zero rows, no extreme norms,
L_16.weight_ih max 4.7 is mildly high only). No targeted repair is identifiable.

## Decision
No edit I am fairly sure about; apply nothing, freeze. Suspicion: diffuse weight perturbation across all tensors (not repairable blind), or a mild blank over-dominance.
