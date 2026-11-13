# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5; settings: low reasoning effort; tools: file read/write and shell only for reading pack and API, no pass run.

Reading: C_3.co.weight and C_3.co.bias are exactly zero (L2 0.00) while every other tensor looks trained. A zeroed conv gives
an all-zero feature map after ReLU, so C_6 sees only its bias and the output is all blank: CER 1.000, all 300 lines collapsed,
every reference character deleted. This fits the baseline exactly. Output layer, LSTMs and other convs look healthy.

H0 (baseline, pass 1): CER 1.000, collapsed 300. Predicted; if not, the defect reading is wrong and I stop editing.
H1 transplant C_3.co.weight = channel-identity (delta at kernel row 1, col 6, gain 1.0), bias stays 0: C_3 becomes a pass-through of
   C_0 features, so C_6 gets pooled C_0 features instead of the lost C_3 features. Predict CER falls from 1.00 to 0.85-0.99, lines un-collapse
   (collapsed < 300). If it fails (still collapsed): feature scale or content is too far from what C_6 expects; try other gains.
H2 identity gain 2.0: restores activation magnitude if the trained C_3 amplified. Predict CER within +-0.05 of H1; better if blank still dominates.
H3 identity gain 0.5: opposite scale. Predict worse than H1 if H1 kept, else a possible un-collapse.
H4 width-smoothed identity (5-tap box along width, best gain so far): C_3's 13-wide kernel likely acted as a smoothing/context filter. Predict CER -0.02 to -0.10 vs best.
H5 transplant donor (bern-hertwig-v2) C_3 weight+bias-free: different lineage features, expected useless; predict CER 1.00 or worse than identity; conclusion if fails: donor convs are not interchangeable.
H6 donor C_3 blended 0.5 with identity: predict no better than identity.
Conclusion rule: the keep rule decides; a variant is kept only if it significantly beats the last kept state. If no edit is kept, ship nothing and report that C_3 is not recoverable without training.
