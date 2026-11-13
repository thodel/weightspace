Model: claude-sonnet-5-5. Settings: default reasoning effort; tools: Read, Bash (cat, py_compile) only. Budget 0, blind.

H1 (applied): `O_18.lin.weight` multiplied by 2 as a whole. Evidence: the rarely used inherited rows (a-acute, A-umlaut, s-caron, e-acute, ...)
have child/parent norm ratios of 2.014, 2.011, 2.013, 2.013 with cosine 0.998 to the parent, which is what a pure scaling gives and
what fine-tuning does not; the confusions are deletions of common characters (blank wins), as expected when logit weights are doubled
against an unscaled bias (space bias -7.85). Edit: scale O_18.lin.weight by 0.5. Predicted: CER falls, perhaps 0.065 -> 0.04.
If wrong (the weights were healthy), the cost is a flatter output; argmax shifts only through the bias balance.

H2 (not applied): conv C_3 bias moved 1.09 rel L2 from the parent. Probably legitimate fine-tuning (weights only moved 0.17); no evidence of defect, nothing applied.
