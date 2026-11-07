# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: Read/Bash (file reading only, no model run).

Reading: CER 0.065 is not bad, but errors are almost all deletions (space 78, e 34, n 28, i 21, r 19 ... -> nothing), 8 collapsed lines.
Output rows of characters shared with the parent are ~2x the parent's norm (e.g. a 23.7 -> 47.7, rel change ~1.0, cosine 0.998, i.e. same direction, doubled length), while conv/LSTM tensors changed far less (cos 0.95-0.99). The blank row is weak (9.3 vs 29.6 mean) while biases are unchanged in scale. Suspect: O_18.lin.weight multiplied by about 2 (bias untouched), which shifts weight/bias balance and favours blank/under-reads.

H1 scale O_18.lin.weight x0.5 (all rows, bias fixed). Predict CER 0.065 -> ~0.04 or below (delta -0.02 to -0.03), collapsed 8 -> fewer. If it fails (CER rises or CI contains 0): the factor is wrong or weights are not the culprit; go to H2.
H2 If H1 kept: refine, scale again x0.85 (net 0.425); then if that is kept x0.85 again. Predict small gains (-0.002 to -0.005) or none; a failure means H1 factor was near the optimum.
H3 If H1 failed: scale x0.7. Predict -0.005..-0.015 if the defect is real but smaller than 2x. Failure -> no uniform scale defect.
H4 If H1 and H3 failed: scale x1.3 (the opposite direction). Predict rise; kept only if I read the direction wrong.
H5 Blank-row-specific: add +/- 0.5 to blank bias (O_18.lin.bias[0]), direction chosen from confusions: deletions dominating -> blank too strong -> bias -0.5. Predict -0.003..-0.01. Failure -> blank balance is fine.
H6 Weak blank row (norm 9.3): scale blank row x1.5 after H1 only if insertions (blank->char) dominate. Predict small. Failure -> leave.
Conclusion rule: if none of H1, H3 helps the readings show no output-scale defect; ship nothing.
