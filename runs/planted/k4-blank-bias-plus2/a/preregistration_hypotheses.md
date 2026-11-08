# Preregistration, programmer (a), budget 10
Model: claude-sonnet-5-5. Settings: default reasoning effort, tools: file read and shell only (no model loaded, no passes).

Evidence: practice CER 0.0789, 14 collapsed lines, length ratio 0.941, and the top confusions are all deletions (reference -> nothing: space 177, e 81, n 53, ...). That is a blank-too-strong signature. Blank bias is +3.23 and has moved +2.11 from the parent, while the blank row norm moved only 0.23 relative; the other rows' biases moved about -2 on the extreme rows too.

H1 (primary): the blank bias is inflated by about +2. Edit class: add, `O_18.lin.bias`, delta -2.1 at row 0. Predicted: CER 0.079 -> about 0.04 to 0.05 (-0.03 or more), fewer collapsed lines, length ratio towards 1. If it fails (CI not below 0): blank bias is not the cause or the shift is wrong in size; try smaller (-1.0) then conclude the deletions come from elsewhere.
H1b: if H1 is kept, a further -0.5 on the blank bias. Predicted: -0.002 to -0.005, may be noise. If it fails, the first step was near the optimum.
H1c: if H1 fails, -1.0 and then -3.0 on the blank bias (size mis-specified). Predicted -0.01 to -0.03 if the sign is right.
H2: C_3.co.bias is drifted (rel L2 1.09 vs parent, cosine 0.68, far out of line with the other tensors). Edit class: interpolate, `C_3.co.bias`, alpha 0.5 toward the parent. Predicted: small (|dCER| < 0.005); if worse, the child legitimately retrained that bias and it is not a defect.
H3 (null): there is no defect beyond H1. If H1 and H2 both fail, ship nothing further.
