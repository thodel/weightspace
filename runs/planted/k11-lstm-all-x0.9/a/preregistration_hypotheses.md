# Preregistration, programmer (a), budget 10
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file read, bash (py_compile only). No model loaded, no pass run.

Evidence: practice CER 0.0727 with 12 collapsed lines; errors are almost all deletions (space, e, n, r, i, s -> nothing), i.e. the
net emits blank too often. Blank bias +1.23 and blank row norm 4.6 look ordinary. Every LSTM tensor sits at a uniform relative distance
0.11-0.33 from the parent with high cosine (0.95-0.999), and bias_hh rel change is only 0.11 with cosine 0.999: consistent with a
uniform global gain on all LSTM tensors, not retraining. Weakened recurrent/input gain -> weaker, less confident logits -> blank wins.

H1 (edit class: scale, all 24 LSTM tensors incl. bias_hh, x1/0.9 = 1.111): the LSTMs were uniformly shrunk by ~0.9. Predict CER 0.0727 -> ~0.04-0.05
(down 0.02-0.03), collapsed lines fall. If it fails (CER up or interval contains 0): the shrink is not uniform or not in biases; go to H2.
H2 (scale, LSTM weight_ih and weight_hh only, x1.111, biases untouched): predict CER down 0.01-0.03. If it fails: gain is not the defect in these tensors; go to H4.
H3 (scale, kept state of H1/H2 further by x1.05): checks whether the optimum is beyond 1/0.9. Predict small change (+-0.003); conclude the correct factor was 1.111 if no gain, and revert.
H4 (add, output blank bias -0.5): blank over-firing; predict CER down 0.005-0.01. If it fails, blank bias is not the cause.
H5 (scale, output weights of non-blank rows x1.05 if H1/H2 failed; otherwise skip): compensates reduced logit margin; predict down 0.005. If it fails, stop.
H6 (interpolate, LSTM weights alpha 0.1 toward parent, only if H1 kept): predict neutral/worse, since the parent is poor on this hand; revert if not significant.
Not hypothesised: conv or output-layer damage; no evidence (C_3 bias is the largest convolution change, 1.09 rel, but its absolute norm is 0.1, so little function lies there).
