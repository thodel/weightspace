# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file reading and shell only, no model loaded, no passes used.

Budget: 1 baseline + up to 9 further measures. Baseline CER 0.1993 (24 collapsed, length ratio 0.83, deletions dominate).

H1. Block L_16 (last BiLSTM) has all weights and biases shrunk (rms 0.12/0.15/0.42 vs 0.22/0.30/0.92 in L_12/L_14, i.e. about half). Edit: scale, all 8 L_16 tensors, x2.0. Predict CER 0.20 -> ~0.12-0.15 (-0.05 or more). If it fails: the shrinkage is not uniform or is natural to the last layer; try H2.
H2. (if H1 fails) Only the input and recurrent weights are shrunk, not the biases: scale L_16 weight_ih/weight_hh x2, biases unchanged. Predict -0.03. If fails: try x1.5 on all; then give up on L_16.
H1b. (if H1 kept) Refine factor: try a further x1.15 on L_16 (total 2.3). Predict -0.005 to -0.01; if not kept, revert (factor 2 stays).
H3. Deletions of every frequent letter and space point to a too-strong blank. Edit: add, O_18.lin.bias[blank] -= 0.5. Predict -0.01 to -0.02. If fails: blank bias is not the issue, retry with -0.25 (then stop).
H4. Space row bias is -7.85 (extreme; space deleted 301 times). Edit: add +1.5 to bias of space. Predict -0.005 to -0.01. If fails: the space row is intrinsically that low; leave it.
Fallbacks never use the donor: its lineage differs and its LSTM/conv features are incompatible.
