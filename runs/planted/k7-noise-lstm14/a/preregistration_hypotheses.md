# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: Read, Bash (reading pack and wsp API files only, py_compile).

Readings: CER 0.0726; errors are overwhelmingly deletions (ref -> nothing: space 103, e 38, n 31, s 23 ...); 10 collapsed lines;
space bias -7.85 (extreme), blank bias +1.23 with weak blank row; L_16.weight_ih_l0 abs max 4.70 (about 2x the others);
L_14 input weights moved most from the parent among the LSTMs (rel L2 0.41 / 0.40 vs 0.20-0.33), which could be noise.
Baseline C0 is pass 1; passes 2-7 are edits, 3 left spare.

H1 (output bias, add): blank bias too high relative to characters -> deletions. Edit: O_18.lin.bias[blank] -= 0.6. Predict CER -0.003 to -0.010
   (0.0726 -> ~0.066). If it fails (CER up): blank is not over-confident; the deletions come from upstream features; revert, do not try larger.
H2 (output bias, add): space bias is implausibly low (-7.85), space deletions are the top error. Edit: bias[' '] += 1.5. Predict CER -0.004 to -0.012,
   space deletions fall, insertions of space rise a little. If it fails: space scarcity is real (spaces are narrow gaps), revert; try +0.7 only if CER change is within noise.
H3 (LSTM, low-rank denoise): L_14 weight_ih_l0 and _reverse carry additive isotropic noise; truncating the SVD to rank 120 of 400 removes noise
   energy and leaves the trained signal. Predict CER -0.005 to -0.03 if noise is planted there; +0.01 or worse if the matrices are genuinely full-rank.
   If it fails: L_14 input weights are not noise-damaged (or noise is not low-rank separable); fall back to H3b rank 250 once, then drop it.
H4 (LSTM, clip outliers): L_16.weight_ih_l0 has |w| up to 4.70, others ~2; clip at 10 robust sigmas of the tensor (median/MAD). Predict CER change -0.001 to -0.005;
   if flat or worse, the spike is genuine, revert.
H5 (conv, interpolate): C_3 bias moved most (rel 1.09, cos 0.68) from the parent; but biases are tiny (L2 0.10), so predict effect near 0 (|dCER| < 0.001).
   Not run unless passes are left over; if run and it fails, conclude it is ordinary fine-tuning drift.
Overall prediction: if the model is healthy, none of H1-H4 passes the paired 95% rule and the program freezes with zero kept edits.
