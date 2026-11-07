Model: claude-sonnet-5-5; reasoning effort low; tools: Read/Bash (pack files, brief, wsp API only), no passes.

H1: the whole last BiLSTM (L_16, both directions, all weights and biases) was scaled by about 0.5.
Evidence: L_16 tensors are uniformly ~0.5x the matching L_14 tensors (ih L2 68/64 vs 123; hh 62/63 vs 120;
bias_hh 12.0/12.4 vs 26.3/26.4; rms 0.12 vs 0.22), in every one of 8 tensors and both directions.
Practice reading: frequent characters (space, e, n, i, r) deleted wholesale, length ratio 0.83, 24 collapsed lines:
a weak L_16 signal leaves the blank row (bias +1.23) dominant.
Edit: scale x2.0 on all L_16 weight and bias tensors. Predicted: CER 0.20 -> clearly lower (maybe 0.10-0.15), collapses fall.
Risk: if L_16 is naturally smaller, x2 hurts; the uniformity across bias and weights makes defect more likely.
Nothing else applied (output layer, convs look normal; donor not used).
