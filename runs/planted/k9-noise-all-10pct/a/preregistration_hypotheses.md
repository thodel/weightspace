Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: Read/Bash (file reads, py_compile only). Budget 10.

Evidence: 123/300 lines collapsed, length ratio 0.55, every top confusion is char -> nothing (CTC emits blank where text should be). Blank row is the weakest (norm 5.0) but has the only positive bias (+1.23) vs -1.9 mean for characters. Every tensor moved 0.15-0.3 rel L2 from parent (uniform, incl. LSTMs), C_3.co.bias 1.09 rel / cos 0.68.

H1 (set blank bias lower, add -1.5 to O_18.lin.bias[0]): blank over-dominant; CER 0.55 -> ~0.40 (-0.10 to -0.2). If it fails: blank bias is not the cause; the collapse is upstream.
H2 (second step, add -1.5 more, or -0.75 if H1 failed): CER another -0.03 to -0.08; fail = H1 already optimal.
H3 (set blank bias to mean of character biases would be too much; instead add +0.5 to all non-blank biases): CER -0.02; fail = bias is not the lever.
H4 (interpolate C_3.co.bias to parent, alpha 1.0): the one anomalously moved tensor is damaged; CER -0.02 to -0.05; fail = its drift is legitimate fine-tuning.
H5 (interpolate all conv tensors 25% toward parent): uniform noise partly averaged out; CER -0.02 to -0.05, or worse since parent is poorer on this hand; fail = drift is learned signal, noise is not removable this way.
H6 (same 25% for LSTM tensors): CER about -0.02; fail as H5.
H7 (scale blank row weight by 1.2 or 0.8 per direction suggested by remaining deletions): +/-0.02.
