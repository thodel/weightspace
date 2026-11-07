Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: Read/Bash (pack files, brief, wsp/ops.py, wsp/session.py only). Budget 0, blind.

H1 (applied): `C_3.co.weight` and `C_3.co.bias` are all zero (L2 0.00; diff vs parent rel L2 exactly 1.0, cosine nan). The practice reading is CER 1.0 with every line collapsed to nothing, consistent with a dead second conv block (ReLU(0)=const, no signal reaches the LSTMs). All other tensors look like a normal fine-tune of the parent (rel L2 0.03-0.33).
Edit: transplant C_3.co.weight and C_3.co.bias from the parent kraken-german-xix-v1 (same shapes, input/output channel layout unchanged; codec growth affects only the output layer). The child's other layers drifted ~0.13-0.3 from the parent, so the parent C_3 is not exact but is the best available estimate.
Predicted effect: CER from 1.00 to somewhere around 0.15-0.5 (the parent is "much worse than a healthy child" on this hand, and the downstream layers were tuned to the child's own C_3, so some mismatch remains).

Not applied: no other anomalies seen (output layer stats unremarkable, bias extremes plausible for rare rows). No scaling or output-layer edits without measurements.
