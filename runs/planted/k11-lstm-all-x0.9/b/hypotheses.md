Model: claude-sonnet-5-5. Settings: default reasoning effort (low), tools: Read/Bash (cat, py_compile only), no passes (budget 0).

H1 (applied): all three BiLSTMs (L_12, L_14, L_16; weights and bias_hh, both directions) were multiplied by one common factor, ~0.9 (or its
inverse). Evidence: LSTM bias_hh_l0 vs parent has rel L2 ~0.11 but cosine 0.999 (direction unchanged, size changed by ~11 %), in all six
bias tensors, whereas fine-tuning would barely move a bias. Weights show cos 0.95-0.98, consistent with ordinary drift plus the same scale. The
CER errors are mostly deletions (space, e, n, r -> nothing), i.e. blank wins too often: shrunk LSTM outputs weaken the character logits.
Edit: scale by 1/0.9 on the 24 non-zero LSTM tensors (bias_ih is zero, skipped). The program checks bias_hh norm against the parent
(donor, read-only) and only applies if the child/parent norm ratio is 0.87-0.93 in every one of the 6 bias_hh tensors; if the ratio is ~1.1 it applies 1/1.1 instead.
Predicted effect: CER 0.073 -> clearly lower (about 0.05-0.06), fewer deletions.
If wrong: no gain/worse; the guard avoids applying anything when the ratio is not near 0.9 or 1.1.

Not applied (not sure): C_3 bias moved 1.09 rel (cos 0.68), probably fine-tuning; output layer rows look ordinary; nothing else shows a clear fault.
