Model: claude-sonnet-5-5. Settings: default reasoning effort (low), tools: Read/Bash (file reading only), no model loaded, no passes (budget 0).

Readings: CER 0.12, 11 collapsed lines, top confusions are all deletions (space, e, n, r, i ... -> nothing).
Parent->child diff: L_14 weights moved far more than the neighbouring LSTMs (weight_ih rel L2 0.75 / cos 0.79 vs 0.30 for L_12 and L_16;
weight_hh 0.55 vs 0.20; both directions). L_14 weight rms is also inflated (0.26 / 0.34 vs 0.22 / 0.30). Pattern fits additive
noise on the whole of L_14's weight matrices, on top of ordinary fine-tuning drift. Biases of L_14 look normal.
(C_3.co.bias rel 1.09 is a tiny-norm vector, 0.10; not trusted as a defect. Output rows look unremarkable.)

H1: L_14 (weight_ih, weight_hh, both directions) carries added noise. Edit: interpolate those 4 tensors 75 % of the way toward the
parent's L_14 (alpha 0.75). Rationale: child = trained + noise (noise var ~0.48 of parent norm^2 for ih, ~0.26 for hh) while child-parent training
difference is small (~0.09 / ~0.04), so the optimal mix is ~0.85; 0.75 is a conservative choice and cuts expected distance from the
undamaged child by ~5x. Predicted: CER 0.12 -> about 0.06-0.08, fewer collapsed lines, fewer deletions. Risk: parent drift loses
some child-specific learning; still expected net gain. If wrong (L_14 was not noised), CER would rise somewhat.

Not applied: anything on C_3 bias, output layer, other LSTMs (no sign of damage).
