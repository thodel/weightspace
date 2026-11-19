Model id: claude-sonnet-5-5. Settings: default/low reasoning effort, budget 0 (blind, with donor). Tools: Read, Bash (cat/awk/grep, py_compile only), Write.

H1 (apply): `dec.05.cross.o.weight` and `.bias` are all zero (norm ratio 0.0, rel. distance 2.0 to the donor), so decoder layer 5 cross-attention contributes nothing. Edit: transplant both tensors from the donor. Predicted: CER drops by several points from 0.232 (the confusions are dominated by dropped spaces and letters).
Not applied: every other tensor lies within the healthy-child range (matrix distances 0.08-0.24, norm ratios 1.0-1.06); no head outliers; tied embedding distance 0.53 is normal (~0.5), no zero or collapsed rows. No other defect is evident.
