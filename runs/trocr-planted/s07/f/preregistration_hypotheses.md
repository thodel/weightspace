Model id: claude-fable-5-2. Settings: reasoning effort low; task (a), budget 10, WITH the donor; MASK = s.unseen_by("trocr-kurrent-XVI-XVII").
Tools used: Read and Bash (cat/grep/python text parsing of the three pack files, the brief, wsp/ops.py, wsp/session.py; py_compile). No model loaded, no pass run.

# Readings
- describe.md: `dec.05.cross.o.weight` (1024x1024) and `dec.05.cross.o.bias` are exactly zero (norm 0, share of zeros 1.0); the same tensors in the other 11 decoder layers have norm about 34 / 0.57. Nothing else is non-finite or zeroed; other norm outliers (enc.09-11, dec.11 norms) follow a smooth depth trend.
- diff_parent_to_subject.md: the same two tensors have relative distance 2.0 and norm ratio 0 to the parent; every other tensor lies in the healthy-child range (matrices 0.08-0.24, layer norms < 0.13, tied embedding 0.527 with no row below cosine 0.5, no outlier head). `dec.embed.pos` identical share 0.47 is taken as untrained positions, not a defect.
- c0_practice.md: CER 0.232, 9 collapsed lines, deletions dominate (space -> nothing 416): the decoder loses its place on the image, consistent with one cross-attention layer writing nothing into the residual stream.

# Hypotheses (one line each)
- H1 [transplant of a block from the donor]: the cross-attention output projection of decoder layer 5 was zeroed; transplanting the parent's `dec.05.cross.o.{weight,bias}` restores it. Prediction: practice CER falls from 0.232 by at least 0.10 (to 0.05-0.12), collapsed lines to <= 3, the space deletion count falls by more than half. If it fails (not kept): the parent's block is not compatible with the child's layer 5 inputs at full size; go to H1b.
- H1b [interpolate toward the donor, alpha 0.5; only if H1 is not kept]: half the parent's block. Prediction: CER falls by 0.03-0.10. If it also fails: the defect is not repairable by this block from the donor; stop, ship the model unchanged.
- H2 [scale, factor 1.05, on the transplanted pair; only if H1 is kept and CER is still > 0.08]: the healthy child's cross.o tensors are slightly larger than the parent's (bias norm ratio about 1.05-1.06, weight 1.004). Prediction: change within noise (|dCER| < 0.005), so the keep rule will most likely refuse it; if refused, conclude the transplant alone is the repair.
- H0 [no further edit]: no second defect is visible in the readings; no other edit is planned. Residual CER after H1 is attributed to the parent block missing the child's fine-tuning step (relative distance about 0.11), which cannot be recovered without training.
