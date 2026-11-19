Model id: claude-fable-5-2. Settings: low reasoning effort, single session, task (a) budget 10, WITH the donor.
Tools used: Read and Bash (cat/grep/awk on the three pack files, the brief, wsp/ops.py, wsp/session.py); py_compile. No model loaded, no pass run.

# Readings that the hypotheses rest on
- Practice C0: CER 0.2321, 5 collapsed lines, top confusion space -> nothing (381); otherwise diffuse deletions. No single-token pattern.
- describe.md: no zeros, no non-finite values anywhere. Layer norm of dec.08 is 161.8 against 140-143 for dec.07/09/10.
- diff to parent: the 10 weight matrices of dec.08 (self q,k,v,o; cross q,k,v,o; ffn in,out) have relative distance 0.38-0.77 and norm
  ratio 1.07-1.35, cosine 0.74-0.93; the same matrices in dec.07 and dec.09 sit at 0.08-0.19 and ratio 1.00-1.02. Biases and layer norms of
  dec.08 are normal. The implied excess std is the same in every one of the 10 matrices (about 0.020 absolute, e.g. ffn.out 0.0295 vs ~0.0218,
  cross.o 0.0394 vs ~0.034): this is additive i.i.d. noise of one sigma on the weight matrices of decoder layer 8.
- Everything else looks like a healthy child: tied embedding 0.527 (reference ~0.5), no outlier rows (min cosine 0.74, no zero rows), no
  outlier heads, encoder within range. dec.embed.pos identical share 0.47 and enc.01.attn.o.bias 0.0013 I read as unused positions / chance, not defects.

# Hypotheses (one line each)
- H1 [donor interpolation, dec.08 weight matrices]: dec.08's 10 weight matrices carry additive Gaussian noise (sigma ~0.020). Edit: interpolate
  each toward the parent with a per-tensor Wiener weight alpha = noise energy / total delta energy (healthy delta energy estimated from the
  mean rel. distance of the same matrix in dec.07 and dec.09; alpha comes out ~0.93-0.96). Prediction: CER falls by at least 5 points
  (0.232 -> 0.10-0.17), collapsed lines 5 -> 0-1, space deletions drop by more than half; kept by the rule. If it fails (no keep): the
  dec.08 excess is not the cause of the errors, or the layer's fine-tuned delta matters more than estimated; I then try H2 and, failing
  that, conclude that the defect is not repairable from the donor by these edits and leave the model unchanged.
- H2 [donor + spectral denoising, dec.08 weight matrices]: the healthy child delta of each matrix is partly low-rank and sticks out of the
  noise spectrum; W = parent + optimally shrunk SVD of (W_orig - parent) (Gavish-Donoho shrinker at the estimated sigma) recovers more of
  the child's own fine-tuning than H1's uniform shrink. Prediction: small further gain over H1, 0 to -1.0 CER points; about even odds that
  it clears the keep rule (-0.5 points, interval). If it fails: the healthy delta is not separable from noise spectrally; H1's state stands.
- H3 [donor transplant, only if neither H1 nor H2 was kept]: transplant the whole of dec.08's 10 matrices (alpha = 1). Prediction: within
  0.5 points of H1's result; if H1 failed this fails too, and I conclude the readings mislead and ship nothing.
- H4 [no edit]: nothing else is wrong (embedding, encoder, other decoder layers, layer norms are in the healthy range). No edit is made
  elsewhere; remaining passes stay unspent. If after the repair the CER is still above 0.20 I note it as an unexplained residual rather
  than fish with further edits.
