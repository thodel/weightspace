# Preregistration, bern-combined-v1, programmer a2 (claude-sonnet-5-5)

Settings: budget 10 passes (pass 1 = C0), keep rule = paired 95 % interval below 0, donor rule applied for the one donor edit (schlussredaktion-v1, mask unseen_by).
Basis (c0_practice.md): CER 0.1386, length ratio 0.949; the dominant errors are deletions (space 195, `/` 98, e 87, n 78, t 56, r 55 ...) against fewer insertions (space 99, `/` 57). The blank row has bias +0.998 against -1.58 for other rows: the model is over-eager to emit blank.

H1 (bias shift, O_18.lin.bias, blank): lowering the blank bias by 0.3 cuts deletions. Predicted CER -0.004 to -0.012 (to ~0.130). If it fails: deletions are not a calibration problem but a feature problem (the evidence is in the rows, not the bias); stop bias edits on the blank.
H2 (same, further -0.3, cumulative -0.6): only if H1 kept. Predicted a further -0.002 to -0.006, with diminishing returns and a risk of insertions. If it fails: optimum is near -0.3.
H3 (bias, space): raising the space bias by +0.3 trades 195 deletions against 99 insertions. Predicted -0.002 to -0.005. If it fails: space errors are a segmentation/context matter, not a prior one.
H4 (bias, `/`): `/` +0.3, deleted 98 vs inserted 57. Predicted -0.001 to -0.003; may be flat (the interval may contain 0). If it fails: `/` is bimodal (line-ending vs. word-separator use) and not fixable by a prior.
H5 (donor interpolation, all C_* and L_* tensors toward schlussredaktion-v1, alpha 0.25; O_18 skipped because codecs differ, rule_mask = unseen_by donor): the sibling shares the hertwig-v2 parent, so averaging is plausible and its unseen-line CER is the lower one of the two donors. Predicted -0.003 to -0.010; if it fails: the models are not linearly connected at this scale (or the sibling has no advantage on unseen lines); no further interpolation.
H6 (only if H5 kept): alpha 0.5 total. Predicted a further -0.002 to -0.005; if it fails, keep alpha 0.25.
H7 (final, bias refinement after the other edits): re-tune the blank bias by -0.15 on the final state, since the rows changed. Predicted -0.001 to -0.003.
Total expected gain if all hold: CER from 0.139 to about 0.12-0.128. No edits of the convolution stack on their own; no donor use for the biases.
