model id: claude-sonnet-5-5
settings: reasoning effort low (default, no extended thinking); no donor; pack_nodonor only (describe.md, c0_practice.md) plus brief and wsp API files
tools used: Read, Bash (ls, wc, cat of own pack, grep of wsp/ops.py and wsp/session.py, py_compile), Write. No model loaded, no pass run.

Readings: no non-finite values, no zeroed tensors, no dead layers. C0 CER 0.2508 (high), dominated by deletions: space -> nothing 319, e/./n -> nothing, 8 collapsed lines. Only visible norm anomaly in the decoder: dec.11.norm_ffn.weight (norm 67.5, std 0.111, mean about 2.1) and bias (3.5) against about 30.6-30.8 and 0.4-1.0 in dec.00-10. The encoder norm_ffn outliers (enc.00, enc.10) look like natural ViT patterns and are weaker candidates.

MASK = s.unseen_by() (all lines). Budget: C0 plus at most 5 edits = 6 of 10 passes.

H1 (scale, dec.11.norm_ffn.weight+bias x0.5): the final-layer FFN norm gain is about 2x too large (planted scaling), which inflates the last FFN update and hurts readout. Prediction: CER -1.0 to -5 points (0.25 -> 0.20-0.24), fewer deletions. If it fails (CER up or no change): scale is natural, go to H1b (weight only x0.5), then abandon; conclude the dec.11 norm is a genuine trained feature.
H1b (only if H1 not kept, scale weight only x0.7): prediction -0.5 to -1 points if the bias is natural and the weight is partly inflated. Failure: conclude no defect here.
H2 (scale, enc.10.norm_ffn.weight+bias x0.6): enc.10 norm_ffn is 1.7x the neighbours; if planted, encoder features are over-amplified. Prediction -0.5 to -3 points if planted; +2 or worse if natural. Failure: revert, encoder norms are natural.
H3 (scale, all dec.NN.cross.o.weight+bias x1.15): deletions of space and characters suggest a weak image signal into the decoder (diffuse attenuation of cross-attention output). Prediction -0.5 to -1.5 points, collapsed lines fewer; if CER rises, conclude cross-attention is not under-scaled.
H4 (scale, all enc.NN.ffn.out.weight+bias x1.1 would be risky; use enc.norm_final.weight x1.1): encoder output slightly under-scaled. Prediction within +-0.5 points; included as a cheap control; expect revert.
Overall expectation: it is plausible the model is healthy for the tested classes (a hard-to-read hand explains CER 0.25); then all edits are reverted and the result is the null (no kept edit).
