model id: claude-sonnet-5-5
settings: reasoning effort low; no donor (pack_nodonor only), MASK = s.unseen_by(); budget 10
tools used: Read (brief, describe.md, c0_practice.md), Bash (ls, grep of wsp/ops.py and wsp/session.py signatures, py_compile), Write. No model loaded, no program run.

Reading: C0 CER 0.298, 33 collapsed lines, length ratio 0.915; the errors are overwhelmingly deletions (space 492, e 140, n 84 ...), i.e. the model under-writes. No non-finite, no zeros, norms smooth across layers. Outliers in describe.md: dec.11.norm_ffn.weight (norm 67.5 vs ~30.6 in every other decoder layer, mean ~2.1 vs ~0.95) and bias (3.5 vs 0.4-1.0); enc.10.norm_ffn weight (52.9 vs 18-22) and bias (19.6 vs 5-7).

H0 baseline: s.measure("C0"). Expect CER ~0.298.
H1 (scale): dec.11.norm_ffn.{weight,bias} x0.5 (a planted gain ~x2.2 on the last decoder layer's FFN norm). Predict CER -0.02 to -0.08 if true; if false, CER rises +0.02 or more. If it fails: conclude the large last-layer gain is native, try x0.75 once only if CER change was within +-0.01, otherwise drop.
H2 (scale): enc.10.norm_ffn.{weight,bias} x0.5. Predict -0.01 to -0.05 if a planted gain; if fails (worse), the encoder outlier is native (massive-activation dims); drop.
H3 (scale): all dec.NN.cross.o.{weight,bias} x1.15 (decoder under-attends to the image, which would give deletions and short outputs). Predict -0.005 to -0.02 if cross-attn was damped; if fails, cross-attn is not globally damped.
H4 (scale): dec.NN.ffn.out.{weight,bias} for NN=08..11 x1.1 (late decoder FFN weak); predict -0.005 to -0.015; if fails, nothing wrong in late decoder FFN.
Overall: if none of H1-H4 is kept, conclude readings show no clearly localisable defect from norms alone and ship the unmodified model (the 0.298 CER may be the native quality of the hand/model). Kept edits are judged by the harness rule (95 % interval below -0.005); I also reject any edit raising collapsed lines by more than 5.
Passes planned: C0, H1, (H1 retry), H2, H3, H4 = at most 6 of 10.
