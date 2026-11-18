Model id: claude-sonnet-5-5. Settings: default reasoning effort (low), no donor. Tools used: Read, Bash (shell, text parsing of describe.md only), Write. No model loaded, no program run.

Reading: C0 CER 0.2426, dominated by deletions (space -> nothing 385, e -> nothing 62), length ratio 0.968, 7 collapsed lines. No zero shares, no non-finite values, no all-zero tensors. Norm anomalies against the layer medians: enc.00 attn.v (21 vs 37), enc.11 norm_attn.weight (6.9 vs 15.6), enc.00 norm_ffn.weight (44 vs 24), dec.11 norm_ffn weight/bias (67/3.5 vs 31/0.6). Many others (enc.09-11, dec.10-11 k bias) look like natural depth trends. Evidence is weak; the model may be healthy or lightly damaged. Each edit is kept only by the harness rule.

H1 scale enc.00.attn.v.weight x1.5 (norm 21 looks shrunk by ~0.6-0.7). Predicted CER -0.5 to -3 points if a scale defect; else +0.5 or worse. If it fails: enc.00 v is natural.
H2 scale enc.11.norm_attn.weight x1.6 (norm 0.44 of median). Predicted -0.5 to -3 points if defect; failure: last-layer norm is natural.
H3 scale dec.11.norm_ffn.weight x0.7 (norm 2.2x median). Predicted -0.5 to -2 points; failure: large last-layer norm is natural.
H4 scale all dec.NN.cross.o.weight x1.15 (deletions suggest too weak a read of the image). Predicted -0.5 to -2 points; failure: no global attenuation of the cross path.
H5 scale all enc.NN.ffn.out.weight and bias x1.05 (mild global encoder gain). Predicted within +-0.5 points; probably undone. Conclusion on failure: no global encoder gain defect.
H6 scale enc.norm_final.weight x1.1. Predicted within +-0.5; failure: final encoder scale fine.
Branching: after each measure, abandon if m.collapsed rises by more than 3 lines. Overall expectation: at most 1 kept edit; if none is kept, report no detectable defect.
