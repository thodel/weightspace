Model id: claude-sonnet-5-5. Settings: reasoning effort low, tools used: Bash, Read, Write (only pack_nodonor/describe.md, c0_practice.md, the brief, wsp/ops.py and wsp/session.py signatures). No donor, MASK = s.unseen_by(). Budget 10, program uses at most 6 measures.

Readings: C0 CER 0.2736; dominant error is deleted spaces (401) and deletions of letters; no zeros, no non-finite tensors. Per-layer norms smooth except the outliers below.

H0 baseline C0. 
H1 (scale, dec.11.norm_ffn.weight+bias x0.5): the last-decoder-layer FFN LayerNorm has weight norm 67.5 (std 0.111) against ~31 (std ~0.02) in every other decoder layer, bias 3.5 vs ~0.6; looks like a scaled/perturbed LN feeding the readout. Predict CER -2 to -6 points if planted, ~0 or worse if natural. If it fails: try x0.75 once; if that fails too, conclude the gain is natural.
H2 (scale, enc.10.norm_ffn.weight+bias x0.6): encoder layer 10 FFN LN has norm 52.9/19.6 vs 20.5/15.0 in neighbours. Predict -1 to -3 points if planted; if it fails, conclude the late-encoder growth is natural (massive-activation channels), stop touching the encoder norms.
H3 (scale, enc.00.attn.v.weight x1.4): enc.00 attn.v norm 20.8 vs ~31 in layers 1-2 (shrunk?). Predict -0.5 to -1.5 points if shrunk; if it fails, concluded natural.
H4 (scale, enc.10.ffn.out.bias+weight x0.85) only if H2 kept: further relief of the same layer. Predict -0.5 point. Fails -> conclude H2 sufficed.
Null: if nothing is kept, the model is healthy or the defect is invisible in per-tensor norms (e.g. a few rows/heads); ship nothing.
