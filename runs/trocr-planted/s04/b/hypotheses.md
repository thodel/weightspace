Model id: claude-sonnet-5-5. Settings: reasoning effort low (default), budget 0, blind, with donor. Tools used: Read, Bash (awk/grep/cat on own pack only), Write; no model loaded, program not run.

## H1 (shipped): all 12 encoder FFN output projections were scaled by about 0.7
Evidence: diff_parent_to_subject.md shows enc.00..enc.11 ffn.out.weight and ffn.out.bias, and only these, with norm ratio 0.700 to 0.704 (biases 0.6998 to 0.7002), cosine ~1.0 (biases exactly 1.0000), relative distance ~0.353 which is what a pure x0.7 gives. A healthy child moves matrices by 0.07 to 0.24 and keeps norm ratio near 1. CER 0.27 is high.
Edit: scale enc.NN.ffn.out.{weight,bias}, NN=00..11, by factor 1/0.7 = 1.428571.
Predicted effect: CER clearly down (several points, perhaps toward the 0.15 to 0.20 range).

## Not shipped
- dec.embed.tokens (rel. distance 0.53, cosine 0.86): matches the normal ~0.5 step of a healthy child; no outlier rows, no head outliers.
- decoder ffn.out weights 0.2 (norm ratio ~1.02) and dec.*.cross.o.bias norm ratio 1.06: within normal child range.
- No zeros, non-finite values or head-level anomalies found.
