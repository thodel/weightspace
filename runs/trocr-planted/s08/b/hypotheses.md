Model id: claude-sonnet-5-5. Settings: default reasoning effort (low), tools: Read, Bash (awk/grep/py_compile only), Write. Task (b): budget 0, with donor. No passes, program not run.

H1. Layer dec.08 weight matrices (self q/k/v/o, cross q/k/v/o, ffn in/out) carry added noise. Evidence: rel. distance to the donor 0.38-0.77
(healthy 0.07-0.24), norm ratio 1.07-1.35 (noise grows the norm), cosine 0.74-0.93, while biases and norms of the same layer and all other
layers look healthy. Edit: interpolate those 10 weight matrices 85% toward the donor (alpha=0.85), keeping 15% of the subject (the expected
healthy drift is about 0.15 rel., noise is much larger, so the donor is the better estimate). Predicted: CER 0.232 drops, probably by 3-8 points.
Biases and norm vectors left alone.

Not edited: embedding (0.53 distance, about normal), other layers, encoder (all normal), no zero or non-finite shares, no heads flagged.
