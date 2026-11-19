model id: claude-sonnet-5-5; settings: reasoning effort low, tools used: Read, Bash (grep/awk on pack files, py_compile only), Write. No passes (budget 0), donor available.

Readings: every weight matrix and norm vector is within the healthy-child range (0.08-0.23 / <0.13), no zero rows, no non-finite values, no outlying heads, no collapsed token rows. The single outlier is `dec.embed.norm.bias`: rel. distance 1.71, norm ratio 6.31, cosine 0.12 to the parent (healthy: <=0.13). C0 CER 0.298 with many deletions fits a decoder input offset.

H1: a large shift was added to `dec.embed.norm.bias`. Edit: transplant the parent's bias (the only edit). Predicted: CER clearly down (several points to >10 points, deletions of space/e/n/r fall). Residual: the parent's value differs from the healthy child's by ~0.1, minor.

Not touched: tied embedding (cosine 0.86, rel 0.53, as in a healthy child), everything else.
