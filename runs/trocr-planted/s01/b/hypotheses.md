Model: claude-sonnet-5-5. Settings: low reasoning effort (default). Tools: Read and Bash (grep/awk/sort on pack files, py_compile only); no passes, program not run.

H1: `enc.06.attn.o.weight` and `enc.06.attn.o.bias` are all zero (norm ratio 0.0000, distance 2.0000 against the donor; all other tensors are within healthy range 0.003 to 0.23, no head or embedding-row outliers, no non-finite values). The practice CER of 0.33 with many pure deletions (space, e, n, r) fits a broken encoder layer.
Edit: transplant both tensors from the donor (a healthy child moves these ~0.1, so the donor is close to the original).
Predicted effect: large CER drop (several points to tens of points).
Not applied: dec.embed.tokens distance 0.527 is the normal ~0.5; the decoder FFN-out distances 0.2 to 0.23 are within the stated normal range.
