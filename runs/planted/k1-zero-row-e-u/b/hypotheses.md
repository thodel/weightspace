# Programmer (b), blind, budget 0
Model: claude-sonnet-5-5. Reasoning effort: low. Tools: Read and Bash (cat, mkdir) on the brief and the three pack files only. No model loaded, no pass run.

## H1: the output row of `e` is zeroed (weights all 0, norm 0.00; every other row has norm >= 7.7)
Evidence: describe.md lists `e` row norm 0.00. c0_practice.md: reference `e` becomes nothing 910 times and a, i, o, r, c, u about 380 times,
all of the top confusions are `e`. CER 0.162 is explained by this. Which character is missing from the diff is not informative (donor is unrelated).
Fix in principle: restore the row of `e` (and check its bias). Predicted effect of a true restoration: CER falls by roughly 0.08 to 0.10.
Why nothing is shipped: the parent is not available and the donor row for `e` reads a different hidden-state basis (different lineage, cosine of other
tensors about 0), so a transplanted row would be noise. A synthetic row (mean of other rows, or bias-only boost) cannot be checked at budget 0 and could make
`e` fire everywhere or not at all, so it could make the model worse. I could not tell whether the `e` bias is intact; it is not in the listed extremes.

## H2: anything else wrong
No evidence: all other tensors have ordinary scale, other row norms and biases look normal. 9 collapsed lines are small and are probably downstream of H1.

Decision: apply nothing; the program only freezes. Conclusion for the scorer: diagnosis `zeroed row e` confident, repair not possible blind.
