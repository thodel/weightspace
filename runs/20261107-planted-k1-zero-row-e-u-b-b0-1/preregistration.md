# Pre-registration — 20261107-planted-k1-zero-row-e-u-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k1-zero-row-e-u/b/program.py`, copied unchanged.

1. **Subject:** `k1-zero-row-e.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `e` (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `f15cb502574f8562a2c7f8fcd819163ba2d17150ceac5f78d7e2e793fe47a7a8`, file sha256 `560e51f8525f14df97da88e09a174a29187993a5adc74bfdce368345bc13ad3c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: `kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

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


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
