# Pre-registration — 20261107-planted-k5-output-weight-x2-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k5-output-weight-x2/a/program.py`, copied unchanged.

1. **Subject:** `k5-output-weight-x2.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output layer weight matrix x2 (bias unchanged); `runs/planted/plant.json`) — tensor digest `2985c3e3f813d9b43bc0dfc951852ac29bb071c03729a7ce67a3a03235ab5fbe`, file sha256 `0e7be185c0f5730a826662ff3087b16b829eba160c7d6bb82170133d0957060b`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: Read/Bash (file reading only, no model run).

Reading: CER 0.065 is not bad, but errors are almost all deletions (space 78, e 34, n 28, i 21, r 19 ... -> nothing), 8 collapsed lines.
Output rows of characters shared with the parent are ~2x the parent's norm (e.g. a 23.7 -> 47.7, rel change ~1.0, cosine 0.998, i.e. same direction, doubled length), while conv/LSTM tensors changed far less (cos 0.95-0.99). The blank row is weak (9.3 vs 29.6 mean) while biases are unchanged in scale. Suspect: O_18.lin.weight multiplied by about 2 (bias untouched), which shifts weight/bias balance and favours blank/under-reads.

H1 scale O_18.lin.weight x0.5 (all rows, bias fixed). Predict CER 0.065 -> ~0.04 or below (delta -0.02 to -0.03), collapsed 8 -> fewer. If it fails (CER rises or CI contains 0): the factor is wrong or weights are not the culprit; go to H2.
H2 If H1 kept: refine, scale again x0.85 (net 0.425); then if that is kept x0.85 again. Predict small gains (-0.002 to -0.005) or none; a failure means H1 factor was near the optimum.
H3 If H1 failed: scale x0.7. Predict -0.005..-0.015 if the defect is real but smaller than 2x. Failure -> no uniform scale defect.
H4 If H1 and H3 failed: scale x1.3 (the opposite direction). Predict rise; kept only if I read the direction wrong.
H5 Blank-row-specific: add +/- 0.5 to blank bias (O_18.lin.bias[0]), direction chosen from confusions: deletions dominating -> blank too strong -> bias -0.5. Predict -0.003..-0.01. Failure -> blank balance is fine.
H6 Weak blank row (norm 9.3): scale blank row x1.5 after H1 only if insertions (blank->char) dominate. Predict small. Failure -> leave.
Conclusion rule: if none of H1, H3 helps the readings show no output-scale defect; ship nothing.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
