# Pre-registration — 20261107-planted-k1-zero-row-e-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k1-zero-row-e/a/program.py`, copied unchanged.

1. **Subject:** `k1-zero-row-e.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `e` (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `f15cb502574f8562a2c7f8fcd819163ba2d17150ceac5f78d7e2e793fe47a7a8`, file sha256 `560e51f8525f14df97da88e09a174a29187993a5adc74bfdce368345bc13ad3c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration, programmer (a), budget 10
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file read, bash (cat, py_compile only). No model loaded, no passes used before the program.

Reading: the output row of `e` has norm 0.00 (child) vs 7.64 (parent); bias -0.142 vs parent. Practice confusions: e -> nothing 910, e -> a/i/o/r/c/u about 380 together; CER 0.162. Everything else (other rows, LSTM/conv drift from the parent of 0.13-0.3 rel L2) looks like ordinary fine-tuning drift. Suspected defect: row `e` of O_18.lin.weight zeroed (a planted defect), so the logit for `e` is only its bias and never beats blank.

H1 (set_rows on O_18.lin.weight and O_18.lin.bias, parent row of `e` by character): restore the parent's `e` row and bias. Predict CER 0.162 -> about 0.05-0.08 (drop of 0.08-0.11; `e` is ~17 % of German characters and its errors are about 1,250 of 2,088 edits). If it fails (CER not down, or collapse rises): the parent row does not live in the child's hidden basis well enough; go to H2.
H2 (set_rows, ridge-mapped parent row): the child's hidden space drifted from the parent's by about 0.2-0.3 rel L2; map the parent's `e` row through a minimum-norm ridge correction fitted on the 224 shared characters. Predict a further CER drop of 0.00-0.01 relative to H1 (or, if H1 was refused, a drop of 0.08-0.10 like H1). If it fails: the row is simply fine as the parent has it; the drift is not the limiting factor.
H3 (set_rows, scale of the restored `e` weight row by 1.15, then 0.87 if refused): the row's norm may be off after the transplant (neighbouring letters have norm about 8); predict +-0.002 CER, probably not kept (interval contains 0). Conclusion if refused: scale is fine.
H4 (add to `e` bias by +0.5, then -0.5 if refused): bias calibration of the restored row; predict +-0.003; probably refused. Conclusion if refused: nothing more to gain from the e row.
Not hypothesised: any other repair; the other changes versus the parent are ordinary fine-tuning. If H1 gives little (CER down less than 0.03), I will conclude `e` was not the only problem, but the budget allows no other edit class in this program beyond the fallbacks above.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
