# Pre-registration — 20261107-planted-k5-output-weight-x2-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k5-output-weight-x2/b/program.py`, copied unchanged.

1. **Subject:** `k5-output-weight-x2.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output layer weight matrix x2 (bias unchanged); `runs/planted/plant.json`) — tensor digest `2985c3e3f813d9b43bc0dfc951852ac29bb071c03729a7ce67a3a03235ab5fbe`, file sha256 `0e7be185c0f5730a826662ff3087b16b829eba160c7d6bb82170133d0957060b`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: claude-sonnet-5-5. Settings: default reasoning effort; tools: Read, Bash (cat, py_compile) only. Budget 0, blind.

H1 (applied): `O_18.lin.weight` multiplied by 2 as a whole. Evidence: the rarely used inherited rows (a-acute, A-umlaut, s-caron, e-acute, ...)
have child/parent norm ratios of 2.014, 2.011, 2.013, 2.013 with cosine 0.998 to the parent, which is what a pure scaling gives and
what fine-tuning does not; the confusions are deletions of common characters (blank wins), as expected when logit weights are doubled
against an unscaled bias (space bias -7.85). Edit: scale O_18.lin.weight by 0.5. Predicted: CER falls, perhaps 0.065 -> 0.04.
If wrong (the weights were healthy), the cost is a flatter output; argmax shifts only through the bias balance.

H2 (not applied): conv C_3 bias moved 1.09 rel L2 from the parent. Probably legitimate fine-tuning (weights only moved 0.17); no evidence of defect, nothing applied.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
