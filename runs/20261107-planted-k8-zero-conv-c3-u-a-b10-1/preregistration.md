# Pre-registration — 20261107-planted-k8-zero-conv-c3-u-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k8-zero-conv-c3-u/a/program.py`, copied unchanged.

1. **Subject:** `k8-zero-conv-c3.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: conv block C_3 (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `df5577f17d54eec6a967ac38de6e09b18aefd467b1b80041c119ece5631304db`, file sha256 `80df9c7b020b643618164105826e3d7ee088addaad390b2e24232a5e6aa4ebd4`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: `kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5; settings: low reasoning effort; tools: file read/write and shell only for reading pack and API, no pass run.

Reading: C_3.co.weight and C_3.co.bias are exactly zero (L2 0.00) while every other tensor looks trained. A zeroed conv gives
an all-zero feature map after ReLU, so C_6 sees only its bias and the output is all blank: CER 1.000, all 300 lines collapsed,
every reference character deleted. This fits the baseline exactly. Output layer, LSTMs and other convs look healthy.

H0 (baseline, pass 1): CER 1.000, collapsed 300. Predicted; if not, the defect reading is wrong and I stop editing.
H1 transplant C_3.co.weight = channel-identity (delta at kernel row 1, col 6, gain 1.0), bias stays 0: C_3 becomes a pass-through of
   C_0 features, so C_6 gets pooled C_0 features instead of the lost C_3 features. Predict CER falls from 1.00 to 0.85-0.99, lines un-collapse
   (collapsed < 300). If it fails (still collapsed): feature scale or content is too far from what C_6 expects; try other gains.
H2 identity gain 2.0: restores activation magnitude if the trained C_3 amplified. Predict CER within +-0.05 of H1; better if blank still dominates.
H3 identity gain 0.5: opposite scale. Predict worse than H1 if H1 kept, else a possible un-collapse.
H4 width-smoothed identity (5-tap box along width, best gain so far): C_3's 13-wide kernel likely acted as a smoothing/context filter. Predict CER -0.02 to -0.10 vs best.
H5 transplant donor (bern-hertwig-v2) C_3 weight+bias-free: different lineage features, expected useless; predict CER 1.00 or worse than identity; conclusion if fails: donor convs are not interchangeable.
H6 donor C_3 blended 0.5 with identity: predict no better than identity.
Conclusion rule: the keep rule decides; a variant is kept only if it significantly beats the last kept state. If no edit is kept, ship nothing and report that C_3 is not recoverable without training.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
