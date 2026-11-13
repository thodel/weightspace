# Pre-registration — 20261107-planted-k4-blank-bias-plus2-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k4-blank-bias-plus2/a/program.py`, copied unchanged.

1. **Subject:** `k4-blank-bias-plus2.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: blank bias +2.0; `runs/planted/plant.json`) — tensor digest `339161afc6ffa62ee51b2d9e5ce3dd0bbd0c3f3cd7939c6300bfa46500dc993b`, file sha256 `2727c5266842c02e67e61b779bf0d900f981af6137e12447ed7f279c833c268c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration, programmer (a), budget 10
Model: claude-sonnet-5-5. Settings: default reasoning effort, tools: file read and shell only (no model loaded, no passes).

Evidence: practice CER 0.0789, 14 collapsed lines, length ratio 0.941, and the top confusions are all deletions (reference -> nothing: space 177, e 81, n 53, ...). That is a blank-too-strong signature. Blank bias is +3.23 and has moved +2.11 from the parent, while the blank row norm moved only 0.23 relative; the other rows' biases moved about -2 on the extreme rows too.

H1 (primary): the blank bias is inflated by about +2. Edit class: add, `O_18.lin.bias`, delta -2.1 at row 0. Predicted: CER 0.079 -> about 0.04 to 0.05 (-0.03 or more), fewer collapsed lines, length ratio towards 1. If it fails (CI not below 0): blank bias is not the cause or the shift is wrong in size; try smaller (-1.0) then conclude the deletions come from elsewhere.
H1b: if H1 is kept, a further -0.5 on the blank bias. Predicted: -0.002 to -0.005, may be noise. If it fails, the first step was near the optimum.
H1c: if H1 fails, -1.0 and then -3.0 on the blank bias (size mis-specified). Predicted -0.01 to -0.03 if the sign is right.
H2: C_3.co.bias is drifted (rel L2 1.09 vs parent, cosine 0.68, far out of line with the other tensors). Edit class: interpolate, `C_3.co.bias`, alpha 0.5 toward the parent. Predicted: small (|dCER| < 0.005); if worse, the child legitimately retrained that bias and it is not a defect.
H3 (null): there is no defect beyond H1. If H1 and H2 both fail, ship nothing further.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
