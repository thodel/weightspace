# Pre-registration — 20261107-planted-k7-noise-lstm14-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k7-noise-lstm14/a/program.py`, copied unchanged.

1. **Subject:** `k7-noise-lstm14.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: Gaussian noise on the weights of BiLSTM L_14, norm 25 % of the block's; `runs/planted/plant.json`) — tensor digest `8caee20d9b623c22c56b655dafd5d95e8aa89e7703c7220073c79443973801bc`, file sha256 `4fe5cd0f681847910dbae3ee44dbe2ab2a2f538e6c93be363731499a65040bf8`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: Read, Bash (reading pack and wsp API files only, py_compile).

Readings: CER 0.0726; errors are overwhelmingly deletions (ref -> nothing: space 103, e 38, n 31, s 23 ...); 10 collapsed lines;
space bias -7.85 (extreme), blank bias +1.23 with weak blank row; L_16.weight_ih_l0 abs max 4.70 (about 2x the others);
L_14 input weights moved most from the parent among the LSTMs (rel L2 0.41 / 0.40 vs 0.20-0.33), which could be noise.
Baseline C0 is pass 1; passes 2-7 are edits, 3 left spare.

H1 (output bias, add): blank bias too high relative to characters -> deletions. Edit: O_18.lin.bias[blank] -= 0.6. Predict CER -0.003 to -0.010
   (0.0726 -> ~0.066). If it fails (CER up): blank is not over-confident; the deletions come from upstream features; revert, do not try larger.
H2 (output bias, add): space bias is implausibly low (-7.85), space deletions are the top error. Edit: bias[' '] += 1.5. Predict CER -0.004 to -0.012,
   space deletions fall, insertions of space rise a little. If it fails: space scarcity is real (spaces are narrow gaps), revert; try +0.7 only if CER change is within noise.
H3 (LSTM, low-rank denoise): L_14 weight_ih_l0 and _reverse carry additive isotropic noise; truncating the SVD to rank 120 of 400 removes noise
   energy and leaves the trained signal. Predict CER -0.005 to -0.03 if noise is planted there; +0.01 or worse if the matrices are genuinely full-rank.
   If it fails: L_14 input weights are not noise-damaged (or noise is not low-rank separable); fall back to H3b rank 250 once, then drop it.
H4 (LSTM, clip outliers): L_16.weight_ih_l0 has |w| up to 4.70, others ~2; clip at 10 robust sigmas of the tensor (median/MAD). Predict CER change -0.001 to -0.005;
   if flat or worse, the spike is genuine, revert.
H5 (conv, interpolate): C_3 bias moved most (rel 1.09, cos 0.68) from the parent; but biases are tiny (L2 0.10), so predict effect near 0 (|dCER| < 0.001).
   Not run unless passes are left over; if run and it fails, conclude it is ordinary fine-tuning drift.
Overall prediction: if the model is healthy, none of H1-H4 passes the paired 95% rule and the program freezes with zero kept edits.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
