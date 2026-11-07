# Pre-registration — 20261107-planted-k11-lstm-all-x0.9-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k11-lstm-all-x0.9/a/program.py`, copied unchanged.

1. **Subject:** `k11-lstm-all-x0.9.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: diffuse: all three BiLSTM blocks (weights and biases) x0.9; `runs/planted/plant.json`) — tensor digest `b126e5d8410e8d8d4f17074561f6f94d07f8d67f05c7d686c5f3c9a9ad2a85fd`, file sha256 `4a7b301b0629c376bf631bceba679cc8b7ef7b10a4e85f7c19d8e6a30ec849c1`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration, programmer (a), budget 10
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file read, bash (py_compile only). No model loaded, no pass run.

Evidence: practice CER 0.0727 with 12 collapsed lines; errors are almost all deletions (space, e, n, r, i, s -> nothing), i.e. the
net emits blank too often. Blank bias +1.23 and blank row norm 4.6 look ordinary. Every LSTM tensor sits at a uniform relative distance
0.11-0.33 from the parent with high cosine (0.95-0.999), and bias_hh rel change is only 0.11 with cosine 0.999: consistent with a
uniform global gain on all LSTM tensors, not retraining. Weakened recurrent/input gain -> weaker, less confident logits -> blank wins.

H1 (edit class: scale, all 24 LSTM tensors incl. bias_hh, x1/0.9 = 1.111): the LSTMs were uniformly shrunk by ~0.9. Predict CER 0.0727 -> ~0.04-0.05
(down 0.02-0.03), collapsed lines fall. If it fails (CER up or interval contains 0): the shrink is not uniform or not in biases; go to H2.
H2 (scale, LSTM weight_ih and weight_hh only, x1.111, biases untouched): predict CER down 0.01-0.03. If it fails: gain is not the defect in these tensors; go to H4.
H3 (scale, kept state of H1/H2 further by x1.05): checks whether the optimum is beyond 1/0.9. Predict small change (+-0.003); conclude the correct factor was 1.111 if no gain, and revert.
H4 (add, output blank bias -0.5): blank over-firing; predict CER down 0.005-0.01. If it fails, blank bias is not the cause.
H5 (scale, output weights of non-blank rows x1.05 if H1/H2 failed; otherwise skip): compensates reduced logit margin; predict down 0.005. If it fails, stop.
H6 (interpolate, LSTM weights alpha 0.1 toward parent, only if H1 kept): predict neutral/worse, since the parent is poor on this hand; revert if not significant.
Not hypothesised: conv or output-layer damage; no evidence (C_3 bias is the largest convolution change, 1.09 rel, but its absolute norm is 0.1, so little function lies there).


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
