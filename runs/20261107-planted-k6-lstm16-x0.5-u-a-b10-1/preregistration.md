# Pre-registration — 20261107-planted-k6-lstm16-x0.5-u-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k6-lstm16-x0.5-u/a/program.py`, copied unchanged.

1. **Subject:** `k6-lstm16-x0.5.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: last BiLSTM block (L_16, all weights and biases) x0.5; `runs/planted/plant.json`) — tensor digest `a339a718a392342d8f0c47c62505ed6b3f9c3f008b0ba5480471f534d1de55f1`, file sha256 `3c7da1f6f6ae42d70ccd49f5cea8b35dc781168bc70b46f557f7af9c3320fae5`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: `kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file reading and shell only, no model loaded, no passes used.

Budget: 1 baseline + up to 9 further measures. Baseline CER 0.1993 (24 collapsed, length ratio 0.83, deletions dominate).

H1. Block L_16 (last BiLSTM) has all weights and biases shrunk (rms 0.12/0.15/0.42 vs 0.22/0.30/0.92 in L_12/L_14, i.e. about half). Edit: scale, all 8 L_16 tensors, x2.0. Predict CER 0.20 -> ~0.12-0.15 (-0.05 or more). If it fails: the shrinkage is not uniform or is natural to the last layer; try H2.
H2. (if H1 fails) Only the input and recurrent weights are shrunk, not the biases: scale L_16 weight_ih/weight_hh x2, biases unchanged. Predict -0.03. If fails: try x1.5 on all; then give up on L_16.
H1b. (if H1 kept) Refine factor: try a further x1.15 on L_16 (total 2.3). Predict -0.005 to -0.01; if not kept, revert (factor 2 stays).
H3. Deletions of every frequent letter and space point to a too-strong blank. Edit: add, O_18.lin.bias[blank] -= 0.5. Predict -0.01 to -0.02. If fails: blank bias is not the issue, retry with -0.25 (then stop).
H4. Space row bias is -7.85 (extreme; space deleted 301 times). Edit: add +1.5 to bias of space. Predict -0.005 to -0.01. If fails: the space row is intrinsically that low; leave it.
Fallbacks never use the donor: its lineage differs and its LSTM/conv features are incompatible.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
