# Pre-registration — 20261107-planted-k6-lstm16-x0.5-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k6-lstm16-x0.5/b/program.py`, copied unchanged.

1. **Subject:** `k6-lstm16-x0.5.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: last BiLSTM block (L_16, all weights and biases) x0.5; `runs/planted/plant.json`) — tensor digest `a339a718a392342d8f0c47c62505ed6b3f9c3f008b0ba5480471f534d1de55f1`, file sha256 `3c7da1f6f6ae42d70ccd49f5cea8b35dc781168bc70b46f557f7af9c3320fae5`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: claude-sonnet-5-5. Reasoning effort: low. Tools: Read, Bash (cat, ls, grep, py_compile only); no passes, no model loaded.

## H1 (applied): the top BiLSTM (L_16, both directions) has all its weights and biases scaled by about 0.5
Evidence: L_16 rms is about half of L_12/L_14 for every tensor (weight_ih 0.121 vs 0.22, weight_hh 0.154 vs 0.30,
bias_hh 0.42 vs 0.93). Against the parent, L_16 changes by rel L2 about 0.5 with cosine 0.999 on the biases (a pure rescale), while
L_12/L_14 move only 0.03-0.3. Practice reading: CER 0.199, length ratio 0.83, and the top confusions are all deletions
(character -> nothing), i.e. weakened emission confidence so the blank wins; fits an under-driven last recurrent layer.
Edit: scale all 8 L_16 tensors by 2.0 (bias_ih are zero, harmless). Predicted: CER falls substantially (0.20 -> roughly 0.10-0.14, toward the parent-child healthy level),
length ratio toward 1, collapsed lines fall from 24.
If wrong (a genuine fine-tune that just has smaller weights), CER would get worse; weak risk since the biases are exactly cosine 0.999.

## Not applied
- C_3.co.bias (rel 1.09, cos 0.68): large change but likely ordinary fine-tuning drift; no evidence of a defect.
- Output rows/biases: blank row and biases look normal; nothing to repair.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
