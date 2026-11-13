# Pre-registration — 20261107-planted-k11-lstm-all-x0.9-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k11-lstm-all-x0.9/b/program.py`, copied unchanged.

1. **Subject:** `k11-lstm-all-x0.9.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: diffuse: all three BiLSTM blocks (weights and biases) x0.9; `runs/planted/plant.json`) — tensor digest `b126e5d8410e8d8d4f17074561f6f94d07f8d67f05c7d686c5f3c9a9ad2a85fd`, file sha256 `4a7b301b0629c376bf631bceba679cc8b7ef7b10a4e85f7c19d8e6a30ec849c1`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: claude-sonnet-5-5. Settings: default reasoning effort (low), tools: Read/Bash (cat, py_compile only), no passes (budget 0).

H1 (applied): all three BiLSTMs (L_12, L_14, L_16; weights and bias_hh, both directions) were multiplied by one common factor, ~0.9 (or its
inverse). Evidence: LSTM bias_hh_l0 vs parent has rel L2 ~0.11 but cosine 0.999 (direction unchanged, size changed by ~11 %), in all six
bias tensors, whereas fine-tuning would barely move a bias. Weights show cos 0.95-0.98, consistent with ordinary drift plus the same scale. The
CER errors are mostly deletions (space, e, n, r -> nothing), i.e. blank wins too often: shrunk LSTM outputs weaken the character logits.
Edit: scale by 1/0.9 on the 24 non-zero LSTM tensors (bias_ih is zero, skipped). The program checks bias_hh norm against the parent
(donor, read-only) and only applies if the child/parent norm ratio is 0.87-0.93 in every one of the 6 bias_hh tensors; if the ratio is ~1.1 it applies 1/1.1 instead.
Predicted effect: CER 0.073 -> clearly lower (about 0.05-0.06), fewer deletions.
If wrong: no gain/worse; the guard avoids applying anything when the ratio is not near 0.9 or 1.1.

Not applied (not sure): C_3 bias moved 1.09 rel (cos 0.68), probably fine-tuning; output layer rows look ordinary; nothing else shows a clear fault.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
