# Pre-registration — 20261107-planted-k9-noise-all-10pct-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k9-noise-all-10pct/a/program.py`, copied unchanged.

1. **Subject:** `k9-noise-all-10pct.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: diffuse: Gaussian noise on every weight tensor, 10 % of each tensor's norm; `runs/planted/plant.json`) — tensor digest `fce9af3f1f4418c4fae94ad695de17f982c7f128c3e84f7de67108e4fd545690`, file sha256 `299cfd10c352e26617380b0dc2a656233b1576497e5b65cc072018ce6a75426d`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: Read/Bash (file reads, py_compile only). Budget 10.

Evidence: 123/300 lines collapsed, length ratio 0.55, every top confusion is char -> nothing (CTC emits blank where text should be). Blank row is the weakest (norm 5.0) but has the only positive bias (+1.23) vs -1.9 mean for characters. Every tensor moved 0.15-0.3 rel L2 from parent (uniform, incl. LSTMs), C_3.co.bias 1.09 rel / cos 0.68.

H1 (set blank bias lower, add -1.5 to O_18.lin.bias[0]): blank over-dominant; CER 0.55 -> ~0.40 (-0.10 to -0.2). If it fails: blank bias is not the cause; the collapse is upstream.
H2 (second step, add -1.5 more, or -0.75 if H1 failed): CER another -0.03 to -0.08; fail = H1 already optimal.
H3 (set blank bias to mean of character biases would be too much; instead add +0.5 to all non-blank biases): CER -0.02; fail = bias is not the lever.
H4 (interpolate C_3.co.bias to parent, alpha 1.0): the one anomalously moved tensor is damaged; CER -0.02 to -0.05; fail = its drift is legitimate fine-tuning.
H5 (interpolate all conv tensors 25% toward parent): uniform noise partly averaged out; CER -0.02 to -0.05, or worse since parent is poorer on this hand; fail = drift is learned signal, noise is not removable this way.
H6 (same 25% for LSTM tensors): CER about -0.02; fail as H5.
H7 (scale blank row weight by 1.2 or 0.8 per direction suggested by remaining deletions): +/-0.02.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
