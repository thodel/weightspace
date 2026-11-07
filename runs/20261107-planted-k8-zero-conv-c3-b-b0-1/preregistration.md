# Pre-registration — 20261107-planted-k8-zero-conv-c3-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k8-zero-conv-c3/b/program.py`, copied unchanged.

1. **Subject:** `k8-zero-conv-c3.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: conv block C_3 (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `df5577f17d54eec6a967ac38de6e09b18aefd467b1b80041c119ece5631304db`, file sha256 `80df9c7b020b643618164105826e3d7ee088addaad390b2e24232a5e6aa4ebd4`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: Read/Bash (pack files, brief, wsp/ops.py, wsp/session.py only). Budget 0, blind.

H1 (applied): `C_3.co.weight` and `C_3.co.bias` are all zero (L2 0.00; diff vs parent rel L2 exactly 1.0, cosine nan). The practice reading is CER 1.0 with every line collapsed to nothing, consistent with a dead second conv block (ReLU(0)=const, no signal reaches the LSTMs). All other tensors look like a normal fine-tune of the parent (rel L2 0.03-0.33).
Edit: transplant C_3.co.weight and C_3.co.bias from the parent kraken-german-xix-v1 (same shapes, input/output channel layout unchanged; codec growth affects only the output layer). The child's other layers drifted ~0.13-0.3 from the parent, so the parent C_3 is not exact but is the best available estimate.
Predicted effect: CER from 1.00 to somewhere around 0.15-0.5 (the parent is "much worse than a healthy child" on this hand, and the downstream layers were tuned to the child's own C_3, so some mismatch remains).

Not applied: no other anomalies seen (output layer stats unremarkable, bias extremes plausible for rare rows). No scaling or output-layer edits without measurements.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
