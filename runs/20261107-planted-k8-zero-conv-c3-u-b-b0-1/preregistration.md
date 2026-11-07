# Pre-registration — 20261107-planted-k8-zero-conv-c3-u-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k8-zero-conv-c3-u/b/program.py`, copied unchanged.

1. **Subject:** `k8-zero-conv-c3.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: conv block C_3 (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `df5577f17d54eec6a967ac38de6e09b18aefd467b1b80041c119ece5631304db`, file sha256 `80df9c7b020b643618164105826e3d7ee088addaad390b2e24232a5e6aa4ebd4`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: `kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

model: claude-sonnet-5-5; reasoning effort: low; tools: Read and Bash (cat, py_compile) on the pack and brief only. Budget 0, no passes.

Note: the pack's diff file is named diff_donor_to_subject.md (unrelated donor), not diff_parent_to_subject.md.

H1 (confident): the second conv block `C_3.co.weight` and `C_3.co.bias` are exactly zero (L2 0.00, abs max 0, all 39936 + 32 entries).
Evidence: describe.md; the donor diff shows rel L2 1.0000 and cosine nan for both. After zero conv + ReLU the feature map is dead, so
everything downstream sees only constant biases. This matches C0: CER 1.0000, 300/300 lines collapsed, length ratio 0, every reference
character deleted (output is all blank).
Predicted effect if repaired with the true weights: CER from 1.00 to roughly the healthy value (unknown, likely well under 0.3).

H2 (not applied): replace C_3 by the donor's C_3 (transplant). The donor's C_3 encodes features for a different hand, and the subject's
C_6 expects its own parent's C_3 features, so the match is arbitrary. I expect at best garbage output, CER still near 1.0, possibly above 1.0
through insertions. Not fairly sure it helps, so not shipped.

H3 (rejected): other tensors look normal (conv rms ~0.19, LSTM and output-layer norms ordinary, blank row weakest as usual, bias_ih zero as by construction).
No other defect is visible, and the zero block alone explains the total collapse.

Decision: apply no edit. The defect is diagnosed (C_3 zeroed) but there is no healthy source for it: the parent is unavailable and the donor is
not a compatible substitute. Expected CER change of the shipped program: 0.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
