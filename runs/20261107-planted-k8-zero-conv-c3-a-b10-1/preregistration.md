# Pre-registration — 20261107-planted-k8-zero-conv-c3-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k8-zero-conv-c3/a/program.py`, copied unchanged.

1. **Subject:** `k8-zero-conv-c3.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: conv block C_3 (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `df5577f17d54eec6a967ac38de6e09b18aefd467b1b80041c119ece5631304db`, file sha256 `80df9c7b020b643618164105826e3d7ee088addaad390b2e24232a5e6aa4ebd4`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file read/shell only, no passes used while writing.

Reading: C_3.co.weight and bias are exactly zero (rel L2 vs parent 1.0), so all downstream features are bias-only constant; C0 CER 1.000, every line empty. Everything else sits 0.13-0.33 from the parent, i.e. ordinary fine-tuning drift.

- H1 (main): C_3 was zeroed. Edit class: transplant C_3.co.weight + C_3.co.bias from the parent (kraken-german-xix-v1). Predict CER from 1.00 to about 0.3-0.6 (child C_0/C_6 drifted from parent, so mismatch remains). If it fails (CER >= 0.9 or not kept): the zeroing is not the only fault or the parent donor's feature space is too far; conclude downstream convs must co-adapt, go to H2.
- H2: the child's C_6/C_9 co-adapted to a C_3 we cannot recover; moving them toward the parent restores a consistent conv stack. Edit class: interpolate C_6 (w,b) toward parent alpha 0.5, then C_9 likewise. Predict a further 0.05-0.2 absolute CER drop. If it fails (not kept): keep child C_6/C_9, the mismatch is tolerable.
- H3: if H2 helped with alpha 0.5, full transplant (alpha 1) of C_6 and C_9 helps more; predict a further 0.02-0.1 drop; if not kept, stay at 0.5.
- H4: C_3 donor magnitude is off for the child's C_6 input: scale C_3 weight by 0.85 or 1.15. Predict at most 0.02 change; likely not kept (interval contains 0). Concluding: leave as is.
- H5: if output is still heavily collapsed (collapsed > 0), blank bias is too high relative to the child's drift: add -0.5 to O_18.lin.bias row 0. Predict small (0.01-0.05) improvement if blanks dominate deletions; else revert.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
