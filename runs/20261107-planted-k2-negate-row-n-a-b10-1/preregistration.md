# Pre-registration — 20261107-planted-k2-negate-row-n-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k2-negate-row-n/a/program.py`, copied unchanged.

1. **Subject:** `k2-negate-row-n.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `n` (weights and bias) negated; `runs/planted/plant.json`) — tensor digest `2daff3da912c8f98552bdde696a9c565cfc6eff847652c81a836d5735660d372`, file sha256 `df3b7f92a0e392169f348393a5fdd0c8db10016ef7dad6edd6d6ed02ef347286`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5. Settings: reasoning effort low, tools: file read/shell only (no model loaded, no passes used before writing).

Evidence: C0 CER 0.7456, 79 collapsed lines, length ratio 1.61; 7667 insertions of `n` (of 9593 edits). Diff to parent: output row `n` moved rel L2 2.04, cosine -0.991 (all other rows cos > 0.88, blank 0.2). The row norm is unchanged (7.96 -> 8.33): sign flip of the row `n` in O_18.lin.weight. Blank row bias is fine; `n` bias moved only -0.142.

H1 (set_rows / sign flip): negate O_18.lin.weight[n]. Predicted CER 0.7456 -> about 0.25-0.40 (drop > 0.30, collapsed lines 79 -> < 10, `∅->n` insertions nearly gone). If it fails (CER drop < 0.05): the flip is not the cause of the insertions; revert, and next suspect the `n` bias and C_3 (H2, H3).
H2 (set_rows, bias, donor): set O_18.lin.bias[n] to the parent's value (small shift, 0.14). Predicted change -0.00 to -0.01 CER; probably not significant, so the keep rule will likely refuse it. If refused: the bias was untouched, fine.
H3 (interpolate, C_3.co.bias toward parent, alpha 0.5): C_3 bias has rel change 1.09 (cos 0.68) vs 0.15-0.33 elsewhere; this may be a second defect (or just fine-tuning drift). Predicted change: 0 to +0.02 (mostly harmful); run only if CER after H1 is still > 0.30. If it fails/worsens: it is training drift, leave it.
H4 (set_rows, donor transplant of row n from parent at alpha 1, instead of negation) only if H1 is kept but `n` confusions (e->n, i->n, n->m) remain large: predicted CER -0.01 to -0.03. If it fails: the child's row is fine, remaining error is the hand domain, stop.
No further edits: the rest of the child differs from the parent by ordinary fine-tuning.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
