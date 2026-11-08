# Pre-registration — 20261107-planted-k2-negate-row-n-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k2-negate-row-n/b/program.py`, copied unchanged.

1. **Subject:** `k2-negate-row-n.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `n` (weights and bias) negated; `runs/planted/plant.json`) — tensor digest `2daff3da912c8f98552bdde696a9c565cfc6eff847652c81a836d5735660d372`, file sha256 `df3b7f92a0e392169f348393a5fdd0c8db10016ef7dad6edd6d6ed02ef347286`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

model: claude-sonnet-5-5; reasoning effort low; tools: Read, Bash (cat, py_compile) on the pack, brief and wsp/ops.py, wsp/session.py only. No passes.

H1 (fairly sure): output row `n` of O_18.lin.weight was sign-flipped. Evidence: diff shows `n` moved most, cosine -0.991 with the parent
(rel L2 2.04, i.e. ~|-x - x|/|x|), norm 7.96 -> 8.33 (a flip with ordinary fine-tuning drift), while every other row has cos > 0.88.
C0 reads: 7667 insertions of `n` (the flipped row fires where the blank/other classes should), n->m/r, e->n, i->n confusions, 79 collapsed lines,
CER 0.75, length ratio 1.61. Bias of `n` changed only -0.142, so the bias was not flipped.
Edit: O_18.lin.weight row cm['n'] multiplied by -1 (set_rows with the negated current row). Predicted: CER drops from 0.75 to roughly the healthy child level (~0.1-0.3), collapses vanish.

Not applied: the other large movers (« ¬ _ » …, cos ~0.9, bias -2.4) look like ordinary rare-character drift; C_3.co.bias rel change 1.09 is large
but it is a small tensor with cosine 0.68 and no sign pattern; I am not sure enough to touch it.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
