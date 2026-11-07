# Pre-registration — 20261107-planted-k7-noise-lstm14-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k7-noise-lstm14/b/program.py`, copied unchanged.

1. **Subject:** `k7-noise-lstm14.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: Gaussian noise on the weights of BiLSTM L_14, norm 25 % of the block's; `runs/planted/plant.json`) — tensor digest `8caee20d9b623c22c56b655dafd5d95e8aa89e7703c7220073c79443973801bc`, file sha256 `4fe5cd0f681847910dbae3ee44dbe2ab2a2f538e6c93be363731499a65040bf8`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: claude-sonnet-5-5. Settings: default reasoning effort (low), tools: Read, Bash (cat, py_compile only). Budget 0, blind.

Readings: CER 0.0726, 10 collapsed lines, length ratio 0.971. The top confusions are almost all deletions
(reference char -> nothing: space 103, e 38, n 31, s 23, i 22, r 21 ...), i.e. the blank wins over real characters
where the logit margin is thin. That is the usual signature of a weak or noisy recurrent stack, and also of a plain
undertrained child. The output rows of the frequent characters are the weakest (e 7.6, r 7.8, s 8.1, ...) and the
blank row is weak too (4.6); that pattern is ordinary for frequent classes.

H1 (suspect, not fixable blind): noise or damage in a recurrent block, most likely L_14 (largest parent->child
rel L2 of the three LSTMs: weight_ih 0.41/0.40, hh 0.32/0.31 vs 0.30/0.21 in L_12 and 0.30/0.33/0.20/0.22 in L_16).
The excess is mild and a fine-tune could produce it, so nothing in the readings separates damage from training.
Repair would be interpolate toward the parent, but the parent is much worse on this hand, so the direction and
size of any gain are unknown. Not applied.

H2: the space row bias (-7.85, about 3 sd below the other rows) causes the 103 space deletions. Frequent classes
usually do not have such a bias, but a large negative bias with a compensating strong row can be a legitimate
trained state, and I cannot see the parent value. Not applied.

H3: C_3.co.bias moved 1.09 relative (cos 0.68) while its weight moved only 0.17. The bias norm is 0.10, so the
effect on activations is tiny; looks like small-number noise in relative terms. Not applied.

Conclusion: the readings are consistent with a mildly weak model; I see no defect I am fairly sure of. No edits.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
