# Pre-registration — 20261107-planted-k4-blank-bias-plus2-b-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b). Written before the first pass; the program is `runs/planted/k4-blank-bias-plus2/b/program.py`, copied unchanged.

1. **Subject:** `k4-blank-bias-plus2.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: blank bias +2.0; `runs/planted/plant.json`) — tensor digest `339161afc6ffa62ee51b2d9e5ce3dd0bbd0c3f3cd7939c6300bfa46500dc993b`, file sha256 `2727c5266842c02e67e61b779bf0d900f981af6137e12447ed7f279c833c268c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: claude-sonnet-5-5. Settings: low reasoning effort; tools used: Read, Bash (reading pack and API files only; no model loaded, no passes).

## H1 (applied): the CTC blank bias is inflated by about +2
Evidence: practice errors are almost all deletions (reference -> nothing: space 177, e 81, n 53, i 38, r 37 ...), 14 collapsed lines,
length ratio 0.941. Blank bias +3.232 against -1.9 +- 2.0 for other rows; parent-to-child blank bias shift +2.113, while every other
aligned row's bias moves little or negatively. A healthy fine-tune would move the blank bias by a few tenths, not 2.
Edit: add -2.0 to `O_18.lin.bias[0]` (class: output-bias add). Predicted: CER falls from 0.079 to roughly 0.03-0.05, deletions drop sharply, collapsed lines near 0.
If wrong (the shift were real training): CER would rise modestly with more insertions; -2.0 is the full anomaly, so some risk.

## Not applied
- C_3.co.bias rel L2 1.09 vs parent: large but its L2 is small (0.10) and C_3 weights moved moderately; no behavioural symptom. Not touched.
- Row norms, the new-character rows, LSTMs: nothing anomalous seen.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and the brief; budget 0.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
