# Pre-registration — 20261107-planted-k6-lstm16-x0.5-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k6-lstm16-x0.5/a/program.py`, copied unchanged.

1. **Subject:** `k6-lstm16-x0.5.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: last BiLSTM block (L_16, all weights and biases) x0.5; `runs/planted/plant.json`) — tensor digest `a339a718a392342d8f0c47c62505ed6b3f9c3f008b0ba5480471f534d1de55f1`, file sha256 `3c7da1f6f6ae42d70ccd49f5cea8b35dc781168bc70b46f557f7af9c3320fae5`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration, programmer (a), budget 10
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file read and shell (py_compile only), no model loaded, no passes.

Evidence: C0 CER 0.199 with errors almost all deletions (reference char -> nothing, length ratio 0.83, 24 collapsed lines): the blank wins too often, i.e. logits are under-confident. In the diff, L_16 (all 8 tensors, including bias_hh) is at rel L2 0.39-0.55 vs the parent with cosine 0.95-0.999, and its rms is about half of L_12/L_14's (bias_hh rms 0.42 vs 0.92); bias_hh cosine 0.999 with rel 0.505 is the signature of a pure uniform scale by 0.5. The other blocks differ from the parent by 0.1-0.3 with no such signature.

| # | edit class | hypothesis | predicted CER change | if it fails |
|---|---|---|---|---|
| H1 | scale L_16 (all 8 tensors) x2.0 | A x0.5 shrink was planted on the last BiLSTM; it starves the output layer, so the blank wins and characters are deleted | 0.199 -> about 0.08-0.12 (delta -0.08 or more), deletions of ` `,e,n,i fall | If CER does not fall: no uniform scaling on L_16; try H1b |
| H1b | scale L_16 weights only (not biases) x2.0, tried only if H1 is refused | gate biases were not touched, only weights | delta -0.05 | scaling is not the defect; go to H3, H4 |
| H1c | scale L_16 by a further x1.15 or x0.9 on top of H1 (best of at most two) | the true factor is not exactly 2 because of fine-tuning drift | delta of -0.005 to -0.01, or nothing | stop refining; keep H1 as is |
| H3 | add to O_18.lin.bias[0] (blank) -0.5, then -1.0 if kept | residual blank over-preference after H1 | delta -0.005 to -0.02 | the blank bias is healthy |
| H4 | interpolate C_3.co.bias toward the parent (alpha 0.5) | C_3.co.bias is far from parent (rel 1.09, cos 0.68), possibly a planted noise; but its rms is tiny (0.018), so little effect | delta around 0 (|delta| < 0.003); expected refused | C_3 bias is natural drift; leave |

Conclusion rules: a defect is confirmed only when s.keep accepts (paired 95 % interval below 0). If the baseline-to-H1 gain is under 0.02 the L_16 scale hypothesis is rejected and the deletion pattern is attributed to something else (blank row or bias); no further guess is shipped.
Measure budget: C0, H1, [H1b], H1c (<=2), H3 (<=2), H4 = at most 8 passes plus spare.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
