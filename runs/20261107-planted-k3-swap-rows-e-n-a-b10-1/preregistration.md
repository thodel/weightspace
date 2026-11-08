# Pre-registration — 20261107-planted-k3-swap-rows-e-n-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k3-swap-rows-e-n/a/program.py`, copied unchanged.

1. **Subject:** `k3-swap-rows-e-n.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output rows of `e` and `n` swapped (weights and bias); `runs/planted/plant.json`) — tensor digest `e04ecfb521c6e879805ac6017fbecdaa4545f1e7e315de4804ea722f62b5ba9c`, file sha256 `c0bc09e5c852501c31d1fd20b48a9a2495071d13fa61edaf919573fb89b580e3`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1` (file kraken-german-xix-v1.mlmodel), as lineage donor only; it was not trained on the practice lines' BLB eval side but **was not trained on BLB at all**, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration (programmer a, budget 10)
Model: claude-sonnet-5-5; reasoning effort low; tools: Read/Bash on the pack and wsp API only, no passes run.

Evidence: C0 CER 0.2443; top confusions are e->n (1364) and n->e (837), nothing else large (the next is 98). The output rows of `e` and `n`
have moved most from the parent (rel L2 1.19 and 1.10, cosine 0.36), every other row <= 0.63. Reads like the rows (and biases) of e and n exchanged.

H1 (edit class: set_rows, output rows swapped): swap rows e and n of O_18.lin.weight and entries of O_18.lin.bias. Predicted CER 0.244 -> about 0.08-0.12
(drop of ~0.13-0.16; e->n and n->e counts fall by >90 %). If it fails (CER not down, or e/n confusions remain): the swap is not the defect or only part of it, go to H2.
H2 (set_rows, weight only): swap the weight rows but leave the biases (if the bias swap is the harmful part, e.g. biases were not part of the corruption). Predict CER 0.09-0.13.
  If it fails: rows were not simply exchanged; go to H3.
H3 (interpolate / set_rows from donor, parent rows by character): replace e and n rows (weight+bias) with parent rows pm['e'], pm['n']. Predict CER 0.12-0.20 (the parent is worse on this hand, but the e/n
  confusion is removed); keep only if the rule accepts. If it fails: the defect is not in the output layer rows; stop, nothing further is applied.
H4 (kept state only, no edit): a collapse check: if collapsed lines rise after an edit, revert regardless.
Residual: 9 collapsed lines and ~98 missing spaces are baseline noise from the finetune, not targeted (space bias -7.85 is as in other rare rows).


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
