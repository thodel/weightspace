# Pre-registration — 20261107-planted-k1-zero-row-e-u-a-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a). Written before the first pass; the program is `runs/planted/k1-zero-row-e-u/a/program.py`, copied unchanged.

1. **Subject:** `k1-zero-row-e.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `e` (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `f15cb502574f8562a2c7f8fcd819163ba2d17150ceac5f78d7e2e793fe47a7a8`, file sha256 `560e51f8525f14df97da88e09a174a29187993a5adc74bfdce368345bc13ad3c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: `kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Preregistration, programmer (a), budget 10
Model: claude-sonnet-5-5. Settings: low reasoning effort; tools: file read and bash on the pack and the brief only, no model run.

Reading: output row `e` has norm 0.00 (all other rows 7.7 to 24), so the logit of `e` is its bias alone. `e` is the most frequent German letter; 910 of its occurrences are read as nothing, the rest as a/i/o/r/c/u. CER 0.162, length ratio 0.90. Conv and LSTM blocks look ordinary. The donor (other hand, other features) cannot supply a usable `e` row; the row has to be rebuilt from the model's own rows.

H1 (set_rows, O_18 row e): the `e` row was zeroed. Rebuild it as the unit direction of (count-weighted mean of rows a,i,o,r,c,u minus the mean of all other rows), norm 6. Predict CER 0.162 -> 0.10 to 0.13 (e deletions are 910 of 2088 edits; a rough row recovers part). If it fails (no CI below 0): the proxy direction is wrong; conclude the e feature is not in the span of confuser rows, try H2/H5.
H2 (set_rows, same row): add 0.5 x unit blank row to the direction (e frames are currently decided as blank). Predict a further 0.005 to 0.015 CER gain over H1, or a loss if e starts to eat blanks (collapse/insertions). If not kept: the pure confuser direction stays.
H3 (set_rows, norm x1.5, then x1.5 again): row norm 6 is below the typical frequent-letter norm of 8; louder e should recover more of the deletions. Predict 0.005 to 0.02 gain per step; stop when not kept (insertions of e rise).
H4 (add, O_18 bias at e, +1 then -1): the bias is whatever training left; with a rebuilt row a shift of 1 logit trades e deletions against e insertions. Predict at most 0.01 either way; failing means bias is fine.
H5 (set_rows, fallback if H1 not kept): e row = 0.6 x mean of rows a,o,u (e's most common wrong readings, similar frames). Predict 0.01 to 0.03 gain, weaker than H1; failing means leave the model with the zero row only and report the defect as found but not repaired by this budget.
No other defect is suspected from the readings (no collapse beyond 9 lines, conv/LSTM statistics regular), so no other edits are tried. Passes used at most 9 of 10.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Sonnet 5.5 (claude-sonnet-5-5) as a fresh subagent in Claude Code, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack and runs/planted/PROGRAMMER_BRIEF.md; wrote its program in advance, did not run it.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
