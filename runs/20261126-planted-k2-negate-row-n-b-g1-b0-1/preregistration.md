# Pre-registration — 20261126-planted-k2-negate-row-n-b-g1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Gemini 4 Argon rerun** of `runs/20261107-planted-k2-negate-row-n-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/s13/b/program.py`, copied unchanged.

1. **Subject:** `k2-negate-row-n.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `n` (weights and bias) negated; `runs/planted/plant.json`) — tensor digest `2daff3da912c8f98552bdde696a9c565cfc6eff847652c81a836d5735660d372`, file sha256 `df3b7f92a0e392169f348393a5fdd0c8db10016ef7dad6edd6d6ed02ef347286`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model ID: `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer, provider default reasoning settings)
Tools used: `read_file`, `write_file`, `finish_round`

# Hypotheses — s13 task (b) (budget 0)

## Hypothesis 1 (`H1`): Sign-flip repair of row `'n'` in `O_18.lin.weight`
- **Diagnosis:** In `diff_parent_to_subject.md`, the output projection row for `'n'` in `O_18.lin.weight` has cosine similarity `-0.991` against the parent row and relative L2 change `2.042`, while its norm (`7.96 -> 8.33`) and bias change (`-0.142`) match normal child fine-tuning (`sqrt(8.33^2 + 7.96^2 + 2*8.33*7.96*0.991) / 7.96 = 2.042`). This shows the child's trained weight row for `'n'` was negated (`w_n -> -w_n`). Because `'n'` normalmente has negative dot product with features at non-`'n'` positions and positive dot product at `'n'` positions, negating the row causes massive spurious `'n'` insertions (`∅ -> n`: 7,667 in `c0_practice.md`, length ratio 1.611, CER 0.7456) and misses true `'n'` (`n -> m`, `n -> ∅`, `n -> r`).
- **Edit:** `set_rows` on `O_18.lin.weight` at index `[cm["n"]]` with `values = -T["O_18.lin.weight"][[cm["n"]]]` (restoring the exact sign of the child's `'n'` output weight row).
- **Edit class:** `set_rows` (output-layer row sign restoration).
- **Predicted effect:** Eliminates the ~7,667 spurious `'n'` insertions and restores true `'n'` recognition, reducing practice CER from `0.7456` by approximately `0.65–0.70` (down to healthy child CER `~0.05–0.09`).


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/g1/s13/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (5 API calls, 57867 input / 1330 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
