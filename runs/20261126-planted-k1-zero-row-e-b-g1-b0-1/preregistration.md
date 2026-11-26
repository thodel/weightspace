# Pre-registration — 20261126-planted-k1-zero-row-e-b-g1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Gemini 4 Argon rerun** of `runs/20261107-planted-k1-zero-row-e-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/s06/b/program.py`, copied unchanged.

1. **Subject:** `k1-zero-row-e.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `e` (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `f15cb502574f8562a2c7f8fcd819163ba2d17150ceac5f78d7e2e793fe47a7a8`, file sha256 `560e51f8525f14df97da88e09a174a29187993a5adc74bfdce368345bc13ad3c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — pack s06, task (b) (budget 0)

- **Model ID:** `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
- **Settings:** provider default reasoning settings; tools used: `read_file`, `write_file`, `finish_round`

## Diagnosis from pack readings

1. **`describe.md`:** In `O_18.lin.weight` (297 × 400), the weakest row is `'e'` with L2 norm `0.00` (all other character rows have norms ≥ 7.68, with mean 14.787 ± 4.138; `<blank>` is 4.637).
2. **`diff_parent_to_subject.md`:** Comparing `kraken-german-xix-v1` (parent) to the subject on output rows aligned by character, row `'e'` has relative L2 change `1.000`, cosine similarity `0.000`, norm `7.64 → 0.00`, and bias Δ `-0.142`. All other shared output rows have cosine similarity ≥ 0.881 and healthy norms, and all conv/LSTM blocks show normal fine-tuning drift (cosines 0.945–0.999, except `C_3.co.bias` which has tiny norm 0.10).
3. **`c0_practice.md`:** Baseline C0 CER is `0.1623` (2,088 character edits across 12,867 reference characters). The top confusions are overwhelmingly dominated by missing or substituted `'e'`:
   - `e → ∅`: 910
   - `e → a`: 131
   - `e → i`: 90
   - `e → o`: 52
   - `e → r`: 36
   - `e → c`: 35
   - `e → u`: 32
   Total listed `'e'` confusions alone account for 1,286 of the 2,088 edits (~10.0 percentage points of CER). Because the weight row for `'e'` in `O_18.lin.weight` was zeroed out (and its bias potentially perturbed or left unable to fire without feature projections), the model cannot emit `'e'`.

## Hypothesis & Edit Plan

- **H1 (Output row transplant for `'e'` from parent):**
  - **Hypothesis:** The output projection row for character `'e'` in `O_18.lin.weight` was zeroed out (norm 7.64 → 0.00, cos 0.000) and its bias in `O_18.lin.bias` shifted by -0.142 relative to the parent model `kraken-german-xix-v1`. Because the recurrent backbone (`L_16`) remains closely aligned with the parent (`cos` > 0.946), transplanting the `'e'` row of `O_18.lin.weight` and the `'e'` entry of `O_18.lin.bias` from the parent model (`kraken-german-xix-v1`) into the subject's `'e'` row will restore the model's ability to recognize `'e'`.
  - **Edit:** `set_rows` on `["O_18.lin.weight", "O_18.lin.bias"]` at index `[cm["e"]]` using the donor values from `KrakenSubject(PARENT)` at index `[pm["e"]]`.
  - **Edit class:** Output-row donor transplant (`set_rows`).
  - **Predicted effect:** CER on the practice set will decrease by approximately **0.09 to 0.11** (from `0.1623` down to `~0.055–0.070`), eliminating the ~1,286 `'e'` deletion and substitution errors.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/g1/s06/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (7 API calls, 124137 input / 2020 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
