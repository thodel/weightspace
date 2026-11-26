# Pre-registration — 20261126-planted-k3-swap-rows-e-n-b-g1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Gemini 4 Argon rerun** of `runs/20261107-planted-k3-swap-rows-e-n-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/s12/b/program.py`, copied unchanged.

1. **Subject:** `k3-swap-rows-e-n.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output rows of `e` and `n` swapped (weights and bias); `runs/planted/plant.json`) — tensor digest `e04ecfb521c6e879805ac6017fbecdaa4545f1e7e315de4804ea722f62b5ba9c`, file sha256 `c0bc09e5c852501c31d1fd20b48a9a2495071d13fa61edaf919573fb89b580e3`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model ID: `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer, provider default reasoning settings)
Tools used: `read_file`, `write_file`, `finish_round`
Task: pack `s12`, task (b) (budget 0, blind programmer)

# Hypotheses

## H1 — Output-layer row swap between `e` and `n` (`O_18.lin.weight` and `O_18.lin.bias`)

- **Diagnosis from pack readings:**
  1. In `diff_parent_to_subject.md`, all convolutional (`C_0`–`C_9`) and recurrent (`L_12`–`L_16`) tensors show normal fine-tuning drift relative to the parent `kraken-german-xix-v1` (cosines 0.945–0.9999 across all major weights; `C_3.co.bias` has L2 0.10 so its relative L2 1.087 reflects small denominator norm).
  2. In the output layer `O_18.lin` (mean relative change 0.2019 across 225 shared rows), exactly two rows stand out as extreme directional outliers against the parent:
     - `e`: rel L2 = 1.189, cosine = 0.356, norm parent → child = 7.64 → 8.33, bias Δ = -0.039
     - `n`: rel L2 = 1.103, cosine = 0.361, norm parent → child = 7.96 → 7.57, bias Δ = +0.180
     Notice that the child's `e` norm (8.33) matches the parent's `n` norm (7.96) and the child's `n` norm (7.57) matches the parent's `e` norm (7.64), while their cosines with the opposite parent rows are symmetric (~0.36).
  3. In `c0_practice.md`, baseline C0 CER is 0.2443 (3,144 character edits on 12,867 reference characters), and the top two confusions are symmetric substitutions between `e` and `n`:
     - `e` → `n`: 1,364 occurrences
     - `n` → `e`: 837 occurrences
     Together these 2,201 confusions account for 70.0% of all character errors on the practice set.
- **Edit class:** `set_rows` (intra-subject row swap of the `e` and `n` rows in `O_18.lin.weight` and `O_18.lin.bias`).
- **Edit specification:** Swap row `cm["e"]` and row `cm["n"]` in `O_18.lin.weight` and `O_18.lin.bias` using `s.edit("set_rows", ["O_18.lin.weight", "O_18.lin.bias"], ...)`.
- **Predicted effect:** Eliminates the ~2,201 `e` ↔ `n` substitution errors on the 300-line practice set (12,867 reference characters), reducing practice CER by approximately **-0.171** (from 0.2443 down to ~0.073).


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/g1/s12/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (6 API calls, 122652 input / 1828 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
