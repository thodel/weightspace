# Notes — 20261119-bern-combined-v1-g1-b10-1

## 2026-11-19 08:04:00 UTC — Round 0 (Pre-registration & Initial Analysis)

- **Model ID:** `gemini-4-argon`
- **Settings:** Provider default reasoning settings (`provider: "gemini"`, `model: "gemini-4-argon"`, `options: {}`), called via `wsp.programmer`, sandboxed CPU execution (`CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `unshare -Un`).
- **Files read:**
  - `runs/20261119-bern-combined-v1-g1-b10-1/PROGRAMMER_BRIEF.md`
  - `runs/20261119-bern-combined-v1-g1-b10-1/preregistration.md`, `notes.md`, `program.py`, `programmer.json`, `session.json`, `env.txt`
  - `runs/20261119-bern-combined-v1-g1-b10-1/pack/c0_c1.md`, `readings_budget0.md`, `per_line_edits.json`
  - `runs/20261119-bern-combined-v1-g1-b10-1/pack/describe_bern-combined-v1.md`, `describe_bern-hertwig-v2.md`, `describe_bern-schlussredaktion-v1.md`
  - `runs/20261119-bern-combined-v1-g1-b10-1/pack/diff_bern-hertwig-v2__bern-combined-v1.md`, `diff_bern-hertwig-v2__bern-schlussredaktion-v1.md`
  - `data/practice/bern-train-300.jsonl`, `data/practice/bern-train-300.meta.json`
  - `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`, `wsp/subject.py`, `wsp/align.py`, `wsp/metrics.py`, `wsp/data.py`
  - Read-only tensor & codec inspection of `models/kraken/kraken-bern-combined-v1.mlmodel`, `models/kraken/kraken-bern-hertwig-v2.mlmodel`, and `models/kraken/kraken-bern-schlussredaktion-v1.mlmodel`.

### Key Findings from Budget-0 Readings & Weight Geometry
1. **The defect in `kraken-bern-combined-v1`:**
   - `combined-v1` was fine-tuned from `hertwig-v2` on both hands, but 44 of its 50 Hertwig training pages came from the corrupted 25.10 export where images were paired with another page's transcription.
   - Consequently, on the 103 Hertwig practice lines (with correct 22.10 images), `combined-v1` degrades to CER `0.2050` compared to `0.1137` for its parent `hertwig-v2`. On the 197 Schlussredaktion practice lines, `combined-v1` achieves `0.1184` (vs `0.1129` for the clean sibling `schlussredaktion-v1`). Overall practice CER is `0.1456` (`1577 / 10829` edits; 29 collapsed lines).
2. **Weight-space comparison of $\Delta_{\text{sub}} = W_{\text{sub}} - W_{\text{par}}$ vs $\Delta_{\text{sib}} = W_{\text{sib}} - W_{\text{par}}$:**
   - In the output layer `O_18` (character-aligned across 267 shared rows), $\cos(\Delta_{\text{sub}}, \Delta_{\text{sib}}) = 0.8458$ for weights and $0.9647$ for biases, but $\Delta_{\text{sub}}$ has a $+0.1184$ blank-bias jump (`+0.8792 -> +0.9976`, compared to `+0.9356` in `sibling`) and $1.33\times$ stronger downward push on rare/umlaut character biases—a classic signature of CTC training under label noise.
   - In the deeper layers, $\cos(\Delta_{\text{sub}}, \Delta_{\text{sib}})$ drops from $0.58$–$0.81$ in `L_16` to $0.42$–$0.70$ in `L_14`, $0.34$–$0.67$ in `C_0`–`C_9`, and only $0.24$–$0.28$ in `L_12.layer.weight_ih/hh`. Thus, the mismatched Hertwig image–text pairs heavily perturbed `L_12` and the conv stack in a direction orthogonal to the clean Schlussredaktion adaptation.
3. **Plan for Round 0 (`program_01.py`, Passes 1–4):**
   - **Pass 1 (`C0`):** Verify baseline CER `0.1456` and collapsed count `29`.
   - **Pass 2 (`H1`):** Test the minimal 1-parameter output edit: restoring `O_18.lin.bias[0]` (CTC blank) to the parent's value `+0.8792`.
   - **Pass 3 (`H2`):** Test character-aligned output-layer repair (`O_18.lin.weight` and `O_18.lin.bias`) using the midpoint $0.5 W_{\text{par}} + 0.5 W_{\text{sib}}$.
   - **Pass 4 (`H3`):** Test full-network character-aligned midpoint merge ($0.5 W_{\text{par}} + 0.5 W_{\text{sib}}$) across all 34 tensors (`C_0`–`O_18`), leaving 6 passes for rounds 1+ to tune the interpolation coefficient $\alpha$, test layer-wise schedules, or blend with subject weights based on the empirical results.

## 2026-11-19 08:08:00 UTC — Round 1 (Analysis of Passes 1–4 & Plan for Passes 5–6)

- **Files read:** `log.jsonl`, `readings/pass_001.jsonl` through `readings/pass_004.jsonl`.
- **Results of `program_01.py` (Passes 1–4):**
  1. **Pass 1 (`C0`):** Reproduced baseline CER `0.1456` (`1577/10829` edits, Hertwig `0.2050`, Schlussredaktion `0.1184`, collapsed `29`, length ratio `0.9180`).
  2. **Pass 2 (`H1_blank_bias_parent`):** Lowered `O_18.lin.bias[0]` to parent's `+0.8792`. Reduced edits by 6 (`1577 -> 1571`, CER `0.1451`, $\Delta = -0.0006$, line 95 % CI `[-0.0014, +0.0003]`, collapsed `29`). Refused by the keep rule (CI crosses 0): shifting the single blank bias scalar without fixing the rest of the classifier rows has too small an effect.
  3. **Pass 3 (`H2_O18_parent_sibling_midpoint`):** Character-aligned output layer `O_18` midpoint ($0.5 W_{\text{par}} + 0.5 W_{\text{sib}}$). Reduced edits by 34 (`1577 -> 1543`, CER `0.1425`, $\Delta = -0.0031$, line 95 % CI `[-0.0061, -0.0002]`, page-clustered 95 % CI `[-0.0070, +0.0005]`), and reduced collapsed lines from `29` to `27` (Hertwig CER `0.2050 -> 0.1968`, Schlussredaktion CER `0.1184 -> 0.1176`). **KEPT.**
  4. **Pass 4 (`H3_full_parent_sibling_midpoint_a050`):** Full-network 50/50 merge ($0.5 W_{\text{par}} + 0.5 W_{\text{sib}}$) across all 34 tensors. Reduced collapsed lines sharply from `27` to `23` and improved Hertwig CER massively from `0.1968` (`670` edits) down to `0.1430` (`487` edits, saving 183 edits on Hertwig!), while Schlussredaktion CER rose from `0.1176` (`873` edits) to `0.1475` (`1095` edits, +222 edits), for a net total of `1582` edits (CER `0.1461`, $\Delta = +0.0036$ vs Pass 3). Refused by the keep rule and reverted to Pass 3.
- **Diagnosis of Pass 4 & Next Hypotheses (Passes 5–6 in `program_02.py`):**
  - **Pass 5 (`H4_blend_50sub_20par_30sib`):** Interpolate `C_0`–`L_16` 50 % from the Pass 3 kept state ($W_{\text{sub}}$) toward the clean donor merge ($0.40 W_{\text{par}} + 0.60 W_{\text{sib}}$), i.e., $W = 0.50 W_{\text{sub}} + 0.20 W_{\text{par}} + 0.30 W_{\text{sib}}$ for `C_0`–`L_16`, while setting `O_18` to $0.40 W_{\text{par}} + 0.60 W_{\text{sib}}$.
  - **Pass 6 (`H5_clean_donor_merge_a068`):** Test the pure clean donor interpolation $0.32 W_{\text{par}} + 0.68 W_{\text{sib}}$ across all 34 tensors.

## 2026-11-19 08:13:00 UTC — Round 2 (Analysis of Passes 5–6 & Plan for Passes 7–8)

- **Files read:** `log.jsonl`, `readings/pass_005.jsonl`, `readings/pass_006.jsonl`.
- **Results of `program_02.py` (Passes 5–6):**
  1. **Pass 5 (`H4_blend_50sub_20par_30sib`):** Three-way blend ($0.50 W_{\text{sub}} + 0.20 W_{\text{par}} + 0.30 W_{\text{sib}}$ in `C_0`–`L_16`, $0.40 W_{\text{par}} + 0.60 W_{\text{sib}}$ in `O_18`).
     - Overall practice CER dropped to **`0.1356`** (`1468 / 10829` edits, saving 109 edits vs C0 baseline and 75 edits vs Pass 3!), with **collapsed lines dropping from `27` down to `21`** (matching the parent `hertwig-v2`'s 21 collapsed lines, the lowest of all models).
     - Per hand: Hertwig CER dropped from `0.1968` (`670` edits, 9 collapsed) to **`0.1639`** (`558` edits, 7 collapsed); Schlussredaktion CER stayed low at **`0.1226`** (`910` edits, 14 collapsed, down from 18 collapsed in Pass 3).
     - Paired intervals:
       - `vs_kept` (vs Pass 3): $\Delta = -0.0069$, line 95 % CI `[-0.0131, -0.0014]`, page-clustered 95 % CI `[-0.0149, +0.0006]`.
       - `vs_baseline` (vs Pass 1 C0): $\Delta = -0.0101$, line 95 % CI `[-0.0177, -0.0033]`, **page-clustered 95 % CI `[-0.0198, -0.0017]`** (entirely below 0 on both line and page-clustered bootstrap!).
     - **KEPT.**
  2. **Pass 6 (`H5_clean_donor_merge_a068`):** Pure clean donor merge $0.32 W_{\text{par}} + 0.68 W_{\text{sib}}$ ($\alpha = 0.68$).
     - Achieved practice CER `0.1405` (`1522 / 10829` edits: Hertwig `0.1706` / `581` edits, Schlussredaktion `0.1268` / `941` edits, collapsed `23`). Reverted (`collapsed` rose `21 -> 23` and CER rose `+0.0050` vs `H4`).

## 2026-11-19 08:17:30 UTC — Round 3 (Analysis of Passes 7–8 & Final Passes 9–10 + Freeze)

- **Files read:** `log.jsonl`, `readings/pass_007.jsonl`, `readings/pass_008.jsonl`.
- **Results of `program_03.py` (Passes 7–8):**
  1. **Pass 7 (`H6_layer_graded_3way_blend`):** Layer-graded blend reducing subject weight in `C_0`–`C_9` (`0.40`) and `L_12` (`0.30`) and shifting toward sibling (`O_18` `0.38/0.62`, `L_14` `0.18/0.32`, `L_16` `0.15/0.30`).
     - Scored CER `0.1361` (`1474 / 10829` edits: Hertwig `0.1671` / `569` edits, Schlussredaktion `0.1219` / `905` edits, collapsed `23`).
     - Reverted because collapsed count rose from `21` to `23` (two short lines, Line 72 `Amfangs` and Line 251 `compt`, shortened from 6 and 3 characters in Pass 5 to 3 and 2 characters in Pass 7, crossing the `< 0.5 * len(ref)` collapse threshold) and overall CER was `+0.0006` above Pass 5.
     - *Lesson:* Co-varying the weights across layers (`0.30` in `L_12` vs `0.55` in `L_16` vs `0.00` in `O_18`) introduces slight cross-layer feature mismatch and reduces the parent (`hertwig-v2`) contribution in `L_14`/`L_16`/`O_18`, costing `+11` edits on Hertwig for only `-5` edits on Schlussredaktion.
  2. **Pass 8 (`H7_blank_bias_minus_005`):** Trimmed `O_18.lin.bias[0]` by `-0.050` (`+0.9130 -> +0.8630`) on top of Pass 5.
     - Lowered CER to **`0.1352`** (`1464 / 10829` edits: Hertwig `0.1639` / `558` edits, Schlussredaktion `0.1220` / `906` edits, **collapsed `21`**).
     - Paired interval vs Pass 5 (`vs_kept`): $\Delta = -0.0004$ (`-4` edits across 6 changed lines: 5 lines improved by `-1` edit, 1 line worsened by `+1` edit), with line 95 % CI `[-0.000835, 0.000000]`. Because the upper 97.5th percentile hit `0.0` (not strictly `< 0`), `s.keep()` refused it and restored Pass 5.
- **Final Hypotheses for Passes 9–10 (`program_04.py`, followed by `s.freeze()`):**
  - Why did Pass 5 (`0.50 sub + 0.20 par + 0.30 sib` in `C_0`–`L_16`, `0.40 par + 0.60 sib` in `O_18`) leave `O_18` at `0.00` subject weight while `C_0`–`L_16` had `0.50` subject weight?
    - In `O_18.lin.weight`, the 45 frequent character rows ($\ge 5$ occurrences in practice) of `subject` differ from `0.40 par + 0.60 sib` because `L_16` in Pass 5 still carries 50 % of `subject`'s recurrent features! Giving `O_18.lin.weight` the same three-way blend ($0.35 W_{\text{sub}} + 0.25 W_{\text{par}} + 0.40 W_{\text{sib}}$ across all 34 tensors, or keeping `O_18.lin.bias` clean while aligning `O_18.lin.weight` with the three-way feature blend and slightly increasing the parent weight from `0.20` to `0.25` to pull Hertwig CER down further) keeps every layer in consistent three-way alignment.
  - Let's test two clean, well-separated hypotheses in our remaining 2 passes:
    1. **Pass 9 (`H8_uniform_3way_35sub_25par_40sib_clean_bias`):** Uniform three-way blend $0.35 W_{\text{sub}} + 0.25 W_{\text{par}} + 0.40 W_{\text{sib}}$ across all 33 weight and recurrent bias tensors (`C_0`–`L_16` and character-aligned `O_18.lin.weight`), paired with clean donor bias $0.38 b_{\text{par}} + 0.62 b_{\text{sib}}$ in `O_18.lin.bias` with blank bias `O_18.lin.bias[0]` set to `+0.840` (slightly below parent's `+0.8792` to counter norm shrinkage from 3-way averaging).
       - *Prediction:* Shifting from `(0.50, 0.20, 0.30)` to `(0.35, 0.25, 0.40)` uniformly across all feature + classifier weights (without cross-layer weight mismatch) reduces the orthogonal label-noise residual in `C_0`–`L_16` from 50 % to 35 %, increases both parent (`0.20 -> 0.25`) and sibling (`0.30 -> 0.40`) clean signals, aligns `O_18.lin.weight` with `L_16`'s 35 % subject representation, and lowers practice CER by `-0.0015` to `-0.0040` vs Pass 5 (`1468` edits) with `collapsed <= 21`.
    2. **Pass 10 (`H9_norm_restored_and_blank_calibrated`):** When three weight tensors with cosines `< 1` are averaged ($w_s W_{\text{sub}} + w_p W_{\text{par}} + w_b W_{\text{sib}}$), the Frobenius norm of `L_12`, `L_14`, `L_16`, and `O_18.lin.weight` shrinks by $\sim 0.6\%\text{--}1.1\%$ relative to the weighted average of the donor norms ($w_s \|W_{\text{sub}}\| + w_p \|W_{\text{par}}\| + w_b \|W_{\text{sib}}\|$), slightly attenuating logit magnitudes relative to the unattenuated biases and favouring the positive-bias CTC blank (`+0.9130`). In Pass 10, on top of the best kept three-way state, we restore each weight tensor's Frobenius norm (and each `O_18.lin.weight` row norm) to the exact convex combination of the three models' norms, combined with a `-0.10` blank-bias reduction (`O_18.lin.bias[0]`), predicted to reduce deletion errors by `-15` to `-30` edits (`-0.0014` to `-0.0028` CER) without raising collapsed lines.
  - After Pass 10 and its keep/revert decision, `program_04.py` calls `s.freeze()`.
