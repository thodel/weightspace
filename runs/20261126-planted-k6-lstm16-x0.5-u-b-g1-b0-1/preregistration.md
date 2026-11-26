# Pre-registration — 20261126-planted-k6-lstm16-x0.5-u-b-g1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Gemini 4 Argon rerun** of `runs/20261107-planted-k6-lstm16-x0.5-u-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/s07/b/program.py`, copied unchanged.

1. **Subject:** `k6-lstm16-x0.5.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: last BiLSTM block (L_16, all weights and biases) x0.5; `runs/planted/plant.json`) — tensor digest `a339a718a392342d8f0c47c62505ed6b3f9c3f008b0ba5480471f534d1de55f1`, file sha256 `3c7da1f6f6ae42d70ccd49f5cea8b35dc781168bc70b46f557f7af9c3320fae5`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: `kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model ID: `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
Settings: provider default reasoning settings; tools used: `read_file`, `list_dir`, `write_file` (with `py_compile`), `finish_round`.

# Hypotheses — s07, Task (b) (budget 0, blind programmer)

## Diagnosis: No defect present (negative control)

1. **Tensor statistics (`describe.md`)**:
   - All four convolutional layers (`C_0`, `C_3`, `C_6`, `C_9`) have normal weight RMS (`0.1841`–`0.2010`) and bias norms.
   - All three BiLSTM layers (`L_12`, `L_14`, `L_16`) have tightly matched forward and reverse norms across `weight_ih`, `weight_hh`, and `bias_hh` (`L_12` `weight_ih` L2 `194.25` vs `193.76`, `weight_hh` `119.24` vs `120.52`; `L_14` `weight_ih` `122.53` vs `123.48`, `weight_hh` `123.90` vs `119.70`; `L_16` `weight_ih` `68.42` vs `64.03`, `weight_hh` `61.77` vs `62.89`), with `bias_hh` cosine `~0.88–0.89` against the unrelated donor `kraken-bern-hertwig-v2`.
   - Output projection `O_18.lin.weight` (`[297, 400]`, L2 `264.33`, RMS `0.7669`, abs max `5.751`) and `O_18.lin.bias` (`[297]`, L2 `47.85`, RMS `2.7768`, abs max `7.845`) show normal distributions: blank row norm `4.637` and bias `+1.232` (only `+0.353` above the unrelated donor's blank bias), non-blank row norms `14.812 ± 4.069` (ranging smoothly from common characters `e` `7.57`, `r` `7.80`, `s` `8.05`, `t` `8.08`, `i` `8.17`, `n` `8.33`, `a` `8.33` up to `23.86`), and non-blank biases `-1.921 ± 2.010` (within `[-7.85, +0.40]`). No row or bias is zeroed, sign-flipped, scaled to an extreme, or corrupted by an outlier spike.

2. **Practice-set readings (`c0_practice.md`)**:
   - Baseline metrics (`CER 0.1993`, `WER 0.4910`, `24` collapsed lines, length ratio `0.828`, `2565` character edits) and top-10 confusions (` ` → `∅`: 301, `e` → `∅`: 251, `n` → `∅`: 170, `i` → `∅`: 127, `r` → `∅`: 111, `c` → `∅`: 85, `s` → `∅`: 85, `u` → `∅`: 85, `t` → `∅`: 80, `m` → `∅`: 79) reflect the unmodified baseline performance of the healthy Kurrent model on `blb-train-300`, where deletions on collapsed/difficult lines occur strictly in proportion to German character frequencies rather than from a character-specific or layer-specific defect.

## Planned Edits and Predicted Effect

- **Edit**: None (0 edits applied).
- **Predicted effect**: `ΔCER = 0.0000` (preserves the healthy model at `CER = 0.1993` without risking unmeasured degradation from blind edits or unrelated-donor transplants).


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF_UNRELATED.md`, its pack (`runs/planted-xv/g1/s07/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (7 API calls, 77646 input / 1674 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).
