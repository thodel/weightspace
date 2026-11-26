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
