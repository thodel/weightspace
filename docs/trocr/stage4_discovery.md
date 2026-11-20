# Stage 4: arm I and the discovery-half reads (20.11.2026)

Ranking rule: amendment 2 of the prereg (written before the reads). Discovery half = 17 objects, 1,147 lines of `blb-eval-hand`; the confirmation half was not read. Δ = CER(model) − CER(unedited subject), same script, page-clustered 95 % interval (2,000 resamples). Rows with per-line edits: `stage4_armI.jsonl`, `stage4_discovery.jsonl`.

## Arm I on practice (global interpolation toward the donor, Δ points, clustered 95 %)

| α | A←XVI-XVII | B←IAM base |
|---:|---|---|
| 0.1 | 21.78 (-1.29) [-1.65, -0.95] | 25.86 (-2.36) [-2.77, -1.91] |
| 0.2 | 21.08 (-1.98) [-2.40, -1.60] | 24.34 (-3.88) [-4.49, -3.30] |
| 0.3 | 20.45 (-2.62) [-3.10, -2.19] | 23.51 (-4.70) [-5.36, -4.06] |
| 0.4 | 19.50 (-3.56) [-4.09, -3.07] | 23.64 (-4.57) [-5.29, -3.85] |
| 0.5 | 18.79 (-4.28) [-4.85, -3.77] | 24.41 (-3.81) [-4.57, -3.01] |
| 0.6 | 18.63 (-4.44) [-5.06, -3.88] | 26.11 (-2.10) [-2.95, -1.24] |
| 0.7 | 18.97 (-4.10) [-4.76, -3.52] | 29.31 (+1.09) [+0.07, +2.14] |
| 0.8 | 19.56 (-3.50) [-4.17, -2.91] | 34.60 (+6.38) [+5.04, +7.77] |
| 0.9 | 20.59 (-2.47) [-3.17, -1.83] | 44.60 (+16.38) [+14.56, +18.22] |

Best α by practice CER: A 0.6, B 0.3 (A's practice baseline 23.07, B's 28.22).

## Subject A (baseline discovery CER 19.12)

| read | practice Δ | discovery CER % | discovery Δ | clustered 95 % |
|---|---:|---:|---:|---|
| cell:dec_all@0.5 | -2.41 | 16.95 | -2.17 | [-2.66, -1.66] |
| cell:enc_all@0.5 | -1.90 | 17.23 | -1.89 | [-2.65, -1.17] |
| cell:dec_all@1.0 | -1.68 | 17.65 | -1.47 | [-2.14, -0.81] |
| cell:enc.04@1.0 | -0.89 | 18.56 | -0.56 | [-0.94, -0.21] |
| cell:enc.03@1.0 | -0.72 | 18.55 | -0.57 | [-0.91, -0.20] |
| cell:enc.06@1.0 | -0.66 | 18.63 | -0.49 | [-0.90, -0.10] |
| armI (α=0.6) | -4.44 | 15.03 | -4.09 | [-5.13, -3.10] |

## Subject B (baseline discovery CER 23.26)

| read | practice Δ | discovery CER % | discovery Δ | clustered 95 % |
|---|---:|---:|---:|---|
| cell:dec_all@0.5 | -1.80 | 22.29 | -0.97 | [-1.71, -0.21] |
| cell:dec.11@0.5 | -1.48 | 21.96 | -1.30 | [-1.76, -0.86] |
| cell:dec.11@1.0 | -1.41 | 22.61 | -0.65 | [-1.27, -0.01] |
| cell:enc_all@0.5 | -1.35 | 24.24 | +0.97 | [-0.40, +2.51] |
| cell:dec.10@0.5 | -0.57 | 22.95 | -0.31 | [-0.81, +0.17] |
| cell:dec.09@0.5 | -0.55 | 22.92 | -0.34 | [-0.75, +0.08] |
| armI (α=0.3) | -4.70 | 19.90 | -3.36 | [-4.51, -2.23] |
