# Stage 4 exchange grid on practice (20.11.2026)

`scripts/trocr_stage4_grid.py`, practice set `trocr-blb-train-1200-seen` (1,194 lines), card 0 free, 6.7 h of cell passes. Delta = CER(cell) − CER(recipient baseline), page-clustered 95 % interval (1,000 resamples). Practice only: nothing here is a held-out result and no cell was read held-out. Full rows with per-line edits: `stage4_grid.jsonl`.

Baselines (CER %): A 23.07, B 28.22, XVI-XVII (D) 21.90, IAM base (I) 64.41.

## Subject A receives from XVI-XVII

| cell | size | CER % | Δ points | clustered 95 % |
|---|---:|---:|---:|---|
| dec_all | 0.5 | 20.65 | -2.41 | [-2.82, -2.01] |
| enc_all | 0.5 | 21.17 | -1.90 | [-2.43, -1.46] |
| dec_all | 1.0 | 21.38 | -1.68 | [-2.17, -1.18] |
| enc.04 | 1.0 | 22.17 | -0.89 | [-1.28, -0.56] |
| enc.03 | 1.0 | 22.35 | -0.72 | [-1.12, -0.37] |
| enc.06 | 1.0 | 22.41 | -0.66 | [-1.05, -0.34] |
| enc.05 | 1.0 | 22.44 | -0.63 | [-1.02, -0.24] |
| enc.04 | 0.5 | 22.45 | -0.62 | [-0.89, -0.35] |

Cells with Δ ≤ −0.5 and interval below 0: 13 of 56. Worst: dec.00@1.0 -0.0, dec.01@0.5 +0.0, enc_all@1.0 +0.8

## Subject B receives from IAM base (its ancestor)

| cell | size | CER % | Δ points | clustered 95 % |
|---|---:|---:|---:|---|
| dec_all | 0.5 | 26.42 | -1.80 | [-2.39, -1.19] |
| dec.11 | 0.5 | 26.74 | -1.48 | [-1.92, -1.03] |
| dec.11 | 1.0 | 26.81 | -1.41 | [-1.93, -0.84] |
| enc_all | 0.5 | 26.87 | -1.35 | [-2.04, -0.62] |
| dec.10 | 0.5 | 27.65 | -0.57 | [-0.97, -0.14] |
| dec.09 | 0.5 | 27.67 | -0.55 | [-0.95, -0.16] |
| embed | 0.5 | 27.75 | -0.47 | [-0.88, -0.08] |
| norms | 0.5 | 27.77 | -0.44 | [-0.80, -0.09] |

Cells with Δ ≤ −0.5 and interval below 0: 6 of 56. Worst: enc.11@1.0 +9.3, enc_all@1.0 +49.7, dec_all@1.0 +61.9

## XVI-XVII receives from A (reverse)

| cell | size | CER % | Δ points | clustered 95 % |
|---|---:|---:|---:|---|
| enc_all | 0.5 | 19.04 | -2.87 | [-3.38, -2.37] |
| enc.05 | 1.0 | 20.64 | -1.26 | [-1.66, -0.87] |
| enc.07 | 1.0 | 20.79 | -1.12 | [-1.46, -0.78] |
| enc.06 | 1.0 | 20.95 | -0.96 | [-1.31, -0.62] |
| enc.05 | 0.5 | 20.97 | -0.93 | [-1.19, -0.65] |
| enc.08 | 1.0 | 21.08 | -0.83 | [-1.18, -0.51] |
| enc.06 | 0.5 | 21.21 | -0.70 | [-0.97, -0.42] |
| enc.07 | 0.5 | 21.33 | -0.57 | [-0.85, -0.31] |

Cells with Δ ≤ −0.5 and interval below 0: 10 of 56. Worst: enc.01@1.0 +0.2, dec.11@1.0 +0.7, dec_all@1.0 +2.0

## IAM base receives from B (reverse)

| cell | size | CER % | Δ points | clustered 95 % |
|---|---:|---:|---:|---|
| enc_all | 0.5 | 44.51 | -19.90 | [-21.72, -18.11] |
| dec_all | 0.5 | 53.28 | -11.13 | [-12.40, -9.91] |
| dec.11 | 0.5 | 59.06 | -5.36 | [-6.22, -4.50] |
| dec.10 | 1.0 | 59.40 | -5.01 | [-5.85, -4.15] |
| dec.09 | 1.0 | 59.59 | -4.82 | [-5.71, -3.99] |
| enc.06 | 0.5 | 60.68 | -3.73 | [-4.69, -2.78] |
| dec.09 | 0.5 | 60.85 | -3.56 | [-4.34, -2.81] |
| dec.10 | 0.5 | 60.92 | -3.49 | [-4.20, -2.81] |

Cells with Δ ≤ −0.5 and interval below 0: 17 of 56. Worst: enc_all@1.0 +30.8, enc.02@1.0 +34.8, enc.11@1.0 +39.3
