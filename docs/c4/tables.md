### T1. Runs

| run | lrate | seed | epochs | val. acc. per epoch (1–5) | kept (best val.) | wall | card 0 free before | card 0 used, peak (all tenants) |
|---|---:|---:|---:|---|---|---:|---:|---:|
| A_seed1 | 1e-5 | 1 | 5 | 0.863 0.864 0.865 0.865 0.866 | 0.8657 (epoch 5) | 4 min 45 s | 34.3 GB | 20.8 GB |
| A_seed2 | 1e-5 | 2 | 5 | 0.863 0.864 0.865 0.866 0.866 | 0.8660 (epoch 5) | 2 min 41 s | 34.3 GB | 26.8 GB |
| A_seed3 | 1e-5 | 3 | 5 | 0.862 0.864 0.864 0.865 0.866 | 0.8657 (epoch 5) | 4 min 47 s | 34.4 GB | 23.4 GB |
| B_seed1 | 5e-5 | 1 | 5 | 0.867 0.870 0.872 0.873 0.874 | 0.8735 (epoch 5) | 4 min 47 s | 34.4 GB | 20.8 GB |
| B_seed2 | 5e-5 | 2 | 5 | 0.866 0.870 0.871 0.872 0.873 | 0.8726 (epoch 5) | 4 min 49 s | 34.3 GB | 26.8 GB |
| B_seed3 | 5e-5 | 3 | 5 | 0.867 0.869 0.871 0.871 0.872 | 0.8718 (epoch 5) | 4 min 49 s | 34.3 GB | 23.6 GB |

### T2. Held-out `bern-eval` (545 lines, 14 pages): each model scored once, CPU

Δ in points of CER; interval = line-level / page-clustered 95 %.

| model | CER all | CER Hertwig | CER Schluss. | collapsed | WER all | Δ vs combined-v1 | Δ vs interpolation edit | practice `bern-val` CER | collapsed (val) |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| combined-v1 | 19.33 | 26.69 | 13.48 | 34 | 51.89 | — | — | 13.86 | 27 |
| interp | 18.94 | 26.19 | 13.18 | 26 | 51.43 | -0.39 [-0.69, -0.08] / [-0.84, +0.04] | — | 12.39 | 20 |
| A_seed1 | 18.79 | 25.44 | 13.50 | 28 | 51.08 | -0.54 [-0.77, -0.31] / [-1.06, -0.09] | -0.15 [-0.42, +0.10] / [-0.63, +0.29] | 13.27 | 23 |
| A_seed2 | 18.77 | 25.33 | 13.55 | 28 | 51.11 | -0.56 [-0.79, -0.35] / [-1.09, -0.09] | -0.18 [-0.44, +0.08] / [-0.72, +0.29] | 13.26 | 23 |
| A_seed3 | 18.77 | 25.39 | 13.52 | 26 | 50.88 | -0.56 [-0.79, -0.34] / [-1.06, -0.10] | -0.17 [-0.45, +0.09] / [-0.68, +0.28] | 13.27 | 23 |
| B_seed1 | 18.34 | 24.49 | 13.46 | 27 | 49.93 | -0.99 [-1.34, -0.66] / [-1.72, -0.30] | -0.60 [-0.90, -0.30] / [-1.32, +0.04] | 12.53 | 18 |
| B_seed2 | 18.31 | 24.50 | 13.40 | 26 | 49.84 | -1.02 [-1.37, -0.69] / [-1.74, -0.35] | -0.63 [-0.92, -0.34] / [-1.34, -0.01] | 12.55 | 19 |
| B_seed3 | 18.35 | 24.39 | 13.55 | 24 | 50.13 | -0.98 [-1.34, -0.65] / [-1.70, -0.29] | -0.59 [-0.90, -0.30] / [-1.33, +0.09] | 12.72 | 19 |

### T3. Arms: mean over three seeds (points)

| arm | mean Δ vs combined-v1 (all) | per seed | Hertwig Δ | Schluss. Δ | mean collapsed | mean practice CER | reading rule (A.4) |
|---|---:|---|---:|---:|---:|---:|---|
| A (1e-5) | -0.55 (sd 0.01) | -0.54 -0.56 -0.56 | -1.30 | +0.04 | 27.3 | 13.27 | detectable gain; equal to the interpolation edit (−0.39) |
| B (5e-5) | -0.99 (sd 0.02) | -0.99 -1.02 -0.98 | -2.22 | -0.01 | 25.7 | 12.60 | detectable gain; larger than the interpolation edit (−0.39) |

combined-v1: held-out 19.33 %, collapsed 34, practice 13.86 %. Interpolation edit rebuilt and re-scored: 18.94 % (Δ -0.39), collapsed 26, practice 12.39 %.
