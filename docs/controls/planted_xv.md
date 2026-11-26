| corruption | damage (pts) | arm | Fable 5.1: held-out CER · recovered · class | Gemini 4 Argon: held-out CER · recovered · class | GPT-6.2 Astra: held-out CER · recovered · class |
|---|---:|---|---|---|---|
| k1-zero-row-e | +10.10 | (a) | 9.56 · 100 % · repaired | failed (program error) | 9.59 · 100 % · repaired |
| k1-zero-row-e | +10.10 | (b) | 9.76 · 98 % · repaired | 9.72 · 98 % · repaired | 9.76 · 98 % · repaired |
| k2-negate-row-n | +60.69 | (a) | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired |
| k2-negate-row-n | +60.69 | (b) | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired |
| k3-swap-rows-e-n | +18.20 | (a) | 9.57 · 100 % · repaired | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired |
| k3-swap-rows-e-n | +18.20 | (b) | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired |
| k4-blank-bias-plus2 | +1.30 | (a) | 9.53 · 102 % · repaired | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired |
| k4-blank-bias-plus2 | +1.30 | (b) | 10.86 · -0 % · not repaired | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired |
| k5-output-weight-x2 | -0.03 | (a) | 9.53 · -0 % · null corruption | 9.53 · -0 % · null corruption | 9.53 · -0 % · null corruption |
| k5-output-weight-x2 | -0.03 | (b) | 9.53 · -0 % · null corruption | 9.53 · -0 % · null corruption | 9.53 · -0 % · null corruption |
| k6-lstm16-x0.5 | +11.62 | (a) | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired |
| k6-lstm16-x0.5 | +11.62 | (b) | 9.54 · 100 % · repaired | 9.55 · 100 % · repaired | 9.55 · 100 % · repaired |
| k7-noise-lstm14 | +0.49 | (a) | 10.04 · 0 % · null corruption | 10.04 · 1 % · null corruption | 10.08 · -7 % · null corruption |
| k7-noise-lstm14 | +0.49 | (b) | 10.04 · -1 % · null corruption | 10.04 · 0 % · null corruption | 10.04 · 0 % · null corruption |
| k9-noise-all-10pct | +54.24 | (a) | 50.99 · 24 % · partly | failed (no program) | 10.88 · 98 % · repaired |
| k9-noise-all-10pct | +54.24 | (b) | 63.80 · 0 % · not repaired | failed (no program) | 63.80 · 0 % · not repaired |
| k10-noise-lstm14-60pct | +3.94 | (a) | 10.88 · 66 % · partly | 10.94 · 65 % · partly | 10.85 · 67 % · partly |
| k10-noise-lstm14-60pct | +3.94 | (b) | 10.84 · 67 % · partly | 11.44 · 52 % · partly | 10.88 · 66 % · partly |
| k11-lstm-all-x0.9 | +0.31 | (a) | 9.67 · 64 % · null corruption | 9.87 · 0 % · null corruption | 9.72 · 48 % · null corruption |
| k11-lstm-all-x0.9 | +0.31 | (b) | 9.87 · 0 % · null corruption | 9.87 · 0 % · null corruption | 9.87 · 0 % · null corruption |
| k8-zero-conv-c3 | +90.45 | (a) | 9.77 · 100 % · repaired | 9.77 · 100 % · repaired | 9.77 · 100 % · repaired |
| k8-zero-conv-c3 | +90.45 | (b) | 9.77 · 100 % · repaired | 9.77 · 100 % · repaired | 9.77 · 100 % · repaired |
| k1-zero-row-e-u | +10.10 | (a) | 19.66 · 0 % · not repaired | failed (no program) | 12.85 · 67 % · partly |
| k1-zero-row-e-u | +10.10 | (b) | 15.28 · 43 % · partly | 19.66 · 0 % · not repaired | 11.32 · 82 % · repaired |
| k6-lstm16-x0.5-u | +11.62 | (a) | 9.55 · 100 % · repaired | 16.05 · 44 % · partly | 9.52 · 100 % · repaired |
| k6-lstm16-x0.5-u | +11.62 | (b) | 9.55 · 100 % · repaired | 21.17 · 0 % · not repaired | 9.55 · 100 % · repaired |
| k8-zero-conv-c3-u | +90.45 | (a) | 86.07 · 15 % · not repaired | 89.92 · 11 % · not repaired | 60.28 · 44 % · partly |
| k8-zero-conv-c3-u | +90.45 | (b) | 100.00 · 0 % · not repaired | 99.45 · 1 % · not repaired | 97.75 · 2 % · not repaired |

| arm | programmer | repaired | partly | not repaired | null corruption | failed (program error) | failed (no program) |
|---|---|---:|---:|---:|---:|---:|---:|
| (a) | Fable 5.1 | 7 | 2 | 2 | 3 | 0 | 0 |
| (a) | Gemini 4 Argon | 5 | 2 | 1 | 3 | 1 | 2 |
| (a) | GPT-6.2 Astra | 8 | 3 | 0 | 3 | 0 | 0 |
| (b) | Fable 5.1 | 6 | 2 | 3 | 3 | 0 | 0 |
| (b) | Gemini 4 Argon | 6 | 1 | 3 | 3 | 0 | 1 |
| (b) | GPT-6.2 Astra | 8 | 1 | 2 | 3 | 0 | 0 |
