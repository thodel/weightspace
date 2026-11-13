| pair | practice set | class | rows | median rel. move | median norm ratio | median bias Δ | norm up | bias down |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `cursive_inzigkofen` → `bern-hertwig-v1` | bern-hertwig-train-300 | absent (0) | 185 | 0.28 | 1.16 | -0.41 | 184 | 184 |
| `cursive_inzigkofen` → `bern-hertwig-v1` | bern-hertwig-train-300 | rare (1–9) | 33 | 0.11 | 1.00 | -0.02 | 18 | 17 |
| `cursive_inzigkofen` → `bern-hertwig-v1` | bern-hertwig-train-300 | frequent (≥10) | 43 | 0.14 | 0.98 | -0.00 | 13 | 22 |
| `cursive_inzigkofen` → `bern-hertwig-v2` | bern-hertwig-train-300 | absent (0) | 185 | 0.41 | 1.23 | -0.72 | 184 | 179 |
| `cursive_inzigkofen` → `bern-hertwig-v2` | bern-hertwig-train-300 | rare (1–9) | 33 | 0.24 | 1.04 | +0.02 | 24 | 15 |
| `cursive_inzigkofen` → `bern-hertwig-v2` | bern-hertwig-train-300 | frequent (≥10) | 43 | 0.25 | 1.01 | -0.00 | 24 | 23 |
| `bern-hertwig-v1` → `bern-hertwig-v2` | bern-hertwig-train-300 | absent (0) | 187 | 0.14 | 1.07 | -0.32 | 186 | 179 |
| `bern-hertwig-v1` → `bern-hertwig-v2` | bern-hertwig-train-300 | rare (1–9) | 35 | 0.17 | 1.03 | +0.04 | 29 | 12 |
| `bern-hertwig-v1` → `bern-hertwig-v2` | bern-hertwig-train-300 | frequent (≥10) | 44 | 0.17 | 1.03 | -0.00 | 34 | 24 |
| `bern-hertwig-v2` → `bern-schlussredaktion-v1` | bern-train-300 | absent (0) | 193 | 0.12 | 1.07 | -0.53 | 178 | 177 |
| `bern-hertwig-v2` → `bern-schlussredaktion-v1` | bern-train-300 | rare (1–9) | 31 | 0.15 | 1.02 | -0.01 | 21 | 21 |
| `bern-hertwig-v2` → `bern-schlussredaktion-v1` | bern-train-300 | frequent (≥10) | 42 | 0.16 | 0.98 | -0.02 | 12 | 29 |
| `bern-hertwig-v2` → `bern-combined-v1` | bern-train-300 | absent (0) | 193 | 0.14 | 1.08 | -0.74 | 176 | 173 |
| `bern-hertwig-v2` → `bern-combined-v1` | bern-train-300 | rare (1–9) | 31 | 0.17 | 1.02 | -0.03 | 19 | 20 |
| `bern-hertwig-v2` → `bern-combined-v1` | bern-train-300 | frequent (≥10) | 42 | 0.20 | 0.97 | -0.02 | 12 | 28 |
| `german-xix-v1` → `german-xix-blb-v1` | blb-train-300 | absent (0) | 134 | 0.17 | 1.02 | -0.75 | 102 | 100 |
| `german-xix-v1` → `german-xix-blb-v1` | blb-train-300 | rare (1–9) | 20 | 0.25 | 1.00 | +0.05 | 10 | 8 |
| `german-xix-v1` → `german-xix-blb-v1` | blb-train-300 | frequent (≥10) | 70 | 0.17 | 1.01 | +0.05 | 42 | 14 |

| pair | ρ(log count, rel. move) | ρ(log count, norm ratio) | ρ(log count, bias Δ) |
|---|---:|---:|---:|
| `cursive_inzigkofen` → `bern-hertwig-v1` | -0.64 (p=0.000) | -0.69 (p=0.000) | +0.75 (p=0.000) |
| `cursive_inzigkofen` → `bern-hertwig-v2` | -0.59 (p=0.000) | -0.67 (p=0.000) | +0.73 (p=0.000) |
| `bern-hertwig-v1` → `bern-hertwig-v2` | +0.35 (p=0.000) | -0.41 (p=0.000) | +0.70 (p=0.000) |
| `bern-hertwig-v2` → `bern-schlussredaktion-v1` | +0.36 (p=0.000) | -0.48 (p=0.000) | +0.57 (p=0.000) |
| `bern-hertwig-v2` → `bern-combined-v1` | +0.36 (p=0.000) | -0.56 (p=0.000) | +0.56 (p=0.000) |
| `german-xix-v1` → `german-xix-blb-v1` | +0.02 (p=0.823) | -0.17 (p=0.013) | +0.44 (p=0.000) |
