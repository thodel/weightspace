# C0 on the practice set (blb-val-300: handwritten lines from 7 objects the model was never trained on, 300 lines)

CER 0.1383, WER 0.3625, collapsed lines 2, length ratio 0.970, 1701 character edits.

Top-10 confusions (reference → hypothesis, ∅ = nothing):

| reference | hypothesis | count |
|---|---|---:|
| ` ` | `∅` | 78 |
| `e` | `∅` | 44 |
| `∅` | ` ` | 30 |
| `i` | `∅` | 29 |
| `n` | `∅` | 26 |
| `r` | `n` | 23 |
| `r` | `∅` | 23 |
| `t` | `∅` | 21 |
| `s` | `∅` | 21 |
| `r` | `e` | 19 |

Donor available: `kraken-german-xix-v1` (the parent; never trained on BLB data), CER 0.3266 on the same lines. `seen_by` is empty for every line.
