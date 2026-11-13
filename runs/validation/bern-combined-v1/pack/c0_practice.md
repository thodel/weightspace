# C0 on the practice set (bern-val: the 17 pages the trainer reserved for validation of the subject; 6 Hertwig pages, 11 Schlussredaktion pages, 513 lines)

CER 0.1386, WER 0.4264, collapsed lines 27, length ratio 0.949, 2408 character edits.

Top-10 confusions (reference → hypothesis, ∅ = nothing):

| reference | hypothesis | count |
|---|---|---:|
| ` ` | `∅` | 195 |
| `∅` | ` ` | 99 |
| `/` | `∅` | 98 |
| `e` | `∅` | 87 |
| `n` | `∅` | 78 |
| `∅` | `/` | 57 |
| `t` | `∅` | 56 |
| `r` | `∅` | 55 |
| `i` | `∅` | 50 |
| `s` | `∅` | 47 |

## Donors

- `kraken-bern-hertwig-v2`: CER 0.2103 on all 513 lines; trained on 212 of them (`seen_by`), CER 0.2766 on the 301 lines it was not trained on. For this donor use `rule_mask=s.unseen_by('kraken-bern-hertwig-v2')`.
- `kraken-bern-schlussredaktion-v1`: CER 0.1179 on all 513 lines; trained on 216 of them (`seen_by`), CER 0.1820 on the 297 lines it was not trained on. For this donor use `rule_mask=s.unseen_by('kraken-bern-schlussredaktion-v1')`.

Practice lines: 513; Hertwig 212, Schlussredaktion 301. Donor paths: `/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel`, `.../kraken-bern-schlussredaktion-v1.mlmodel`. Note the subject was initialised from hertwig-v2, which trained on every Hertwig page except its own 5 validation pages.
