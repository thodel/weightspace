# Notes — 20261107-kurrent-page-v2-b10-1

## Passes 1-2 (07.11.2026, evening)

- **Pass 1, baseline:** byte-identical to C0 on 64 of 64 pages. CER 0.4515, 9 pages at the cap.
- **Pass 2, H1:** direction 0 of `layers.31.self_attn.o_proj` was removed from `B`. The edit was **kept**.
  - CER 0.4515 → 0.3470; paired interval [−20.3, −2.4] points.
  - `truncated_cer` 0.2564 → 0.2341, `length_ratio` 1.226 → 1.126.
  - Pages at the cap 9 → 4, loops 12 → 8.
- **Against the prediction:** the cap and length predictions were met. CER fell
  much more than predicted, because five pages that ran to the cap now stop
  (−11,000 of −11,309 edits). The reading side improved too, which was not
  predicted (`truncated_cer` −2.2 points).
- **Per page:** 28 pages better, 16 worse (at most +115 edits), 20 identical.
  The gain is a stopping gain and rests on few pages, so the held-out set has to
  carry it.

## Passes 3-6 and the freeze (07./08.11.2026)

- **Pass 3, H1b:** refused. Against H1, +4.5 points [−2.4, +15.8], pages at the cap 4 → 6.
- **Pass 4, H2:** refused. Against H1, +5.6 points [+0.3, +13.5], pages at the cap 4 → 5.
  - The logged comparisons for passes 3 and 4 are wrong because of a harness
    bug, and the log's deviation note corrects them.
- **Passes 5-6, the dose of H1:** both refused.

  | dose | CER | at the cap |
  |---|---:|---:|
  | 0 | 0.4515 | 9 |
  | 0.5 | 0.3990 | 7 |
  | **1.0** | **0.3470** | **4** |
  | 1.5 | 0.3616 | 5 |

  The curve falls monotonically up to full removal and turns up beyond it, so
  the edit is a mechanism and not noise on five pages.
- **Freeze:** with H1 as the final state and 4 passes unused. The remaining
  lens candidates did not survive (H1b), and no other direction of the
  attention outputs comes near direction 0 of layer 31 (all |z| < 1.8 on
  `<|im_end|>`).
