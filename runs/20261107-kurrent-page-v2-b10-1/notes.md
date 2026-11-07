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
