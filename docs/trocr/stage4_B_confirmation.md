# Stage 4, subject B: confirmation-half read of the `cross.o` edit (25.-26.11.2026)

Rule and scope: amendment 6 of `docs/TROCR_STAGE4_PREREG.md` (written before the read). First and only read of the confirmation half: 12 objects, 1,125 lines of `blb-eval-hand`, each model read once, card 0. Δ = CER(model) − CER(unedited `trocr-kurrent`), points, page-clustered 95 % (2,000 resamples). Rows: `stage4_B_confirmation.jsonl`; script `scripts/trocr_stage4_confirm.py`.

| read | CER % | Δ vs subject | clustered 95 % |
|---|---:|---:|---|
| subject (baseline) | 29.83 | – | – |
| **edit** (`dec.NN.cross.o.{weight,bias}` × 1.15, all 12 layers; final state of P2) | 27.77 | **−2.06** | [−2.86, −1.31] |
| sham 0 | 29.95 | +0.12 | [−0.42, +0.66] |
| sham 1 | 29.99 | +0.16 | [−0.25, +0.56] |
| sham 2 | 30.19 | +0.36 | [−0.12, +0.89] |
| sham 3 | 29.97 | +0.15 | [−0.35, +0.60] |
| sham 4 | 29.89 | +0.07 | [−0.32, +0.43] |
| arm I, α 0.3 (donor interpolation) | 22.47 | −7.36 | [−9.19, −5.47] |

Edit minus arm I: +5.30 [+3.81, +6.71] (worse).

**Reading by the rule of amendment 6.** The edit **confirms**: interval below 0, and the point estimate is below all five shams (best sham +0.07). It does **not** beat arm I, which is far better (−7.36 against −2.06), so it stays "below arm I" and Gate 4 (section 7) is **not met** by it. The effect is real and donor-free (P3-Fable reached the same state without the donor) but small next to a donor interpolation, which is itself a merge with the donor.

**Limits.** One edit, one subject, 12 objects. The subject's baseline CER is much higher on this half (29.83) than on discovery (23.26), and the edit's gain is larger here (−2.06 against −0.70 on discovery, −0.96 on practice), so the half-to-half difference is a reminder that object-level variation is large. Arm I is also larger here (−7.36 against −3.36). Whether "below arm I" counts as *merge* under section 7 is still left to Tobias. Subject A was not read.
