# Stage 4, subject A (`from-kurrent-v2`): programmer sessions, 20.11.2026

Practice set only for the programmers; frozen models read on the **discovery half** only; the confirmation half was not read. Δ = CER(final) − CER(subject), points, page-clustered 95 %; "vs arm I" = CER(final) − CER(arm I α 0.6), paired on the discovery half. Rows: `stage4_A_sessions.jsonl` (rows of the discarded first runs `…-b10-1` of p2, p3f are not results), `stage4_A_followup.jsonl` (shams, margins).

| arm | who / donor | passes, kept | what the program did | discovery Δ | vs arm I (−4.09) |
|---|---|---|---|---|---|
| P1 | Sonnet 5.5, donor | 7, 2 | 30 % pull of all non-embedding tensors toward the donor (kept), then 60 % (kept); space-row, EOS and embedding-row edits rejected | −3.73 [−4.75, −2.75] | +0.37 [+0.12, +0.60] (worse) |
| P2 | Fable 5.2, donor | 7, 1 | encoder 30 % toward the donor (kept); space and EOS logit offsets, extrapolation away from the donor rejected | −1.67 [−2.21, −1.14] | +2.42 [+1.81, +3.06] |
| P3 | Sonnet 5.5, no donor | 7, 0 | scalings of cross-attention output, tied embedding, encoder final norm, last FFN: all rejected | 0 (null) | – |
| P3 | Fable 5.2, no donor | 7, 0 | readout logit offsets for word-initial tokens and EOS, temperature, cross-attention gain: all rejected | 0 (null) | – |
| B | Sonnet 5.5, blind | 0, 0 | no edit | 0 (null) | – |
| R | replay of P1's family (fresh Sonnet), donor | 5, 2 | space offset, EOS rejected; 25 % pull toward the donor twice (kept, about 44 %) | −3.09 [−3.89, −2.28] | +1.01 [+0.56, +1.40] |

**Shams** (five per edit, same tensors, same Frobenius norm): the best sham Δ on the discovery half is −0.02 for P1 and −0.09 for P2 (practice −0.02, −0.06): every real edit is far above its shams. **Margins:** P1 at 0.5× −2.40, at 2× +3.32 (an optimum near 1×); P2 at 0.5× −1.10, at 2× −1.61 (still improving, not at an optimum).

**Reading.** Every kept edit is a pull toward the donor: a lookup/interpolation, not a new finding. No programmer beat arm I; P1 reaches −3.73 against arm I's −4.09, and the donor-free sessions (P3, both models) and the blind session found nothing and changed nothing. Gate 4 is not met on subject A: no frozen model exceeds arm I. Crashes: none apart from the harness defect of amendment 3 (first runs discarded).
