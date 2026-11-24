# Stage 4, subject B (`trocr-kurrent`, donor = IAM base `microsoft/trocr-base-handwritten`): programmer sessions, 24.11.2026

Practice set only for the programmers; frozen models read on the **discovery half** only; the confirmation half was not read. Δ = CER(final) − CER(subject, discovery baseline 23.26 %), points, page-clustered 95 %; "vs arm I" = CER(final) − CER(arm I α 0.3, discovery −3.36 [−4.51, −2.23]), paired on the same lines.

| arm | who / donor | passes, kept | what the program did | practice Δ | discovery Δ | vs arm I |
|---|---|---|---|---|---|---|
| P1 | Sonnet 5.5, donor | 5, 0 | small magnitude edits (cross.o bias ×0.9, EOS row ×1.05, enc.norm_final.bias ×0.9, tied embedding 10 % toward the donor): none passed the keep rule | 0 | 0 (null) | – |
| P2 | Fable 5.2, donor | 9, 1 | cross.q ×1.15, EOS row ×0.9, **cross.o ×1.15 (kept)**, 15 % extrapolation of the encoder away from the donor (rejected), temperature via dec.11.norm_ffn | −0.96 | −0.70 [−1.16, −0.29] | +2.66 [+1.71, +3.65] (worse) |
| P3 | Sonnet 5.5, no donor | crash, 0 | program crashed at its fifth `attempt()` call (missing positional argument `names`, program.py line 68) before any pass; counted as Δ 0, not rerun | – | 0 (crash) | – |
| P3 | Fable 5.2, no donor | 10, 1 | one-scalar knobs; **cross.o ×1.15 (kept)**, the other knobs (cross.q, self-attn write, EOS shift, word-initial shift, temperature) rejected | −0.96 | −0.70 [−1.16, −0.29] | +2.66 [+1.71, +3.65] |
| B | Sonnet 5.5, blind | 0, 0 | no edit | 0 | 0 (null) | – |
| R | replay of P2's family (fresh Fable, donor) | 8, 1 | extrapolation away from the donor (rejected, twice), **cross.o ×1.15 (kept)**, cross.q ×1.15 and ×0.87, EOS row (rejected) | −0.96 | −0.70 [−1.16, −0.29] | +2.66 [+1.71, +3.65] |

P2, P3 (Fable) and R ended in the **same final state** (identical weight digest): `dec.NN.cross.o.{weight,bias}` × 1.15 in all 12 decoder layers. Two of the three had no access to the donor (P3) or no knowledge of the other session (R), so this is a reproducible scaling edit that needs no donor, not a lookup or interpolation.

**Shams** (five per edit, same tensors, same Frobenius norm; net change): best sham Δ on the discovery half is +0.10 / −0.11 (practice −0.00 to +0.27), all intervals contain 0; the real edit (−0.70, interval below 0) is above all five. **Margins:** 0.5× practice −0.66, discovery −0.56 [−0.91, −0.19]; 2× practice −1.18, discovery −0.45 [−1.02, +0.13]. The effect grows on practice with the dose and saturates on the discovery half.

**Reading.** The gain exists and is far above shams, but it is about a fifth of arm I's (−0.70 vs −3.36); arm I is significantly better (+2.66 [+1.71, +3.65]). No program here matches or exceeds arm I. The section-7 class is not decided here (P2, P3-Fable and R are below arm I, as P1, P2 and R were on A). The confirmation half was not read.
