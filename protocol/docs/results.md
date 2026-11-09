# Results

Each result names its run, the practice and held-out set hashes and the model
digests. Practice-set numbers are labelled as such; only held-out numbers are
results in the paper sense.

## R1 — Lossless rewrite `Cr255 → Cr1` { #r1 }

A kraken+ output head convolves over a one-row feature map with a 255-row
kernel, so 254 of the rows only ever multiply zero padding. Keeping row 127
and rebuilding the model from the rewritten architecture string:

| | Cr255 | Cr1 |
|---|---:|---:|
| parameters | 15,207,871 | 4,153,791 |
| text identical on 300 practice lines | — | 300 / 300 |
| max \|Δ softmax\| | — | 1.1 × 10⁻⁶ |

<figure markdown="span">
  ![Kernel-row L2 of the Cr255 conv: row 127 receives gradient, the other 254 stay at their initialisation norm.](figures/padding-cr255.svg#only-light)
  ![Kernel-row L2 of the Cr255 conv: row 127 receives gradient, the other 254 stay at their initialisation norm.](figures/padding-cr255-dark.svg#only-dark)
  <figcaption>Kernel-row L2 of the Cr255 conv: row 127 receives gradient, the other 254 stay at their initialisation norm.</figcaption>
</figure>

The padding rows sit at their initialisation norm (12.045 ± 0.027 against
12.38 for the live row), as expected for weights that never receive a
gradient. Source model: a collapsed from-scratch run (CER 0.99 on both sides,
by construction). Value: the same function at a quarter of the size.

## R2 — Output-layer levers do not move `german-xix-blb-v1` { #r2 }

Session `20261106-german-xix-blb-v1-b10-1`, budget 10, practice
`blb-train-300` (sha256 `4a803b49…`), held-out `blb-eval` (5,628 lines,
sha256 `0729b93b…`, audit clean). Subject tensor digest `f35e075a…`, frozen
`78d6b7d5…`.

| | practice CER | held-out CER | collapsed (held-out) |
|---|---:|---:|---:|
| before (C0) | 0.06458 | 0.12335 | 428 |
| after | 0.06435 | 0.12325 | 403 |
| Δ, 95 % interval | −0.023 pt [−0.34, +0.30] | **−0.010 pt [−0.048, +0.026]** | |
| parent (C1) | 0.3039 | 0.3196 | 455 |

**Not a gain.** Run under the keep rule before [A1](amendments.md#a1).

## R3 — Label-noise damage is not removable by transplant { #r3 }

Session `20261106-bern-combined-v1-b10-1`, budget 10, practice `bern-train-300`
(sha256 `474414e3…`, training side), held-out `bern-eval` (545 lines, sha256
`f7b024e0…`). One edit kept: the output layer from the parent.

| | practice CER | held-out all | Hertwig | Schlussredaktion |
|---|---:|---:|---:|---:|
| before (C0) | 0.1456 | 0.1933 | 0.2669 | 0.1348 |
| after | 0.1414 | 0.1948 | 0.2639 | 0.1399 |
| Δ, 95 % interval (points) | −0.42 [−0.85, −0.04] | **+0.15 [−0.10, +0.42]** | −0.30 [−0.72, +0.13] | +0.51 [+0.21, +0.82] |

**Not a gain:** the practice gain came from the donor's own training lines.
[Journal](journal/2026-11-06-e-phase3.md).

*Note to R2 (7 Nov):* re-aggregated on `blb-eval-hand`, the handwriting
held-out set chosen by data properties: 9.553 % → 9.553 %, Δ −0.001 points
[−0.055, +0.053].

## R4 — A block exchange for `latin_incunabula` { #r4 }

Two group norms from its base, output layer half-way. Held-out, 4,374 lines of
its own collection: 5.01 % → 4.99 % (−0.03 [−0.07, +0.02]); serving path, 58
pages, median page 4.16 % → 4.05 % (−0.19 [−0.27, −0.10]). Abbreviation signs
get worse: MUFI characters 532 → 731 errors (+0.11 points, interval excluding
0). [Journal](journal/2026-11-07-b-model-family.md).

## R5 — Weight survey of the kraken zoo { #r5 }

30 public models read, no forward pass. Weights agree with the documented
lineage in 11 of 13 pairs; a shared name predicts relatedness in 1 of 17
pairs, a near-identical codec in 23 of 23. Blank-bias rise relative to the
other biases in 11 of 11 parent → child pairs. [Journal](journal/2026-11-07-c-zoo-survey.md).

## R6 — Seed-variance null { #r6 }

Five seeds of one recipe. Seed noise is about a third of the output rows'
movement in a fine-tune and about 90 % of the pooled stacks'. Survives: the
relative blank-bias rise (+0.53 to +0.55 vs −0.02 to +0.01 between seeds),
rare rows growing and losing bias. Held-out spread between seeds sd 0.18
points; differences under about one point between two runs are not
distinguishable from seed. [Journal](journal/2026-11-07-d-seed-null.md).

## R7 — Planted defects and unseen practice { #r7 }

All six planted corruptions that cost ≥ 1 point were repaired to within 0.22
points by the programmer, the blind programmer and a rule list alike; diffuse
damage only by the programmer with passes (83 %). Unseen practice: Kurrent
model 0 of 6 attempts kept; `combined-v1` −0.19 to −0.39 points held-out, none
with a page-clustered interval excluding 0. [Journal](journal/2026-11-07-e-controls.md).

## R8 — Sham, global interpolation, margins { #r8 }

20 norm-matched shams: held-out mean −0.013 (sd 0.062); 0 of 20 reach the real
edit's −0.39, which equals plain quarter-step interpolation. Whole-network
interpolation costs +3.81 points [+2.57, +4.98] more than the sum of single
blocks. Blank-bias edits recover one deletion in 40 / 9 / 5 at −0.1 / −0.5 /
−1.0 and pay with nearly as many insertions. [Journal](journal/2026-11-07-e-controls.md).

## R9 — Neither `latin_incunabula` candidate qualifies { #r9 }

Preregistered rule. Both read held-out Fraktur better (v1 −2.04 / −0.44, v2
−1.08 / −0.19 points) and both lose abbreviations; v2 also +0.04 on its own
collection. Neither enabled. [Journal](journal/2026-11-07-g-candidate-decision.md).

## R10 — Adapter session A4 { #r10 }

`qwen3.5-4b-german-kurrent-xvi-xix-page-v2`, budget 10, one edit kept
(layer-27 `q_proj` × 0.8). 134 held-out pages: CER 0.685 → 0.569, Δ −0.115
[−0.271, −0.001] by pages; sham +0.032; clipped CER unchanged (0.354 → 0.351).
**A gain in stopping, not in reading,** on a quantity shown to be chaotic.
[Journal](journal/2026-11-07-h-adapter-track.md).

## R11 — C4: retraining `combined-v1` on the corrected data { #r11 }

5 epochs, lr 1e-5, three seeds: held-out 19.33 % → 18.77–18.79 %, Δ −0.54 to
−0.56 points (page-clustered about [−1.1, −0.1]); at lr 5e-5, −0.98 to −1.02.
At least as good as the best weight edit (−0.39). Gain on the Hertwig lines.
Not shown: that the correction rather than more training causes it.
[Journal](journal/2026-11-08-b-c4-retrain.md).

## Phase 5 adapter session — no held-out gain

`20261107-kurrent-page-v2-b10-1`: practice CER 0.4515 → 0.3470; 733 held-out
pages 0.4012 → 0.4036, Δ +0.24 points [−3.5, +3.9]. The edit swaps which pages
loop. [Journal](journal/2026-11-08-c-adapter-heldout.md).

## R12 — `latin_incunabula` v3 qualifies { #r12 }

Abbreviation rows spared. Held-out: NewsEye −1.66 [−2.02, −1.30] (pages),
GT-Fraktur −0.21, own collection −0.02; MUFI share +0.014 (v1 +0.110). Serving
path: median page −0.09 [−0.16, −0.03]. Registered and disabled.
[Journal](journal/2026-11-08-d-incunabula-v3.md).

## R13 — Exchange on out-of-domain material and across lineages { #r13 }

First-convolution exchange stabilises the Kurrent model on held-out
out-of-domain documents (−0.35 to −1.07 points, no own-material cost), below the
preregistered minimum; Bern null; unrelated models cannot exchange units
(correlation 0.2–0.4, every transplant breaks the model).
[Journal](journal/2026-11-08-e-exchange-ood.md).
## Controls

- **C2 average, C5 sham, global interpolation:** R7, R8.
- **C4 retrain** on corrected data: R11.

- **C3 per-block α-sweep, `german-xix-v1 → german-xix-blb-v1`** (practice
  only): see the [journal entry](journal/2026-11-06-d-alpha-sweep.md).
