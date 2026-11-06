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

## Controls

- **C3 per-block α-sweep, `german-xix-v1 → german-xix-blb-v1`** (practice
  only): see the [journal entry](journal/2026-11-06-d-alpha-sweep.md).
