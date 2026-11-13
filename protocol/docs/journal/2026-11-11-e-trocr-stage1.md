# 2026-11-11 · TrOCR track, stage 1: the family read as weights

**Type:** readings (weights only, no forward pass) · **Source:** `docs/TROCR_READINGS.md`,
`docs/trocr/readings_tables.md`, `wsp/trocr_family.py` · **Gate 1:** pass, with caveats

## Plan

Read the seven models of [stage 0](2026-11-11-d-trocr-stage0.md) as weights, the way the [zoo survey](2026-11-07-c-zoo-survey.md) read the kraken models:
- per-tensor distances;
- lineage;
- depth;
- token rows;
- move directions.

No null yet: the seed runs come in stage 2.

## Found

- **Lineage is recoverable for the small fine-tunes.** The documented parent is the nearest model for `v2` (0.090), `from-kurrent-v2` (0.088) and `full-v1` (0.140, against 0.154 to its sibling).
  - The result depends on pooling only the weight matrices. That choice was made after seeing the first table, so the reading is exploratory.
  - Its two narrowest margins (0.002 and 0.009) carry no weight.
  - The unused ViT pooler is an exact lineage marker. It confirms the chain `kurrent → xvi-xvii → {from-kurrent-v2, full-v1}` bit for bit, and shows that `v2` and `medieval` do not descend from `kurrent`.
- **The decoder moves, the encoder hardly does.**
  - After 20 epochs: decoder 0.51, encoder 0.11. The tied embedding moves most (0.83).
  - This is the transformer analogue of kraken's "the stack that reads moves least, the stack that writes moves most", and it holds by a larger ratio than in the kraken zoo.
- **Depth:**
  - The encoder moves in a hump, most at layers 4 and 5, so its first layers are not a "generic front end" as in kraken.
  - The decoder's movement rises with depth only in the long fine-tunes.
- **Token rows:** all 50,265 rows move in every pair, a little more for high token ids. Because the output projection is tied, every row gets a gradient at every step.
- **Directions of the moves:**
  - consecutive fine-tunes move orthogonally in the encoder and decoder, and the embedding step partly undoes the previous one (cosine −0.29 to −0.48);
  - the same training data from two different origins moves the weights almost orthogonally;
  - **the data does not determine the move; the starting point and the run do.**

## Changes

- **Gate 1 passed.** The readings go to stage 2, the seed null, as hypotheses, not findings.
- Fetching the third-party fine-tunes for the comparison needs Tobias's yes.
