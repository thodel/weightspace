# 2026-11-07 · Weight survey of the kraken zoo

**Type:** reading, no forward pass · **Source:** `docs/ZOO_SURVEY.md`,
`docs/zoo/` · **Result:** [R5](../results.md#r5)

## Plan

Test the readings of [phase 0](2026-11-06-a-phase0.md), made on six in-house
models, against the public models: do the fine-tuning traces hold elsewhere,
and can lineage be read from the weights?

## Done

30 of the 37 public kraken models the project serves were downloaded and read
(seven could not be obtained); compared with the documented lineage on Zenodo.

## Found

- 21 of 30 (70 %) have exactly the in-house models' stack; all 30 fall into
  three architecture families.
- **Lineage is readable** for fine-tunes that did not move far: weights agree
  with the documented lineage in 11 of 13 documented pairs (one contradiction,
  one architecture mismatch). With unrelated models at a distance of about
  1.41, close fine-tunes sit at 0.1–0.3, heavy retrainings at 0.7–0.9.
- **Names mislead, codecs don't:** 1 of 17 name-sharing pairs is related; 23 of
  23 pairs with a near-identical codec are.
- **Fine-tuning traces:** the blank-bias rise relative to the other biases holds
  in 11 of 11 pairs. Rare-row inflation holds in direction, not as "the rows
  that moved most". "The conv stack moves less than the recurrent stack" is not
  supported (2 of 5); the first conv block moves least and movement grows with
  depth.

<figure markdown="span">
  ![Blank-row bias of every model, raw (left) and relative to the mean of the other biases (right).](../figures/zoo/zoo-blank-bias.svg#only-light)
  ![Blank-row bias of every model, raw (left) and relative to the mean of the other biases (right).](../figures/zoo/zoo-blank-bias-dark.svg#only-dark)
  <figcaption>Blank-row bias of every model, raw (left) and relative to the mean of the other biases (right).</figcaption>
</figure>

<figure markdown="span">
  ![Distance (non-output weights) between the 28 models of the standard stack, average-linkage order; ≈ 1.41 means unrelated.](../figures/zoo/zoo-distance-standard.svg#only-light)
  ![Distance (non-output weights) between the 28 models of the standard stack, average-linkage order; ≈ 1.41 means unrelated.](../figures/zoo/zoo-distance-standard-dark.svg#only-dark)
  <figcaption>Distance (non-output weights) between the 28 models of the standard stack, average-linkage order; ≈ 1.41 means unrelated.</figcaption>
</figure>

<figure markdown="span">
  ![Every parent → child relation: green where record and weights agree, orange where the weights contradict the record, blue where only weights and codec suggest it.](../figures/zoo/zoo-lineage.svg#only-light)
  ![Every parent → child relation: green where record and weights agree, orange where the weights contradict the record, blue where only weights and codec suggest it.](../figures/zoo/zoo-lineage-dark.svg#only-dark)
  <figcaption>Every parent → child relation: green where record and weights agree, orange where the weights contradict the record, blue where only weights and codec suggest it.</figcaption>
</figure>

<figure markdown="span">
  ![Appearance against weights. Left: codec overlap against weight distance for 395 same-architecture pairs. Right: the pairs that share a project or a name.](../figures/zoo/zoo-appearance.svg#only-light)
  ![Appearance against weights. Left: codec overlap against weight distance for 395 same-architecture pairs. Right: the pairs that share a project or a name.](../figures/zoo/zoo-appearance-dark.svg#only-dark)
  <figcaption>Appearance against weights. Left: codec overlap against weight distance for 395 same-architecture pairs. Right: the pairs that share a project or a name.</figcaption>
</figure>

<figure markdown="span">
  ![Parent → child pairs: pooled relative L2 by stack (left) and the change of the blank bias (right).](../figures/zoo/zoo-pairs.svg#only-light)
  ![Parent → child pairs: pooled relative L2 by stack (left) and the change of the blank bias (right).](../figures/zoo/zoo-pairs-dark.svg#only-dark)
  <figcaption>Parent → child pairs: pooled relative L2 by stack (left) and the change of the blank bias (right).</figcaption>
</figure>

## Changes

The manuscript's model-zoo section rests on this survey; which traces are
specific to fine-tuning at all is tested by the [seed null](2026-11-07-d-seed-null.md).
