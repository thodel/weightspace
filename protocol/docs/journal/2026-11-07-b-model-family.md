# 2026-11-07 · The served kraken family compared as networks

**Track:** C · **Budget spent:** none (exchange grid on practice lines;
one candidate scored held-out) · **Source:** `docs/MODEL_FAMILY.md`,
`docs/family/` · **Result:** [R4](../results.md#r4)

## Plan

The project serves 43 kraken models (37 published on Zenodo, 6 trained
in-house). How do they compare *as networks*?

- **RQ1** How much of them is identical, and could have been frozen in training?
- **RQ2** Which layers differ, and how strongly?
- **RQ3** Can a network be stabilised by exchanging parts, e.g. from a more
  general to a more specific model?

## Done

All 43 weight files obtained (checksums equal to Zenodo's) and sorted into five
architecture groups; group A (33 models) is the standard kraken stack of the
in-house models. Per pair and block: share of identical parameters, relative
distance, cosine; per layer, linear CKA on shared probe lines. For RQ3, every
block of five specific ← general pairs at β = 0.5 and 1, read on three practice
sets (240 readings).

## Found

**Lineage from weights alone.** Seven lineages, matching the records — and
adding to them: the base of `german-xix-v1`, recorded as "to confirm", is
`bohemian_19th`; two models without a stated base descend from known ones; two
stated bases are contradicted (one model is unrelated to its named base, one
has an architecture its named base cannot produce).

**RQ1 — nothing trained stays identical; what is identical is dead.** Within
lineages the only bit-identical parameters sit in the first two convolution
layers, and they are dead units: filters whose ReLU never fires (so no gradient
reaches them) and the next layer's weights that read them. 11 of 43 models have
no dead first-layer filter, 11 have a third or more; along a fine-tune they
mostly accumulate.

<figure markdown="span">
  ![Dead first-layer filters for all 43 models: 11 have none, 11 have a third or more, up to 19 of 32.](../figures/family/family-dead-filters.svg#only-light)
  ![Dead first-layer filters for all 43 models: 11 have none, 11 have a third or more, up to 19 of 32.](../figures/family/family-dead-filters-dark.svg#only-dark)
  <figcaption>Dead first-layer filters for all 43 models: 11 have none, 11 have a third or more, up to 19 of 32.</figcaption>
</figure>

**RQ2 — distance grows with depth; the front end is shared across the whole
family.** Every fine-tune moves late layers more than the first. In function,
independently trained models compute much the same first layer (CKA 0.80 for
unrelated pairs in group A, 0.99 in group B) and diverge in the recurrent
stack; the last LSTM and the output layer are where models are individual.

<figure markdown="span">
  ![Per-layer CKA by relation class (same lineage, distant relatives, unrelated) in groups A and B: high in the convolutional front end, falling through the recurrent stack.](../figures/family/family-cka-profile.svg#only-light)
  ![Per-layer CKA by relation class (same lineage, distant relatives, unrelated) in groups A and B: high in the convolutional front end, falling through the recurrent stack.](../figures/family/family-cka-profile-dark.svg#only-dark)
  <figcaption>Per-layer CKA by relation class (same lineage, distant relatives, unrelated) in groups A and B: high in the convolutional front end, falling through the recurrent stack.</figcaption>
</figure>

<figure markdown="span">
  ![Group A's 33 models, model × model CKA per layer: nearly uniform at the first convolution, lineage blocks from the second LSTM on, almost only the diagonal at the output.](../figures/family/family-cka-matrices.svg#only-light)
  ![Group A's 33 models, model × model CKA per layer: nearly uniform at the first convolution, lineage blocks from the second LSTM on, almost only the diagonal at the output.](../figures/family/family-cka-matrices-dark.svg#only-dark)
  <figcaption>Group A's 33 models, model × model CKA per layer: nearly uniform at the first convolution, lineage blocks from the second LSTM on, almost only the diagonal at the output.</figcaption>
</figure>

<figure markdown="span">
  ![Relative weight distance parent → child per layer for thirteen fine-tunes: the first convolution barely moves in small fine-tunes; large ones move the whole network.](../figures/family/family-weight-drift.svg#only-light)
  ![Relative weight distance parent → child per layer for thirteen fine-tunes: the first convolution barely moves in small fine-tunes; large ones move the whole network.](../figures/family/family-weight-drift-dark.svg#only-dark)
  <figcaption>Relative weight distance parent → child per layer for thirteen fine-tunes: the first convolution barely moves in small fine-tunes; large ones move the whole network.</figcaption>
</figure>

**RQ3 — exchange.** No block's fine-tune can be undone for free; the first
convolution comes closest (half-way back costs nothing). Small stabilisation
without cost exists only in the convolutional blocks (e.g. `bern-hertwig-v2`
with `C_9` half-way to its parent: own material +0.18 points, interval
including 0; two other sets −1.1 each). The recurrent layers trade one for one
or worse: moving them toward the general model undoes the fine-tune. Each
front-end block alone could be set back at no cost, but not all of them
together: they compensate for each other.

<figure markdown="span">
  ![ΔCER per block and practice set when one block moves half-way (β = 0.5) to the more general model, for eight specific ← general pairs.](../figures/family/family-exchange-b0.5.svg#only-light)
  ![ΔCER per block and practice set when one block moves half-way (β = 0.5) to the more general model, for eight specific ← general pairs.](../figures/family/family-exchange-b0.5-dark.svg#only-dark)
  <figcaption>ΔCER per block and practice set when one block moves half-way (β = 0.5) to the more general model, for eight specific ← general pairs.</figcaption>
</figure>

<figure markdown="span">
  ![The same with the general model's block transplanted (β = 1).](../figures/family/family-exchange-b1.svg#only-light)
  ![The same with the general model's block transplanted (β = 1).](../figures/family/family-exchange-b1-dark.svg#only-dark)
  <figcaption>The same with the general model's block transplanted (β = 1).</figcaption>
</figure>

A first candidate came out of the grid: `latin_incunabula` with its first two
group norms from its base and its output layer half-way (R4). On held-out lines
of its own collection it is as good (5.01 % → 4.99 %, interval including 0),
slightly better through the serving path (median page 4.16 % → 4.05 %), but
**abbreviation signs get worse** (MUFI characters 532 → 731 errors) — a
regression the total CER hides.

<figure markdown="span">
  ![Combined block moves for latin_incunabula: ΔCER on its own material and two Fraktur sets, practice lines next to fresh confirmation lines.](../figures/family/family-combine-latin_incunabula-from-catmus_print_large.svg#only-light)
  ![Combined block moves for latin_incunabula: ΔCER on its own material and two Fraktur sets, practice lines next to fresh confirmation lines.](../figures/family/family-combine-latin_incunabula-from-catmus_print_large-dark.svg#only-dark)
  <figcaption>Combined block moves for latin_incunabula: ΔCER on its own material and two Fraktur sets, practice lines next to fresh confirmation lines.</figcaption>
</figure>

**Data finding on the way:** in the Reichenau incunable dataset used here, 39 of
the 59 validation pages are byte-identical to training pages.

## Changes

The candidate was registered for the evaluation only and disabled again. Which
of two candidates to enable was preregistered and decided in
[R9](2026-11-07-g-candidate-decision.md).
