# 2026-11-07 · Where `german-xix-blb-v1` finds and loses characters

**Type:** reading on practice lines, no session budget, no held-out line ·
**Source:** `docs/viz/README.md` (draft pull request 11, not yet merged)

!!! warning "Not yet merged"
    The code and figures of this entry are on a draft pull request. The figures
    are included here because they illustrate the readings; they will be
    regenerated from the merged code before publication.

## Plan

Make the layers' decisions visible: at which depth does a letter become
identifiable, and which block's share of the fine-tune does what?

## Done

- Forced CTC alignment of the reference, including the characters the model
  deletes; one output frame is 8 px of the line.
- Two one-frame ridge probes per layer, cross-fitted over five document folds:
  *identity* (which character) and *emission* (character vs blank).
- The C3 α-sweep re-read per line and per character (α ∈ {0, 0.5, 1.5}); its
  16 CERs reproduce [the sweep](2026-11-06-d-alpha-sweep.md) exactly.
- An exact bias trade-off curve, re-decoded from one capture of the logits
  (verified against real edited passes: 300/300 lines identical).

## Found

- **Letters become identifiable over the BiLSTMs:** probe accuracy for
  correctly read letters 0.36 at the last convolution, then 0.59, 0.82 and 0.94
  after the three LSTMs.
- **Each LSTM has a job in the fine-tune** (that block alone set back to the
  parent): the first finds the letters (+366 letter deletions without it), the
  second tells them apart (+342 substitutions), the third sets the word gaps
  (+362 deleted spaces). The conv blocks and the output layer contribute little.
- At α = 1.5 each block trades one kind of error for another; the last LSTM's
  near-null is −20 substitutions against +29 deletions.

<figure markdown="span">
  ![Per-layer probes on practice lines: how well a single frame identifies its character, and whether it emits a character or blank, from the convolutions through the three LSTMs.](../figures/probes/probe-kraken-german-xix-blb-v1__blb-train-300.svg#only-light)
  ![Per-layer probes on practice lines: how well a single frame identifies its character, and whether it emits a character or blank, from the convolutions through the three LSTMs.](../figures/probes/probe-kraken-german-xix-blb-v1__blb-train-300-dark.svg#only-dark)
  <figcaption>Per-layer probes on practice lines: how well a single frame identifies its character, and whether it emits a character or blank, from the convolutions through the three LSTMs.</figcaption>
</figure>

<figure markdown="span">
  ![The C3 α-sweep per block at α = 0 (block set back to the parent), 0.5 and 1.5: CER and edits by kind (letters deleted, spaces deleted, substituted, inserted) against the unedited child.](../figures/probes/alpha-overlay-kraken-german-xix-blb-v1__blb-train-300.svg#only-light)
  ![The C3 α-sweep per block at α = 0 (block set back to the parent), 0.5 and 1.5: CER and edits by kind (letters deleted, spaces deleted, substituted, inserted) against the unedited child.](../figures/probes/alpha-overlay-kraken-german-xix-blb-v1__blb-train-300-dark.svg#only-dark)
  <figcaption>The C3 α-sweep per block at α = 0 (block set back to the parent), 0.5 and 1.5: CER and edits by kind (letters deleted, spaces deleted, substituted, inserted) against the unedited child.</figcaption>
</figure>

<figure markdown="span">
  ![Output-bias levers: change in CER as one bias is shifted, re-decoded exactly from one capture of the logits.](../figures/probes/bias-kraken-german-xix-blb-v1__blb-train-300.svg#only-light)
  ![Output-bias levers: change in CER as one bias is shifted, re-decoded exactly from one capture of the logits.](../figures/probes/bias-kraken-german-xix-blb-v1__blb-train-300-dark.svg#only-dark)
  <figcaption>Output-bias levers: change in CER as one bias is shifted, re-decoded exactly from one capture of the logits.</figcaption>
</figure>

## Changes

A first piece of the proposed activation atlas of the whole network.
