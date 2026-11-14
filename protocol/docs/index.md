# Weight-space programming on HTR models

This is the research protocol of a project that reads the weights of
handwritten-text-recognition (HTR) models, forms hypotheses about what their
components do, edits the weights in code — **without gradient descent** — and
measures the result against held-out data and against the cheapest
alternatives (weight averaging, interpolation, a short retrain).

The method follows R. Takehi, J. Teoh, A. Zhang and O. Khattab, ["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/)
(2 Nov 2026), which applied it to MLPs and small language models. This project
transfers it to a different architecture: kraken line recognisers (VGSL
convolution stack → three bidirectional LSTMs → CTC output over ~100–300
characters, about 4 M parameters) fine-tuned on historical German handwriting.

!!! note "Status — 14 November 2026"
    **No weight edit has yet done better than a cheap alternative.**

    - **Kraken models:** one budgeted session gave a held-out gain whose page
      interval excludes 0: the Fable 5.1 rerun of session 2, −0.57 points
      [−1.15, −0.05]. Its gain is a step of 12.5 % toward the parent model, the
      same size and place as a short retrain on corrected data (−0.55). The
      same retrain on the uncorrected pairs gains nothing (−0.07), so it is the
      correction that helps. Every other session gave no held-out gain.
    - **Positive control:** the method repairs planted, localised defects, but
      so do a blind programmer and a rule list. That holds with Sonnet 5.5 and
      Fable 5.1 programmers. The one diffuse defect repaired by measuring
      was repaired by one of the two models only.
    - **Vision-language adapter:** two sessions with different programmer models
      (Opus 5.5, Fable 5.1) independently kept the same edit. It gave a large
      practice gain from pages that stop looping, and none on 733 held-out
      pages. One random direction passed the same keep rule.
    - **Block exchange between related models does help** on material the
      model was not trained for. One candidate (`latin_incunabula` v3) reads
      held-out Fraktur 1.7 points better while keeping its abbreviations.
    - **Second architecture:** a TrOCR track (transformer encoder-decoder,
      334 M parameters) has started. Readings are deterministic and lineage can
      be read from the weights. No edit yet.
    - **Layer views on unseen lines:** the probe findings hold; part of the
      block roles do not, and four part-block hypotheses are withdrawn.
    - **Activation atlas:** no single unit matters much or is selective on its
      own; characters are read from the later blocks as a whole. The atlas
      stays descriptive, and neither model has a circuit for the deleted
      spaces.
    - **Data:** along the way the work found several defects in the project's
      own data.

    Negative and null results are reported in full, as the protocol requires.

## The models

All in-house models share one stack; only the output rows (one per character) differ.

<figure markdown="span">
  ![The VGSL stack, one rectangle per tensor, width proportional to parameter count; the conv stack (5 % of the parameters) magnified underneath.](figures/architecture.svg#only-light)
  ![The VGSL stack, one rectangle per tensor, width proportional to parameter count; the conv stack (5 % of the parameters) magnified underneath.](figures/architecture-dark.svg#only-dark)
  <figcaption>The VGSL stack, one rectangle per tensor, width proportional to parameter count; the conv stack (5 % of the parameters) magnified underneath.</figcaption>
</figure>

## How to read this protocol

- **[Method and rules](method.md)** — units, preregistration, keep rule,
  controls, statistics. Fixed before the sessions they govern.
- **[Protocol amendments](amendments.md)** — every change to the rules, dated,
  with the reason and the sessions affected.
- **[Journal](journal/index.md)** — dated entries in the order the work
  happened: what was planned, what was done, what was found, what changed.
  Entries are not rewritten afterwards; corrections are new entries.
- **[Results](results.md)** — the numbers that would go into a paper, each
  traceable to a run, a data-set hash and the model digests.

## Principles

1. **Preregister.** Hypotheses, with predicted direction and size, are written
   down before the first forward pass of a session.
2. **Held-out once.** The held-out set is unreachable from the editing sandbox
   and scored once, after the edit is frozen.
3. **Intervals, not point estimates.** Every difference carries a paired
   per-line bootstrap 95 % interval; a gain whose interval contains 0 is not
   a gain.
4. **Report everything.** Reverted edits, failed sessions and data defects are
   results.
5. **Compare with the cheap alternatives.** An edit is only interesting if it
   beats averaging, interpolation and a short retrain.
