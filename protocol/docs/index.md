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

!!! note "Status — 11 November 2026"
    **No weight edit has yet repaired a model.**

    - **Kraken models:** budgeted sessions gave no held-out gain. The one small
      gain (−0.39 points) is a quarter-step merge toward a sibling, and a short
      retrain on corrected data does at least as well (−0.55).
    - **Positive control:** the method repairs planted, localised defects, but
      so do a blind programmer and a rule list.
    - **Vision-language adapter:** two sessions with different programmer models
      (Opus 5.5, Fable 5.1) independently kept the same edit. It gave a large
      practice gain from pages that stop looping, and none on 733 held-out
      pages. One random direction passed the same keep rule. The replicate is
      in thodel/weightspace#25, not yet merged.
    - **Block exchange between related models does help** on material the
      model was not trained for. One candidate (`latin_incunabula` v3) reads
      held-out Fraktur 1.7 points better while keeping its abbreviations.
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
