# Weight-space programming on HTR models

This is the research protocol of a project that reads the weights of
handwritten-text-recognition (HTR) models, forms hypotheses about what their
components do, edits the weights in code — **without gradient descent** — and
measures the result against held-out data and against the cheapest
alternatives (weight averaging, interpolation, a short retrain).

The method follows R. Tanaka, ["Weight-space programming"](https://rikiyat.github.io/blog/weight-space-programming/)
(2 Nov 2026), which applied it to MLPs and small language models. This project
transfers it to a different architecture: kraken line recognisers (VGSL
convolution stack → three bidirectional LSTMs → CTC output over ~100–300
characters, about 4 M parameters) fine-tuned on historical German handwriting.

!!! note "Status — 6 November 2026"
    Phase 0 (harness, practice sets, baselines) and phase 1 (first budgeted
    session) are done. **No gain has been found yet**: the one session so far
    kept an edit whose held-out effect is −0.010 CER points with a 95 %
    interval containing 0. Negative and null results are reported here in
    full, as the protocol requires.

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
