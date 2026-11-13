# 2026-11-11 · TrOCR track, stage 0: fetch, subject, practice set, C0

**Type:** new track, set-up (no edit) · **Source:** `docs/trocr/STAGE0.md`,
`docs/epics/EPIC_TROCR_TRACK.md`, `docs/MODELS_TROCR.md` · **Gate 0:** pass, with conditions

## Plan

Repeat the kraken experiments on a second architecture: the in-house **TrOCR** family. These models are transformer encoder-decoders:
- a ViT encoder with 12 layers;
- a text decoder with 12 layers;
- 334 M parameters;
- a byte-level BPE vocabulary of 50,265 tokens.

Six base-size models of the Bern chain descend from `microsoft/trocr-base-handwritten`. Stage 0 only checks feasibility: whether reading these models is deterministic, what a pass costs, and which practice and held-out sets to use.

## Done

- **Weights:** seven models fetched read-only, with revisions and sha256 pinned in `docs/trocr/DIGESTS.json`. Two are private; Tobias approved fetching them.
- **Subject interface:** `wsp/trocr_subject.py`, with tests.
- **Practice set:** `trocr-blb-train-300`, audited for overlap with the held-out set `blb-eval-hand` and with the training data.
- **C0:** baseline readings for both candidate subjects and their parent, on card 0 and on CPU.

## Found

- **Tokenizers:** vocabulary, merges and special tokens are identical in all seven models; the files differ only in how they are serialised. Tensors can therefore be aligned by name, with no codec alignment as kraken needs.
- **Correction to the model inventory:** input embedding and output projection are **one tied tensor** (50,265 × 1,024). Editing it changes input and output together.
- **Deterministic:** repeated readings are byte-identical, on card 0 and on CPU, and across settings and processes.
- **C0 on `blb-eval-hand` (2,272 lines, card 0):**

  | model | CER |
  |---|---:|
  | `from-kurrent-v2` | 19.74 % |
  | `XVI-XVII` (parent) | 19.83 % |
  | `trocr-kurrent` | 26.41 % |

  The CERs on the model cards were measured on other material and cannot be reproduced.
- **Cost:** a practice pass takes about 35 s on card 0. On CPU it takes 10 min to several hours, depending on load. Card 0 is the engine.

## Changes

- **Gate 0 passed**, with five conditions for later stages:
  - pin the generation settings, including length penalty 1.0;
  - redefine the planted defects for a tied embedding;
  - ask the parents' authors whether Laßberg or BLB material was in their corpora;
  - read the held-out set once per frozen model;
  - choose the subject.
- **Decisions (Tobias, 11 Nov):**
  - both `from-kurrent-v2` and `trocr-kurrent` are subjects;
  - the stage-2 seeds and the stage-5 retrain go through the trainer;
  - a comparison with large models (the Swedish and CATMuS fine-tunes) follows at the end.
