# Epic: the same experiments on TrOCR (a transformer line recogniser), after kraken (CNN/LSTM) and Qwen (QLoRA)

Proposed 11 November 2026. Status: prepared, not started. Owner: Tobias Hodel. Related: docs/MODELS_TROCR.md (the inventory this epic rests on), docs/PROTOCOL.md,
docs/RESULTS.md (R1-R13), docs/MODEL_FAMILY.md, docs/ZOO_SURVEY.md, docs/SEED_NULL.md, docs/CONTROLS_REPORT.md, docs/ADAPTER_REPORT.md, docs/PHASE5_REPORT.md,
`wsp/subject.py` (the `Subject` interface), `wsp/kraken_subject.py`, `wsp/lora_subject.py`.

## Question

Do the findings about weight reading and weight editing transfer to a third architecture: a vision transformer encoder with an autoregressive text decoder, trained in-house
(`dh-unibe/trocr-*`)? The kraken results were: lineage is readable, the front end is generic and the recurrent stack individual, two traces of fine-tuning survive a seed null,
edits that look good on practice lines do not transfer, the one gain is an interpolation, a retraining does as well, an exchange of blocks can stabilise a model at a price. The
adapter results were: no donor, the metric (looping) is chaotic, one practice gain that does not transfer. TrOCR sits between them: **full fine-tunes of one shared base with an
identical tokenizer** (so donors exist, as in kraken, and tensors align by name), but a decoder that is a text model with its own language prior (so the "implicit language model"
question of the article becomes testable), and an autoregressive readout (so loops and length matter, as in Qwen).

## Verdict of the preparation (11 November 2026)

**Feasible, and richer in lineage than the other two tracks; the cost is in the engine and not in the idea.**

For:

- **The family exists and is ours.** Six base-size in-house models descend from `microsoft/trocr-base-handwritten` (docs/MODELS_TROCR.md): a documented chain of four, two models that share data but not the parent,
  four of them public and two (`...-v2`, `...-full-v1`) private but readable by us. Fine-tune distance spans 1 to 20 epochs and 54 k to 450 k samples. No training is needed for stages 0 to 3.
- **Alignment is trivial.** Same shapes and tokenizer, so a parent, sibling or earlier model is a donor for every tensor, with the donor rule of the proposed amendment applicable as it stands.
- **Same lines as the other tracks.** `data/heldout/blb-eval-hand` (2,272 lines) is read by the kraken `xix-blb-v1` (9.55 %, and by `xix-v1`, its parent) and can be read by TrOCR lines; the Qwen page models were scored on other pages (733 test pages), so the page models stay a separate comparison. The
  BLB train side is unseen by the TrOCR models, which satisfies the practice-side rule for this subject.
- **The harness has the seams.** `Subject` needs `describe`, `tensors`, `set_tensor`, `forward`, `save`; a `TrocrSubject` is one class and an ops table of names; the Session, keep rule, controls and
  aggregation scripts are architecture-independent.
- **The gateway serves four TrOCR models** (line level, 1,500 MB each), so an exported edited model can be read through the serving path as for R4 and R12 (with the user's OK, registry writes).

Against, and what the epic must design around:

1. **Cost per pass.** 334 M parameters, beam search 4, up to 256 tokens per line: a pass over 300 lines is minutes on a card and slow on CPU. The kraken rules (CPU unless stated; card 0 only with 12 GB free) are
   kept as written; the model needs only 2 to 4 GB, so the 12 GB rule is the binding one and has to be met, not loosened. Gate 0 measures the pass time on CPU and on card 0 and sets the budget.
2. **The readout is not CTC.** CER by generation depends on beam, repeat constraint, length cap, and can loop. A **teacher-forced token loss on the reference** (one forward pass, deterministic, no loops, per-token margins)
   is added as a second readout; CER by generation stays the primary and is what the keep rule uses. Generation settings are pinned in the preregistration (the trainer's eval uses beam 4 and `no_repeat_ngram_size` 3).
3. **The input is distorted by design.** A line is resized to 384 x 384, so the horizontal axis is compressed by a factor that depends on the aspect ratio. Any "where in the line" claim (activation atlas)
   must go through that resizing; the kraken atlas assumes 120 px height and an unscaled width.
4. **No seed null exists.** The kraken claims were corrected by five seeds. For TrOCR the null is built from (a) the two same-data models with different parents, and (b) 5 seeded repetitions of the 3-epoch recipe
   from the same parent (about 0.44 kWh, 1 to 1.5 GPU hours each at the card, through the trainer). Option (b) is a **trainer run and needs the user's OK**; stage 2 is split so that (a) can run without it.
5. **Tied or untied tensors and token rows.** Embedding and output projection are separate; the 50,265 token rows do not correspond to characters, so the "rare rows move most" check of the kraken output layer has no
   direct equivalent: it is redefined per token frequency in the training text (tokenised with the shared tokenizer) and tested against the null before being claimed.
6. **The decoder carries a prior from RoBERTa pre-training and from 292 k lines of German.** An edit that "helps" may only be moving the language prior toward the practice text. The control is the same edit tested
   on lines from a different corpus and on shuffled-word lines (the readout must not improve when the text is scrambled).

## Mapping: what runs where

| kraken / Qwen experiment | TrOCR counterpart | change |
|---|---|---|
| Lineage and zoo survey (R5, docs/ZOO_SURVEY.md) | per-tensor distance and per-component change over the family of docs/MODELS_TROCR.md, **encoder vs decoder, per layer, embedding vs output, norms and biases** | no alignment step; includes third-party fine-tunes of the same base (Riksarkivet, LaMOP, medieval-data, kazars24, ...) |
| Dead filters, identical parameters | identical parameters between a child and its parent (frozen parts), dead heads or FFN units by activation | check whether the trainer froze anything (`training_args.bin`) |
| Seed null (R6) | stage 2: same-data different-parent pair, then 5 seeds | needs the trainer for 5 seeds |
| Planted defects k1-k11 and diffuse (R7) | analogues: zero a token row, scale a layer norm, scale cross-attention output, noise on a layer, zero a head, halve the encoder FFN, bias shift in the output | the positive control comes before any real session |
| Two sessions on training lines, then on non-training practice sets (R2, R3, R7) | sessions on `from-kurrent-v2` with practice = BLB train side, held-out = `blb-eval-hand`; second subject `trocr-kurrent` (19th c.) | three programmers as before; budget 10 passes; keep rule, donor rule, page-clustered intervals from the start |
| Sham, global interpolation, margins (R8) | the same; margins are the teacher-forced per-token margins (the CTC blank is not there) | |
| C2 average, C3 alpha sweep | average of `from-kurrent-v2`, `v2`, `full-v1`; alpha sweep parent to child, per component (encoder only, decoder only, cross-attention only) | the parts can be separated cleanly |
| Block exchange (R4, R9, R12, R13) | exchange of the encoder or of single layers with the parent, sibling or `microsoft/trocr-base-handwritten`; practice, confirmation and held-out sets disjoint **by document**, minimum effects and the page-clustered interval set in advance | applies the lessons of R12/R13 from the start |
| C4 retrain | 1 epoch more on the same data through the trainer, 3 seeds | needs the trainer and the user's OK |
| Adapter track (A0-A4, phase 5) | `slawa188/trocr-kurrent-galicia-lora` is the only TrOCR adapter seen; optional | only if it adapts a model we can read |
| Activation atlas (epic #13) | the same questions for the encoder (patch tokens and attention maps) and for the decoder (cross-attention to image regions) | after the atlas of the kraken model passes its first gate |

## Stages, gates and stops

- **Stage 0, preparation (CPU, asteraix; about one session).** Fetch the models of docs/MODELS_TROCR.md into `~/atr-cache/trocr/` with `scripts/trocr_fetch.py` (read-only on the hub, revision and sha256 recorded in
  `docs/trocr/DIGESTS.md`, no hub write). Build `wsp/trocr_subject.py` (`TrocrSubject`: tensors by canonical name, float32, save in the hub format so that the serving engine loads it; a pluggable
  `forward` for generation and for the teacher-forced loss), the tests on a tiny random `VisionEncoderDecoderModel` (no download), a C0 for each subject on `blb-eval-hand` and on a 300-line practice set,
  the pass-time measurement, and the **tokenizer identity check** across the family. *Gate 0:* C0 on the held-out set reproduces the recorded behaviour
  (CER in the plausible range of the card and byte-identical on repetition) and the time per pass allows a session of 10 passes in under a day; otherwise the practice set is reduced or the track stops.
- **Stage 1, readings (no pass beyond C0).** Distances, component profile, lineage table over the family, the same two surveys as R5. *Gate 1:* the lineage must be recoverable from weights for the small fine-tunes
  (as in kraken); if not, the readings are reported as a negative result and the sessions go on without a lineage argument.
- **Stage 2, nulls.** (a) same-data, different-parent pair; (b) five seeds of the recipe (trainer, with the user's OK); (c) the five documented traces of R6 recomputed. *Gate 2:* only traces that survive the null are used as
  hypotheses in sessions.
- **Stage 3, positive control.** The planted-defect set (analogues above) with the programmer, the blind programmer, a hand-written rule list and a bias/scale search, with a donor and without. *Gate 3:* at least one planted
  defect is repaired by the keep rule; otherwise the sessions of stage 4 are not run, because a null there would mean nothing.
- **Stage 4, sessions.** Practice not trained on, held-out by `blb-eval-hand` (and `bern-eval` read out-of-domain as a control), donor rule and page-clustered intervals from the first pass, three programmers (at least two
  model families), one replay, shams of the same norm, global interpolation, margins; exchange grid over encoder, decoder and layers with document-disjoint confirmation sets.
- **Stage 5, comparator.** C4 (trainer) and the average of siblings, the same way as R11. Needs the OK.
- **Stage 6, write-up.** `docs/TROCR_REPORT.md`; a section for the article if the results change a claim (for example whether the generic front end and individual recurrent stack of the kraken results also appear as a generic encoder and individual decoder), or a second short paper if they do not fit.

Stop gates: after Gate 0 (cost), after Gate 1 (readings), after Gate 3 (positive control). A negative result at any gate is reported as the result.

## Rules carried over (not to be relaxed)

- Asteraix: CPU unless stated; card 1 is the trainer's; card 0 only with at least 12 GB free (a TrOCR pass needs about 2 to 4 GB, but the rule stays); research share read-only; no registry, HF or trainer writes without the user's OK for that run; `TMPDIR=~/atr-cache/tmp`; kraken stays pinned at 7.0.2 in its own venv (TrOCR gets its own venv with `transformers` pinned and recorded; no change to the kraken venv).
- Hub: read-only. The private repositories are ours but are read with the user's credentials only; nothing from them is committed (weights never; digests yes).
- Held-out sets are read once per frozen model; a practice set is never the held-out set; the overlap audit (object level, `wsp.audit`) is run for every subject against the 20 datasets of its card **before** the preregistration.
- Preregistration states the generation settings, the practice and held-out set hashes, the donor list with `seen_by`, the minimum effect, the interval type and the controls; the harness refuses a keep otherwise.
- No children's first names or personal data in anything committed (the repository is private, but the same rule holds for text that may move to the article).
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; session branches are not pushed to main or merged without the user's say-so.

## Deliverables

1. `docs/MODELS_TROCR.md` (done, from the hub), `docs/trocr/DIGESTS.md`, `scripts/trocr_fetch.py`.
2. `wsp/trocr_subject.py` and tests; `data/practice/trocr-blb-train-300.*` and the audit result.
3. `docs/TROCR_READINGS.md` (stage 1), `docs/TROCR_NULLS.md` (stage 2), `docs/TROCR_CONTROLS.md` (stage 3), `docs/TROCR_REPORT.md` (stages 4 to 6) and run folders under `runs/`.
4. A row for each stage in docs/RESULTS.md and docs/PROJECT_STATE.md.

## Risks and costs

- Disk: about 14 GB for the stage-1 models plus run folders. Not on h1 (5.5 GB free); on asteraix under `~/atr-cache/trocr/`.
- Time: stage 0 about one session; stages 1 and 3 each one to two sessions; stage 2(b) and stage 5 about 15 GPU hours in total on the trainer's card (the 3-epoch recipe took 0.44 kWh); stage 4 depends on Gate 0.
- The practice-to-held-out gap seen in every earlier track (practice gains of up to ten points that vanish on held-out) is the expected outcome; the track is worth doing because the **architecture is different**, so that "it also fails here" or "it works here" is informative either way.
- The `no_repeat_ngram_size` 3 and the beam of the trainer's eval are part of how the recorded CERs were produced: our C0 may differ from the card by a fraction of a point; this is reported, not tuned away.
- Comparability across architectures needs one metric: NFD CER by `wsp.metrics` on the same lines; the three tracks' held-out sets overlap only in `blb-eval-hand`.

## Open decisions for Tobias

1. Primary subject: `from-kurrent-v2` (recommended), or `trocr-kurrent`, or both in parallel.
2. May the stage-2 seeds and the stage-5 retrain go through the trainer (about 15 GPU hours, card as the trainer allocates), and when?
3. May stage 0 download the two private models to asteraix (read, with the user's hub credentials)?
4. Is a second architecture-family comparison wanted at the end (large models, the Swedish and CATMuS fine-tunes), or only the Bern chain?
