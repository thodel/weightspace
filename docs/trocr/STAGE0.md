# TrOCR track, stage 0 report (11 November 2026)

Session on asteraix (dhserver03), C0 runs 09:52 to 11:38 CET. Brief: `docs/BRIEF_TROCR_STAGE0.md`; epic: `docs/epics/EPIC_TROCR_TRACK.md`. Nothing here trains, uploads or
writes to the hub, the registry or the trainer; card 1 was not touched; the kraken venv was not touched. Weights were fetched read-only to `~/atr-cache/trocr/` and are not committed.

**Gate 0 verdict: pass, with three conditions (section 8).** C0 is byte-identical on repetition on card 0 for all three subjects on both sets, and CPU and card 0 read the same text; a pass over the
300 practice lines costs about 35 s on card 0 and about 9 to 11 min on a quiet CPU; the recorded CER of the cards is not reproduced because it cannot be (other material), and the number we get is the baseline.

## 1. What was fetched

`python scripts/trocr_fetch.py --set core --private-ok` (open decision 3 answered yes by Tobias; credentials as `huggingface_hub` finds them, never printed). Seven repositories, revision pinned at fetch,
sha256 in `docs/trocr/DIGESTS.json` (model.safetensors, tokenizer files, configs; plus `docs/trocr/stage0_checks.json` for everything the checks computed). Not fetched: `extra`, `all`.

| repo | visibility | revision | model.safetensors sha256 (first 16) | bytes |
|---|---|---|---|---:|
| `microsoft/trocr-base-handwritten` | public | eaacaf452b06 | 25a40cddc7e61201 | 1,333,384,464 |
| `dh-unibe/trocr-kurrent` | public | 66e374e1e465 | 7752d0843f90ff19 | 1,335,747,112 |
| `dh-unibe/trocr-kurrent-XVI-XVII` | public | a6bb6086ba3f | dfb88d0a11fecc1f | 1,335,747,112 |
| `dh-unibe/trocr-german-kurrent-xvi-xix-from-kurrent-v2` | public | ea9383f3c89d | 054474de1f621f61 | 1,335,747,032 |
| `dh-unibe/trocr-german-kurrent-xvi-xix-v2` | private | f8a97c30931c | 0d8fca80f4dd8baf | 1,335,747,032 |
| `dh-unibe/trocr-german-kurrent-xvi-xix-full-v1` | private | d8dce8518ada | 870235a8c7612b5b | 1,335,747,032 |
| `dh-unibe/trocr-medieval-escriptmask` | public | 3a65a4d93cf7 | e7105d440a774249 | 1,335,747,032 |

Environment (own venv `~/atr-cache/trocr/venv`, versions as the trainer's `trocr-train` venv): torch 2.8.0, transformers 4.57.6, safetensors 0.8.0, tokenizers 0.22.2, huggingface_hub 0.36.2,
numpy 2.5.4, pillow 13.0.0, Python 3.12.3.

## 2. Tokenizer identity

`scripts/trocr_checks.py`, result in `docs/trocr/stage0_checks.json`.

- **Vocabulary, merge list and special tokens are identical in all seven models:** 50,265 entries, 50,002 merge lines, `<s>`=0, `<pad>`=1, `</s>`=2, `<unk>`=3, `<mask>`=50264. Encoding the 300 practice lines with each
  model's own tokenizer gives identical id sequences in all seven (`encoding_identical: true`).
- **The files are not byte-identical,** and the byte-level comparison therefore says "differ" although the content does not: `vocab.json` has three serialisations (798,293 bytes in `trocr-kurrent` and the three
  v2/full-v1 models, 898,822 bytes in `microsoft`, `trocr-kurrent-XVI-XVII`; absent in `trocr-medieval-escriptmask`); `merges.txt` is byte-identical in six and absent in `trocr-medieval-escriptmask`;
  `tokenizer.json` has four variants (`trocr-kurrent` differs in the `normalized` flag of `<mask>`; `trocr-medieval-escriptmask` carries a fixed padding of 512; `microsoft` and `-XVI-XVII` have no `tokenizer.json`);
  `special_tokens_map.json` and `tokenizer_config.json` differ in formatting. None of this changes an id.
- **Embedding shape:** `decoder.model.decoder.embed_tokens.weight` is 50,265 x 1,024 in all seven.
- **What it does to name-aligned tensor comparison:** nothing. Every token row means the same thing in every model, so a tensor comparison by name (and a token-frequency analysis) is valid across the family.
  The tensor shapes are identical in all seven, with one difference that is not a shape: `microsoft/trocr-base-handwritten` has no `encoder.pooler.*` (478 tensors in the file against 480), see section 3.

## 3. Architecture as counted (`TrocrSubject.describe`, on `from-kurrent-v2`; all seven have the same shapes)

| component | parameters |
|---|---:|
| encoder attention (q, k, v, o, weights and biases) | 28,320,768 |
| encoder FFN | 56,669,184 |
| encoder norms (layer norms, final) | 38,400 |
| encoder embeddings (patch projection, cls, positions) | 1,034,496 |
| **encoder, in the forward** | **86,062,848** |
| decoder self-attention | 50,380,800 |
| decoder cross-attention | 44,089,344 |
| decoder FFN | 100,724,736 |
| decoder norms | 73,728 |
| decoder embeddings (tokens 51,471,360, positions 526,336, embedding norm 2,048) | 51,999,744 |
| **decoder** | **247,268,352** |
| ViT pooler (not read by the TrOCR forward) | 590,592 |
| **stored parameters** | **333,921,792** (forward: 333,331,200) |

Two corrections to `docs/MODELS_TROCR.md` (edited there):

1. **The decoder is tied.** The hub files hold one embedding tensor and no `output_projection`; transformers ties the output projection to it on load (the decoder sub-config says `tie_word_embeddings: true`
   although the top-level config says false). Editing the token matrix therefore edits input and output together. `TrocrSubject` exposes one tensor, `dec.embed.tokens`, and accepts `dec.out.weight` as an alias.
   Consequences for the epic: item 5 of its "against" list (embedding and output projection are separate) does not hold; "rare rows move most" is a property of one matrix; the planted-defect analogue
   "bias shift in the output" has no bias to shift (the output layer has none); "zero a token row" damages input and output of that token.
2. **The base has no pooler,** the children do: the pooler is initialised at random when a model is first trained from the base and then never trained (no gradient reaches it), so it is a **free lineage marker**: its weight is
   byte-identical in `trocr-kurrent`, `trocr-kurrent-XVI-XVII`, `from-kurrent-v2` and `full-v1` (one chain), and different in `v2` and in `medieval-escriptmask` (each started from the base separately), which
   agrees with the cards. It does not touch the forward. `TrocrSubject` leaves it out of `tensors()` (and so out of the digest) and writes it back unchanged on save; loading the base would otherwise draw it at random each time.

## 4. `wsp/trocr_subject.py`

`TrocrSubject(Subject)`: canonical names (`enc.07.attn.q.weight`, `enc.03.ffn.in.bias`, `dec.11.cross.k.weight`, `dec.00.self.o.weight`, `dec.05.norm_cross.weight`, `dec.embed.tokens`, `dec.embed.pos`, `dec.embed.norm.*`,
`enc.embed.patch.*`, ...; 478 editable tensors; unknown keys would appear under `other.`, none do); float32 numpy views and float32 model on any device; `set_tensor` (shape fixed); `describe` (counted); `forward` (generation, pinned
settings, `Prediction.extra` = tokens, at_cap, beam score); `teacher_forced` (per-token NLL and margin on the reference, labels `<s> text </s>`, decoder input shifted right behind the start token 0, which is the
alignment the model generates and was trained with: generated ids start `[0, 0, ...]`; the alternative alignment scores 3.5 against 1.6 nats on 8 lines); `logits`; `digest`; `save` (hub format, tokenizer and
preprocessor files copied). Tests: `tests/test_trocr_subject.py`, 10 tests on a tiny random `VisionEncoderDecoderModel` and a tiny byte-level BPE built offline (naming is a bijection, counted parameters, tied
alias, edit changes the model and the readout, digest moves and returns, generation deterministic, teacher-forced numbers match the logits, save and reload round trip, 3-pixel crop). The tests skip cleanly where torch or
transformers are missing (the repository's `.venv`).

Round trip on the real `from-kurrent-v2`: `save` then reload gives the same digest (3027fea2a056...), 480 tensors under the same keys, **every tensor byte-identical to the hub file** (file size also identical,
1,335,747,032 bytes). Not tested: that the serving stack loads a saved model (no registry writes were allowed).

**Generation settings pinned** (`wsp.trocr_subject.GENERATION`): beam 4, `max_new_tokens` 256, `no_repeat_ngram_size` 3, `length_penalty` 1.0, `early_stopping` True, no sampling, `use_cache` True, float32,
batch of 8 lines in the order of the list, image processor slow (`use_fast=False`) with `input_data_format="channels_last"`. The trainer's eval (`engines/trocr_train_svc/evaluate_trocr.py`) passes beam 4,
length penalty 1.0 and early stopping, repeat 3, one line at a time. Three discoveries behind this list:

- the hub's `generation_config.json` of the models says `length_penalty` 2.0 and `use_cache` false; the trainer's job says 1.0. **Length penalty 2.0 changes 30 of 300 practice lines** (CER 21.45 to 21.64 %): it must be pinned, and it is.
- `use_cache` false and batch 1 give **byte-identical text** to the pinned setting on all 300 practice lines (card 0): they are speed settings only (`docs/trocr/c0_variants.json`).
- **A crop 3 px high has the array shape (3, W, 3), which transformers cannot tell from channels-first** and, without an explicit format, decides from the first image of the batch for the whole batch. The practice set has
  one such crop (`manuskripte/K3123-52_0027/r1l25`, 749 x 3 px); it sat at position 6 of its batch, so the first C0 runs read it correctly by luck, and batch 1 would have misread it (1 line differed). The format is now
  explicit and tested; all six card-0 runs were repeated after the fix and the predictions are byte-identical to the ones before it.

## 5. Sets and audit

`data/practice/trocr-blb-train-300.jsonl` (+ `.meta.json`): the kraken family-2 practice list **unchanged** (byte-identical file, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`;
300 lines, 114 BLB objects, BLB train side only), so that kraken, Qwen-page and TrOCR numbers on these lines are comparable. 7 crops are taller than wide and 7 references have 1 to 3 characters (rules that
`blb-eval-hand` applies and this list does not; kept as in every earlier use). Held-out: `data/heldout/blb-eval-hand.jsonl`, 2,272 lines, 29 objects, sha256 `a784ae084058...`.

- **Object level (`wsp.audit`):** practice list clean against the held-out identities (`data/practice/overlap-trocr-blb-train-300.json`), held-out set clean against the practice list
  (`data/heldout/overlap-blb-eval-hand-trocr-practice.json`): 0 shared objects, 0 shared pages.
- **Against what the models were trained on (`scripts/trocr_audit_trained.py`, `docs/trocr/audit_trained_overlap.json`).** The three trainer jobs (`from-kurrent-v2`, `v2`, `full-v1`) list 20 datasets (Zurich Regierungsratsprotokolle,
  kurrent-xix with its project names, Rats- und Richtebücher, Königsfelden charters, Bullinger, AAEB, Grundbuch Basel, German 20th century, Parlamentsdienste, Bundesversammlung, Weisthümer, gt-fraktur, fibeln, dtgt, dach-gt,
  Reichsanzeiger, sg-missiven, ...); none is a Laßberg/BLB dataset or project. The job folders hold every page XML (1,681 pages, 79,918 lines for `from-kurrent-v2` and for `v2`, the same count as the card says for the
  shared data; 16,633 pages, 542,340 lines for `full-v1`). Scan of all page XML for BLB shelfmarks, "Laßberg", "Badische Landesbibliothek", "BLB": **0 identity hits**; one page of `full-v1` (a printed Fraktur book of 1875)
  mentions "Laßberg" inside a transcribed line (a literary-history text about his correspondence); that is text of another document, not the BLB document, and is listed in the JSON as `in_transcribed_text`.
  Exact line matches (NFD, whitespace collapsed, at least 20 characters; 249 of the 300 practice lines and 2,007 of the 2,272 held-out lines qualify): **0 in every job**, practice and held-out.
- **Limit stated, not worked around:** the parents `trocr-kurrent` (292,997 lines, "many other smaller collections") and `trocr-kurrent-XVI-XVII` (about 1.58 M words) were trained elsewhere; their corpora are described in prose on
  their cards (no mention of BLB or Laßberg) and cannot be audited at document level here. Indirect evidence that the BLB lines are unseen: CER of 20 to 28 % on both sets, far from the 3 to 9 % that each model reaches on
  what it saw, and equal on practice and held-out (21.5 / 19.7 %). The microsoft base is English (IAM).

No shared documents were found: the stop condition of the brief did not trigger.

## 6. C0

All numbers: NFD CER and WER by `wsp.metrics` (blocks off), collapsed by the current rule (07.11.2026), length ratio = hypothesis characters over reference characters, tokens = generated tokens (without the start token)
summed over lines, at cap = lines that reached 256 tokens without an end token. Table produced by `scripts/trocr_c0_table.py` from `~/atr-cache/trocr/c0/*/*.json` (summary in `docs/trocr/c0_summary.json`; the reports
carry the prediction hashes). "CER no spaces" removes every space from both sides: the BLB references put a space before punctuation, a convention the models partly lack, but it is not what drives the error.

| model | set | device | lines | CER % | WER % | collapsed | length ratio | tokens | at cap | CER no spaces % | s per pass (1st, 2nd) | repeat byte-identical |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| trocr-german-kurrent-xvi-xix-from-kurrent-v2 | blb-train-300 | cpu | 300 | 21.45 | 62.38 | 6 | 0.964 | 9548 | 0 | 20.90 | 5807 / 551 | True |
| trocr-german-kurrent-xvi-xix-from-kurrent-v2 | trocr-blb-train-300 | cpu | 40 | 17.43 | 55.66 | 2 | 0.962 | 1352 | 0 | 16.47 | 88 / 82 | True |
| trocr-kurrent-XVI-XVII | trocr-blb-train-300 | cpu | 40 | 18.06 | 41.59 | 1 | 0.996 | 1424 | 0 | 20.04 | 2693 / 188 | True |
| trocr-kurrent | trocr-blb-train-300 | cpu | 40 | 21.96 | 47.40 | 1 | 0.977 | 1216 | 0 | 24.16 | 81 / 82 | True |
| trocr-german-kurrent-xvi-xix-from-kurrent-v2 | blb-eval-hand | cuda:0 | 2272 | 19.74 | 60.41 | 9 | 0.965 | 64912 | 0 | 19.51 | 238 / 234 | True |
| trocr-kurrent-XVI-XVII | blb-eval-hand | cuda:0 | 2272 | 19.83 | 49.66 | 14 | 0.987 | 67728 | 0 | 21.65 | 240 / 242 | True |
| trocr-kurrent | blb-eval-hand | cuda:0 | 2272 | 26.41 | 56.80 | 24 | 0.964 | 61840 | 0 | 28.73 | 232 / 230 | True |
| trocr-german-kurrent-xvi-xix-from-kurrent-v2 | trocr-blb-train-300 | cuda:0 | 300 | 21.45 | 62.38 | 6 | 0.964 | 9548 | 0 | 20.90 | 37 / 35 | True |
| trocr-kurrent-XVI-XVII | trocr-blb-train-300 | cuda:0 | 300 | 21.17 | 48.58 | 6 | 0.991 | 10064 | 0 | 22.67 | 34 / 35 | True |
| trocr-kurrent | trocr-blb-train-300 | cuda:0 | 300 | 27.59 | 56.10 | 9 | 0.961 | 9124 | 0 | 29.74 | 34 / 32 | True |

| model | lines compared | texts differing CPU vs card 0 | max abs beam-score difference |
|---|---:|---:|---:|
| trocr-german-kurrent-xvi-xix-from-kurrent-v2 | 40 | 0 | 3.4e-05 |
| trocr-kurrent-XVI-XVII | 40 | 0 | 1.1e-05 |
| trocr-kurrent | 40 | 0 | 7.5e-06 |

Notes on the table. (a) Rows `cuda:0`: card 0, float32, batch 8, 2 passes in one process; the first pass includes warm-up (a few seconds). Card 0 had 28 to 45 GB free at every start (`wsp.card.ready`, nothing waiting);
peak allocated by a pass 5.6 to 6.3 GB. (b) Row `blb-train-300 / cpu / 300`: the full practice pass on CPU, 16 threads, made with the first version of the subject (before the image-format fix, which does not change this
set's predictions) from the identical list under its earlier name; its texts are identical to card 0's on all 300 lines (largest beam-score difference 3.4e-5). (c) Rows `cpu / 40`: the first 40 lines of the practice set, 16 threads,
for the other two subjects and for `from-kurrent-v2` again. (d) **CPU times depend entirely on the load of the box,** which other sessions' planted-defect runs drove to 100 to 175 (48 cores): the first pass of the 300-line
run took 5,807 s (19.4 s/line) at load 100 to 170 and its second pass 551 s (1.84 s/line) at load 14 to 37; `trocr-kurrent-XVI-XVII` took 2,693 s for its first 40 lines (67 s/line, load 174) and 188 s for the second pass; the runs at
load 29 to 41 took 2.0 to 2.2 s/line. Times are CET-labelled in the reports (`passes[].started`).

CPU against card 0 on the same lines:

| model | lines compared | texts differing CPU vs card 0 | max abs beam-score difference |
|---|---:|---:|---:|
| trocr-german-kurrent-xvi-xix-from-kurrent-v2 | 40 | 0 | 3.4e-05 |
| trocr-kurrent-XVI-XVII | 40 | 0 | 1.1e-05 |
| trocr-kurrent | 40 | 0 | 7.5e-06 |


**Repetition.** All 6 card-0 runs and all CPU runs repeated the pass in the same process: prediction files (text, tokens, beam score) byte-identical in every case (`byte_identical_repeat: true`). Across processes: the
card-0 runs before and after the image-format fix gave byte-identical predictions on both sets for all three subjects. Across devices: identical text, scores to 3e-5.

**Teacher-forced readout** (second readout, one forward pass over the reference, `TrocrSubject.teacher_forced`; card 0): mean NLL per label token (nats, including `<s>` and `</s>`) and the share of tokens whose reference logit is
not the maximum.

| model | held-out NLL | held-out share margin <= 0 | practice NLL | practice share margin <= 0 |
|---|---:|---:|---:|---:|
| from-kurrent-v2 | 2.082 | 31.6 % | 2.292 | 33.2 % |
| trocr-kurrent-XVI-XVII | 2.351 | 29.1 % | 2.454 | 28.9 % |
| trocr-kurrent | 2.873 | 36.1 % | 3.257 | 36.9 % |

**Variants** on the 300 practice lines, `from-kurrent-v2`, card 0 (`docs/trocr/c0_variants.json`): pinned 21.45 % (32 s); `use_cache` false 21.45 %, 0 lines differ (100 s); batch 1 21.45 %, 0 lines differ (66 s);
hub length penalty 2.0 21.64 %, 30 lines differ (29 s).

## 7. Against the cards' recorded CER

| model | card CER (its own split) | our C0 held-out `blb-eval-hand` | our C0 practice |
|---|---|---:|---:|
| `trocr-kurrent` | eval 2.83 %, test 2.66 % (Kurrent 19th c.) | 26.41 % | 27.59 % |
| `trocr-kurrent-XVI-XVII` | eval 5.67 %, test 5.42 % (16th to 18th c., Swiss-biased) | 19.83 % | 21.17 % |
| `from-kurrent-v2` | 8.89 %, WER 24.71 % (500 lines of its 20 datasets) | 19.74 % (WER 60.41 %) | 21.45 % |

The cards measure each model on lines from its own training corpora; ours are Laßberg's correspondence and manuscripts (BLB), seen by none of the three (audit, section 5). A factor of 2 to 10 is what unseen hands cost, and
the order of the models on BLB differs from the card order (the 19th-c. model, best on its own split, is worst here; `from-kurrent-v2` and `XVI-XVII` are equal in CER while `XVI-XVII` has a much lower WER, 49.7 against
60.4 %, and a length ratio closer to 1). This is reported, not tuned: no setting was chosen to approach a card.
Reference point for the track: the kraken `xix-blb-v1`, trained on the BLB train side, reads `blb-eval-hand` at 9.55 %; the TrOCR models read it unseen at 19.7 to 26.4 %. The two are not the same experiment
(trained against not trained on this material), and the practice-to-held-out gap of the earlier tracks has no TrOCR counterpart yet: practice CER (21.5 %) is not lower than held-out CER (19.7 %), as it must be for
lines that no model has seen.

## 8. Gate 0

Gate text: "C0 on the held-out set reproduces the recorded behaviour (CER in the plausible range of the card and byte-identical on repetition) and the time per pass allows a session of 10 passes in under a day".

- **Deterministic: yes.** Byte-identical on repetition (6 of 6 on card 0, 4 of 4 on CPU), stable across the speed settings (cache, batch), across processes and across devices (text). Save and reload is lossless to the byte.
- **Plausible: yes, with the reading above.** The card CERs are on other material and cannot be reproduced; the C0 numbers (19.7 to 26.4 %) are consistent with unseen Kurrent hands, the sets are audited, no model has seen
  the practice or held-out lines as far as the corpora can be examined (parents excepted, section 5).
- **Cost per pass.** On card 0: 300 practice lines 32 to 38 s (0.11 to 0.13 s/line), 2,272 held-out lines about 4 min. A session of 10 passes plus baseline and one held-out reading is about 8 min of card time, with 6 GB
  of the card: far inside the 12 GB rule and inside "a day" by two orders of magnitude (100 sessions would be about 13 GPU hours). On CPU: 2.0 s/line on a box at load below 40, so 10 minutes
  per practice pass and about 2 hours per session; at the load seen this morning (100 to 175) 19 to 67 s/line, so one pass takes 1.6 to 5.6 hours and a session does not fit a day. **CPU is a fallback only; card 0 is the
  engine**, as the epic expected, and the pass budget per session (10) is not limited by cost.

**Recommendation: continue to stage 1** (readings; no pass beyond C0), with these conditions for the later stages:

1. Decision 1 (primary subject) is still open; C0 exists for both candidates and for the parent: `from-kurrent-v2` and `XVI-XVII` have equal CER on BLB, `trocr-kurrent` is 5 to 6 points worse. The recommended subject stays `from-kurrent-v2`.
2. Pin the generation settings of section 4 in every preregistration (length penalty 1.0, not the hub's 2.0), and keep the batch size fixed between compared passes.
3. Redefine the stage-3 planted defects for a tied embedding and an output layer without bias (section 3); and say in the preregistration that edits of `dec.embed.tokens` change input and output together.
4. Audit gap: the parents' corpora are not enumerated; ask the model's authors (Tobias is one) whether any Laßberg or BLB material was in the Kurrent dataset of `trocr-kurrent` or in the XVI-XVII corpus. Until then
   the "unseen" claim for BLB rests on the CER level and the cards' prose.
5. Held-out lines are read once per frozen model: the C0 readings above used `blb-eval-hand` for the three baselines only, as the epic allows (C0 is the baseline, no edit exists yet).

Not done, by instruction: C0 for `v2`, `full-v1`, `medieval-escriptmask`, the base (digests only); stages 1 to 5; registry or hub writes; serving-stack load of a saved model.

## 9. Reproduction

```bash
export TMPDIR=~/atr-cache/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
V=~/atr-cache/trocr/venv/bin/python
$V scripts/trocr_fetch.py --set core --private-ok          # once, needs hub credentials
$V scripts/trocr_checks.py                                  # tokenizer identity, shapes, counts, adds digests
$V scripts/trocr_audit_trained.py                           # training-data audit (reads the research share, slow on CIFS)
$V scripts/trocr_c0.py ~/atr-cache/trocr/dh-unibe__trocr-german-kurrent-xvi-xix-from-kurrent-v2 data/heldout/blb-eval-hand.jsonl \
   --device cuda:0 --repeat --teacher-forced --roundtrip --out ~/atr-cache/trocr/c0/gpu     # card 0 after wsp.card.ready
$V scripts/trocr_c0_variants.py <model dir> data/practice/trocr-blb-train-300.jsonl --out <file>
$V scripts/trocr_c0_table.py
$V -m pytest tests -q          # TrOCR venv: 89 passed, 5 skipped (10 of the passes are tests/test_trocr_subject.py); repository .venv: 91 passed, 3 skipped (that module skips without torch)
```
