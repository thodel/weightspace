# 2026-11-07 · Adapter track: a QLoRA adapter of a vision-language model

**Type:** harness, readings, positive control, one session (A0–A4) ·
**Session:** `20261107-adapter-v2-b10-1` · **Budget:** 10 (7 spent) ·
**Held-out touched:** once, after freeze · **Source:** `docs/ADAPTER_REPORT.md`
(branch not yet merged) · **Result:** [R10](../results.md#r10)

## Plan

Transfer the method from kraken to a page-level transcription adapter:
`qwen3.5-4b-german-kurrent-xvi-xix-page-v2`, a rank-64 LoRA (α 128) on 128
modules of the language tower of Qwen3.5-4B. Its recorded symptom: held-out
CER 0.418 against 0.250 when the reading is clipped to the reference's length,
with 49 of 733 pages running to the token cap — it loops.

## Done

- **A0, the forward pass.** The trainer's own scorer (transformers, greedy,
  deterministic: byte-identical text across processes) for sessions; vLLM only
  for the planted-defect control.
- **A1, readings (budget 0).** Norms tight across three healthy siblings;
  adapters of one base nearly orthogonal (cosine 0.10), so no donor operations.
- **A3, positive control:** six planted adapter defects.
- **A4, one session** by a fresh programmer (Claude Sonnet 5.5) that wrote its
  whole program in advance: practice 32 validation pages, held-out 134 pages of
  the trainer's frozen test draw.

## Found

- **Whether a page loops is chaotic.** Merged weights and the peft-wrapped model
  give identical text on 3 of 10 pages (the merge itself is exact); vLLM and the
  scorer on 10 of 61. A change below bf16 resolution (every `B` × 1.002) moved
  practice CER by −8 points. So a loop-fixing edit cannot be judged from one
  pass — and merging an adapter for serving changes the very behaviour edited.
- **Positive control:** of six planted defects only two did damage beyond pass
  noise. A doubled scale (CER 3.76 → 0.52, clean 0.51) was repaired by
  programmer, blind programmer and rule list alike; four single-module defects
  did no damage — one module of 128 carries little.
- **Session A4:** six edits, one kept: layer-27 `q_proj` scaled by 0.8.

  | 134 held-out pages | CER | clipped CER | length ratio | at cap |
  |---|---:|---:|---:|---:|
  | unedited | 0.685 | 0.354 | 1.33 | 22 |
  | **frozen** | **0.569** | 0.351 | 1.22 | 18 |
  | sham, same norm, same tensor (C5) | 0.716 | 0.388 | 1.32 | 24 |

  Δ −0.115 [−0.271, −0.001] by pages ([−0.252, −0.009] clustered by
  document); frozen vs sham −0.147 [−0.300, −0.041]. **The gain is in the
  length of what is written, not in reading:** the clipped CER is unchanged,
  4 of 22 capped pages stop — the chaotic quantity above. The kept edit was the
  one whose predicted effect was smallest; the stop-token hypotheses were not
  borne out.
- **The base alone loops more** (12 of 32 practice pages at the cap, against 8
  with the adapter): the adapter already suppresses most looping, so "the loop
  is written by the adapter" is wrong as stated.

## Changes

Nothing here is a claim about language-model weights in general: it concerns a
rank-64 update of one 4 B model on one task. A second session on the same
adapter, with practice and held-out sets of its own, followed in
[phase 5](2026-11-08-a-adapter-session-v2.md).
