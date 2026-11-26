# Cross-vendor programmers: Gemini 4 Argon and GPT-6.2 Astra (dhserver03, 19.11.2026)

Plan: `docs/XVENDOR_PLAN.md`. This report covers **E2**, kraken session 2 on `kraken-bern-combined-v1`
(§1–5), **E1**, the phase-5 Kurrent adapter (§6), and **E3**, the planted-defect controls (§7, added
26.11.2026).

**E3 in short:** on the 14 kraken planted defects and the adapter defect, GPT did as well as Sonnet and
Fable, and Gemini somewhat worse: it shipped nothing in four of its 30 cells and only partly
repaired the one unrelated-donor defect the others repaired. Where the parent is the donor, all four programmers repaired
the same defects with the same one-tensor edits. Where an unrelated donor stands in, GPT repaired or
partly repaired five of six cells, against two for Sonnet, three for Fable and one for Gemini; that is one
attempt per cell. All four adapter cells reached
the adapter Sonnet and Fable reached.

**E1 in short:** Gemini and GPT kept the same single edit and froze on byte-identical adapters. On
practice it gains 7.8 points; on held-out it is a null (+1.13 points, page interval [−3.0, +5.2]). Two of
three random directions removed from the same tensors would also have been kept on practice. This
repeats Fable's phase-5 result with a slightly different edit.

**E2 in short:** on kraken session 2, both new programmers did what Fable did. Each moved the whole
network toward the parent, and each gained about as much on held-out. Gemini kept two edits, and GPT
one blend of all three models.

| Gemini 4 Argon | GPT-6.2 Astra |
|---|---|
| held-out **−0.65 points**, page interval [−1.38, +0.00] | held-out **−0.57 points**, page interval [−1.46, +0.24] |

All of both gains lies on the Hertwig pages: −1.76 points for Gemini, −2.13 for GPT. GPT also paid
+0.67 on Schlussredaktion, with a page interval [+0.18, +1.21] that excludes 0.

Neither page interval lies wholly below 0. Gemini's touches 0, and Fable 5.1's excluded it at −0.05.
The three programmers are indistinguishable from one another: pairwise, every difference is under 0.1
point. All ten random edits of the same size (C5 shams) made held-out worse.

A third Fable run, with Fable 5.2 as programmer (`runs/20261115-bern-combined-v1-f2-b10-1`,
`docs/KRAKEN_S2_FABLE52.md`), fits the same picture. It lowered held-out CER by 0.65 points, with a page
interval of [−1.64, +0.24], by stepping 25 % toward the parent. It was added to §3 and §5 on 19.11.

## 1. Set-up

The set-up follows plan §3:
- **The same as the Fable run:** subject, practice set (`bern-train-300`, the training side), held-out
  set (`bern-eval`), budget 10, keep rule, seed and brief. The runs were built by
  `scripts/xv_prepare.py` from `runs/20261111-bern-combined-v1-f1-b10-1`, and the two briefs are
  identical apart from the run name.
- **The programmers:**
  - Gemini: `gemini-4-argon`, through `generateContent`;
  - GPT: `gpt-6.2-astra`, through the OpenAI Responses API;
  - both with the provider's default settings, through `wsp.programmer`, with the tool layer of
    plan §3.2.
- **Gate 0 (19.11.):**
  - both ids were in the keys' model lists, and both answered a tool call;
  - the transcripts show the served model on every call as the pinned id.
- **A change found at Gate 0:** GPT-6.2 Astra refuses function tools on Chat Completions unless
  reasoning is switched off. The OpenAI client therefore uses the Responses API (commit `b82fd59`),
  so the model runs at its own default reasoning setting.
- **Operator:** a Claude Opus 5.5 session. It ran each program in the sandbox (CPU) and passed back
  only the stdout. A driver script, `~/atr-cache/wsp/xv/drive.sh`, alternated sandbox runs and
  rounds until freeze. There were no nudges, and every round ended with `finish_round`.

| | Gemini 4 Argon (g1) | GPT-6.2 Astra (o1) |
|---|---:|---:|
| rounds | 4 | 5 |
| API calls | 54 | 29 |
| input / output tokens | 6.65 M / 63.6 k | 0.87 M / 16.5 k |
| reasoning tokens reported | 0 (none reported) | 500 |
| `run_python` reading scripts | 34 | 7 |
| refused tool calls | 0 | 2: writes to `preregistration.md` outside §5, corrected by the model |

**Reasoning:** at the provider defaults, GPT-6.2 Astra reasoned very little (500 tokens over the whole
session). Gemini reported no thought tokens. Fable also ran at its default effort. Whether more
reasoning changes the outcome is open.

## 2. The sessions (practice, 300 lines)

Each row is against the last kept state, with the line interval in CER points.

**Gemini 4 Argon** (`runs/20261119-bern-combined-v1-g1-b10-1`): 10 of 10 passes, 9 edits, 2 kept.

| pass | edit | CER | Δ vs kept [line 95 %] | decision |
|---:|---|---:|---|---|
| 1 | C0 | 0.1456 | | baseline, reproduced |
| 2 | H1: blank bias back to the parent's (+0.998 → +0.879) | 0.1451 | −0.06 [−0.14, +0.03] | refused |
| 3 | H2: `O_18` to the midpoint of parent and sibling, rows aligned by character | 0.1425 | −0.31 [−0.61, −0.02] | **kept** |
| 4 | H3: whole network to the parent–sibling midpoint | 0.1461 | +0.36 [−0.63, +1.24] | refused |
| 5 | H4: whole network blend 50 % subject, 20 % parent, 30 % sibling | 0.1356 | −0.69 [−1.31, −0.14] | **kept** |
| 6 | H5: parent–sibling merge, α 0.68 | 0.1405 | +0.50 [+0.14, +0.83] | refused |
| 7 | H6: layer-graded three-way blend | 0.1361 | +0.06 [−0.14, +0.26] | refused |
| 8 | H7: blank bias −0.05 | 0.1352 | −0.04 [−0.08, 0.00] | refused |
| 9 | H8: uniform blend 35 / 25 / 40 | 0.1353 | −0.03 [−0.22, +0.16] | refused |
| 10 | H9: norm-restored blank bias −0.10 | 0.1346 | −0.09 [−0.19, +0.01] | refused |

**GPT-6.2 Astra** (`runs/20261119-bern-combined-v1-o1-b10-1`): 9 of 10 passes, 8 edits, 1 kept.

| pass | edit | CER | Δ vs kept [line 95 %] | decision |
|---:|---|---:|---|---|
| 1 | C0 | 0.1456 | | baseline, reproduced |
| 2 | whole network: 35 % subject, 30 % parent, 35 % sibling | 0.1366 | −0.90 [−1.78, −0.14] | **kept** |
| 3 | `L_16` and head 40 % toward the sibling | 0.1356 | −0.10 [−0.29, +0.09] | refused |
| 4 | blank bias −0.8 | 0.1350 | −0.16 [−0.43, +0.09] | refused |
| 5 | clean blend 40 % parent / 60 % sibling | 0.1400 | +0.34 [+0.06, +0.63] | refused |
| 6 | blank −1, space −0.5, `/` −0.95, `D` +1.4 | 0.1350 | −0.16 [−0.52, +0.17] | refused |
| 7 | early layers toward the parent, late layers toward the sibling | 0.1439 | +0.73 [+0.36, +1.10] | refused |
| 8 | restore the subject's `L_16` and head | 0.1383 | +0.18 [−0.13, +0.52] | refused |
| 9 | early convs and `L_12` halfway toward the sibling | 0.1370 | +0.05 [−0.20, +0.29] | refused |

## 3. Held-out (545 lines, 14 pages, scored once)

Each row is against C0, the unedited subject. "page" is the page-clustered interval (A4). All numbers
are CER points; 2,000 resamples.

| programmer | held-out | Δ all, page [95 %] | Δ Hertwig, page | Δ Schlussred., page |
|---|---:|---|---|---|
| C0 | 0.1933 | | | |
| Opus 5.5 (session 2) | 0.1948 | +0.15 [−0.29, +0.60] | −0.30 [−0.82, +0.21] | +0.51 [−0.04, +1.08] |
| Fable 5.1 | 0.1875 | −0.57 [−1.15, −0.05] | −1.32 [−2.36, −0.60] | +0.02 [−0.41, +0.52] |
| Fable 5.2 (f2, added 19.11.) | 0.1868 | −0.65 [−1.64, +0.24] | −2.35 [−3.51, −1.40] | +0.69 [+0.19, +1.33] |
| **Gemini 4 Argon** | **0.1868** | **−0.65 [−1.38, +0.00]** | −1.76 [−2.74, −0.96] | +0.23 [−0.22, +0.76] |
| **GPT-6.2 Astra** | **0.1876** | **−0.57 [−1.46, +0.24]** | −2.13 [−3.28, −1.21] | **+0.67 [+0.18, +1.21]** |

The line intervals are:
- Gemini −0.65 [−1.05, −0.29];
- GPT −0.57 [−1.00, −0.18].

Pairwise on held-out, page interval:

| comparison | Δ, page [95 %] |
|---|---|
| Gemini − Fable | −0.08 [−0.48, +0.29] |
| GPT − Fable | +0.01 [−0.53, +0.56] |
| GPT − Gemini | +0.08 [−0.25, +0.44] |

No pairwise intervals against Fable 5.2 were computed. Its held-out CER is 0.1868, the same as
Gemini's to four places.

Collapsed held-out lines: C0 34; Fable, Gemini and GPT 22–23.

## 4. Controls (C5 sham)

Script: `scripts/sham_combined_xv.py`. Results: `runs/…-{g1,o1}-b10-1/controls.json`.

How the shams were built:
- every tensor of the kept diff gets Gaussian noise with that tensor's Frobenius norm (Gemini 42.19,
  GPT 47.61);
- seeds 20261111–20261115, the same seeds as Fable's shams;
- each is scored on practice and held-out against C0.

| | practice Δ (range of 5) | keep rule | held-out Δ (range of 5) |
|---|---|---|---|
| Gemini shams | +0.03 to +0.93 | 0 of 5 kept | +0.44 to +0.75 |
| GPT shams | +0.06 to +0.76 | 0 of 5 kept | +0.37 to +0.68 |

So the direction of the edits matters, not their size, as it did for Fable. C0, C1 and C6 are
unchanged from `docs/baselines/` and phase 0.

No component split was run:
- GPT's kept edit is a single blend;
- Gemini's two kept edits overlap on `O_18`: H4 rewrote the output layer that H2 had set.

## 5. The five programmers compared

| | Opus 5.5 | Fable 5.1 | Fable 5.2 | Gemini 4 Argon | GPT-6.2 Astra |
|---|---|---|---|---|---|
| passes / edits / kept | 9 / 8 / 1 | 9 / 8 / 3 | 8 / 7 / 2 | 10 / 9 / 2 | 9 / 8 / 1 |
| kept move | parent's output layer, transplanted | blank −0.40, then 12.5 % toward the parent | 25 % toward the parent, then blank −0.30 | `O_18` to the parent–sibling midpoint, then 50/20/30 subject/parent/sibling | 35/30/35 subject/parent/sibling |
| practice Δ | −0.42 | −0.99 | −0.96 | −1.01 | −0.90 |
| held-out Δ, page | +0.15 [−0.29, +0.60] | −0.57 [−1.15, −0.05] | −0.65 [−1.64, +0.24] | −0.65 [−1.38, +0.00] | −0.57 [−1.46, +0.24] |
| Hertwig / Schlussred. | −0.30 / +0.51 | −1.32 / +0.02 | −2.35 / +0.69 | −1.76 / +0.23 | −2.13 / +0.67 |

- **One direction, five programmers.** Every programmer gave the parent, `hertwig-v2`, weight against
  the subject. The parent was trained on the correct Hertwig images; the subject learnt Hertwig from
  mismatched images. Whatever they kept, the held-out gain sits on the Hertwig pages.
- **What differs is how much Schlussredaktion pays.**
  - Fable 5.1 kept the subject at 87.5 % and Schlussredaktion was unchanged.
  - Fable 5.2 kept it at 75 %: Hertwig −2.35, Schlussredaktion +0.69 with a page interval that
    excludes 0. Its own component split puts the whole held-out effect in the step toward the parent
    (−0.63); the blank step alone gave +0.01.
  - Gemini kept the subject at 50 %, plus 30 % sibling: Schlussredaktion +0.23, not resolved.
  - GPT kept the subject at only 35 %: the largest Hertwig gain, and a Schlussredaktion loss whose
    page interval excludes 0.
  - Opus transplanted the parent's whole output layer and lost the most on Schlussredaktion.
- **On practice, the training-side confound of amendment A3 is still at work.**
  - Gemini and GPT both state in their pre-registrations that the parent was trained on the
    103 Hertwig practice lines.
  - Both still ended with practice gains of 0.9–1.0 points, larger than their held-out gains of
    0.57–0.65.
- **Limits.**
  - One attempt per programmer model. The two Fable runs differ by version as well as by run, so
    they don't measure run-to-run variation either.
  - Gemini's page interval reaches 0, and GPT's and Fable 5.2's include it. Fable 5.1's excluded 0,
    but by 0.05.
  - Pairwise, no two programmers differ by more than 0.1 point on held-out.
  - GPT and Gemini reasoned little or not visibly at their default settings.

## 6. E1: the phase-5 Kurrent adapter (`kurrent-page-v2`)

### 6.1 Set-up

The same as §1, with these differences:
- template `runs/20261110-kurrent-page-v2-f1-b10-1`; practice `kurrent-val-64` (64 pages); held-out
  `kurrent-test` (733 pages); budget 10; the engine is `evaluate_qlora` on card 0;
- the brief carries the two errata Fable found (plan §3.4), recorded as a deviation in each
  pre-registration;
- **deviation:** both sessions ran **in parallel** on card 0, not one after the other as plan §4 says,
  and the shams ran in parallel with held-out. Card 1 was never used. C0 reproduced to the last digit in
  both runs (0.4515, 9 pages at the cap) and the two H1 passes were identical, so sharing the card did
  not touch determinism. It only halved the speed of each job.

| | Gemini 4 Argon (g1) | GPT-6.2 Astra (o1) |
|---|---:|---:|
| rounds | 7 | 4 |
| API calls | 64 | 24 |
| input / output tokens | 7.02 M / 64.4 k | 0.82 M / 11.8 k |
| reasoning tokens reported | 0 | 376 |
| `run_python` reading scripts | 37 | 6 |
| refused tool calls | 0 | 1: a write to `preregistration.md` outside §5, corrected by the model |

Every call was served as the pinned id. There were no nudges, and every round ended with `finish_round`.

### 6.2 The sessions (practice, 64 pages)

Each row is against the last kept state, with the paired page interval in CER points.

**Gemini 4 Argon** (`runs/20261119-kurrent-page-v2-g1-b10-1`): 7 of 10 passes, 6 edits, 1 kept.

| pass | edit | CER | at cap | Δ vs kept [page 95 %] | decision |
|---:|---|---:|---:|---|---|
| 1 | C0 | 0.4515 | 9 | | baseline, reproduced |
| 2 | H1: direction 0 out of layer-31 `o_proj` and `down_proj` `lora_B` | 0.3737 | 5 | −7.78 [−17.3, −0.4] | **kept** |
| 3 | H2: direction 0 out of `down_proj` in layers 27 and 29 | 0.4464 | 7 | +7.26 [−3.0, +18.9] | refused |
| 4 | H3: a shared direction out of layers 27–29 | 0.3726 | 4 | −0.11 [−3.2, +3.0] | refused |
| 5 | H4: the rest of layer 31's `o_proj`/`down_proj` × 0.5 | 0.3962 | 5 | +2.25 [−2.9, +9.4] | refused |
| 6 | H5: the logit-lens `<\|im_end\|>` direction out of 25 suppressing writers | 0.4257 | 6 | +5.20 [−0.4, +13.0] | refused |
| 7 | H6: H1's direction inverted at −0.5 | 0.4213 | 6 | +4.76 [−0.3, +12.3] | refused |

Froze on H1 with three passes unused. The notes argue that the remaining capped pages are not driven by
the end-token direction.

**GPT-6.2 Astra** (`runs/20261119-kurrent-page-v2-o1-b10-1`): 5 of 10 passes, 4 edits, 1 kept.

| pass | edit | CER | at cap | Δ vs kept [page 95 %] | decision |
|---:|---|---:|---:|---|---|
| 1 | C0 | 0.4515 | 9 | | baseline, reproduced |
| 2 | H1: the same edit as Gemini's H1 | 0.3737 | 5 | −7.78 [−17.3, −0.4] | **kept** |
| 3 | H2: `q_proj` B × 0.65 in layers 19/23/27/31 | 0.4610 | 9 | +8.72 [0.0, +18.8] | refused |
| 4 | H3: `gate_proj` B × 0.75 in layers 24–31 | 0.3898 | 6 | +1.60 [−0.4, +4.2] | refused |
| 5 | H4: H1 extended to layers 26–27 | 0.3986 | 6 | +2.49 [−0.1, +6.6] | refused |

Froze on H1 with five passes unused: "there is no supported reason to vary another layer or dose on the
same 64 pages".

**One adapter:** both `final.adapter/adapter_model.safetensors` have sha256 `c5cde457…` (digest
`d9fc39be…`). It is not Fable's adapter (`3f68caa5…`, direction 0 out of `o_proj` only), so held-out was
scored anew, once, in the Gemini run. The GPT run's `summary.json` reuses those readings (plan §3.3) and
says so.

### 6.3 Held-out (733 test pages, scored once)

Paired page bootstrap and the document-clustered interval (A4, 70 documents), 2,000 resamples. Script
`scripts/xv_e1_compare.py`, results `docs/baselines/xv-e1-compare.json`.

| | wsp CER | trainer CER | at cap | Δ vs v2, page [doc] | pages better / worse |
|---|---:|---:|---:|---|---|
| v2, unedited (`p5-v2-test`) | 0.4012 | 0.4053 | 44 | | |
| Fable's H1 (`o_proj` only) | 0.4036 | 0.4079 | 43 | +0.24 [−3.5, +3.9] [−5.0, +6.7] | 225 / 233 |
| **Gemini/GPT H1** (`o_proj` + `down_proj`) | **0.4125** | 0.4174 | 42 | **+1.13 [−3.0, +5.2] [−4.9, +12.4]** | 256 / 250 |

Gemini/GPT H1 − Fable's H1: +0.90 [−3.4, +5.2], doc [−4.8, +9.7]. Held-out took 8.3 h on card 0, part of it
shared with the shams.

### 6.4 Controls (C5 sham, practice)

`scripts/xv_e1_sham.py`: for each seed, one seeded random unit direction out of each of the two `B`
tensors H1 edited. Seeds 20261108/10/11, the seeds of the earlier E1 shams. Results
`docs/baselines/xv-e1-sham-*.json`.

| practice, 64 pages | CER | at cap | − C0, page [doc] | H1 − it, page |
|---|---:|---:|---|---|
| C0 | 0.4515 | 9 | | |
| H1 | 0.3737 | 5 | −7.78 [−17.3, −0.4] [−16.7, −0.6] | |
| sham 20261108 | 0.3701 | 6 | **−8.14 [−16.6, −1.2]** [−16.5, −1.2] | +0.37 [−5.0, +3.9] |
| sham 20261110 | 0.3654 | 6 | **−8.61 [−18.1, −0.1]** [−18.6, −0.2] | +0.84 [−6.5, +7.7] |
| sham 20261111 | 0.4293 | 8 | −2.22 [−5.9, +0.8] [−5.7, +0.8] | −5.55 [−14.8, +0.9] |

The keep rule would have kept two of the three shams (interval below 0, pages at the cap 9 → 6). H1 does
not differ from any of them. C1, C3 and the decode comparator do not depend on the edit and are those of
phase 5 (`docs/baselines/p5-*.json`).

### 6.5 What E1 shows

- **Three vendors, one family of edit.** Opus, Fable, Gemini and GPT all removed the top reading
  direction from layer 31's writers. Gemini and GPT agreed to the byte; Fable and Opus took `o_proj`
  alone. The reading decides the edit, for every programmer tried.
- **The practice gain does not transfer.** On held-out, H1 is +1.13 points (worse, but the interval
  includes 0); Fable's version was +0.24. Neither moves the 733 test pages.
- **The shams explain the practice gain.** Removing random directions from the same tensors gains as
  much on practice. What the keep rule saw on 64 pages was mostly a reshuffle of which pages loop, as the
  Fable report argued with one tensor.
- By plan §5, this is the "different kept edit" case: it has its own held-out score and shams, and it
  lies inside the sham spread on practice and at the null on held-out. It is not a finding for the edit.
  It strengthens the phase-5 conclusion that on this adapter the 64-page keep rule does not separate a
  meaningful direction from noise.

## 7. E3: the planted-defect controls (26.11.2026)

### 7.1 Set-up

As the Fable rerun (`docs/CONTROLS_REPORT.md` §9c and the 11.11. section), with the programmer swapped; the
decisions of plan §7.1 were fixed on 20.11., before any cell was scored except the shakedown `s06/a`.

- **Cells:** 30 per vendor: the 14 kraken corruptions (k1–k11 and the three unrelated-donor `-u` variants)
  in arms (a), the programmer with 10 passes, and (b), the blind programmer with budget 0, plus adapter
  p3 (`lora_alpha` doubled), pack `q01`, arms (a) with 6 passes and (b). Same subjects, packs (copied
  unchanged under the same opaque labels), practice sets, budgets, keep rule, held-out sets and outcome
  classes as the Sonnet and Fable runs.
- **Programmers:** Gemini 4 Argon (`gemini-4-argon`, tag `g1`) and GPT-6.2 Astra (`gpt-6.2-astra`, `o1`),
  each cell a fresh conversation through `wsp.programmer`, one round, tools `read_file`, `list_dir` and
  `write_file` (with py_compile), provider default reasoning settings. Readable: the brief, the cell's own
  pack and the API files; writable: the arm's two files. Every reply was served as the pinned id (Gemini
  177 calls, GPT 194). No read was refused. Arm (a) programs were written in advance and never run by the
  programmer, as for Sonnet and Fable.
- **Classes added by plan §7.1:** a program that does not freeze ships nothing and is **failed (program
  error)**; a conversation that ends without writing a program is **failed (no program)**. Both count as
  not repaired, and neither cell is rerun.
- **Runs:** the 28 kraken cells per vendor on CPU (20.11. for `s06/a`, 26.11. for the rest; six at a time,
  four threads each), held-out `blb-eval-hand` once per frozen model. The adapter cells on card 0 only,
  26.11. 16:27–17:09 CET, through a vLLM pass server holding 0.35 of the card while 39 GB were free; card 1
  not used. Run folders `runs/202611{20,26}-planted-*-{g1,o1}-b*-1` (60).
- **Code:** `scripts/run_planted_xv.py`, `scripts/run_planted_adapter_xv.py` (the Fable adapter runner,
  now taking the cell folder, tag, programmer text and spool as arguments), `scripts/aggregate_planted_xv.py`;
  data `docs/controls/planted_xv.{json,md}`.

### 7.2 Results

Classes per arm on the 14 kraken corruptions (Fable 5.1 = the 11.11. rerun; held-out CER, recovered share
and class per cell in `docs/controls/planted_xv.md`):

| arm | programmer | repaired | partly | not repaired | null corruption | failed (program error) | failed (no program) |
|---|---|---:|---:|---:|---:|---:|---:|
| (a) | Sonnet 5.5 | 8 | 1 | 2 | 3 | 0 | 0 |
| (a) | Fable 5.1 | 7 | 2 | 2 | 3 | 0 | 0 |
| (a) | Gemini 4 Argon | 5 | 2 | 1 | 3 | 1 | 2 |
| (a) | GPT-6.2 Astra | 8 | 3 | 0 | 3 | 0 | 0 |
| (b) | Sonnet 5.5 | 7 | 1 | 3 | 3 | 0 | 0 |
| (b) | Fable 5.1 | 6 | 2 | 3 | 3 | 0 | 0 |
| (b) | Gemini 4 Argon | 6 | 1 | 3 | 3 | 0 | 1 |
| (b) | GPT-6.2 Astra | 8 | 1 | 2 | 3 | 0 | 0 |

Split by donor (11 corruptions with damage, per arm):

| | Sonnet (a) / (b) | Fable (a) / (b) | Gemini (a) / (b) | GPT (a) / (b) |
|---|---|---|---|---|
| parent donor, 8 defects: repaired + partly | 8 / 7 | 8 / 6 | 6 / 7 | 8 / 7 |
| unrelated donor, 3 defects: repaired + partly | 1 / 1 | 1 / 2 | 1 / 0 | 3 / 2 |

**Adapter p3 (score set, 64 val pages; clean 0.507, corrupted 3.764):** all four cells halve every
`lora_B` and freeze on the adapter `f5adb9da…`, byte-identical to the Sonnet and both Fable attempts:
0.517 · 100 % · repaired in each. GPT (a) 2/1/1, Gemini (a) 5/4/1 (the further edits refused), both (b)
0/1/1.

### 7.3 What E3 shows

- **With the parent as donor, the programmer barely matters.** All four repair the same row, sign, swap,
  bias, scale and zeroed-layer defects, most with the same one-tensor restore. The blind arm matches the
  arm with passes, as before: the donor does the work. The only parent-donor cell where they part is k9
  (10 % noise on every tensor): GPT (a) repaired it (98 %), Sonnet (a) 83 %, Fable (a) 24 %, and Gemini
  shipped nothing in either arm.
- **With an unrelated donor, GPT did better on this one attempt.** It repaired k6-u in both arms (as Sonnet
  and Fable did), k1-u in (b) (82 %), partly repaired k1-u in (a) (67 %) and k8-u in (a) (44 %). Sonnet
  and Fable reached 1 % and 15 % on k8-u (a). One attempt per cell, and the repeats of kraken session 2 showed
  that one model's attempts can differ as much as two models' do (`docs/S2_REPEATS.md`), so this is a
  lead, not a ranking.
- **Gemini lost cells to its harness, not to its edits.** In three cells (s08/a = k1-u (a), s10/a and
  s10/b = k9) both of its replies used the whole default output budget on reasoning (finish `MAX_TOKENS`,
  32,765 reasoning tokens, no text, no tool call), the one allowed nudge included, so no program was
  written. In the shakedown cell `s06/a` its program read `m.confusions` entries as dicts (they are
  tuples) and raised before freezing. Where Gemini shipped a program on a parent-donor defect, it repaired
  what the others repaired; with the unrelated donor it did worst, reaching 44 % on k6-u (a), which every
  other programmer repaired.
  A second Gemini attempt with a larger output cap would be a new arm (`g2`, plan §7 item 5), not a rerun.
- **Null corruptions stay null for everyone** (k5, k7, k11): no programmer harmed them beyond the class
  bound; GPT (a) on k11 recovered 48 % of a 0.31-point damage, inside the null.

### 7.4 Limits

- One attempt per cell and vendor; differences of one or two cells per arm are within what repeats of the
  same model produce.
- The packs show the diff to the parent, where almost every lineage defect is visible; the parent-donor
  cells test reading a diff, not diagnosis without a reference.
- Gemini's three empty cells depend on the client's output cap (the provider default); they say nothing
  about what Gemini would have written.

## 8. What was not done

- Nothing was registered, exported or served.
