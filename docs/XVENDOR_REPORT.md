# Cross-vendor programmers: Gemini 4 Argon and GPT-6.2 Astra (dhserver03, 19.11.2026)

Plan: `docs/XVENDOR_PLAN.md`. This report covers **E2**, kraken session 2 on `kraken-bern-combined-v1`.
E1, the phase-5 adapter, has not run yet.

**In short:** on kraken session 2, both new programmers did what Fable did. Each moved the whole
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

## 6. What was not done

- E1, the phase-5 adapter on card 0, has not run yet. Plan §2 sets it as the next step.
- E3, the planted-defect controls, waits for a go-ahead.
- Nothing was registered, exported or served.
