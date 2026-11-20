# Pre-registration — 20261119-kurrent-page-v2-g1-b10-1

Filled in **before the first pass** (PROTOCOL.md §2), 19.11.2026. This session replicates
`runs/20261107-kurrent-page-v2-b10-1`, which Claude Opus 5.5 programmed. It uses the same
subject, practice set, held-out set, budget, engine and keep rule, with a different
programmer (Gemini 4 Argon). Sections 1–4 and 6–7 were written by the operator; section 5 was written by the
programmer.

**Deviation from the template:** the brief carries 2 corrections that the Fable programmer found in the template brief (runs/20261110-kurrent-page-v2-f1-b10-1/notes.md, round 0); docs/XVENDOR_PLAN.md §3.4.

1. **Subject:** `qwen3.5-4b-german-kurrent-xvi-xix-page-v2`
   (`~/atr-cache/wsp/adapters/…`). Tensor digest
   `fb4f55ee7386abc7292e29bbf2a24f502fc4079cb4732934cbead229b300504c`, file
   sha256 `1b70ce4a2b2125bd007bd1c81789bd43c8d72c12ef49e4f50cf8ad9ca3c67897`.
   peft LoRA with r 64 and α 128 on 128 language-tower modules of
   `Qwen/Qwen3.5-4B`. **Only the adapter is edited** (PHASE5_PLAN decision 2); the base is
   read, never written.
   - **Donors:** none.
2. **Practice set:** `data/practice/kurrent-val-64.jsonl`, 64 pages, list sha256
   `2cc5278ea215b72d5f2faad9977abcde6dde36bc327876d08643852827d46f9b`.
   - **Source:** the registered **val** side of `german-kurrent-xvi-xix-v1`, which is not
     the training side (decision 1; amendment A3, item 2).
   - **Selection rule:** one page per document, and three from each of the two 160-page
     documents. Seed 20261107 (`data/practice/kurrent-val-64.meta.json`).
   - **What a pass is:** the trainer's own `evaluate_qlora` on these 64 pages. That means
     transformers 5.17, greedy decoding, cap 3,072, `max_pixels` 2,097,152, and the prompt
     from the training render. Card 0 only, and only after `wsp.card.ready`.
3. **Held-out set:** `german-kurrent-xvi-xix-v1` **test**, 733 pages from 70 documents.
   - **Identity:** `vlm-pages-test/pages.jsonl`, sha256
     `c56698a79a8f628b1bc905241696ce394044973773915bd9b11a90ca24a07bc1`; `draw_md5`
     `ae8bf1fd315e3a06eee272bb3ca7c7ad`.
   - **Scoring:** once, after freeze, by `evaluate_qlora`
     (`wsp.run heldout --set kurrent-test`).
   - **Paired comparisons:** against the unedited v2 scored on the same card and engine,
     and against the Opus session's frozen adapter. Both readings already exist from phase 5.
   - **Overlap audit** (`data/practice/kurrent-val-64.audit.json`):
     - no practice document is among the 70 test documents;
     - none is in v2's training manifest;
     - six practice pages are on v2's own validation split, which the trainer never
       trained on.
4. **Budget:** 10 passes, the first being the in-session baseline. `min_effect` 0.
   - **Keep rule:** PROTOCOL §3 on the paired page bootstrap (2,000 resamples), **and** the
     pages at the cap must not rise against the last kept state.
   - **Intervals for claims:** page-clustered, as amendment A4 requires. On a page set the
     unit is already the page, so the two coincide.
5. **Hypotheses** (written by the programmer before the first pass):

   **Structural diagnosis from budget-0 reading:**
   On the 64 practice pages (`docs/baselines/c0-kurrent-val-v2.json`), C0 has CER 0.4515
   (48,873 edits / 108,243 chars), `truncated_cer` 0.2564, `length_ratio` 1.226, and 9 pages
   at the 3,072-token cap. Those 9 at-cap pages account for 33,936 of the 48,873 total edits
   (~19.5 points of CER are text written past the end of the page). Moreover, a paired
   page-bootstrap simulation over the 64 pages shows that at least 4 of the 9 at-cap pages
   must stop looping for the 95 % bootstrap interval of ΔCER to clear 0 (`ci95` entirely
   below 0).
   In the logit-lens reading (`docs/readings/adapter_kurrent-page-v2.json`) and SVD geometry
   of the residual-writing modules (`down_proj` and `o_proj`), there is a striking late-layer
   direction cluster:
   - `layers.31.self_attn.o_proj` dir 0 ($\sigma = 2.226$) and `layers.31.mlp.down_proj` dir 0
     ($\sigma = 2.612$) are the two strongest suppressors of `<|im_end|>` across the entire
     adapter ($z = -3.60$ and $z = -3.32$), while also strongly suppressing `\n` ($z = -5.52$
     and $-1.95$) and `¬` ($z = -8.06$ and $-0.93$) and promoting out-of-domain tokens.
   - Geometrically, these two top singular directions in the final layer are strongly aligned
     ($\cos = +0.516$), and both are anti-aligned ($\cos = -0.483$ and $-0.430$) with
     `layers.30.mlp.down_proj` dir 0 ($\sigma = 3.705$, $z(\text{<|im\_end|>}) = +2.85$,
     $z(\text{\\n}) = +2.05$), directly counteracting layer 30's stopping signal right before
     the final norm and `lm_head`.
   - Upstream in the late block, `layers.29.mlp.down_proj` dir 0 ($\sigma = 2.802$,
     $z(\text{<|im\_end|>}) = -1.51$, $z(\text{\\n}) = -4.04$, $z(\neg) = -3.40$,
     $\cos(\text{L31.o\_proj:0}) = +0.296$) and `layers.27.mlp.down_proj` dir 0
     ($\sigma = 2.678$, $z(\text{<|im\_end|>}) = -2.80$, $\cos(\text{L31.o\_proj:0}) = +0.165$)
     belong to the same `<|im_end|>`-suppressing subspace.

   **Plan across the 10-pass budget:**
   - **Pass 1 (Baseline C0):** Run `s.measure("C0")` on the unedited adapter.
     - *Prediction:* Reproduces C0 exactly: CER 0.4515, `truncated_cer` 0.2564, 9 pages at cap.
       If it does not reproduce C0, stop immediately and report the discrepancy.
   - **Pass 2 — H1 (`project_out`, rank-1 direction removal on both layer-31 residual writers):**
     Project out dir 0 ($u_0$) from `layers.31.self_attn.o_proj.lora_B.weight` (`axis=1`) and
     dir 0 ($u_0$) from `layers.31.mlp.down_proj.lora_B.weight` (`axis=1`).
     - *Predicted direction and magnitude:* CER drops by 0.08 to 0.16 (to ~0.29–0.37),
       `truncated_cer` stays unchanged within $\pm 0.003$ (~0.256), and pages at the cap drop
       by 4 to 7 pages (from 9 to 2–5).
     - *What we conclude if it fails:* If at-cap pages drop by fewer than 4 (so the 95 % CI
       contains 0) or do not drop at all, removing the two layer-31 `<|im_end|>`-suppressing
       singular directions alone is either too weak to flip enough pages out of repetition
       loops or the two modules interact differently individually; we will inspect the per-page
       readings from Pass 2 (`passes/`) to see whether individual modules or upstream late-layer
       suppressors (`layers.27` / `layers.29`) are needed.
   - **Pass 3 — H2 (Conditional on Pass 2; `project_out`, upstream/downstream late-layer
     `<|im_end|>` suppressors):**
     - If H1 is **kept**: extend the edit by projecting out `layers.27.mlp.down_proj` dir 0
       ($\sigma = 2.678, z = -2.80$) and `layers.29.mlp.down_proj` dir 0
       ($\sigma = 2.802, z = -1.51, z(\text{\\n}) = -4.04$) (`axis=1` on their `lora_B.weight`).
       *Prediction:* CER drops by a further 0.03 to 0.08 against H1, `truncated_cer` remains
       within $\pm 0.003$ of baseline, and pages at the cap drop by an additional 1 to 3 pages.
       *If it fails:* The layer-31 projection already captured the removable late-layer
       stopping obstacle; upstream rank-1 removals add no statistically significant gain.
     - If H1 is **refused**: test projecting out `layers.31.self_attn.o_proj` dir 0,
       `layers.31.mlp.down_proj` dir 0, `layers.29.mlp.down_proj` dir 0, and
       `layers.27.mlp.down_proj` dir 0 jointly in a single multi-tensor `project_out` edit
       (or isolating the single strongest module if H1 hurt `truncated_cer`), guided by the
       Pass 2 per-page readings.
       *Prediction:* CER drops by 0.09 to 0.18 vs C0, `truncated_cer` stays within $\pm 0.004$,
       and pages at the cap drop by 4 to 7 pages (from 9 to 2–5).
       *If it fails:* Rank-1 removal of the top `<|im_end|>`-suppressing singular vectors in
       layers 27–31 does not resolve the page-end looping symptom.
   - **Pass 4 — H3 (`project_out`, cross-layer shared `<|im_end|>`-suppression vector or
     isolated single-module ablations):**
     - Decide based on Passes 2–3: either project the single shared layer-31
       `<|im_end|>`-suppressing unit direction $u_{\text{L31}}$ out of all late residual writers
       (`layers.{27..31}.{mlp.down_proj, self_attn.o_proj}`) simultaneously (`project_out`,
       `axis=1`), or isolate single-module dir-0 projections (`layers.31.self_attn.o_proj`
       alone vs `layers.31.mlp.down_proj` alone vs `layers.26.mlp.down_proj` dir 2).
     - *Prediction:* CER decreases by 0.03 to 0.12 vs the last kept state, `truncated_cer`
       changes by at most $\pm 0.003$, and at-cap pages decrease by 1 to 5.
     - *If it fails:* Conclude that further rank-1 direction projections in the late residual
       stream do not improve stopping beyond the already-kept state (or that the loop is not
       driven by a single linear residual direction).
   - **Pass 5 — H4 (`scale`, late-layer residual writer attenuation):**
     - Test scaling down `lora_B.weight` of layer 31 (or layers 27–31) residual-writing or
       MLP modules (`scale`, factor $0.5$ or $0.7$), where energy surges in layers 27–30.
     - *Prediction:* CER drops by 0.04 to 0.10 vs the last kept state, `truncated_cer` stays
       within $\pm 0.005$, and pages at the cap drop by 2 to 5.
     - *If it fails:* Whole-module damping in late layers trades off reading accuracy against
       stopping or does not beat surgical rank-1 projection.
   - **Passes 6–10 — H5–H9 (Follow-up hypotheses decided from Passes 1–5 results):**
     - Each follow-up will be registered as a hypothesis with an explicit predicted direction
       and magnitude for CER, `truncated_cer` and at-cap pages before its pass is run, exploring
       either: (a) mid/early-layer `<|im_end|>` suppressors (`layers.14.mlp.down_proj` dir 2,
       $z = -2.67$; `layers.3.mlp.down_proj` dir 0, $z = -2.48$; `layers.8.mlp.down_proj` dir 0,
       $z = -2.29$), (b) newline/line-break suppressors on pages that loop on punctuation/lines,
       or (c) early `s.freeze()` if the obstacle has been cleanly removed and remaining at-cap
       pages are genuine long/dense pages or base-model loops that cannot be improved without
       degrading `truncated_cer`.

6. **Controls.** If an edit is kept, they run outside the budget on the same 64 pages, with
   the same engine, before the held-out score:
   - **C0:** the adapter as it is. Done in phase 5: CER 0.4515, `truncated_cer` 0.2564,
     9 pages at the cap (`docs/baselines/c0-kurrent-val-v2.json`).
   - **C1:** the base without the adapter. Already measured in phase 5 on the same pages
     and engine; it is reused, not rerun.
   - **C3:** the α-sweep of the whole adapter at 0.75, 0.9 and 1.1. Already measured in
     phase 5; reused.
   - **C5, sham:** a random unit direction of the same module(s) as the kept edit, removed
     the same way.
   - **Decode comparator:** `repetition_penalty` 1.05 with the adapter unchanged. Already
     measured in phase 5; reused.
   - **C4:** a short QLoRA continuation. Not submitted without the user's go-ahead.
   - **C2 and C6:** not run, as in phase 5.

   Controls that do not depend on the edit (C1, C3 and the decode comparator) are a
   function of (weights, pages, engine). They were measured in phase 5 and are reused here.
   C5 depends on the kept edit and is run anew.
7. **Programmer and operator:**
   - **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer`
     with the provider's default reasoning settings. It has the tools `read_file`, `list_dir`,
     `write_file`, `run_python` (CPU-only reading in the sandbox, on a throw-away copy of this
     folder) and `finish_round`. The tool layer lets it read only the files the brief lists and
     write only section 5 of this file, `program_NN.py` and `notes.md`.
     - **Reading:** only `PROGRAMMER_BRIEF.md` in this folder and the files it lists. It does
       not read any other run, report or plan.
     - **Rounds:** it writes `program_NN.py` in rounds. The operator runs each one in the
       sandbox and gives it the stdout; the conversation carries over between rounds.
     - **Record:** `programmer/transcript.jsonl` (every call and tool result),
       `programmer/usage.json` (tokens), `programmer/rounds.json`.
   - **Operator:** a Claude session. It set up the run, runs the programs, scores held-out once,
     runs the controls and writes the comparison. It writes no hypotheses and no edits. What it
     knew before the first pass is written in `docs/XVENDOR_PLAN.md` §6, which the programmer
     does not read. The brief and this file name no edit and no result of an earlier session.
