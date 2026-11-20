# Pre-registration — 20261119-kurrent-page-v2-o1-b10-1

Filled in **before the first pass** (PROTOCOL.md §2), 19.11.2026. This session replicates
`runs/20261107-kurrent-page-v2-b10-1`, which Claude Opus 5.5 programmed. It uses the same
subject, practice set, held-out set, budget, engine and keep rule, with a different
programmer (GPT-6.2 Astra). Sections 1–4 and 6–7 were written by the operator; section 5 was written by the
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

   - Pass 1: unedited C0. Expect exact agreement with the archived practice scores; stop on mismatch. All predictions below are absolute differences from the last kept state.
   - Pass 2, H1 (project_out): remove the first left singular direction from each of layer 31's `self_attn.o_proj` and `mlp.down_proj` B factors in one edit. These last two residual writers both have leading axes with unusually large end-token contrasts. The lens selects axes, not activation signs: whether either disfavors stopping depends on its input. If both carry excess continuation, expect CER −0.055, truncated CER roughly unchanged (within 0.005), and 3 fewer capped pages. A refusal means this pair is not a demonstrated remedy, not that end signals are absent. Shorter output with worse truncated CER suggests premature stopping.
   - Pass 3, H2 (scale): reduce `self_attn.q_proj` B to 0.65 of current strength at layers 19, 23, 27, 31 only. Late attention may favor generated text over the exhausted image, sustaining repeats even with intact end-token readout. Alter the query update, not keys or values. Expect CER −0.045, truncated CER +0.003 from slightly worse alignment, caps −2. If it fails, late query attenuation at this dose has not separated stopping from reading.
   - Pass 4, H3 (scale): reduce `mlp.gate_proj` B in layers 24–31 to 0.75. These large late gate updates could maintain transcription features after the last visible line. Expect CER −0.04, truncated CER +0.005, caps −2. Failure argues against this block and dose; update energy alone does not diagnose a loop.
   - Passes 5–10 reserved for follow-ups, or early freeze. Inspect which capped/loop pages recover, new failures, and whether the CER change survives dropping the single biggest beneficiary descriptively. For a multi-page reduction with a reading penalty, narrow the layer set or soften the dose; for broad improvement without a reading penalty, test a distinct remaining block. Do not chase a lone cap flip. Specify modules, operation, numeric predictions and failure interpretation before each follow-up. Every comparison is to the last kept state under section 4. Freeze if no supported avenue remains, even if nothing was kept.

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
   - **Programmer:** GPT-6.2 Astra (`gpt-6.2-astra`), called through the OpenAI API by `wsp.programmer`
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
