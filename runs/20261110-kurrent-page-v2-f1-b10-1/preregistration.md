# Pre-registration — 20261110-kurrent-page-v2-f1-b10-1

Filled in **before the first pass** (PROTOCOL.md §2), 10.11.2026. This session replicates
`runs/20261107-kurrent-page-v2-b10-1`, which Claude Opus 5.5 programmed. It uses the same
subject, practice set, held-out set, budget, engine and keep rule, with a different
programmer. Sections 1–4 and 6–7 were written by the operator; section 5 was written by the
programmer.

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

   **Programmer:** Claude Fable 5.1 (`claude-fable-5-1`), 10.11.2026, before any pass.
   Written from the brief, the budget-0 reading (`docs/readings/adapter_kurrent-page-v2.{md,json}`),
   C0 (`docs/baselines/c0-kurrent-val-v2.json`), the practice list and the harness source only.

   **What the readings say.** C0 on the 64 pages: 48,874 edits on 108,243 reference characters.
   The 9 pages at the cap carry 33,936 of those edits (31.4 of the 45.2 CER points); four of them
   are short archival cover notes (129–895 reference characters) that run on for 2,000–6,500
   characters of invented text. `practice_per_page` holds `[edits, reference characters]`, not
   `[reference length, hypothesis length]` as the brief says. One page at the cap that stops is
   worth about −0.02 to −0.05 CER on its own, so a single flipped page moves the corpus CER more
   than most reading improvements would; the paired page bootstrap will not exclude zero for a
   one-page change, which is the protection the keep rule gives against the chaotic flips the
   brief warns of. The adapter's energy sits in `gate_proj` (47 %) and `up_proj` (28 %);
   `down_proj` has 15 %, all attention 10 %. The logit lens of the residual-writing modules
   (`down_proj`, `o_proj`) names three late directions that push `<|im_end|>` down most:
   `layers.31.self_attn.o_proj` dir 0 (σ 2.23, z −3.6 on `<|im_end|>`, −5.5 on newline, −8.1
   on `¬`), `layers.31.mlp.down_proj` dir 0 (σ 2.61, z −3.3) and `layers.27.mlp.down_proj`
   dir 0 (σ 2.68, z −2.8). Each is the top singular direction of its module and carries
   11–17 % of that module's energy, but only 0.03–0.05 % of the adapter's. The sign of a
   singular direction is arbitrary, so the lens cannot say whether the adapter uses the
   direction to suppress or to promote stopping at inference; `project_out` removes the
   direction whichever way it is used, so each hypothesis below is a test of whether that
   direction carries the stop decision at all, with the favourable sign as the prediction.

   **Edit mechanics.** `project_out` on `…lora_B.weight` (out × r) with `axis=1` and the unit
   output direction `u = U[:, k]` from `sub.directions(module)`: B ← (I − uuᵀ)B, which is
   `LoraSubject.remove_direction`. Checked on the CPU: the module's top singular value goes
   to zero and the rest are unchanged. (`axis=0`, as in the brief's example, raises.)
   `scale` on `…lora_B.weight` multiplies that module's whole update.

   **Keep rule as I read it.** An edit is kept only if the 95 % paired page-bootstrap interval of
   CER(after) − CER(last kept) lies below 0 and the pages at the cap do not rise. In practice
   that needs three or more cap pages to stop, or a spread gain across many pages, with no
   collateral loss on the 55 ordinary pages. Any edit that is refused is undone by the harness,
   so each hypothesis is applied to the last kept state.

   - **H0 (pass 1, no edit): baseline `s.measure("C0")`.** Predicted to reproduce C0 exactly:
     CER 0.4515, `truncated_cer` 0.2564, 9 pages at the cap, 12 loops, `length_ratio` 1.226.
     `program_01.py` stops before any edit if CER or `truncated_cer` differ by more than 1e-4
     or the cap count is not 9. A mismatch means the engine is not deterministic on this card
     and no paired comparison in this session can be trusted.

   - **H1 (pass 2): the top direction of the last attention output carries the stop.**
     Edit class `project_out`, module `layers.31.self_attn.o_proj`, `lora_B`, direction
     `U[:, 0]` (σ 2.23, 17 % of the module's energy, 0.035 % of the adapter's). This is the
     most direct lens read: the last residual write before the final norm, and the one direction
     in the adapter whose lens touches all three structural tokens (`<|im_end|>`, newline, `¬`)
     at |z| > 3. Prediction: pages at the cap 9 → 7 or fewer; CER −0.03 (range −0.08 to +0.01);
     `truncated_cer` within ±0.005 (the reading itself is untouched); loops 12 → 10. Possible
     collateral: `¬` and line-end handling change, which the whitespace-flattened CER barely
     sees except through `¬`. If it fails (cap count unchanged, CER within ±0.01): a single
     rank-1 direction through one final-layer module does not carry the stop decision; the
     lens read of this direction is not causal, or the adapter uses it with the sign that
     promotes stopping. Then the stop is not a single late residual direction.

   - **H2 (pass 3): the two late MLP directions that suppress `<|im_end|>` carry it.**
     Edit class `project_out`, one edit over two tensors: `layers.31.mlp.down_proj` `lora_B`
     with its `U[:, 0]` (σ 2.61) and `layers.27.mlp.down_proj` `lora_B` with its `U[:, 0]`
     (σ 2.68); together 0.10 % of the adapter's energy. Applied to the last kept state, so it
     is independent of H1's decision. Prediction: pages at the cap −2 against the last kept
     state; CER −0.03 (range −0.08 to +0.01); `truncated_cer` within ±0.005; loops −2. If it
     fails together with H1: the lens-selected `<|im_end|>` directions are not where the loop
     is written, and I stop editing by lens and go to coarse, sign-free edits (H3, H4).

   - **H3 (pass 4, unless H1/H2 call for a follow-up first): the adapter's late attention
     rewrite stops the model from seeing what it has already written.** Edit class `scale`,
     factor 0.5 on `lora_B` of `q_proj`, `k_proj`, `v_proj`, `o_proj` in the three last
     full-attention layers (23, 27, 31; 12 tensors, 3.9 % of the adapter's energy; the
     `q_proj` of layer 27 holds the adapter's strongest single attention direction, σ 8.2,
     36 % of that module). In a hybrid model the eight full-attention layers are where a
     loop is noticed or not; halving the adapter there moves those layers half-way back to the
     base's attention. Prediction: loops 12 → 9 or fewer, cap −3, CER −0.04 (range −0.10 to
     +0.03), `truncated_cer` +0.01 (reading slightly worse, the price of the edit). Failure
     (cap not down, or CER up): the loop is not written by the adapter's attention rewrite;
     either the base loops on these pages too or the loop sits in the Gated-DeltaNet layers,
     which the adapter does not touch and I cannot edit.

   - **H4 (pass 5): the late MLP residual writes as a whole are the obstacle.** Edit class
     `scale`, factor 0.5 on `lora_B` of `mlp.down_proj` in layers 27–31 (5 tensors, 2.2 % of
     the adapter's energy): the coarse, sign-free version of H2 (every direction of those
     modules at half strength rather than one direction removed). Prediction: cap −2, CER
     −0.02 (range −0.07 to +0.03), `truncated_cer` +0.01, loops −2. Failure: the late MLP
     writes are not the obstacle either; with H1–H4 all refused, the session's finding is
     that the no-stop behaviour is not removable by rank-1 or module-scale edits of the late
     layers, and the C1 control (base alone) decides whether it is the base's.

   - **Passes 6–10: follow-ups decided from these results**, each written as a hypothesis with
     a prediction before its pass, in the next programs. The decision rules, fixed now:
     1. If H1 or H2 moved the cap count by 2 or more (even if refused by the interval), the
        follow-up extends the same module(s) to directions 0–2 (the module's lens rows) in
        one edit, predicting the same direction of change at larger size; a second step, if
        warranted, removes the lens-named suppressor directions of the mid layers (3, 8, 14,
        22) in one edit.
     2. If H3 or H4 was kept or moved the cap by 2 or more, the follow-up sweeps the factor
        (0.25 if 0.5 helped and the reading held, 0.75 if 0.5 cost `truncated_cer` more than
        0.02) on the same tensors.
     3. If an edit was kept, one pass re-tests the next hypothesis on top of it as the rule
        requires (the harness does this anyway); no edit is ever re-measured for luck.
     4. If nothing moved the cap by 2 or more after H1–H4, I do not spend the remaining
        passes on variants; I freeze with "nothing found, nothing kept" and leave the budget
        unused. Edits on `gate_proj`/`up_proj` (the inputs of the MLPs, 75 % of the energy)
        are out of scope: the lens cannot read them and scaling them is the C3 α-sweep the
        operator already has.

   **What a null result means.** If no edit meets the rule, the loop is not a direction in ΔW
   that a rank-1 removal or a late-layer scale can take out, and `C1` (base alone) and the
   decode comparator (`repetition_penalty` 1.05) in §6, which I have not seen, carry the
   answer to whether the base loops too. That outcome is acceptable and will be reported as
   such; I will not add passes to find a gain.

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
7. **Programmer:**
   - **Model:** Claude Fable 5.1 (`claude-fable-5-1`), as a Claude Code subagent (`model:
     fable`) with the default reasoning effort. Its tools are Read, Write and Bash (CPU-only
     reading, `py_compile`).
   - **Reading:** only `PROGRAMMER_BRIEF.md` in this folder and the files it lists, which are
     the budget-0 reading, C0, the practice list and the harness source. It did not read
     the earlier session, the phase-5 plan or the report.
   - **Rounds:** it writes `program_NN.py` in rounds. The operator runs each one in the
     sandbox and reports its stdout; the programmer reads its own log and passes.
   - **Operator:** a Claude Opus 5.5 (`claude-opus-5-5`) session. It set up the run, ran
     the programs, and does the held-out score, the controls and the write-up. It does not
     write hypotheses or edits. It read the phase-5 plan, the Opus pre-registration and §1–3
     of the phase-5 report before launching the programmer, so **it knows the Opus session's
     kept edit**. It passes on none of this to the programmer: the brief was written so that
     it names no edit of that session, and no result.
