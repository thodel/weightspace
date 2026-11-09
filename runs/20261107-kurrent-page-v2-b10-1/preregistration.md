# Pre-registration — 20261107-kurrent-page-v2-b10-1

Filled in **before the first pass** (PROTOCOL.md §2), 07.11.2026, from
docs/PHASE5_PLAN.md (decisions §7, status §8).

1. **Subject:** `qwen3.5-4b-german-kurrent-xvi-xix-page-v2`
   (`~/atr-cache/wsp/adapters/…`). Tensor digest
   `fb4f55ee7386abc7292e29bbf2a24f502fc4079cb4732934cbead229b300504c`, file
   sha256 `1b70ce4a2b2125bd007bd1c81789bd43c8d72c12ef49e4f50cf8ad9ca3c67897`.
   peft LoRA with r 64 and α 128 on 128 language-tower modules of
   `Qwen/Qwen3.5-4B`. **Only the adapter is edited** (decision 2); the base is
   read, never written.
   - **Donors:** none for edits. v1 (`52f94e10…`) is used for reading only,
     because it is `crossed` on the held-out set.
2. **Practice set:** `data/practice/kurrent-val-64.jsonl`, 64 pages, list sha256
   `2cc5278ea215b72d5f2faad9977abcde6dde36bc327876d08643852827d46f9b`.
   - **Source:** the registered **val** side of `german-kurrent-xvi-xix-v1`.
     This is not the train side, a deviation from the kraken protocol taken as
     decision 1.
   - **Selection rule:** one page per document, three from each of the two
     160-page documents, seed 20261107
     (`data/practice/kurrent-val-64.meta.json`).
   - **What a pass is:** the trainer's own `evaluate_qlora` on these 64 pages.
     That means transformers 5.17, greedy, cap 3,072, `max_pixels` 2,097,152,
     and the prompt from the training render.
   - **Engine:** gate G1 refused vLLM. The reference repeats byte-identically.
3. **Held-out set:** `german-kurrent-xvi-xix-v1` **test**, 733 pages from 70
   documents (`vlm-pages-test/pages.jsonl`, sha256
   `c56698a79a8f628b1bc905241696ce394044973773915bd9b11a90ca24a07bc1`,
   `draw_md5` `ae8bf1fd315e3a06eee272bb3ca7c7ad`, registry page digest
   `22814f9ebb7b58cc`).
   - **Scoring:** once, after freeze, by `evaluate_qlora`
     (`wsp.run heldout --set kurrent-test`).
   - **The number to beat:** the trainer's own score of v2 on it. CER 0.4181,
     `truncated_cer` 0.2502, `length_ratio` 1.181, 49 of 733 at the cap
     (HELDOUT_BOUNDARY.md, #341).
   - **Overlap audit** (`data/practice/kurrent-val-64.audit.json`):
     - no practice document is among the 70 test documents;
     - none is in v2's training manifest (1,191 documents);
     - six practice pages are on v2's own validation split, which the run
       scored but never trained on.
4. **Budget:** 10 passes, the first being the in-session baseline. `min_effect`
   is 0. **Keep rule:** PROTOCOL §3 on the paired page bootstrap (2,000
   resamples), **and** the pages at the cap must not rise against the last kept
   state.
5. **Starting point (C0, outside the budget, `docs/baselines/c0-kurrent-val-v2.json`).**
   On the 64 practice pages:
   - CER **0.4515**, `truncated_cer` **0.2564**, `length_ratio` 1.226;
   - **9 pages at the cap**, 12 pages with a loop (`wsp.pages.page_loop`).

   About 19.5 points of CER are text past the page.

   **Hypotheses**, from the budget-0 reading (`docs/readings/adapter_kurrent-page-v2.md`).
   The logit lens ignores everything after the module, so each prediction is
   a guess at the order of magnitude:
   - **Pass 1, baseline.** It is predicted byte-identical to C0's readings of
     the same 64 pages: CER 0.4515 and 9 at the cap. Any other result means the
     engine or the setup changed, and the session stops.
   - **H1 (direction subtraction):** direction 0 of
     `layers.31.self_attn.o_proj` writes against the end of turn. Its lens
     scores are z −3.60 for `<|im_end|>`, −5.52 for `\n` and −8.06 for `¬`.
     The adapter taught the last layer to keep writing, and that is the
     not-stopping.
     - **Edit:** `remove_direction` of u₀ from that module's `B`.
     - **Prediction:** pages at the cap 9 → 5–7, CER −2 to −6 points,
       `truncated_cer` within ±1 point, `length_ratio` down by 0.03–0.10.
     - **Risk:** the same direction suppresses newline and `¬`. Removing it may
       add line breaks and hyphens, or start new loops, and the cap rule would
       refuse that.
   - **H1b (direction subtraction):** the next two directions that lower the
     end of turn: direction 0 of `layers.31.mlp.down_proj` (z −3.32) and
     direction 0 of `layers.27.mlp.down_proj` (−2.80).
     - **Edit:** remove both, alone or on top of H1 if H1 was kept.
     - **Prediction:** smaller than H1, cap −1 to −2 and CER −1 to −3 points.
   - **H2 (scaling):** the stopping behaviour sits in the last five layers
     (27–31 hold 20 % of the adapter's energy).
     - **Edit:** `scale_module` 0.75 on every adapted module of layers 27–31.
     - **Prediction:** cap −1 to −3, `truncated_cer` +0.5 to +2 points (some
       reading lost), CER −1 to +1 point. Of the three, this is the most
       likely to be refused by the keep rule.
   - **Passes 5–9:** follow-ups decided from these results. Each is written
     into the log as a hypothesis with its prediction before its pass. Typical
     candidates are a partial removal (u₀ scaled out by 0.5 or 1.5 instead of
     1), combinations of kept edits, and the next lens candidates.
   - **Pass 10:** reserve.
6. **Controls.** Run outside the budget on the same 64 pages, before the
   held-out score, with the same engine:
   - **C0:** done, as above.
   - **C1, base without the adapter:** does the base loop on these pages?
   - **C3, α-sweep of the whole adapter** (every module × α), at α ∈ {0.75, 0.9, 1.1}.
   - **C5, sham:** a random unit direction of the same module(s) as the kept
     edit, removed the same way.
   - **Decode comparator:** `repetition_penalty` 1.05 with the adapter
     unchanged. `evaluate_qlora` has no flag for it. It will be run through a
     wrapper that changes only that argument of `generate`, and recorded as
     such.
   - **C4, a short QLoRA continuation:** deferred (decision 4). It is submitted
     only if an edit is kept, and only with a separate go-ahead.
   - **C2 (average) and C6 (blind):** not run. There is no clean sibling
     adapter of the same base to average with, and the blind control belongs
     to the kraken track.
