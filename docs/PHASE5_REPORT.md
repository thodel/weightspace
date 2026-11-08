# Phase 5 report: one direction that stops the v2 Kurrent adapter on practice pages, and not on held-out pages (asteraix, 07.–08.11.2026)

Branch `phase5`. The plan and its decisions are in `docs/PHASE5_PLAN.md`, and the
session is `runs/20261107-kurrent-page-v2-b10-1/`. Everything ran on card 0
through the trainer's own `evaluate_qlora`. Nothing was written to the share,
the registry, HF or the trainer.

**In one sentence:** removing one direction from the adapter's last attention
output (layer 31, `o_proj`) cut practice CER from 45.2 % to 34.7 % and the
pages at the cap from 9 to 4, with a clean dose curve. **On the 733 held-out
pages it changed nothing:** +0.24 points [−3.5, +3.9], 43 against 44 pages at
the cap. Which pages loop is too unstable for a 64-page practice set to find a
fix that transfers.

## 1. Subject, engine, sets

- **Subject:** `qwen3.5-4b-german-kurrent-xvi-xix-page-v2`, sha256 `1b70ce4a…`. peft
  LoRA, r 64, α 128, 128 language-tower modules of Qwen3.5-4B. Only the adapter
  was edited.
- **Engine:** the trainer's `evaluate_qlora` (transformers 5.17, greedy, cap
  3,072).
  - **Why not vLLM:** gate G1 refused it. On 32 pages the CER between the two
    engines' readings was 7.2 %, and the two engines put different pages at the
    cap.
  - **Determinism:** the reference repeated byte-identically, and the in-session
    baseline matched C0 on 64 of 64 pages.
- **Practice:** 64 pages from the registered `val` side of
  `german-kurrent-xvi-xix-v1`, one per document, three from each 160-page
  document (`data/practice/kurrent-val-64.*`). On it v2 reads like on the
  held-out set: CER 0.4515, `truncated_cer` 0.2564, 9 pages at the cap.
- **Held-out:** the 733 test pages (70 documents), scored once for the frozen
  adapter. Unedited v2 was scored on the same card with the same engine, so the
  comparison is paired.
  - **Engine gap:** here v2 reads 40.53 %. The trainer's UBELIX score of the
    same adapter on the same pages is 41.81 %. That 1.3-point gap is the engine
    and the hardware, and it is why the pair was scored here.

## 2. The session (budget 10, 6 used)

| pass | edit | practice CER | `truncated_cer` | at cap | decision |
|---|---|---:|---:|---:|---|
| 1 | none (baseline = C0, byte-identical) | 0.4515 | 0.2564 | 9 | — |
| 2 | **H1:** direction 0 of `layers.31.self_attn.o_proj` out of `B` | **0.3470** | 0.2341 | **4** | **kept**, −10.5 points [−20.3, −2.4] |
| 3 | H1b: direction 0 of `layers.31` and `layers.27.mlp.down_proj` | 0.3922 | 0.2346 | 6 | refused (+4.5 against H1) |
| 4 | H2: layers 27–31 scaled to 0.75 | 0.4025 | 0.2452 | 5 | refused (+5.6, CER rose) |
| 5 | H1 at dose 1.5 | 0.3616 | 0.2480 | 5 | refused |
| 6 | H1 at dose 0.5 | 0.3990 | 0.2509 | 7 | refused |

H1 came from the budget-0 reading (`docs/readings/adapter_kurrent-page-v2.md`).
In the logit lens this direction has by far the strongest scores against the
end of turn (z −3.60), the newline (−5.52) and `¬` (−8.06). The dose curve over
0 / 0.5 / 1 / 1.5 gives CER 0.452 / 0.399 / 0.347 / 0.362, with its minimum at
full removal. On practice the edit behaves like a mechanism.

## 3. Held-out (paired, same card and engine)

| 733 test pages | CER (trainer) | CER (wsp, NFD) | `truncated_cer` | `length_ratio` | at cap | loops |
|---|---:|---:|---:|---:|---:|---:|
| v2 unedited | 0.4053 | 0.4012 | 0.2430 | 1.177 | 44 | 63 |
| v2 + H1 (frozen session) | 0.4079 | 0.4036 | 0.2460 | 1.175 | 43 | 61 |

- **Paired page bootstrap, H1 − v2:** **+0.24 points [−3.5, +3.9]**. 225 pages read
  better, 233 worse, and 275 identically.
- **Pages at the cap:** 37 run to the cap under both, 6 only with H1, and 7
  only without it. The edit swaps which pages loop more than it removes loops.

## 4. Controls on the practice pages (outside the budget)

| practice, 64 pages | CER | `truncated_cer` | at cap |
|---|---:|---:|---:|
| C0, v2 | 0.4515 | 0.2564 | 9 |
| **H1 (session)** | **0.3470** | 0.2341 | **4** |
| sham: random direction out of the same module (C5) | 0.4128 | 0.2525 | 7 |
| whole adapter × 0.75 / 0.9 / 1.1 (C3) | 0.3983 / 0.4228 / 0.5037 | 0.261 / 0.261 / 0.271 | 5 / 7 / 6 |
| `repetition_penalty` 1.05, no edit | 0.3680 | **0.2287** | 4 |
| base without adapter (C1) | 1.6237 | 0.5581 | 23 |

A random direction moves practice CER by 3.9 points, and the whole-adapter α
sweep by up to 5 points in either direction. So the "noise" a practice set of
this size must beat is not the bootstrap interval of one fixed reading. It is
the reshuffling of looping pages under any perturbation. H1 beat the sham on
practice, but not by a margin that survives new pages.

The decode guard needs no weight edit and nearly matches H1 on practice,
with better reading (`truncated_cer` 0.229). It was not scored on held-out (one
held-out score per session; it is a comparator, not a candidate).

The base model loops on a third of the pages, so stopping is something the
adapter taught, and partly.

## 5. What this says about the method

- **What held:** the budget-0 reading found the one direction with an outsized
  effect on stopping. The edit did what the hypothesis said on the pages it was
  tuned on. It also has a dose curve, which a lucky perturbation would not.
- **What failed:** the step from practice to held-out. Running to the cap on
  long Kurrent pages is a property of individual pages under greedy decoding,
  and it flips under small changes. Whether a page loops depends on whether
  greedy decoding happens to enter a repetition, so whatever lowers the
  end-of-turn logit, from an edit or a random perturbation, changes a different
  handful of pages each time. The keep rule's interval is about one fixed set of
  readings and does not see that variance.
- **For a next attempt** (not run):
  1. Use the variance under sham edits as the noise floor: several seeded random
     directions per pass, against which an edit must stand out.
  2. Use a practice set large enough in looping pages: the defect hits about one
     page in fifteen, so roughly 200–300 pages for 15–20 looping ones.
  3. Measure stopping without decoding to the end: the margin of `<|im_end|>`
     at the reference's true end, in one forward pass with teacher forcing. That
     is a smoother target than the page-level cap count.
- **Is C4 (retrain) worth running?** No edit was kept on held-out evidence, so
  decision 4 says no.

## 6. Harness changes in this phase (all tested)

- `LoraSubject`:
  - peft's scale and an SVD via QR of the factors;
  - `scale_module` and `remove_direction`;
  - `vllm_runner` and `reference_runner`, with the work copy in the run folder.
- `wsp.pages`: page metrics and a numpy Levenshtein.
- `wsp.card`: card 0 memory, the trainer's `waiting` (#396), and a watcher file
  for the sandbox, which has no loopback. The runners wait for room instead of
  failing.
- `wsp.vlm_render`, `wsp.vllm_forward` and `wsp.vlm_reference`, plus
  `wsp.rp_eval` (the decode comparator).
- `Session` in page mode: the cap condition in the keep rule.
- Three cross-process bugs fixed, all logged as deviations:
  - a pass refused before it started was counted;
  - reference states lived in memory only, so passes 3–4 were compared wrongly
    (recomputed, and both refusals stand);
  - `freeze` counted only its own process's events.
- `wsp.run`:
  - adapter subjects;
  - chunked, resumable held-out scoring;
  - `scripts/p5_queue.py` for the controls.

Readings, per-page results and the logs are in `docs/baselines/p5-*.json`,
`docs/baselines/c0-kurrent-val-v2.json` and the session folder.
