# Notes — 20261119-kurrent-page-v2-g1-b10-1

## Round 0 — 2026-11-19 09:32:33 UTC

- **Programmer model id:** `gemini-4-argon`
- **Settings:** Called through the Google API by `wsp.programmer` with the provider's default reasoning settings; sandbox execution with `wsp` and `numpy` (CPU-only for reading scripts; operator runs `program_NN.py` on card 0 via `evaluate_qlora`, greedy decoding, cap 3,072).
- **Files read:**
  - `runs/20261119-kurrent-page-v2-g1-b10-1/PROGRAMMER_BRIEF.md`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/preregistration.md`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/notes.md`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/session.json`
  - `docs/readings/adapter_kurrent-page-v2.md` and `docs/readings/adapter_kurrent-page-v2.json`
  - `docs/baselines/c0-kurrent-val-v2.json`
  - `data/practice/kurrent-val-64.jsonl` and `data/practice/kurrent-val-64.meta.json`
  - `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/lora_subject.py`, `wsp/pages.py`, `wsp/metrics.py`
  - The adapter tensors (`~/atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v2`) via `LoraSubject` in CPU reading scripts.
- **Budget-0 observations:**
  1. **Practice set error budget:** On the 64 practice pages, C0 has 48,873 total character edits across 108,243 reference characters (CER 0.451512, `truncated_cer` 0.256386, `length_ratio` 1.226370, 9 pages at the 3,072-token cap). The 9 at-cap pages alone contribute 33,936 edits on 23,508 reference characters. A paired page-bootstrap check on `practice_per_page` shows that at least 4 of the 9 at-cap pages must stop looping for the 95 % bootstrap interval of $\Delta\text{CER}$ to clear 0.
  2. **Late-layer `<|im_end|>` suppression geometry:**
     - `layers.31.self_attn.o_proj` dir 0 ($\sigma = 2.226, z(\text{<|im\_end|>}) = -3.60, z(\text{\\n}) = -5.52, z(\neg) = -8.06$) and `layers.31.mlp.down_proj` dir 0 ($\sigma = 2.612, z(\text{<|im\_end|>}) = -3.32, z(\text{\\n}) = -1.95$) are the two strongest `<|im_end|>` suppressors across all 128 modules.
     - Their left singular vectors (output directions written into the residual stream) have cosine similarity $+0.516$ with each other, and cosine similarities $-0.483$ and $-0.430$ with `layers.30.mlp.down_proj` dir 0 ($\sigma = 3.705, z(\text{<|im\_end|>}) = +2.85, z(\text{\\n}) = +2.05$). Thus layer 31's two residual writers directly oppose layer 30's dominant stopping direction.
     - Upstream in the late layers, `layers.29.mlp.down_proj` dir 0 ($\sigma = 2.802, z(\text{<|im\_end|>}) = -1.51, z(\text{\\n}) = -4.04, \cos(\text{L31.o\_proj:0}) = +0.296$) and `layers.27.mlp.down_proj` dir 0 ($\sigma = 2.678, z(\text{<|im\_end|>}) = -2.80, \cos(\text{L31.o\_proj:0}) = +0.165$) share positive alignment with this same `<|im_end|>`-suppressing subspace.
- **Actions taken in Round 0:**
  - Completed section 5 of `preregistration.md` with hypotheses H1–H9 across the 10-pass budget.
  - Wrote `program_01.py` to execute Pass 1 (baseline C0 check) and Pass 2 (H1: projecting out dir 0 from `layers.31.self_attn.o_proj.lora_B.weight` and dir 0 from `layers.31.mlp.down_proj.lora_B.weight` via `project_out` with `axis=1`, followed by `s.measure` and `s.keep`).

## Round 1 — 2026-11-19 12:00:36 UTC

- **Files read:**
  - `runs/20261119-kurrent-page-v2-g1-b10-1/log.jsonl`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/passes/pass-001.raw.jsonl`, `pass-001.draw.jsonl`, `pass-002.raw.jsonl`
- **Results of `program_01.py` (Passes 1 and 2):**
  1. **Pass 1 (C0):** Reproduced the baseline exactly: CER `0.451512` (48,873 edits / 108,243 chars), `truncated_cer` `0.256386`, `length_ratio` `1.226370`, `at_cap` `9`, `loops` `12`.
  2. **Pass 2 (H1 — `project_out` dir 0 of `layers.31.self_attn.o_proj` and dir 0 of `layers.31.mlp.down_proj`):**
     - **KEPT (`rule: met`)**: CER dropped by `-0.077769` to `0.373742` (40,455 edits, $-8,418$ edits), with 2,000-resample paired page-bootstrap 95 % CI `[-0.173227, -0.004299]` (entirely below 0).
     - `truncated_cer` **improved** from `0.256386` to `0.241180` ($-0.015207$), `length_ratio` dropped from `1.226370` to `1.147640`, `at_cap` dropped from `9` to `5` ($-4$ pages at cap), and `loops` dropped from `12` to `10`.
     - **Per-page breakdown:**
       - `007154_023822_0002_1913639` (ref 222): stopped looping `\n✳`! Length dropped $2,141 \to 200$, edits $2,042 \to 154$ ($-1,888$).
       - `007218_024284_0002_656610` (ref 188): stopped looping `\n✳`! Length dropped $2,112 \to 177$, edits $2,040 \to 118$ ($-1,922$).
       - `007967_036171_0001_1210969` (ref 5,369): stopped at end of document (`...vierzechenhundertvierzigundsechs\njar.`)! Length dropped $8,247 \to 5,399$, edits $5,137 \to 1,583$ ($-3,554$), `truncated_cer` edits $3,157 \to 1,613$.
       - `008241_036296_0001_1211427` (ref 5,582): stopped looping `, vˀsitzenn` and terminated cleanly (`...vierzechenhundert\nund sibentzig jar.`)! Length dropped $6,106 \to 5,643$, edits $3,179 \to 1,808$ ($-1,371$).
       - Total Dingbats (`✳`) insertions across the corpus fell from $3,537$ to $1,253$, and Space insertions fell from $7,220$ to $4,550$.
       - The 5 remaining at-cap pages are: `006694_018260_0002_443451` (`\n✳` loop, 1,955 excess edits), `007884_027397_0001_772042` (clause loop, 1,194 excess edits), `008159_036203_0001_1211042` (charter loop, 6,316 excess edits), `008291_036276_0001_1211387` (`hievor,` loop, 1,551 excess edits), and `011629_1503666_0006_57465490` (`\n"` loop, 2,470 excess edits).
- **Plan for Round 1 (`program_02.py`, Pass 3 — H2):**
  - Follow the pre-registered H2 (branch for H1 kept): project out dir 0 of `layers.27.mlp.down_proj` ($\sigma = 2.678, z(\text{<|im\_end|>}) = -2.80$) and dir 0 of `layers.29.mlp.down_proj` ($\sigma = 2.802, z(\text{<|im\_end|>}) = -1.51, z(\text{\\n}) = -4.04$) on top of the kept H1 state, measure, and call `s.keep()`.

## Round 2 — 2026-11-19 13:15:10 UTC

- **Files read:**
  - `runs/20261119-kurrent-page-v2-g1-b10-1/log.jsonl`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/passes/pass-003.raw.jsonl`
- **Results of `program_02.py` (Pass 3 — H2):**
  - **REFUSED & REVERTED (`rule: refused`, state restored to H1)**: CER rose vs H1 by `+0.072642` to `0.446385` (ci95 `[-0.029788, +0.188632]`), `truncated_cer` degraded from `0.241180` to `0.275297` (`+0.034118`), and `at_cap` rose from `5` to `7`.
  - **Diagnosis from per-page readings (`pass-003.raw.jsonl`):**
    - Removing the entire dominant singular direction (`dir 0`) of `layers.27.mlp.down_proj` ($\sigma = 2.678$, energy $7.17$) and `layers.29.mlp.down_proj` ($\sigma = 2.802$, energy $7.85$) damaged intermediate text-reading features (`truncated_cer` worsened by 3.4 points, with Latin lowercase errors rising from 26,436 to 32,739), which in turn caused `007967`, `008241`, and `016379` to fall into mid-page repetition loops and hit the token cap, even though `008159` stopped looping ($-2,519$ edits).
    - Why? Because `layers.27.mlp.down_proj` dir 0 and `layers.29.mlp.down_proj` dir 0 only have cosine similarities $+0.176$ and $+0.272$ with the layer-31 `<|im_end|>`-suppressing direction $u_{\text{comb}} = \text{norm}(u_{\text{o31},0} + u_{\text{d31},0})$; over 92% of their energy is orthogonal to $u_{\text{comb}}$ and carries useful Kurrent transcription features.
- **Plan for Round 2 (`program_03.py`, Pass 4 — H3):**
  - Follow pre-registered H3: instead of removing the upstream modules' own dominant singular directions, project out only the exact shared layer-31 `<|im_end|>`-suppressing unit vector $u_{\text{comb}} = \text{norm}(u_{\text{o31},0} + u_{\text{d31},0})$ from the upstream late-layer residual writers `layers.{27,28,29}.mlp.down_proj` and `layers.27.self_attn.o_proj` (`project_out`, `axis=1`). This removes less than 1% of their Frobenius energy (0.46%, 0.19%, 0.25%, 0.97%) while eliminating their contribution along the layer-31 stopping-suppression direction, and leaves layer 30 ($z(\text{<|im\_end|>}) = +2.85$) completely untouched.
