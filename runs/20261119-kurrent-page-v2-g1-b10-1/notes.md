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
