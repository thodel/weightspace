# Notes — 20261110-kurrent-page-v2-f1-b10-1

## 2026-11-10T17:35Z — round 0 (Claude Fable 5.1, `claude-fable-5-1`)

**Settings.** Claude Code subagent, model `claude-fable-5-1`, default reasoning effort, tools Read,
Write, Edit, Bash (CPU reading with `.venv/bin/python`, `CUDA_VISIBLE_DEVICES=`, `py_compile`).
No GPU, no `measure`, no `program_*.py` executed by me.

**Read.** `PROGRAMMER_BRIEF.md`; `docs/readings/adapter_kurrent-page-v2.md` and `.json` (all
160 lens rows of the 40 `down_proj`/`o_proj` modules); `docs/baselines/c0-kurrent-val-v2.json`;
`data/practice/kurrent-val-64.jsonl` (head) and `.meta.json`; `wsp/ops.py`, `wsp/session.py`,
`wsp/run.py`, `wsp/lora_subject.py`, `wsp/pages.py`, `wsp/metrics.py`; `session.json`,
`program.py`, `env.txt` in this folder. Nothing else: no other run, nothing else under
`docs/` or `protocol/`, no git history, no `wsp/subject.py`.

**CPU readings I ran** (adapter through `LoraSubject`, numpy only, nothing written).
- Shapes: `lora_A` 64 × in, `lora_B` 2560 × 64 (hidden 2560); `o_proj` in = 4096.
- Energy shares: gate 47 %, up 28 %, down 15 %, q 7.4 %, o 1.8 %, k 0.7 %, v 0.6 %.
  Largest modules are `gate_proj` of layers 27, 0, 28, 29, 25 (≈2 % each). The strongest
  attention direction is `layers.27.self_attn.q_proj` dir 0, σ 8.2, 36 % of its module.
- Top direction shares: `o_proj.31` dir 0 σ 2.23 = 17 % of its module, `down_proj.31` dir 0
  σ 2.61 = 17 %, `down_proj.27` dir 0 σ 2.68 = 11 %. The three together are 0.13 % of the
  adapter's energy.
- `ops.project_out(B, direction=u, axis=1)` equals `remove_direction` (max |diff| 1e-8) and
  zeroes that singular value, the others unchanged. `axis=0` raises
  ("direction length must equal the number of columns"): **the brief's example `axis=0` is
  wrong for `lora_B`; `axis=1` is the right call.**
- C0 per-page values in `practice_per_page` are `[edits, reference characters]`, not
  `[reference length, hypothesis length]` as the brief says (e.g. `013876_…` is `[0, 1255]`
  and `006694_…` at the cap is `[2054, 129]`). The 9 cap pages carry 33,936 of the 48,874
  edits; the 13 loop-list pages in the practice set vs 12 counted loops is a one-page
  difference I note and ignore.

**Reasoning.** The symptom is the stop, not the reading (`truncated_cer` 0.256 vs CER 0.452).
The only direction-level read I have is the lens, whose sign is arbitrary for singular
directions, so each lens hypothesis is "this direction carries the stop decision" tested by
removing it (sign-free), with the favourable sign as the prediction. I start with the three
late directions the lens names (layer 31 attention output, layer 31 and 27 MLP output) because
they are the last writes before the final norm and the only ones where the lens read is
direct. Then coarse sign-free scales of the late attention (H3) and late MLP output (H4)
adapters. A single flipped cap page is worth 0.02–0.05 CER but the paired bootstrap will not
exclude zero for it; a keep needs ≥3 cap pages to stop or a spread gain, which is the
protection against the chaotic flips. If H1–H4 move nothing by ≥2 cap pages, I freeze with
nothing kept and leave the budget unused (preregistration §5, rule 4).

**program_01.py** (compiled): pass 1 `measure("C0")` with a guard that stops the program
before any edit if CER or `truncated_cer` deviate by >1e-4 from C0 or the cap count is not 9;
pass 2 H1 (`project_out`, `layers.31.self_attn.o_proj` `lora_B`, `U[:,0]`, `axis=1`);
pass 3 H2 (`project_out`, one edit over `layers.31.mlp.down_proj` and `layers.27.mlp.down_proj`
`lora_B`, each its own `U[:,0]` via `per_tensor`). Each followed by `measure` and `keep`.
No freeze.
