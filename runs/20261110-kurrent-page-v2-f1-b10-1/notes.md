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

## 2026-11-10T20:40Z — round 1 (Claude Fable 5.1, `claude-fable-5-1`)

**Read.** The operator's stdout of program_01 (rc 0, 17:30–20:32 UTC); my own `log.jsonl`,
`state.json`, `readings/pass_001..003.jsonl` (per-page texts of C0, H1, H2) and
`current.adapter` (through `LoraSubject`, CPU). Nothing else.

**Results.** Baseline reproduced C0 exactly (CER 0.4515, truncated 0.2564, 9 at cap, 12 loops).
- **H1 kept**: CER 0.3470 (−0.1045, ci95 [−0.203, −0.024]), truncated 0.2341, at cap 4,
  loops 8, length_ratio 1.126. Five of the nine cap pages stopped (007967, 008159, 007218,
  007154, 008241); 28 pages better, 16 worse, 19 of the 44 changes within ±5 edits. The gain
  is concentrated in the five pages (−1,900 to −3,500 edits each) but the interval clears zero.
  `truncated_cer` fell too because the long cap pages (007967, 008241) had been looping inside
  the reference length. Still at the cap: 006694 (129 chars; identical 2,054 edits under C0,
  H1 and H2 — insensitive), 011629 (unchanged), 008291 and 007884 (long pages whose truncated
  edits, 3,400–3,500, say the reading itself is bad there, not only the stop).
- **H2 refused** (on top of H1): CER 0.3922 (+0.045 vs H1, ci95 [−0.024, +0.158]); 008159 and
  007218 went back to the cap; 007884 improved by 1,000 edits. Undone. The late MLP output
  directions the lens named are not the obstacle; the attention output of layer 31 was.
- CPU check of `current.adapter`: only `layers.31.self_attn.o_proj.lora_B.weight` changed;
  its σ are now [1.586, 1.397, 1.277, …, 6e-8]; the current dirs 0 and 1 coincide with C0's
  dirs 1 and 2 (overlap 1.000).

**Decisions (preregistration §5 rules).** Rule 1 applies (H1 moved the cap by 5 and was kept):
the follow-up extends the same module to its lens rows 1–2. The second step of rule 1
(mid-layer MLP suppressors) I judge not warranted after H2 re-broke two pages: the locus is the
attention output, not the MLP writes. H3 (late attention at 0.5) runs next as preregistered,
on top of the kept state. H4 (late down_proj at 0.5) is the coarse form of H2 and is
deprioritised; it runs only if passes remain after H5/H3 and nothing better is indicated.
Expectation stated honestly: the two insensitive cap pages are probably not adapter-written,
and the two long ones are reading failures, so further stop edits have a small ceiling.

**program_02.py** (compiled), 2 passes, no freeze:
- pass 4, **H5**: remove the current dirs 0 and 1 of `layers.31.self_attn.o_proj` (C0's dirs 1
  and 2) from `lora_B` in one edit, op `add` with `delta = −(U₂U₂ᵀ)B` (exact two-direction
  projection; CPU check: top σ becomes 1.277, delta norm 0.954). Prediction: cap 4 → ≤3,
  CER −0.01 (range −0.05 to +0.04), truncated ±0.01, loops 8 → 7. Failure = the H1 effect
  is specific to dir 0.
- pass 5, **H3**: scale 0.5 on q/k/v/o `lora_B` of layers 23, 27, 31 (12 tensors). Prediction:
  loops −2, cap ≤3, CER −0.02 (range −0.06 to +0.04), truncated +0.01.
An assertion checks that the restored state is the H1 state before any edit.

## 2026-11-10T22:35Z — round 2 (Claude Fable 5.1, `claude-fable-5-1`): freeze

**Read.** The operator's stdout of program_02 (rc 0, 20:36–22:31 UTC); my own `log.jsonl` and
`readings/pass_004..005.jsonl` against pass_002 (H1). Nothing else.

**Results (both on top of the kept H1 state, both refused and undone).**
- **H5** (C0's dirs 1–2 of `layers.31.self_attn.o_proj` removed as well): CER 0.3378
  (−0.009, ci95 [−0.076, +0.047]), cap 4, loops 8. Per page: 011629 stopped (−2,682 edits)
  and 008159 went back to the cap (+1,847); 19 pages better, 25 worse, the rest small. That is
  one page swapped for another, inside the interval — the chaotic flip the brief warns of, not
  an effect. The H1 effect is specific to dir 0; the module's other directions add nothing.
- **H3** (q/k/v/o `lora_B` of layers 23, 27, 31 at 0.5): CER 0.3859 (+0.039, ci95 [−0.014,
  +0.111]), truncated 0.2608 (+0.027), cap 6, loops 10; 42 of 64 pages worse, 007967 back at
  the cap (+3,262). The late attention adapters carry the reading; the loop is not their work.

**Decision: freeze with H1 as the only kept edit.** Preregistration §5 rules: rule 2 gives no
sweep (H3 was neither kept nor did it lower the cap); rule 1's second step was ruled out in
round 1 (H2 re-broke pages); H4 is the coarse form of H2 and I do not run it; rule 4 says not
to spend passes on variants. A dose-response extrapolation of the H1 direction (removing more
than 100 % of it) would add a direction the adapter did not learn, which the brief excludes
("edits remove obstacles; they do not add ability"), so I do not do it either. 5 of 10 passes
are left unused on purpose.

**What the session says.** The adapter's top singular direction of the last attention output
(`layers.31.self_attn.o_proj`, σ 2.23, 17 % of that module and 0.035 % of the adapter; lens
z −3.6 on `<|im_end|>`, −5.5 on newline, −8.1 on `¬`) carries a stop suppression: removing
it stopped 5 of 9 cap pages on the practice set, CER 0.4515 → 0.3470 (ci95 [−0.203, −0.024]),
truncated 0.2564 → 0.2341, loops 12 → 8, with 16 of 64 pages slightly worse. The two cap
pages that never moved (006694, 011629 except under H5's flip) and the two long cap pages with
3,400–3,500 truncated edits are not this direction's doing. The late MLP output directions
(H2) and the late attention adapters as a whole (H3) are not obstacles. Whether the base
alone loops on the four remaining pages is for C1; whether the H1 gain is the direction and
not any perturbation of that module is for C5 (sham), which H5 partly anticipates: the
module's next two directions gave nothing.

**program_03.py** (compiled): no pass; asserts the restored state is H1, logs a note, `s.freeze()`,
prints the summary's used/edits/kept/final digest and the changed tensors. Final state =
C0 with `layers.31.self_attn.o_proj.lora_B.weight ← (I − u₀u₀ᵀ)B`.
