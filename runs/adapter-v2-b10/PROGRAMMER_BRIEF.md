# Brief for the programmer (real adapter session, stage A4, 07.11.2026)

You are the *programmer* of a weight-space session on a real **peft LoRA adapter**: `qwen3.5-4b-german-kurrent-xvi-xix-page-v2`
(rank 64, `lora_alpha` 128, scale 2, on the language tower of Qwen3.5-4B; 128 modules: `mlp.{gate,up,down}_proj` in all 32 layers,
`self_attn.{q,k,v,o}_proj` in the 8 full-attention layers; `lm_head`, embeddings, the Gated-DeltaNet layers and the vision tower are not adapted).
It transcribes whole pages of handwritten 16th–19th-century German (Kurrent). You edit the adapter only (never the base). No gradient, no retraining.

## The symptom the trainer recorded
On the trainer's frozen held-out draw (733 pages, 70 documents it never saw) this adapter reads at CER 0.418 but `truncated_cer` 0.250
(the reading clipped to the reference's length), length ratio 1.18, and 49 of 733 pages run to the 3,072-token cap. About 17 of the 42 points are text
written past the page: the adapter sometimes does not stop and loops (dashes, full stops, `✳`, repeated lines). Your practice pages are loop-rich:
on the 32 practice pages C0 gives CER 0.73, `truncated_cer` 0.33, 8 pages at the cap. Whether the loop is written by a direction in ΔW, or whether the
base alone would loop too, is **not known** (the base without adapter was not run). The brief of the harness says edits remove obstacles, they do not add ability;
no obstacle may exist. A result of "nothing found, nothing kept" is a fully acceptable outcome. Do not invent a gain.

## What you may read
- your pack (`pack/`): `pack.md` (config, per-module norms/effective rank and the ratio to two sibling adapters from other runs; all modules sit at 0.80–0.93 of
  the siblings' median, i.e. no module stands out by norm), `reading_energy_and_logit_lens.md` (energy per layer and kind; which tokens the largest singular directions of
  the output-writing modules `down_proj`/`o_proj` promote or suppress, read through the base's final norm and `lm_head`; z-scores of `<|im_end|>`, newline, `¬`), and `c0_practice.json`
  (this adapter's reading of the 32 practice pages: per-page CER, reference/hypothesis length, at-cap, loop, tails of the looping outputs);
- this brief, and `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/lora_subject.py`, `wsp/pages.py`.
- no donor: sibling adapters have independent A initialisations (cosine of ΔW ≈ 0.1); only statistics transfer. The donor rule is therefore moot; if your program uses
  a sibling's weights anyway, say so in the preregistration.

**Do not read** anything else: not other `runs/` folders, not `docs/`, not `~/atr-cache`, not the test side. Do not run the program or load anything: no GPU for you. `python -m py_compile` is fine.
Write only `preregistration_hypotheses.md` and `program.py` into the folder named in your task.

## The API (inside the sandbox: `wsp` + numpy only; one pass = the 32 practice pages decoded greedily, transformers, deterministic, about 30 minutes)
```python
from wsp.run import open_session
import numpy as np
s = open_session(".")                  # s.budget_left passes remain (10)
sub = s.subject                        # LoraSubject
T = sub.tensors()                      # name -> float32 array; ...layers.27.mlp.down_proj.lora_B.weight (out × r), ...lora_A.weight (r × in)
sub.modules(); sub.config["lora_alpha"]; sub.scale
U, S, Vt = sub.directions(module)      # SVD of the module's ΔW: U (out × r) directions written into the residual stream, S their strengths
sub.remove_direction(module, u)        # (not logged) use s.edit("project_out", [B name], direction=u, axis=0) for a logged edit; see wsp/ops.py
h = s.hypothesis("text", "predicted direction and size of the change in CER, truncated CER, pages at the cap", "edit class")   # BEFORE the edit
s.edit("scale", [names...], hypothesis=h, factor=0.8)   # ops: scale, add, set_rows, set_cols, interpolate, project_out (one edit = one op over named tensors;
                                                          #   per_tensor={name: {...}} when arguments differ by tensor). Scaling B scales the module's update.
m = s.measure("label")                 # one counted pass; m.cer; page metrics are in the log
kept = s.keep("reason")                # the harness's rule: kept only if the paired page-bootstrap 95 % interval of CER(after) - CER(last kept) lies entirely
                                       #   below 0 AND pages at the cap did not rise; else the edit is undone
s.revert("reason"); s.note("free text"); summary = s.freeze()    # always end with freeze
```
Budget 10 passes; pass 1 is the C0 baseline `s.measure("C0")` (it reproduces the 0.73 above; the engine is deterministic). So at most 9 edits. Each pass costs half an hour: do not waste them.
`s.measure` returns Metrics (`.cer`); the page metrics (`truncated_cer`, `at_cap`, `loops`, `length_ratio`) are in `s._log[-1]["page"]` (a dict) after a pass; use them for your own branching.
Whether a looping page stops is a chaotic event: a numerically tiny change (the same adapter merged into the weights instead of applied beside them) flipped one page of ten from loop to stop. An edit has to beat that: prefer edits whose
predicted effect is on many pages and larger than one page flipping.

## Task
Write `preregistration_hypotheses.md` first: the hypotheses you will test (from the readings, with the edit class, the modules, and a **predicted direction and magnitude** for CER, truncated CER and pages at the cap on the practice set; what you will conclude if it fails). Then `program.py`,
written in full in advance (it cannot be changed after the first pass; it may branch on results), at most 10 `measure` calls. State in the first lines of your preregistration the model id you are and your settings.
Candidates the reader of the pack may consider (you are free to add, drop or refine): (i) the directions that lower the `<|im_end|>` logit most (layers 31 `o_proj`/`down_proj`, 27 `down_proj`; note they also suppress newline and `¬`);
(ii) the directions that raise it (layer 30 `down_proj`) — strengthen; (iii) a per-group scale of the update (late MLP layers 24–31; the whole adapter by 0.9/1.1) as the cheapest family; (iv) per-module scale of the modules the loop
tails point to. Whichever you choose, say what would falsify it.
