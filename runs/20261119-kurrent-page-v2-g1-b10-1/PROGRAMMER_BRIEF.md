# Brief for the programmer (replicate of the phase-5 adapter session, 10.11.2026)

You are the **programmer** of a weight-space session on a real **peft LoRA adapter**,
`qwen3.5-4b-german-kurrent-xvi-xix-page-v2`. It has rank 64 and `lora_alpha` 128, so its
scale is 2. It sits on the language tower of Qwen3.5-4B, on 128 modules:
`mlp.{gate,up,down}_proj` in all 32 layers, and `self_attn.{q,k,v,o}_proj` in the 8
full-attention layers (3, 7, 11, …, 31). `lm_head`, the embeddings, the Gated-DeltaNet layers
and the vision tower are not adapted. The adapter transcribes whole pages of handwritten
16th–19th-century German (Kurrent). **You edit the adapter only, never the base.** There is
no gradient and no retraining.

Your run folder is `runs/20261119-kurrent-page-v2-g1-b10-1/` in the worktree
`/home/tobias/Repo/weightspace`. All paths below
are relative to that worktree.

## The symptom

On the trainer's frozen held-out draw (733 pages from 70 documents it never saw), the adapter
reads:
- CER 0.418, but `truncated_cer` 0.250, which is the reading clipped to the reference's length;
- `length_ratio` 1.18;
- 49 of 733 pages run to the 3,072-token cap.

About 17 of the 42 points are text written past the end of the page. The adapter sometimes
does not stop, and loops on dashes, full stops, `✳` or repeated lines.

On your 64 practice pages (C0, the adapter as it is):
- CER 0.4515, `truncated_cer` 0.2564, `length_ratio` 1.226;
- 9 pages at the cap, 12 pages with a loop.

About 19.5 points of that CER are text past the page. Nobody knows whether the loop is
written by a direction in ΔW, or whether the base alone would loop too. Edits remove
obstacles; they do not add ability, and there may be no obstacle to remove. "Nothing found,
nothing kept" is a fully acceptable outcome. Do not invent a gain.

## What you may read

- **This brief.**
- **The budget-0 reading of this adapter:** `docs/readings/adapter_kurrent-page-v2.md` and
  `.json`. They give the energy per layer and module kind, and the logit lens of the
  residual-writing directions. The lens reads which tokens the largest singular directions of
  `down_proj`/`o_proj` promote or suppress through the base's final norm and `lm_head`, as
  z-scores for `<|im_end|>`, newline and `¬`.
- **C0 on the practice pages:** `docs/baselines/c0-kurrent-val-v2.json`. `practice_per_page`
  maps each key to [edits, reference characters]. `at_cap_pages` and `loop_pages`
  list the defective pages.
- **The practice list and its metadata:** `data/practice/kurrent-val-64.jsonl` and `.meta.json`.
- **The harness source:** `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`,
  `wsp/lora_subject.py`, `wsp/pages.py` and `wsp/metrics.py`.
- **Everything in your own run folder**, once it exists: `log.jsonl`, `passes/` with the
  per-page readings of your own passes, `state.json` and `budget.json`.

**Do not read anything else.** That means:
- no other `runs/` folder;
- nothing else under `docs/` or `protocol/`, including the plans and reports;
- no git history (`git log`, `git show`);
- nothing under `~/atr-cache` except, through the API, the adapter itself;
- nothing from the test side.

Another programmer has worked on this adapter before. This session must be independent of
that work, so do not look for it.

## What you may run

- `python -m py_compile`.
- CPU-only reading scripts with `.venv/bin/python` that open the adapter through the API
  (`LoraSubject`, `sub.directions(...)`, numpy) or the lens JSON. They must not call
  `measure`, load a model onto a GPU, or write anything outside your run folder. Use
  `CUDA_VISIBLE_DEVICES=` to be safe.
- **Never execute `program_*.py` yourself.** The operator runs each program in the sandbox
  and tells you when it has finished. A pass is a GPU job of about 35 minutes.

## The API

The API runs inside the sandbox, with `wsp` and numpy only. One pass is the 64 practice
pages, decoded greedily by the trainer's own `evaluate_qlora` (transformers, deterministic,
cap 3,072), and takes about 35 minutes.

```python
from wsp.run import open_session
import numpy as np
s = open_session(".")                  # cwd = run folder; restores the last KEPT state; s.budget_left passes remain
sub = s.subject                        # LoraSubject
T = sub.tensors()                      # name -> float32 array; e.g. base_model.model...layers.27.mlp.down_proj.lora_B.weight (out × r), lora_A (r × in)
sub.modules(); sub.scale
U, S, Vt = sub.directions(module)      # SVD of the module's ΔW: U (out × r) directions written into the residual stream, S their strengths
h = s.hypothesis("text", "predicted direction and size of the change in CER, truncated CER, pages at the cap", "edit class")   # BEFORE the edit
s.edit("project_out", [B_name], hypothesis=h, direction=u, axis=1)   # logged edit; ops in wsp/ops.py: scale, add, set_rows, set_cols, interpolate, project_out
s.edit("scale", [names...], hypothesis=h, factor=0.8)                 # scaling B scales that module's update
m = s.measure("label")                 # one counted pass; m.cer; page metrics in s._log[-1]["page"] (truncated_cer, at_cap, loops, length_ratio)
kept = s.keep("reason")                # the harness's rule decides (see below); a refused edit is undone
s.revert("reason"); s.note("free text")
summary = s.freeze()                   # only in your last program
```

For the edit classes and their arguments, read `wsp/ops.py`, and `wsp/lora_subject.py`
for the tensor names.

## The rules

- **Budget:** 10 passes. **Pass 1 is the baseline, `s.measure("C0")`, and it comes first.**
  It is predicted to reproduce C0 above exactly, because the engine is deterministic. If it
  does not, stop and say so. That leaves at most 9 edits.
- **Keep rule (PROTOCOL §3, page unit, `min_effect` 0):** an edit is kept only if both hold:
  - the paired page-bootstrap 95 % interval of CER(after) − CER(last kept), with 2,000
    resamples, lies entirely below 0;
  - the pages at the cap did not rise against the last kept state.

  `s.keep()` enforces this. **A claim needs the page-clustered interval too (amendment A4).**
  The harness logs both.
- **Whether a looping page stops is a chaotic event.** A numerically tiny change can flip a
  page from loop to stop. Prefer edits whose predicted effect spans many pages and is larger
  than one page flipping.
- **No donor.** Sibling adapters have independent A initialisations, and v1 trained on the
  held-out documents. Do not use another adapter's weights.

## How the session runs (rounds)

**Round 0, before any pass:**
1. Fill in section 5, "Hypotheses", of `preregistration.md` in your run folder. Leave the other
   sections as they are. For each hypothesis give:
   - the edit class and the modules;
   - a **predicted direction and magnitude** for CER, `truncated_cer` and the pages at the cap
     on the practice set;
   - what you will conclude if it fails.

   State your plan for all passes. Later passes may be "follow-ups decided from these
   results", each written as a hypothesis with a prediction before its pass.
2. Write `program_01.py`: the baseline pass plus as many of your first hypotheses as you
   want to run before you see results. Each edit is followed by `measure` and then by
   `keep` (or `revert`).
3. Append a timestamped entry (UTC) to `notes.md` with your model id, your settings and what
   you read.

**Rounds 1, 2, …:** the operator runs `program_NN.py` and tells you its stdout. You then read
`log.jsonl` and `passes/`, append to `notes.md`, and write `program_NN+1.py`. Your **last**
program ends with `s.freeze()`. A program that does no pass, only `freeze`, is fine. Freeze
when you are done, or when the budget is used up.

Write only into your run folder: `preregistration.md` (section 5 only), `program_NN.py` and
`notes.md`.
