# Brief for a programmer (planted-defect control on an adapter, cross-vendor rerun, 20.11.2026)

You are the *programmer* of a weight-space session on one **peft LoRA adapter**: rank r = 64, `lora_alpha` in its config,
on the language tower of Qwen3.5-4B (a page model that transcribes whole pages of handwritten 16th–19th-century German).
The adapter has 128 modules: `mlp.{gate,up,down}_proj` in all 32 layers and `self_attn.{q,k,v,o}_proj` in the 8 full-attention
layers. For each module the base receives the update ΔW = (lora_alpha / r) · B · A. **The adapter may contain one or more
defects, or none.** Your job: from the reading pack, form hypotheses about what is wrong with the weights, and write a program
that tests and applies repairs by editing the adapter directly (no gradient, no retraining, the base model is never edited).

## What you may read
- the files of your pack (`pack/pack.md`): the adapter's config, per-module statistics (Frobenius norm of ΔW, effective rank,
  largest singular value, ‖B‖, ‖A‖) and the ratio of each module's norm to the median of two **sibling adapters** (same base, same
  recipe, other training runs, healthy as far as known), and a C0 reading of the practice set (this adapter as it is);
- this brief, and `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/lora_subject.py` (the API);
- the two sibling adapters may be used **as donors inside the program**: `LoraSubject("/home/tobias/atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v1")`
  and `LoraSubject("/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/trained/qwen3.5-4b-german-kurrent-xvi-xix-blb-page-v1")`.
  Note: each adapter's A matrices were initialised independently, so a sibling's `B`, `A` cannot be mixed into this adapter's tensors;
  only statistics (norms, ranks) transfer.

**Do not read** anything else in the repository or on disk: not other `runs/` folders (in `runs/planted-xv/g1/` only `q01/pack/` and your own `a/` or `b/` subfolder), not `docs/`, not `wsp/plant_adapter.py`,
not `~/atr-cache`. Do not run the program or load any adapter: you have no passes. A `python -m py_compile` of your file is fine.
Write only the files named in your task.

## The API (runs inside the sandbox: `wsp` + numpy only, no network; a pass is one batch of 32 pages decoded greedily, up to 3,072 tokens each)
```python
from wsp.run import open_session
from wsp.lora_subject import LoraSubject
import numpy as np

s = open_session(".")                  # cwd is the run folder; s.budget_left passes remain
sub = s.subject                        # a LoraSubject
T = sub.tensors()                      # dict name -> float32 array; names like
                                       #   base_model.model.model.language_model.layers.27.mlp.down_proj.lora_B.weight  (out × r)
                                       #   ...lora_A.weight (r × in)       (a live view: re-read after edits)
sub.modules()                          # the 128 module prefixes (tensor name minus '.lora_A.weight' / '.lora_B.weight')
sub.config["lora_alpha"], sub.scale    # alpha and alpha / r
U, S, Vt = sub.directions(module)      # SVD of the module's ΔW without forming it (U: out × r, S: r, Vt: r × in)
h = s.hypothesis("text", "predicted direction and size of the change in CER / pages at the cap", "edit class")   # BEFORE the edit
s.edit("scale", [name_B1, name_B2], hypothesis=h, factor=0.5)   # ops of wsp/ops.py: scale, add, set_rows, set_cols, interpolate,
                                                                 #   transplant, project_out. One edit = one op over named tensors
                                                                 #   (per_tensor={name: {...}} for arguments that differ by tensor).
                                                                 # Scaling a module's B scales its update; scaling every B by f is a change of
                                                                 #   lora_alpha by f.
m = s.measure("label")                 # one counted pass over the 32 practice pages -> Metrics (.cer, .collapsed, ...)
kept = s.keep("reason")                # the harness's rule decides: kept only if the paired 95 % interval of CER(after) - CER(last kept) lies
                                       #   entirely below 0 AND the number of pages that ran to the token cap did not rise; else the edit is undone
s.revert("reason")                     # undo the edit yourself
s.note("free text")
summary = s.freeze()                   # always end with this
```
Pass 1 is normally the C0 baseline `s.measure("C0")`. Every edit needs a `s.hypothesis` first and a `s.measure` before `s.keep`/`s.revert`.
`Metrics` after a page pass: `m.cer` (page-level CER, includes text written past the page), `m.collapsed`. Passes are decoded by a batched engine and
are **noisy**: repeating the same adapter changes the practice CER by several points because a few pages loop to the cap or not. An edit has to
beat that.

## Task (a): budget 6 (programmer with passes)
Write `preregistration_hypotheses.md` first: one line per hypothesis with the edit class and a **predicted direction and magnitude** of the CER change
on the practice set, and what you will conclude if it fails. Then `program.py`: written in full in advance (it cannot be changed after the first pass),
at most 6 `measure` calls including the baseline, deciding each edit by `s.keep(...)` (the keep rule is enforced) and by your own checks. You may branch on results.

## Task (b): budget 0 (blind programmer)
No pass exists. Write `hypotheses.md` (hypothesis, edit, predicted effect) and `program.py` that applies the edits you would ship, each as `s.hypothesis(...)`,
`s.edit(...)`, `s.keep_unmeasured("reason")` (only legal at budget 0; no `measure`, no `keep`), then `s.freeze()`. Apply nothing you are not fairly sure about: an
edit that is wrong makes the adapter worse and nobody will check before it is scored. Apply nothing if the readings show nothing wrong.

Both: state in the first lines of your `hypotheses` file the model id you are (as your task gives it) and your settings (reasoning effort, tools used).
