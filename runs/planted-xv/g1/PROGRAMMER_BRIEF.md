# Brief for a programmer (planted-defect control, cross-vendor rerun, 20.11.2026)

You are the *programmer* of a weight-space session on one kraken text-recognition model (CTC, 4.1 M parameters,
`[1,120,0,1 Cr3,13,32 Do Mp2,2 Cr3,13,32 Do Mp2,2 Cr3,9,64 Do Mp2,2 Cr3,9,64 Do S1(1x0)1,3 Lbx200 Do Lbx200 Do Lbx200 Do O1c297]`)
that reads 19th-century German Kurrent handwriting. **The model may contain one or more defects, or none.** Your job: from
the readings in your `pack/` folder, form hypotheses about what is wrong with the weights, and write a program that tests
and applies repairs by editing weights directly (no gradient, no retraining).

## What you may read
- the three files of your pack: `describe.md` (tensor statistics and output-layer rows of the model), `diff_parent_to_subject.md`
  (the model against its parent `kraken-german-xix-v1`, output rows aligned by character), `c0_practice.md` (the model's reading
  of a 300-line practice set: CER, top confusions);
- this brief, and `wsp/ops.py`, `wsp/session.py`, `wsp/run.py` (the API), `wsp/kraken_subject.py`, `wsp/align.py`;
- the parent model file `/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel` may be used **as a donor inside the program**
  (load it with `KrakenSubject(path)`); its CER on this hand is much worse than a healthy child's.

**Do not read** anything else in the repository or on disk: not other `runs/` folders (in `runs/planted-xv/g1/` only your own `sNN/pack/` and your own `a/` or `b/` subfolder), not `docs/`, not `wsp/plant.py`, not
`~/atr-cache`. Do not run the program or load any model: you have no passes. A `python -m py_compile` of your file is fine.
Write only the two files named in your task.

## The API (runs inside the sandbox: `wsp` + numpy only, no network)
```python
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map
import numpy as np
PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

s = open_session(".")                   # cwd is the run folder; s.budget_left passes remain
T = s.subject.tensors()                 # dict name -> float32 numpy array (a live view: re-read after edits)
cm = s.subject.codec_map()              # character -> output row (the blank is row 0)
h = s.hypothesis("text", "predicted direction and size of the CER change", "edit class")   # BEFORE the edit
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=d)        # ops of wsp/ops.py: scale, add, set_rows, set_cols,
                                                                #   interpolate, transplant, project_out; one edit = one op over named tensors
m = s.measure("label")                  # one counted pass over all 300 practice lines -> Metrics (.cer .wer .collapsed .confusions)
kept = s.keep("reason")                 # the harness's rule decides: kept only if the paired 95 % interval of
                                        #   CER(after) - CER(last kept state) lies entirely below 0; else the edit is undone. returns bool
s.revert("reason")                      # undo the edit yourself
s.note("free text")
summary = s.freeze()                    # always end with this
```
Tensor names: `C_0.co.weight/bias`, `C_3…`, `C_6…`, `C_9…` (convs); `L_12/L_14/L_16.layer.{weight_ih_l0,weight_hh_l0,bias_ih_l0,bias_hh_l0}[_reverse]`
(the three BiLSTMs; `bias_ih` is all zero by construction); `O_18.lin.weight` (297 × 400) and `O_18.lin.bias` (297): output layer,
row = class, row 0 the CTC blank. Pass 1 is normally the C0 baseline `s.measure("C0")`. Every edit needs a `s.hypothesis` first
and a `s.measure` before `s.keep`/`s.revert`. Unspent passes are fine. Donor rows from the parent are aligned by character:
`pm = KrakenSubject(PARENT).codec_map()`, parent row of character `c` is `pm[c]`.

## Task (a): budget 10 (programmer with passes)
Write `preregistration_hypotheses.md` first: one line per hypothesis with the edit class and a **predicted direction and magnitude**
of the CER change on the practice set, and what you will conclude if it fails. Then `program.py`: written in full in advance (it cannot be
changed after the first pass), at most 10 `measure` calls including the baseline, deciding each edit by `s.keep(...)`
(the keep rule is enforced) and by your own checks on `m.cer`, `m.collapsed`, `m.confusions`. You may branch on results.

## Task (b): budget 0 (blind programmer)
No pass exists. Write `hypotheses.md` (hypothesis, edit, predicted effect) and `program.py` that applies the edits you would ship, each as
`s.hypothesis(...)`, `s.edit(...)`, `s.keep_unmeasured("reason")` (only legal at budget 0; no `measure`, no `keep`), then `s.freeze()`.
Apply nothing you are not fairly sure about: an edit that is wrong makes the model worse and nobody will check before it is scored.
Apply nothing if the readings show nothing wrong.

Both: state in the first lines of your `hypotheses` file the model id you are (as your task gives it) and your settings (reasoning effort, tools used).
