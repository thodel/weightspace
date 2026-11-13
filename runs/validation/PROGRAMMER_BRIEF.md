# Brief for a programmer (validation-practice sessions, 07.11.2026)

You are the *programmer* of a weight-space session on one real kraken text-recognition model (CTC, 4.1 M parameters, BiLSTM ×3 + conv stack,
output layer `O_18` with row 0 = CTC blank). Goal: find edits of its weights (no gradient, no retraining) that lower its character error rate on text it
has not been trained on. Your program is written **in advance** from the readings in your `pack/` folder, then run once in a sandbox with a budget of 10 passes
(pass 1 = C0 baseline) on a practice set of lines the model was *not* trained on. A held-out score is taken once at the end by someone else; you do not see it.

## What you may read
- the files of your pack (`describe*.md` tensor statistics and output-layer rows; `diff*.md` against parent and sibling models, rows aligned by character;
  `c0_practice.md` the unedited model's reading of the practice set, with CERs of the donor models on the same lines);
- this brief and `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`, `wsp/align.py` (the API).
**Do not read** other files of the repository or disk (no `runs/`, `docs/`, `~/atr-cache`). Do not run the program or load a model. `python -m py_compile` is fine.

## API (sandbox: `wsp` + numpy only)
```python
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np
s = open_session(".")
T = s.subject.tensors(); cm = s.subject.codec_map()          # name -> array (live), character -> output row (blank = 0)
donor = KrakenSubject("/home/tobias/Repo/weightspace/models/kraken/<id>.mlmodel")   # donor models listed in your pack; read-only; donor.codec_map() aligns rows by character
h = s.hypothesis("text", "predicted direction and size of the CER change", "edit class")       # before every edit
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=d)      # ops: scale, add, set_rows, set_cols, interpolate, transplant, project_out
    # several tensors with different arguments each: s.edit("transplant", names, hypothesis=h, per_tensor={name: {"donor": array}})
m = s.measure("label")                                        # one counted pass over all practice lines -> Metrics (.cer .collapsed .confusions ...)
m = s.measure("label", rule_mask=s.unseen_by("<donor id>"))   # DONOR RULE: judge an edit that uses a donor on the lines that donor was not trained on
kept = s.keep("reason")                                       # the harness's rule: kept only if the paired 95 % interval of CER(after) - CER(last kept) lies below 0
s.revert("reason"); s.note("text"); summary = s.freeze()      # always end with freeze()
```
**Donor rule:** a donor (any other model whose weights an edit copies, averages or transplants) must not have been trained on the lines the keep rule is computed on.
Your pack says which practice lines each donor trained on (`seen_by`); for an edit that uses donor `X`, pass `rule_mask=s.unseen_by("X")` to `measure`. Edits that use no donor need no mask.
Tensor names: `C_0/C_3/C_6/C_9 .co.weight|bias`; `L_12/L_14/L_16 .layer.{weight_ih_l0,weight_hh_l0,bias_ih_l0,bias_hh_l0}[_reverse]`; `O_18.lin.weight|bias`.

## Task
Write `preregistration_hypotheses.md` first (model id you are: claude-sonnet-5-5 unless you know otherwise; settings; one line per hypothesis with edit class and
**predicted direction and magnitude** of the CER change, and the conclusion if it fails). Then `program.py`: complete in advance, ≤ 10 `measure` calls including the baseline,
deciding each edit with `s.keep(...)`, free to branch on results. Your two files go to the paths named in your task; write nothing else.
