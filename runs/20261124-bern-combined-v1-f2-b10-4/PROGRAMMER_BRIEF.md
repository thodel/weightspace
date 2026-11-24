# Brief for the programmer (kraken session on `bern-combined-v1`, 24.11.2026)

You are the **programmer** of a weight-space session on one real kraken text-recognition model,
`kraken-bern-combined-v1`. It is a CTC line recogniser with 4.1 M parameters: a conv stack
(`C_0 C_3 C_6 C_9`), three bidirectional LSTMs (`L_12 L_14 L_16`, 200 units each way) and an output
layer `O_18.lin` (270 rows × 400; **row 0 is the CTC blank**). It reads the 1528 Bern Disputation
manuscripts, two hands: `Hertwig` and `Schlussredaktion`. Your goal is to find edits of its weights
that lower its character error rate (CER). There is no gradient and no retraining. Edits remove
obstacles; they do not add ability, and there may be no obstacle that an edit can remove.
"Nothing found, nothing kept" is a fully acceptable outcome. Do not invent a gain.

Your run folder is `runs/20261124-bern-combined-v1-f2-b10-4/` in the worktree
`/home/tobias/Repo/weightspace/.claude/worktrees/kraken-s2-repeats-4f8360`. All paths below are
relative to that worktree.

## The models

- **Subject:** `/home/tobias/Repo/weightspace/models/kraken/kraken-bern-combined-v1.mlmodel`.
  It was fine-tuned from `hertwig-v2` on the training pages of both hands. Its Hertwig training
  pages came from an export in which 44 of the 50 Hertwig pages carry another page's image for
  their transcription, so it learnt part of Hertwig from mismatched image–text pairs.
- **Donors you may read (read-only):**
  - `/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel`, the **parent**.
    It was trained on the Hertwig training pages with correct images.
  - `/home/tobias/Repo/weightspace/models/kraken/kraken-bern-schlussredaktion-v1.mlmodel`, a
    **sibling**. It has the same parent and was fine-tuned on clean Schlussredaktion pages.

  Codecs differ. Align output rows **by character** (`donor.codec_map()`, `wsp/align.py`),
  never by index.

## The practice set

`data/practice/bern-train-300.jsonl` has 300 lines from 146 pages: 103 Hertwig lines and
197 Schlussredaktion lines. They are drawn from the **training side** of both hands, with the
correct Hertwig images. So the subject was trained on these pages, though with wrong images for
most Hertwig pages. The parent was trained on the Hertwig lines, and the sibling on the
Schlussredaktion lines, both with correct images. Every line carries `key`
(`<project>/<page>/<line>`), `text` (the reference) and `doc`. You may read the reference texts
and look at the line images (`image`). One pass reads all 300 lines on the CPU in under a minute.

The held-out set is 545 lines from 14 test pages of the two hands. No page in it is in the
practice set or among the training pages. It is scored once, after you freeze, by the operator.
You do not see it.

## What you may read

- **This brief.**
- **Your pack**, `runs/20261124-bern-combined-v1-f2-b10-4/pack/`:
  - `readings_budget0.md` holds the budget-0 readings of the model family from phase 0, before
    any session;
  - `describe_*.{md,json}` holds tensor statistics and output-layer rows of the subject, the
    parent and the sibling;
  - `diff_bern-hertwig-v2__bern-combined-v1.{md,json}` and
    `diff_bern-hertwig-v2__bern-schlussredaktion-v1.{md,json}` give per-tensor and per-row
    changes, parent → subject and parent → sibling, with rows aligned by character;
  - `c0_c1.md` and `per_line_edits.json` hold the unedited subject and the donors on the practice
    set (CER per hand, confusions, blank share, per-line edits), and the held-out totals.
- **The practice list:** `data/practice/bern-train-300.jsonl` and `.meta.json`.
- **The harness source:** `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`,
  `wsp/subject.py`, `wsp/align.py`, `wsp/metrics.py`, `wsp/data.py`.
- **In your own run folder, only these files:** `preregistration.md`, `notes.md`, your own
  `program_NN.py`, and, once your passes exist, `log.jsonl`, `readings/pass_*.jsonl` (your own
  passes, line by line), `state.json`, `budget.json`, `session.json`. Nothing else there
  (no `stdout_*`, `sandbox_*`, `summary.json` or any other file the operator writes).

**Do not read anything else.** That means:
- no other `runs/` folder;
- nothing else under `docs/`, `protocol/` or `paper/`, including plans, reports and results;
- no git history (`git log`, `git show`, other branches);
- nothing under `~/atr-cache` except the practice line images;
- nothing from the held-out side (`data/heldout/`);
- nothing under `.claude/` and no other session's transcripts.

Other programmers have worked on this model before. This session must be independent of that
work, so do not look for it.

## What you may run

- `python -m py_compile`.
- CPU-only reading scripts with `.venv/bin/python` that open the three models read-only
  (`KrakenSubject(path).tensors()`, `.codec_map()`, numpy) or read your pack. They must not call
  `forward`, `measure` or `open_session`, and must not write anything outside your run folder.
  Prefix them with `CUDA_VISIBLE_DEVICES=`.
- **Never execute `program_*.py` yourself.** The operator runs each program in the sandbox and
  gives you its stdout.

## The API

The sandbox has `wsp` and numpy only.

```python
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np
s = open_session(".")                 # cwd = run folder; restores the last KEPT state; s.budget_left passes remain
T = s.subject.tensors()               # name -> float32 array (live)
cm = s.subject.codec_map()            # character -> output row (blank = row 0)
donor = KrakenSubject("/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel")   # read-only
h = s.hypothesis("text", "predicted direction and size of the CER change", "edit class")   # BEFORE every edit
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=d)    # ops in wsp/ops.py: scale, add, set_rows, set_cols, interpolate, transplant, project_out
s.edit("transplant", names, hypothesis=h, per_tensor={name: {"donor": array}})   # several tensors, one argument set each
m = s.measure("label")                # one counted pass over all 300 lines -> Metrics (.cer .collapsed .length_ratio .confusions ...)
kept = s.keep("reason")               # the harness's rule decides (below); a refused edit is undone
s.revert("reason"); s.note("free text")
summary = s.freeze()                  # only in your last program
```

Tensor names: `C_0/C_3/C_6/C_9 .co.weight|bias`;
`L_12/L_14/L_16 .layer.{weight_ih_l0,weight_hh_l0,bias_ih_l0,bias_hh_l0}[_reverse]`;
`O_18.lin.weight|bias`. The pass log entry (`s._log[-1]`) holds `vs_kept` (the paired interval
against the last kept state) and `vs_baseline`.

## The rules

- **Budget:** 10 passes. **Pass 1 is the baseline, `s.measure("C0")`, and it comes first.** It
  is predicted to reproduce the subject's line of `c0_c1.md` (CER 0.1456). If it does not, stop
  and say so. That leaves at most 9 edits.
- **Keep rule (PROTOCOL §3, `min_effect` 0):** an edit is kept only if the paired per-line
  bootstrap 95 % interval of CER(after) − CER(last kept), with 2,000 resamples and seed 20261106,
  lies entirely below 0. `s.keep()` enforces this and undoes a refused edit. You also revert any
  edit that raises the collapsed-line count (`m.collapsed`) against the last kept state, even if
  the harness would keep it.
- **For a claim, the interval is also clustered by page (amendment A4).** The operator computes
  it after the session. Lines of one page share a hand, an image and a segmentation, so the
  page-clustered interval is wider than the line interval.
- **Donors** may be used: copy, transplant, average or interpolate toward them. No donor rule is
  in force; the keep rule is computed on all 300 lines. State in your hypotheses what each donor
  was trained on and what that means for the practice set.

## How the session runs (rounds)

**Round 0, before any pass:**
1. Fill in section 5, "Hypotheses", of `preregistration.md` in your run folder. Leave the other
   sections as they are. For each hypothesis give:
   - the edit class and the tensors;
   - a **predicted direction and magnitude** of the practice CER change (and, where you expect
     it, per hand);
   - what you will conclude if it fails.

   State your plan for all passes. Later passes may be "follow-ups decided from these results",
   each written as a hypothesis with a prediction before its pass.
2. Write `program_01.py`: the baseline pass plus as many hypotheses as you want to run before
   you see results. It may be the whole session, with branches on the keep decisions. Each edit
   is followed by `measure` and then by `keep` (or `revert`). Print what you need to stdout.
3. Append a timestamped entry (UTC) to `notes.md` with your model id, your settings and what you
   read.

**Every round** (round 0 included), end your reply to the operator with one line confirming
whether anything you read named an edit, a kept decision or a result of an earlier session on
this model. If something did, say what and where, and stop.

**Rounds 1, 2, …:** the operator runs `program_NN.py` and gives you its stdout. You then read
`log.jsonl` and `readings/`, append to `notes.md`, and write `program_NN+1.py`. Your **last**
program ends with `s.freeze()`. A program that does no pass, only `freeze`, is fine. Freeze when
you are done or when the budget is used up. Unspent passes may stay unspent.

Write only into your run folder: `preregistration.md` (section 5 only), `program_NN.py` and
`notes.md`.
