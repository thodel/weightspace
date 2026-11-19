# Brief for a programmer (TrOCR planted-defect control, 18.11.2026)

You are the *programmer* of a weight-space session on one TrOCR text-recognition model (a Hugging Face `VisionEncoderDecoderModel`, 334 M
parameters: a 12-layer ViT encoder of width 768 that reads a text-line image, a 12-layer transformer decoder of width 1024 with 16 heads and
FFN width 4096 that writes the line as BPE tokens, beam search 4) that reads 16th to 19th century German Kurrent handwriting. **The model may
contain one or more defects, or none.** Your job: from the readings in your `pack/` folder, form hypotheses about what is wrong with the
weights, and write a program that tests and applies repairs by editing weights directly (no gradient, no retraining).

## What you may read
- the files of your pack (the folder your task names): `describe.md` (shape, L2 norm, std, share of zeros, non-finite count of every tensor, and
  the per-layer norms), `c0_practice.md` (the model's reading of a 300-line practice set: CER, top confusions), and, **only if your task says
  you have the donor**, `diff_parent_to_subject.md` (the model against its parent donor: per-tensor distance, norm ratio, cosine; the token rows of
  the tied embedding that moved most; the attention heads that moved most);
- this brief, and `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/trocr_subject.py` (the API);
- with the donor, the parent model directory `/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII` may be loaded **inside the program**
  (see below). Without the donor you must not read, load or name it.

**Do not read** anything else in the repository or on disk: not other `runs/` folders (including other folders of `runs/trocr-planted/`), not
`docs/`, not `wsp/trocr_plant.py`, not `scripts/`, not `~/atr-cache` (except the donor directory when you have it). Do not run the program or load any
model: you have no passes. A `python -m py_compile` of your file is fine. Write only the files named in your task.

## The model, tensor names, shapes
Canonical names (zero-padded layer numbers, so sorted order is layer order). `describe.md` lists every tensor with its shape.
- Encoder, layers `enc.00 … enc.11`: `enc.NN.attn.{q,k,v}.weight` (768×768; rows are output features; 12 heads of 64), `enc.NN.attn.o.{weight,bias}`, `enc.NN.ffn.in.{weight,bias}` (3072×768), `enc.NN.ffn.out.{weight,bias}`, `enc.NN.norm_attn.*`,
  `enc.NN.norm_ffn.*`; `enc.embed.{cls,patch.weight,patch.bias,pos}`, `enc.norm_final.*`.
- Decoder, layers `dec.00 … dec.11`: `dec.NN.self.{q,k,v,o}.{weight,bias}` and `dec.NN.cross.{q,k,v,o}.{weight,bias}` (1024-wide, 16 heads of
  64; `cross.k/v` read the 768-wide encoder output), `dec.NN.ffn.{in,out}.{weight,bias}`, `dec.NN.norm_{self,cross,ffn}.{weight,bias}`.
- `dec.embed.tokens` (vocabulary × 1024), `dec.embed.pos`, `dec.embed.norm.{weight,bias}`.
- **`dec.embed.tokens` is tied to the output projection: one tensor, used as input embedding *and* as the readout. There is no output bias.**
  A change of a row changes both how the token is read in and how likely it is to be written.
- A head of an output projection `*.o.weight` is a block of 64 *columns*; of `q,k,v` a block of 64 *rows*.
- Token ids and strings: the row index of `dec.embed.tokens` is the BPE id; a leading `U+0120` in a token string is a word-initial space.
  The pack prints every character that is not plain printable ASCII with its code point (`<U+00A0>` is a no-break space, not a plain space).

## The API (runs inside the sandbox: `wsp`, `transformers`, `torch` (CPU/card) and numpy; no network)
```python
from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np
PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"       # only with the donor

s = open_session(".")                   # cwd is the run folder; s.budget_left passes remain
T = s.subject.tensors()                 # dict name -> float32 numpy array (a live view: re-read after edits)
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()             # donor tensors, same names (only with the donor)
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")    # donor rule: lines the donor was not trained on; with the donor, pass it to measure.
                                                # Without the donor use MASK = s.unseen_by() (all lines).
h = s.hypothesis("text", "predicted direction and size of the CER change", "edit class")   # BEFORE the edit
s.edit("scale", ["dec.03.cross.o.weight", "dec.03.cross.o.bias"], hypothesis=h, factor=1.4)  # ops of wsp/ops.py: scale, add, set_rows,
                                        # set_cols, interpolate, transplant, project_out; one edit = one op over named tensors
                                        # (per_tensor={name: {...}} gives arguments that differ by tensor, e.g. one donor array each)
m = s.measure("label", rule_mask=MASK)  # one counted pass over all 300 practice lines -> Metrics (.cer .wer .collapsed .confusions)
kept = s.keep("reason")                 # the harness's rule decides: kept only if the page-clustered paired 95 % interval of
                                        #   CER(after) - CER(last kept state) lies entirely below -0.005 (half a CER point); else the edit is undone
s.revert("reason")                      # undo the edit yourself
s.note("free text")
summary = s.freeze()                    # always end with this
```
Pass 1 is normally the C0 baseline `s.measure("C0", rule_mask=MASK)`. Every edit needs an `s.hypothesis` first and a `s.measure` before
`s.keep` / `s.revert`. Unspent passes are fine. A pass decodes 300 lines with the pinned beam-search settings and takes about half a minute. The
noise of the metric across retrained models is about 0.5 CER points; the practice set is 300 lines of one hand, so a kept edit can still be a
small overfit. Donor tensors with the same name have the same shape; `interpolate(w, other, alpha)` and `transplant(w, donor)` take donor arrays
through `per_tensor`. The model is a *child* of the donor: the donor reads this hand worse than a healthy child does, so a full transplant of a
tensor is usually not the answer.

## Task (a): budget 10 (programmer with passes)
Write `preregistration_hypotheses.md` first: one line per hypothesis with the edit class and a **predicted direction and magnitude** of the CER
change on the practice set, and what you will conclude if it fails. Then `program.py`: written in full in advance (it cannot be changed after the
first pass), at most 10 `measure` calls including the baseline, deciding each edit by `s.keep(...)` (the keep rule is enforced) and by your own
checks on `m.cer`, `m.collapsed`, `m.confusions`. You may branch on results.

## Task (b): budget 0 (blind programmer, with the donor)
No pass exists. Write `hypotheses.md` (hypothesis, edit, predicted effect) and `program.py` that applies the edits you would ship, each as
`s.hypothesis(...)`, `s.edit(...)`, `s.keep_unmeasured("reason")` (only legal at budget 0; no `measure`, no `keep`), then `s.freeze()`. Apply
nothing you are not fairly sure about: an edit that is wrong makes the model worse and nobody will check before it is scored. Apply nothing if the
readings show nothing wrong.

Both: state in the first lines of your `hypotheses` file the model id you are and your settings (reasoning effort, tools used). Your task says
whether you have the donor.
