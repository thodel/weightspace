# Brief — phase 0 on asteraix

For the session (Claude Code or a person) that executes phase 0 of
[PLAN.md](../PLAN.md) on asteraix. Written 06.11.2026 on h1, where nothing can
run. Read [ASTERAIX_RUNBOOK.md](ASTERAIX_RUNBOOK.md), [DESIGN.md](DESIGN.md)
and [PROTOCOL.md](PROTOCOL.md) first; this brief says what to do, in order,
and what "done" looks like. **Phase 0 is harness work only: no edits to any
model's behaviour are attempted, no held-out set is scored, no forward pass
is "spent" against a budget.**

## 0. Ground rules on that machine

- Card 1 (`ATR_TRAIN_GPU=1`) is the trainer's. Export
  `CUDA_VISIBLE_DEVICES=0` in every shell, and run `nvidia-smi` first: card 0
  may host the neighbours' workers. Everything in phase 0 also works on CPU.
- The research share is read-only for this work. The only writes are under
  `~/Repo/weightspace/` and `~/atr-cache/wsp/`.
- The trainer's venvs (`~/Repo/training-atr-models/.venvs/*`) are services;
  do not install anything into them.
- `TMPDIR=~/atr-cache/tmp`, never the CIFS share (pip and `shutil` break there).
- Secrets: `.env` files on that box are mode 600; never print them.

## 1. Checkout and venv — `bash deploy/asteraix_bootstrap.sh` does §1 and §2 (docs/DEPLOY_ASTERAIX.md)

```bash
cd ~/Repo && git clone git@github.com:thodel/weightspace.git && cd weightspace
export TMPDIR=~/atr-cache/tmp; mkdir -p "$TMPDIR" ~/atr-cache/wsp
python3.12 -m venv .venv
.venv/bin/pip install --no-cache-dir -e '.[kraken,dev]'
.venv/bin/python -c "import kraken, torch; print(kraken.__version__, torch.__version__, torch.cuda.is_available())"
.venv/bin/python -m pytest -q tests          # the 18 engine-free tests must pass there too
```

If `kraken==7.0.2` refuses to resolve with the torch that pip picks, pin torch
to what the trainer's `kraken-train` venv has (`.venvs/kraken-train/bin/pip
show torch` — 2.8.0+cu128 on 16.09.2026) and record the final `pip freeze`
in `docs/ENV_ASTERAIX.md`.

## 2. Weights and digests

Copy the six fine-tunes as the runbook §2 shows, plus the parent and the
kraken+ checkpoint:

| what | where to find it |
|---|---|
| six in-house `.mlmodel` | `/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/trained/<id>/<id>.mlmodel` |
| parent `cursive_inzigkofen` | the HF cache or `htrmopo` by DOI `10.5281/zenodo.18207767`; the serving box caches it under `~/.kraken/` — if neither is reachable, download once from Zenodo and note the sha256 |
| kraken+ run (2026-09-01) | the three runs in `serving-atr-inference/docs/KRAKEN_PLUS.md`; take the `Cr1,1,85` **and** the `Cr255,1,85`-spec checkpoint if both still exist under the jobs tree (`…/training_folder/jobs/`) — the second is the one the lossless rewrite needs |

Write `models/DIGESTS.md`: id, path on the share, sha256, size, and for the
four the gateway knows (`combined-v1`, `schlussredaktion-v1`,
`german-xix-blb-v1`, and whichever else carries `weights_digest`) whether the
sha256 matches `gateway_models`. A mismatch is a finding, not something to fix.

## 3. Verify the kraken 7.0.2 API against `wsp/kraken_subject.py`

The subject was written on h1 without kraken installed. Check, and fix the
code where the names differ:

- `kraken.lib.vgsl.TorchVGSLModel.load_model(path)` → object with `.nn`,
  `.spec`, `.codec`, `.input`, `.user_metadata`, `.save_model(path)`;
- the codec's character→labels map (`codec.c2l`) and its blank index;
- `kraken.lib.models.TorchSeqRecognizer(model, device=…)`: `predict_string`
  and the raw `forward` signature (line tensor + lengths);
- `kraken.lib.dataset.ImageInputTransforms` arguments for a line crop at the
  model's input height, and whether lines must be dewarped/centred first;
- whether `save_model` on a model loaded from CoreML writes CoreML again
  byte-identically, or re-serialises (both are fine; the **text** must match).

Record what you found in `docs/KRAKEN_API_NOTES.md` with the exact version.

## 4. Practice sets (train side only)

Build `wsp/data.py` with `sample_lines(source, n, seed) -> jsonl` writing
`{"key", "text", "image"}` per line, where `image` is a line crop on local
disk under `~/atr-cache/wsp/practice/<set>/`. Sources, in this order of
preference:

1. the compiled arrows the trainer already has (`~/atr-cache/arrows/`,
   `~/atr-cache/artefacts/`): `kraken.lib.dataset.ArrowIPCRecognitionDataset`
   yields line images and text directly — **use only train/val arrows, never
   `german_test.arrow`**;
2. PAGE XML + image from the dataset trees under
   `/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt/<project>/` for the BLB
   *train* projects (`lassberg`, `korrespondenten`, `unbestimmt`,
   `manuskripte`; never `*_eval`), cutting lines from the baseline polygons as
   the trainer does (`atr_training.pagexml`);
3. the public `dh-unibe/image-text_*-lines` parquet sets for anything else.

Make three sets of 300 lines with seed 20261106: `bern-hertwig-train-300`,
`blb-train-300`, `medieval-val-300`. Write each list's sha256 into
`data/practice/README.md` with its source and selection rule.

Then the overlap audit: `wsp/audit.py overlap` listing, per practice set,
every document id it draws from, and confirming none of them is in the
held-out identities (`_eval` projects; the Hertwig eval pages of doc
13922499; `config/heldout_eval_documents.json` in training-atr-models). Save
the result next to the README.

## 5. The round-trip, the first real test

```bash
export WSP_MODEL=models/kraken/kraken-german-xix-blb-v1.mlmodel
export WSP_PRACTICE=data/practice/blb-train-300.jsonl
.venv/bin/python -m pytest -q tests/asteraix
```

Done when: the kraken version check passes, the saved copy has the same
digest, and the 20 lines read identically before and after. Then score the
full 300 lines once per family model (not budgeted — this is C0 of the
protocol) and write the numbers into `docs/BASELINES.md`: CER, WER,
collapsed lines, length ratio, top-10 confusions, with the practice-set hash.
These are practice numbers, not results; say so in the file.

## 6. The lossless rewrite (kraken+ `Cr255 → Cr1`)

Write `wsp/rewrite.py`: load the `Cr255,1,85,1,1` checkpoint, build a model
from the same spec with `Cr1,1,85,1,1`, copy every tensor, and for that conv
copy only kernel row 127 (`ops.drop_padding_rows`). Save as `.mlmodel`. Done
when the two models give identical text on `medieval-val-300` and the
parameter count drops by 11,054,080. This is the first entry in
`docs/RESULTS.md` (a function-preserving edit, CER unchanged by construction,
measured anyway).

## 7. The session runner

`wsp/run.py` with four sub-commands, as the runbook §4–5 sketches:

- `new --subject … --practice … --budget N` creates `runs/<YYYYMMDD>-<id>-b<N>-<n>/`
  with `preregistration.md` from a template (PROTOCOL §2 fields, empty), copies
  nothing heavy, and prints the sandbox command line:
  `env -i PATH=… HOME=<run dir> CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 unshare -n .venv/bin/python -I …`
  (test whether `unshare -Un true` works for this account; if not, drop
  `unshare -n` and say so in the run's `summary.json` deviations);
- `freeze <run>` calls `Session.freeze` on a saved session state;
- `heldout <run> --set <name>` scores the frozen artefact once, from outside
  the sandbox, with the set's identity hash;
- `export <run> --as <id>` writes the `.mlmodel` and a registry YAML in the
  trainer's `trained/<id>.yaml` layout (copy the fields from an existing one;
  **do not** copy the result to the share or register it — that is phase 4
  and needs an OK).

A smoke test: a budget-2 session on `blb-train-300` whose only edit is
`scale` of the output bias by 1.0 (a no-op), kept, frozen; `diff.json` must
be empty and `summary.json` complete.

## 8. The first reading, passes = 0

Not an edit session: `wsp/stats.py` with per-tensor norms, per-row energies
of the output layer, and the lineage diff `wsp.diff.compare` for the pairs
`hertwig-v2 → combined-v1`, `hertwig-v2 → schlussredaktion-v1`,
`xix-v1 → xix-blb-v1`, with codec rows aligned by character
(`wsp.align`). Output: `docs/READINGS.md`, one section per pair, with the
tables and three sentences of what moved. No forward passes are needed; the
confusion pairing (PROTOCOL, track B.1) comes from the C0 numbers of §5.

## 9. Report back

Commit as you go on a branch `phase0`, push, and leave in the PR description
or in `docs/PHASE0_REPORT.md`: what was verified, every deviation from this
brief, the numbers of §5 and §6, the open questions for phase 1, and the
exact commands that reproduce §5. Add a line to h1's handoff file
(`~/HANDOFF-hodelweb.md`, section 5) only if you touched h1; otherwise this
repo is the log.

## What not to do in phase 0

- No budgeted session on a real target, no held-out score, no registry write,
  no upload to HF, no change to the trainer or the serving config.
- No practice line from an `_eval` project, from `german_test.arrow`, or from
  the Hertwig eval pages.
- No use of card 1.
