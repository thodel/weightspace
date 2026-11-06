# Running on asteraix

asteraix (dhserver03, 130.92.59.242) is the training box: two A40s, the
research share under `/mnt/wbkolleg_dh_1`, the trainer's venvs under
`~/Repo/training-atr-models/.venvs/`, local scratch under `~/atr-cache/`.
Read `training-atr-models/docs/INFRASTRUCTURE.md` first; this runbook only
adds what this repo needs. **Card 1 is the trainer's (`ATR_TRAIN_GPU=1`).
Sessions use card 0 or the CPU, never card 1.**

## 1. Checkout and venv (once)

```bash
cd ~/Repo && git clone git@github.com:thodel/kraken-weightspace.git
cd kraken-weightspace
export TMPDIR=~/atr-cache/tmp            # never the CIFS share
python3.12 -m venv .venv
.venv/bin/pip install --no-cache-dir -e '.[kraken]'   # pins kraken==7.0.2, torch as the trainer's
.venv/bin/python -c "import kraken, torch; print(kraken.__version__, torch.__version__, torch.cuda.is_available())"
```

Why an own venv and not `.venvs/kraken-train`: the trainer's venv is a service
dependency; this repo must not change it. kraken is pinned to **7.0.2** because
that is what `load_any` on idhefix reads (CoreML only); a model saved by a
newer kraken may not serve.

## 2. Weights (read-only copies)

```bash
S=/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/trained
mkdir -p models/kraken
for id in kraken-bern-hertwig-v1 kraken-bern-hertwig-v2 kraken-bern-schlussredaktion-v1 \
          kraken-bern-combined-v1 kraken-german-xix-v1 kraken-german-xix-blb-v1; do
  cp "$S/$id/$id.mlmodel" models/kraken/ && sha256sum "models/kraken/$id.mlmodel"
done
chmod 444 models/kraken/*.mlmodel
```

Record every digest in `models/DIGESTS.md`; the gateway's `weights_digest`
(`gateway_models`) must match for the four it knows. The parent
`cursive_inzigkofen` comes from Zenodo 10.5281/zenodo.18207767 (the serving
box's `~/.kraken/` cache has it). kraken+ checkpoints: the three runs of
2026-09-01 under the jobs tree (`docs/KRAKEN_PLUS.md` names them).

Adapters (phase 5): peft directories under the jobs tree or `dh-unibe/*` on
HF; copy the directory, never only the safetensors file (`adapter_config.json`
carries rank, alpha and target modules).

## 3. Practice and held-out sets

- Practice (train side only): a few hundred lines per family. Family 1: lines
  from the Hertwig *train* pages of `image-text_bern-disputation-1528`;
  family 2: lines from the BLB *train* projects (`lassberg`, `korrespondenten`,
  …, never `*_eval`); family 3: `~/atr-cache/arrows/german_val.arrow` is the
  *validation* arrow and may serve as practice, `german_test.arrow` never.
  `wsp.data.sample_lines` draws deterministically (seed in the
  preregistration) and writes the line list and its sha256.
- Held-out: see `docs/MODELS.md`; scored outside the sandbox by
  `wsp.heldout`, or through the gateway (`start_batch` on
  `datasets:lassberg-blb-gt/<project>_eval`, then `score_ground_truth`).
- Overlap audit: `wsp.audit overlap --practice <list> --heldout <ids> --trained-on <corpora>`
  before the first pass; the result goes into the preregistration.

## 4. A session

```bash
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m wsp.run new --subject models/kraken/kraken-bern-combined-v1.mlmodel \
    --practice data/practice/bern-hertwig-train-300.jsonl --budget 25
# prints runs/<session>/ and the sandbox command line for the programmer
```

The programmer (a Claude Code session on asteraix, or any agent) works inside
`runs/<session>/` with `python -I` and the `wsp` API only. Network is cut with
`unshare -n` if the account may use user namespaces (`unshare -Un true` says
so); otherwise the offline variables above and the runner's deny-list apply.
`wsp.run freeze <session>` closes the log, writes the diff and the summary;
`wsp.run heldout <session> --set <name>` scores once.

## 5. Handing a candidate to the serving stack

A kept edit becomes a candidate only after its held-out score. Then, by hand
and never from inside a session:

1. `wsp.run export <session> --as kraken-<parent>-wsp-v1` writes the `.mlmodel`
   and a registry YAML next to it (fields as `trained/<id>.yaml` in the
   trainer's registration, with `weights_digest`);
2. copy both to `…/trained/kraken-<parent>-wsp-v1/` on the share;
3. register on idhefix as `training-atr-models/docs/INFRASTRUCTURE.md`
   describes (a deployment act: ask before doing it);
4. read the held-out projects through the gateway and score them with
   `score_ground_truth`, so the number comes from the serving path and not
   only from the harness.

Adapters: `wsp.run export` writes a peft directory; merging into the base for
vLLM is `serving-atr-inference/scripts/merge_loras.py` from the right venv
(`docs/GERMAN_XIX_MODELS.md` there has the peft-version trap).

## 6. Housekeeping

- `runs/*/readings/` (predictions per pass) can be large; git-ignored. Logs,
  diffs and summaries are committed.
- Nothing in this repo writes to the share except step 5.2 above.
- `nvidia-smi` before a GPU pass: card 0 may host the neighbours' RAG workers
  (`gateway_models` → `headroom.theirs`).
