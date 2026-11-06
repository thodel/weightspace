# Harness design

The harness is what makes a weight-space programming session repeatable: the
same loader, the same budget rule, the same log format, whether the subject is
a 4 M-parameter kraken net or a rank-16 adapter on a 7 B VLM. Code in `wsp/`.

## 1. Subject — one interface for every model kind

```
Subject
  describe()  -> dict        architecture, parameter count, codec/vocab, lineage
  tensors()   -> dict[name, ndarray-like]   editable views of the weights
  apply(edit) -> None        one EditOp on named tensors; logged by the Session
  forward(practice) -> list[Prediction]      ONE forward pass; counted
  save(path)  -> Path        the format the serving stack reads
  digest()    -> sha256 of the tensors as saved
```

| subject | weights | forward | save format | where |
|---|---|---|---|---|
| `KrakenSubject` | `TorchVGSLModel.nn` state dict (conv, 3 × BiLSTM, output), codec | `rpred` / `TorchSeqRecognizer` over line crops, CTC greedy decode | CoreML `.mlmodel` (kraken 7.0.2 `load_any` reads nothing else) | asteraix, CPU or card 0 |
| `LoraSubject` | peft `adapter_model.safetensors`: `lora_A`/`lora_B` per target module, plus `adapter_config.json` | vLLM (adapter merged on the fly or served with `--enable-lora`) or transformers+peft, pinned sampling | peft adapter directory, then `scripts/merge_loras.py` for serving | asteraix card 0 |
| `FullWeightsSubject` | safetensors shards | same | safetensors | later, if ever |

The programmer never touches a file format; it reads `tensors()` and applies
`EditOp`s. A subject is responsible for naming: kraken tensors are named as the
state dict names them (`nn.L_3.layer.weight_ih_l0` …) with a `describe()` map to
VGSL blocks; adapter tensors keep peft's names
(`base_model.model.model.layers.17.mlp.down_proj.lora_B.weight`).

## 2. EditOp — pure functions, logged with their arguments

`wsp/ops.py` holds the operations; each is a pure function on arrays, so an
edit is fully described by `(op, tensor names, args)` and can be replayed.

| op | arguments | meaning |
|---|---|---|
| `scale` | `factor` | `W ← factor·W` |
| `add` | `delta` (array or name of another tensor) | `W ← W + delta` |
| `set_rows` / `set_cols` | `index`, `values` | overwrite rows/columns (codec rows, `B` rows) |
| `project_out` | `direction`, `axis`, optional `index` | remove the component along a unit direction from the selected columns — the Exp-8 edit of Takehi et al. |
| `interpolate` | `other`, `alpha` | `W ← (1−α)W + α·other` |
| `transplant` | `donor tensor` | copy a whole tensor from a sibling |
| `drop_padding_rows` | `keep` | for the kraken+ `Cr255,1,85` kernel: keep the centre row, drop the 254 rows that only ever saw padding |

Donor tensors come from a second `Subject` opened read-only. Codec alignment
for kraken (rows matched by character, not index) and module-name alignment
for adapters live in `wsp/align.py`, so merges never silently pair the wrong
rows.

## 3. Session — the budget and the log

```
Session(subject, practice_set, budget=25, seed=…)
  hypothesis(text, prediction)       -> id          written before any pass
  edit(op, names, **args)            -> edit id     applied, digest recorded
  measure(label)                     -> Metrics     ONE pass; refuses past the budget
  keep() / revert()                  decision on the last edit, with the measured delta
  freeze()                           -> final diff, final digest, log closed
```

`runs/<session>/log.jsonl`, one JSON object per event:
`{"t": …, "kind": "hypothesis|edit|pass|decision|note", "budget_used": n, …}`.
A pass records the full metrics; a decision records `kept: bool` and the
metric delta it was based on. `freeze()` writes `diff.safetensors` (sparse:
names, indices, values against the starting digest, like the published
Exp-8 diffs) and `summary.json`. The budget counter is in-process **and** in a
file in the run folder, so a restarted sandbox cannot forget it.

## 4. Sandbox

The programmer's Python runs as `python -I` (no site customisation, no
current-dir imports) in a per-session working directory, with network
disabled where the host allows (`unshare -n` on asteraix if the user may; else
`HF_HUB_OFFLINE=1`, `no_proxy=*` and a deny-list on the runner), a time limit,
and only two paths visible: the run folder and the read-only `models/` and
`data/practice/` trees. Held-out data is **not mounted** in the sandbox;
`wsp.heldout` scores a frozen artefact from outside, once, and writes the
number into `summary.json` with the held-out set's identity hash.

GPU: the subject chooses the device; on asteraix `CUDA_VISIBLE_DEVICES=0` is
set by the runner and never card 1.

## 5. Metrics (`wsp/metrics.py`)

Pure Python, no kraken dependency, so the same code scores kraken lines, VLM
lines and VLM pages:

- `cer`, `wer`: corpus-level (Σ edits / Σ reference characters), NFD, after
  the same whitespace flattening the trainer applies;
- `collapse(hyp, ref)`: a line is collapsed when its length ratio is outside
  [0.5, 2] or a character n-gram repeats more than `k` times — the two shapes
  the pipeline saw (#PIPELINE_CONSEQUENCES: kraken collapses bounded by image
  width, VLM loops unbounded);
- `confusions(hyps, refs)`: aligned substitutions, top-N;
- `blank_share` for CTC subjects (from the logits, supplied by the subject);
- `bootstrap_delta(a, b)`: paired per-line bootstrap of the CER difference,
  95 % interval, 2,000 resamples, fixed seed.

## 6. Scaling notes

- **Context:** no weights are ever pasted into a prompt. The programmer sees
  `describe()`, summary statistics (`wsp.stats`: norms, spectra, row/column
  energies, top singular directions) and whatever it computes itself in the
  sandbox. This is the same for 4 M and 7 B parameters.
- **Pass cost:** kraken, 300 lines on CPU ≈ seconds; Qwen3.5-4B, 64 pages on
  an A40 ≈ minutes. The budget is in passes; the practice set size is fixed per
  session and logged, so a pass means the same thing in every log.
- **Adapters are already low-rank:** `ΔW = B·A` with rank 16 gives at most
  16 directions per module; `wsp.stats.adapter_directions` lists them with
  their singular values, which is the natural starting point for a reading.
- **Merging into the base for serving** is `scripts/merge_loras.py` in
  serving-atr-inference; the harness never serves, it hands over a file and a
  digest.
