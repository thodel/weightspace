# kraken 7.0.2 API, as verified on asteraix (06.11.2026)

Environment: `.venv` of this repo on asteraix, `kraken==7.0.2`,
`torch==2.10.0+cu128`, `coremltools==9.0`, python 3.12 (`docs/ENV_ASTERAIX.txt`).
Everything below was checked by introspection and by running it against
`models/kraken/kraken-german-xix-blb-v1.mlmodel`; `wsp/kraken_subject.py` was
patched where it mattered (§5). kraken 7 has no `kraken.__version__`; use
`importlib.metadata.version("kraken")`.

## 1. Loading

`kraken.lib.vgsl.TorchVGSLModel.load_model(path)` → `TorchVGSLModel` with

| attribute | what we saw |
|---|---|
| `.nn` | `torch.nn.Module`; `state_dict()` has 34 tensors for the production spec, all `float32`, keys like `C_0.co.weight`, `L_12.layer.weight_ih_l0`, `O_18.lin.weight` (block names from the spec's `{…}` labels, not `nn.L_3.layer…` as DESIGN.md guessed) |
| `.spec` | the VGSL string with block labels, e.g. `[1,120,0,1 Cr{C_0}3,13,32 … Lbx{L_16}200 Do{Do_17} O{O_18}1c297]` |
| `.input` | `(batch, channels, height, width)` = `(1, 1, 120, 0)` for the in-house models, `(256, 1, 64, 0)` for kraken+ |
| `.codec` | `kraken.lib.codec.PytorchCodec`; `.c2l: dict[str, list[int]]` character → labels (labels start at **1**; 0 is the CTC blank), `.l2c` the inverse, `.max_label` |
| `.user_metadata` | dict: `accuracy`, `metrics`, `seg_type`, `one_channel_mode`, `model_type`, `codec`, `legacy_polygons`, `vgsl` (+ `hyper_params` on the Zenodo parent) |
| `.model_type`, `.one_channel_mode`, `.seg_type`, `.use_legacy_polygons` | properties over `user_metadata`: `'recognition'`, `'L'`, `'baselines'`, `False` for all seven in-house models |
| `.save_model(path)` | CoreML writer, see §5 |

`named_parameters()` keys equal the `state_dict()` keys (no buffers), so
`KrakenSubject.set_tensor` writes back by name as written.

Parameter counts: production spec 4,111,961; kraken+ `Cr1,1,85` run
4,148,545; kraken+ `Cr255,1,85` (collapse probe) 15,207,871, of which
`C_12.co.weight` is `(85, 512, 255, 1)` = 11,097,600.

## 2. Codec

`codec.c2l` maps every character to a list of labels; for all in-house models
every list has length 1, so `KrakenSubject.codec_map()` (character → output
row) is total. The output layer `O{…}1c297` has `max_label + 1` rows: row 0 is
the blank. `codec.encode(text)` / `codec.decode(labels)` exist for the
confusion pairing later.

## 3. Recognition

`kraken.lib.models.TorchSeqRecognizer(nn, decoder=greedy_decoder, temperature=1.0, train=False, device='cpu')`.

- `predict_string(line: Tensor[N,C,H,W], lens: Tensor | None) -> list[str]`
- `forward(line, lens) -> (probs: ndarray[N, W, C], lengths: ndarray)` — the
  array is already **softmaxed** (`(logits / temperature).softmax(1)`), not raw
  logits; blank share is `probs[..., 0]`.
- `load_any(path)` is what the serving box calls; it wraps the same class and
  checks `'recognition' in nn.model_type` (§5).

## 4. Line input

`kraken.lib.dataset.ImageInputTransforms(batch, height, width, channels, pad, valid_norm=True, force_binarization=False, dtype=float32)`.

The trainer (`kraken/train/vgsl.py` l.142) builds it as
`ImageInputTransforms(1, height, width, channels, (padding, 0), valid_norm=False)`
with the ketos default `padding=16`; the baseline path of `rpred`
(`kraken/rpred.py` l.132) uses `(pad, 0)` with `pad=16` and `valid_norm=False`
as well. `KrakenSubject` uses exactly `(16, 0), valid_norm=False`.

Lines must be the already-cut line crops: `rpred` cuts them from the page with
`kraken.lib.segmentation.extract_polygons(im, Segmentation(...))` (baseline +
boundary polygon, dewarped onto the baseline), and `ketos compile` stores those
crops in the arrow's `lines.im`. `wsp/data.py` cuts PAGE XML lines with the same
`extract_polygons` call (`legacy=False`, matching `legacy_polygons: false` in the
models) and takes arrow rows as they are, so a practice line is what the serving
path and the trainer see. The transform does the resize to the model's height,
the padding and the tensor conversion; no further centring is needed.

Arrow files (`ketos compile`): a `lines: struct<text, im: binary>` column plus
`train/validation/test` booleans; `ArrowIPCRecognitionDataset.add(file)` and
`__getitem__` → `{"image": tensor, "target": …}`. No document ids are stored.

## 5. Saving — the one real difference

`save_model(path)` **re-serialises** (coremltools builder); it does not copy the
input bytes. On the models here the weights, the spec and the codec survive
exactly (every tensor `array_equal`, `spec` and `codec.c2l` equal), but

1. the file is **not byte-identical** (`kraken_meta` JSON key order differs,
   same size for the fixed path; 10 bytes shorter on the naive path), and
2. **kraken 7.0.2 corrupts `model_type`** on a model loaded from CoreML:
   `save_model` does `coreml_metadata['model_type'][0]` to turn the kraken-7
   list form into the legacy string, but a CoreML-loaded model carries the
   string `'recognition'`, so the saved file says `'r'`. `load_any` then fails:
   `ValueError: Models of type r are not supported by TorchSeqRecognizer`.
   A naively re-saved model would **not serve**.

Fix in `KrakenSubject.save`: set `user_metadata['model_type'] = ['recognition']`
(list) before `save_model`; the written metadata is then equal to the
original's and `load_any` loads the copy. Verified for `xix-blb-v1`,
`hertwig-v2` and the Zenodo parent. The tensor digest
(`Subject.digest()`, over names, shapes and bytes) is identical before and
after; the file sha256 is not, and `models/DIGESTS.md` records file digests,
so a re-saved copy gets a new file digest with the same tensor digest. Both are
written to a run's `summary.json`.

## 6. Building a model from a spec (for the `Cr255 → Cr1` rewrite)

`TorchVGSLModel(vgsl=spec, codec=c2l_dict, **user_metadata)` builds the net;
`.add_codec(codec)`, `.init_weights()`, `.load_state_dict` on `.nn` are
available. `load_model` itself does
`cls(vgsl=spec, codec=codec, **coreml_metadata)` and then deserialises layer by
layer, so the same constructor call with the rewritten spec and a copied,
reshaped state dict is the supported route (`wsp/rewrite.py`).

## 7. Odds and ends

- `pip install huggingface_hub` pulled `click 8.5`, which breaks kraken's
  `click<8.3` pin; `huggingface_hub<1.0` (0.36.2) is what the venv now has, and
  `pip check` is clean. It is only used by `wsp/data.py` to fetch the Hertwig
  page shard once.
- `unshare -Un true` works for this account (user namespaces allowed), so the
  session sandbox can cut the network with `unshare -n`.
- Card 0 carried 42.6 GB of neighbours' memory on 06.11.2026; every phase 0
  pass ran on CPU (`CUDA_VISIBLE_DEVICES=""`), seconds per 300 lines.
