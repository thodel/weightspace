"""TrocrSubject — a ``VisionEncoderDecoderModel`` (TrOCR) directory as a Subject (docs/epics/EPIC_TROCR_TRACK.md, stage 0).

Needs ``torch`` and ``transformers`` (own venv on asteraix, never the kraken venv). Tensors are exposed under canonical
names (``enc.03.attn.q.weight``, ``dec.07.cross.k.weight``, ``dec.out.weight``, ...) as float32 numpy arrays; the model
itself is held and run in float32 so that a pass does not depend on a reduced-precision cast. ``forward`` is generation
with pinned settings (GENERATION, the trainer's eval: beam 4, 256 new tokens, no repeat of a trigram); ``teacher_forced``
is the second readout: one deterministic forward pass over the reference text with per-token loss and margin.
``save`` writes the hub format (``config.json``, ``model.safetensors``, the tokenizer and preprocessor files of the source).
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from wsp.subject import Subject, Prediction

#: pinned generation settings; part of the preregistration of any session (the repeat constraint changes
#: lines that legitimately repeat a trigram). `early_stopping` and `length_penalty` are written out so that a
#: later transformers default cannot change them silently: the hub's generation_config.json says length_penalty 2.0 and
#: use_cache false, the trainer's eval (evaluate_trocr.py, job params) passes length_penalty 1.0 and early_stopping.
#: `use_cache` is a speed setting (no change in the maths); Stage 0 measures whether it moves any output.
GENERATION = {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0,
              "early_stopping": True, "do_sample": False, "use_cache": True}

# --- canonical names -------------------------------------------------------------------------------------------------
# (regex over the Hugging Face state-dict key, canonical template). First match wins; ``{i}`` is the layer index.
_ENC = r"^encoder\.encoder\.layer\.(\d+)\."
_DEC = r"^decoder\.model\.decoder\.layers\.(\d+)\."
_RULES: list[tuple[str, str]] = [
    (_ENC + r"attention\.attention\.query\.(weight|bias)$", "enc.{0}.attn.q.{1}"),
    (_ENC + r"attention\.attention\.key\.(weight|bias)$", "enc.{0}.attn.k.{1}"),
    (_ENC + r"attention\.attention\.value\.(weight|bias)$", "enc.{0}.attn.v.{1}"),
    (_ENC + r"attention\.output\.dense\.(weight|bias)$", "enc.{0}.attn.o.{1}"),
    (_ENC + r"intermediate\.dense\.(weight|bias)$", "enc.{0}.ffn.in.{1}"),
    (_ENC + r"output\.dense\.(weight|bias)$", "enc.{0}.ffn.out.{1}"),
    (_ENC + r"layernorm_before\.(weight|bias)$", "enc.{0}.norm_attn.{1}"),
    (_ENC + r"layernorm_after\.(weight|bias)$", "enc.{0}.norm_ffn.{1}"),
    (r"^encoder\.embeddings\.cls_token$", "enc.embed.cls"),
    (r"^encoder\.embeddings\.position_embeddings$", "enc.embed.pos"),
    (r"^encoder\.embeddings\.patch_embeddings\.projection\.(weight|bias)$", "enc.embed.patch.{0}"),
    (r"^encoder\.layernorm\.(weight|bias)$", "enc.norm_final.{0}"),
    (r"^encoder\.pooler\.dense\.(weight|bias)$", "enc.pooler.{0}"),   # unused by the TrOCR forward; present in some hub files
    (_DEC + r"self_attn\.q_proj\.(weight|bias)$", "dec.{0}.self.q.{1}"),
    (_DEC + r"self_attn\.k_proj\.(weight|bias)$", "dec.{0}.self.k.{1}"),
    (_DEC + r"self_attn\.v_proj\.(weight|bias)$", "dec.{0}.self.v.{1}"),
    (_DEC + r"self_attn\.out_proj\.(weight|bias)$", "dec.{0}.self.o.{1}"),
    (_DEC + r"self_attn_layer_norm\.(weight|bias)$", "dec.{0}.norm_self.{1}"),
    (_DEC + r"encoder_attn\.q_proj\.(weight|bias)$", "dec.{0}.cross.q.{1}"),
    (_DEC + r"encoder_attn\.k_proj\.(weight|bias)$", "dec.{0}.cross.k.{1}"),
    (_DEC + r"encoder_attn\.v_proj\.(weight|bias)$", "dec.{0}.cross.v.{1}"),
    (_DEC + r"encoder_attn\.out_proj\.(weight|bias)$", "dec.{0}.cross.o.{1}"),
    (_DEC + r"encoder_attn_layer_norm\.(weight|bias)$", "dec.{0}.norm_cross.{1}"),
    (_DEC + r"fc1\.(weight|bias)$", "dec.{0}.ffn.in.{1}"),
    (_DEC + r"fc2\.(weight|bias)$", "dec.{0}.ffn.out.{1}"),
    (_DEC + r"final_layer_norm\.(weight|bias)$", "dec.{0}.norm_ffn.{1}"),
    (r"^decoder\.model\.decoder\.embed_tokens\.weight$", "dec.embed.tokens"),
    (r"^decoder\.model\.decoder\.embed_positions\.weight$", "dec.embed.pos"),
    (r"^decoder\.model\.decoder\.layernorm_embedding\.(weight|bias)$", "dec.embed.norm.{0}"),
    (r"^decoder\.output_projection\.weight$", "dec.out.weight"),
    (r"^enc_to_dec_proj\.(weight|bias)$", "bridge.{0}"),
]
_COMPILED = [(re.compile(p), t) for p, t in _RULES]


def canonical_name(hf_name: str) -> str:
    """Hugging Face state-dict key -> canonical name. Layer indices are zero-padded to two digits so that sorted order is
    layer order. A key no rule knows keeps its own name under ``other.`` (never dropped; ``describe`` lists them)."""
    for rx, tpl in _COMPILED:
        m = rx.match(hf_name)
        if m:
            g = list(m.groups())
            if "{1}" in tpl:
                g[0] = f"{int(g[0]):02d}"
            return tpl.format(*g)
    return "other." + hf_name


def component(name: str) -> str:
    """Coarse component of a canonical name, the unit of ``describe`` and of the readings of stage 1."""
    p = name.split(".")
    if p[0] == "enc":
        if p[1] == "embed":
            return "enc.embed"
        if p[1] == "norm_final":
            return "enc.norm"
        if p[1] == "pooler":
            return "enc.pooler"
        kind = p[2]
        return {"attn": "enc.attn", "ffn": "enc.ffn"}.get(kind, "enc.norm")
    if p[0] == "dec":
        if p[1] == "embed":
            return "dec.embed"
        if p[1] == "out":
            return "dec.out"
        kind = p[2]
        return {"self": "dec.self", "cross": "dec.cross", "ffn": "dec.ffn"}.get(kind, "dec.norm")
    return p[0]


UNUSED = ("enc.pooler",)


class TrocrSubject(Subject):
    """``path``: a hub-format TrOCR directory. ``model`` / ``tokenizer`` / ``image_processor`` may be passed instead
    (tests build a tiny random model from a config, no download)."""

    def __init__(self, path: str | Path | None = None, device: str = "cpu", batch: int = 8,
                 generation: dict | None = None, model=None, tokenizer=None, image_processor=None, check_card: bool = True):
        import torch
        from transformers import VisionEncoderDecoderModel

        self._torch = torch
        self.path = Path(path) if path else None
        self.device = device
        self.check_card = check_card         # a pass on a cuda device first asks wsp.card.ready (memory, no trainer job waiting)
        self.batch = batch
        self.generation = {**GENERATION, **(generation or {})}
        if model is None:
            model = VisionEncoderDecoderModel.from_pretrained(str(self.path), torch_dtype=torch.float32)
        self.model = model.to(device=device, dtype=torch.float32).eval()
        if tokenizer is None and self.path is not None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(self.path))
        if image_processor is None and self.path is not None:
            from transformers import AutoImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(str(self.path), use_fast=False)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        sd = self.model.state_dict()
        self._all = {canonical_name(k): k for k in sd}
        if len(self._all) != len(sd):
            raise RuntimeError("two state-dict keys share a canonical name")
        # editable tensors: one name per storage (the hub files of this family hold ONE embedding tensor; the decoder's
        # output projection is tied to it, so ``dec.out.weight`` is an alias of ``dec.embed.tokens``), and not the
        # ViT pooler (not read by the TrOCR forward; absent from microsoft/trocr-base-handwritten, where transformers
        # would initialise it at random on every load and so make the digest depend on the load).
        self._params: dict[str, str] = {}
        self._alias: dict[str, str] = {}
        self.unused = sorted(c for c in self._all if c.startswith(UNUSED))
        seen: dict[int, str] = {}
        for cn in sorted(self._all):
            if cn.startswith(UNUSED):
                continue
            ptr = sd[self._all[cn]].data_ptr()
            if ptr in seen:
                self._alias[cn] = seen[ptr]
            else:
                seen[ptr] = cn
                self._params[cn] = self._all[cn]
        self._cache: dict[str, np.ndarray] = {}

    # -- Subject ------------------------------------------------------------------------------------------------------
    def _state(self) -> dict:
        return self.model.state_dict()

    def describe(self) -> dict:
        """Counted, not assumed: parameters per component from the tensors themselves; a tensor that shares storage
        with another (tied) is counted once. ``parameters`` is every distinct stored parameter (the hub's 333.9 M);
        ``forward_parameters`` leaves out the ViT pooler, which the TrOCR forward does not read."""
        sd = self._state()
        seen: set[int] = set()
        by_comp: dict[str, int] = {}
        total = 0
        shapes = {}
        for cn in sorted(self._all):
            t = sd[self._all[cn]]
            shapes[cn] = list(t.shape)
            ptr = t.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            c = component(cn)
            by_comp[c] = by_comp.get(c, 0) + t.numel()
            total += t.numel()
        unused = sum(by_comp.get(c, 0) for c in UNUSED)
        enc = sum(v for k, v in by_comp.items() if k.startswith("enc.") and not k.startswith(UNUSED))
        dec = sum(v for k, v in by_comp.items() if k.startswith("dec."))
        cfg = self.model.config
        return {"kind": "trocr", "path": str(self.path) if self.path else None, "parameters": total,
                "forward_parameters": total - unused, "unused_by_forward": unused,
                "encoder_parameters": enc, "decoder_parameters": dec,
                "bridge_and_other": total - unused - enc - dec,
                "by_component": dict(sorted(by_comp.items())),
                "tensors": len(self._params), "state_dict_keys": len(self._all), "shapes": shapes,
                "aliases": dict(self._alias), "unused_tensors": list(self.unused),
                "other_tensors": sorted(k for k in self._all if k.startswith("other.")),
                "encoder_layers": int(cfg.encoder.num_hidden_layers), "decoder_layers": int(cfg.decoder.decoder_layers),
                "vocab_size": int(cfg.decoder.vocab_size),
                "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
                "decoder_tie_word_embeddings": bool(getattr(cfg.decoder, "tie_word_embeddings", False)),
                "generation": dict(self.generation)}

    def tensors(self) -> dict[str, np.ndarray]:
        if not self._cache:
            sd = self._state()
            self._cache = {cn: sd[hf].detach().cpu().to(self._torch.float32).numpy().copy()
                           for cn, hf in self._params.items()}
        return self._cache

    def set_tensor(self, name: str, value: np.ndarray) -> None:
        name = self._alias.get(name, name)
        if name not in self._params:
            raise KeyError(name)
        value = np.asarray(value, dtype=np.float32)
        param = self._state()[self._params[name]]
        if tuple(param.shape) != value.shape:
            raise ValueError(f"{name}: shape {tuple(param.shape)} -> {value.shape}; a TrOCR tensor keeps its shape")
        with self._torch.no_grad():
            param.copy_(self._torch.from_numpy(np.ascontiguousarray(value)).to(param.device, param.dtype))
        if self._cache:
            self._cache[name] = value.copy()

    # -- readouts -------------------------------------------------------------------------------------------------------
    def _pixels(self, items: Sequence[dict]):
        from PIL import Image

        imgs = []
        for it in items:
            im = it["image"]
            im = Image.open(im) if isinstance(im, (str, Path)) else im
            imgs.append(im.convert("RGB"))
        # input_data_format is explicit: a crop that is 3 px high has the shape (3, W, 3), which transformers cannot tell from
        # channels-first and, without this, decides from the FIRST image of the batch for all images of the batch.
        px = self.image_processor(images=imgs, return_tensors="pt", input_data_format="channels_last").pixel_values
        return px.to(self.device, self._torch.float32)

    def _batches(self, items: Sequence[dict]):
        for i in range(0, len(items), self.batch):
            yield items[i:i + self.batch]

    def forward(self, practice: Sequence[dict]) -> list[Prediction]:
        """ONE generation pass over ``practice`` (items need ``key`` and ``image``: a path or a PIL image), in the order
        given and in fixed batches of ``self.batch`` (batch composition can move the last float bits; keep it fixed
        between passes that are compared). ``extra``: tokens generated, ``at_cap`` (no end token within the cap), the
        beam's sequence score."""
        torch = self._torch
        if self.check_card and str(self.device).startswith("cuda") and os.environ.get("WSP_CARD_GATE") != "outside":
            from wsp import card
            ok, why = card.ready()
            if not ok:
                raise RuntimeError(f"no pass: {why}")        # refused before the pass: a session does not count it
        g = self.generation
        eos = self.model.generation_config.eos_token_id
        eos = eos if eos is not None else self.model.config.decoder.eos_token_id
        eos = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        out: list[Prediction] = []
        with torch.inference_mode():
            for chunk in self._batches(practice):
                res = self.model.generate(self._pixels(chunk), return_dict_in_generate=True, output_scores=True, **g)
                seqs = res.sequences.cpu()
                texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
                scores = res.sequences_scores.cpu().tolist() if getattr(res, "sequences_scores", None) is not None else [None] * len(chunk)
                for it, t, s, row in zip(chunk, texts, scores, seqs):
                    ids = row.tolist()
                    n = len(ids) - 1                                   # without the decoder start token
                    has_eos = any(i in eos for i in ids[1:])
                    out.append(Prediction(key=it["key"], text=t.strip() if isinstance(t, str) else t,
                                          extra={"tokens": n, "at_cap": (not has_eos) and n >= g["max_new_tokens"],
                                                 "score": s}))
        return out

    def _label_ids(self, text: str) -> list[int]:
        return list(self.tokenizer(text).input_ids)       # <s> text </s>

    def teacher_forced(self, practice: Sequence[dict]) -> list[dict]:
        """The second readout: reference text forced through the decoder, no search, no loops, deterministic.
        Items need ``key``, ``image``, ``text``. Per line: ``nll`` (per label token, nats), ``margin`` (logit of the
        reference token minus the best other logit; negative = the model would have picked something else),
        ``tokens``, and the line's mean loss. Labels are ``<s> text </s>``; the decoder input is the labels shifted
        right behind the decoder start token."""
        torch = self._torch
        cfg = self.model.config
        start = cfg.decoder_start_token_id if cfg.decoder_start_token_id is not None else cfg.decoder.decoder_start_token_id
        pad = cfg.pad_token_id if cfg.pad_token_id is not None else cfg.decoder.pad_token_id
        res: list[dict] = []
        with torch.inference_mode():
            for chunk in self._batches(practice):
                labs = [self._label_ids(it["text"]) for it in chunk]
                n = max(len(l) for l in labs)
                lab = torch.full((len(labs), n), -100, dtype=torch.long)
                dec_in = torch.full((len(labs), n), pad, dtype=torch.long)
                for i, l in enumerate(labs):
                    lab[i, :len(l)] = torch.tensor(l)
                    dec_in[i, 0] = start
                    dec_in[i, 1:len(l)] = torch.tensor(l[:-1])
                att = (lab != -100).long()
                logits = self.model(pixel_values=self._pixels(chunk), decoder_input_ids=dec_in.to(self.device),
                                    decoder_attention_mask=att.to(self.device)).logits.float().cpu()
                logp = torch.log_softmax(logits, dim=-1)
                for i, it in enumerate(chunk):
                    k = len(labs[i])
                    tgt = lab[i, :k]
                    lp = logp[i, :k]
                    nll = -lp.gather(1, tgt[:, None])[:, 0]
                    lg = logits[i, :k]
                    ref_logit = lg.gather(1, tgt[:, None])[:, 0]
                    other = lg.clone()
                    other.scatter_(1, tgt[:, None], float("-inf"))
                    margin = ref_logit - other.max(dim=1).values
                    res.append({"key": it["key"], "tokens": k, "nll": nll.tolist(), "margin": margin.tolist(),
                                "loss": float(nll.mean())})
        return res

    def logits(self, item: dict) -> np.ndarray:
        """Teacher-forced logits for one line, ``(tokens, vocab)`` float32 (labels as in ``teacher_forced``)."""
        torch = self._torch
        cfg = self.model.config
        start = cfg.decoder_start_token_id if cfg.decoder_start_token_id is not None else cfg.decoder.decoder_start_token_id
        lab = self._label_ids(item["text"])
        dec_in = torch.tensor([[start] + lab[:-1]], device=self.device)
        with torch.inference_mode():
            return self.model(pixel_values=self._pixels([item]), decoder_input_ids=dec_in).logits[0].float().cpu().numpy()

    # -- save -------------------------------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Hub format: ``config.json``, ``generation_config.json``, ``model.safetensors`` (written by transformers from
        the held tensors), plus every tokenizer and preprocessor file of the source directory, unchanged."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").unlink(missing_ok=True)
        self.model.save_pretrained(str(path), safe_serialization=True)
        if self.path is not None and self.path.resolve() != path.resolve():
            for f in self.path.iterdir():
                if f.is_file() and f.name not in {"model.safetensors", "config.json", "generation_config.json"} \
                        and not f.name.startswith("training_args"):
                    dst = path / f.name
                    dst.unlink(missing_ok=True)
                    shutil.copyfile(f, dst)
        elif self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(path))
            if self.image_processor is not None:
                self.image_processor.save_pretrained(str(path))
        return path
