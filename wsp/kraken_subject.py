"""KrakenSubject — a CoreML ``.mlmodel`` recogniser as a Subject.

Runs only where kraken 7.0.2 is installed (asteraix). The API names are those of kraken 7.0.2 as verified on asteraix on
06.11.2026 (docs/KRAKEN_API_NOTES.md); ``tests/asteraix/test_roundtrip.py`` is
the no-op round-trip that re-verifies them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from wsp.subject import Subject, Prediction


class KrakenSubject(Subject):
    def __init__(self, path: str | Path, device: str = "cpu", batch: int = 32):
        from kraken.lib.vgsl import TorchVGSLModel  # lazy: engine only on asteraix
        import torch

        self.path = Path(path)
        self.device = device
        self.batch = batch
        self.model = TorchVGSLModel.load_model(str(self.path))
        self.model.nn.to(device).eval()
        self._torch = torch
        self._recognizer = None
        # one-time view of the state dict as numpy; writes go back through set_tensor
        self._state = {k: v.detach().cpu().numpy().astype(np.float32)
                       for k, v in self.model.nn.state_dict().items()}

    # -- Subject ----------------------------------------------------------
    def describe(self) -> dict:
        codec = self.model.codec
        n_params = sum(int(np.prod(v.shape)) for v in self._state.values())
        return {
            "kind": "kraken", "path": str(self.path), "spec": self.model.spec,
            "codec_size": len(codec.c2l),            # characters; labels start at 1
            "output_classes": int(codec.max_label) + 1,  # + the CTC blank at 0
            "blank_index": 0,
            "input_height": int(self.model.input[2]),
            "one_channel_mode": self.model.one_channel_mode,
            "model_type": self.model.model_type,
            "layers": [{"name": k, "shape": list(v.shape), "numel": int(np.prod(v.shape))}
                       for k, v in self._state.items()],
            "parameters": n_params,
            "user_metadata": dict(getattr(self.model, "user_metadata", {}) or {}),
        }

    def codec_map(self) -> dict[str, int]:
        """character -> output row (single-label codes only; multi-label chars are listed under '_multi')."""
        out: dict[str, int] = {}
        for ch, labels in self.model.codec.c2l.items():
            if len(labels) == 1:
                out[ch] = int(labels[0])
        return out

    def tensors(self) -> dict[str, np.ndarray]:
        return self._state

    def set_tensor(self, name: str, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32)
        self._state[name] = value
        with self._torch.no_grad():
            param = dict(self.model.nn.named_parameters()).get(name) or dict(self.model.nn.named_buffers()).get(name)
            if param is None:
                raise KeyError(name)
            if tuple(param.shape) != value.shape:
                # a shape change (drop_padding_rows) needs the layer rebuilt; refuse here, do it in rewrite.py
                raise ValueError(f"shape change {tuple(param.shape)} -> {value.shape} needs wsp.rewrite")
            param.copy_(self._torch.from_numpy(value).to(param.device, param.dtype))
        self._recognizer = None

    def forward(self, practice: Sequence[dict]) -> list[Prediction]:
        """practice items: {"key", "text", "image": path to a line crop}.

        Transforms as the trainer and the baseline path of ``rpred`` use them:
        ``ImageInputTransforms(batch, height, width, channels, (16, 0), valid_norm=False)``
        (kraken 7.0.2: ``kraken/train/vgsl.py`` l.142, ``kraken/rpred.py`` l.132)."""
        from kraken.lib.models import TorchSeqRecognizer
        from kraken.lib.dataset import ImageInputTransforms
        from PIL import Image

        if self._recognizer is None:
            self._recognizer = TorchSeqRecognizer(self.model, device=self.device)
        batch, channels, height, width = self.model.input
        tf = ImageInputTransforms(batch, height, width, channels, (16, 0), valid_norm=False)
        preds: list[Prediction] = []
        for item in practice:
            im = Image.open(item["image"]).convert("L")
            line = tf(im).unsqueeze(0).to(self.device)
            lens = self._torch.tensor([line.shape[3]], dtype=self._torch.int64)
            with self._torch.no_grad():
                out = self._recognizer.predict_string(line, lens)
            text = out[0] if isinstance(out, (list, tuple)) else str(out)
            preds.append(Prediction(key=item["key"], text=text))
        return preds

    def logits(self, practice: Sequence[dict]) -> list[np.ndarray]:
        """Softmaxed CTC output per line (T x classes; class 0 is the blank), for blank
        share and direction reading. ``TorchSeqRecognizer.forward`` returns
        ``(probs, lengths)`` with probs already through softmax, not raw logits. Not a counted pass."""
        from kraken.lib.models import TorchSeqRecognizer
        from kraken.lib.dataset import ImageInputTransforms
        from PIL import Image

        rec = TorchSeqRecognizer(self.model, device=self.device)
        batch, channels, height, width = self.model.input
        tf = ImageInputTransforms(batch, height, width, channels, (16, 0), valid_norm=False)
        out = []
        for item in practice:
            im = Image.open(item["image"]).convert("L")
            line = tf(im).unsqueeze(0).to(self.device)
            lens = self._torch.tensor([line.shape[3]], dtype=self._torch.int64)
            with self._torch.no_grad():
                o = rec.forward(line, lens)
            probs = o[0] if isinstance(o, (list, tuple)) else o
            out.append(np.asarray(probs)[0])      # (W, C) for the single line
        return out

    def save(self, path: str | Path) -> Path:
        """Write CoreML. kraken 7.0.2's ``save_model`` takes ``model_type[0]`` of the
        metadata; a model loaded from CoreML carries the *string* ``'recognition'``,
        so a naive save writes ``'r'`` and ``load_any`` (the serving loader) refuses
        the file. Storing the list form first makes the saved metadata equal to the
        original's (docs/KRAKEN_API_NOTES.md §5)."""
        path = Path(path)
        if path.suffix != ".mlmodel":
            raise ValueError("the serving loader reads CoreML only; save as .mlmodel")
        mt = self.model.user_metadata.get("model_type")
        if isinstance(mt, str):
            self.model.user_metadata["model_type"] = [mt]
        if "legacy_polygons" not in self.model.user_metadata:
            self.model.user_metadata["legacy_polygons"] = bool(self.model.use_legacy_polygons)
        self.model.save_model(str(path))
        return path
