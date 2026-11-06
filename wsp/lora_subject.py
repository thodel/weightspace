"""LoraSubject — a peft adapter directory as a Subject (phase 5, skeleton).

Tensors are the adapter's ``lora_A`` / ``lora_B`` weights (safetensors), named
as peft names them. ``forward`` is pluggable: a callable that takes the adapter
directory and the practice items and returns texts — on asteraix a vLLM batch
with the adapter merged or served with ``--enable-lora``. Saving writes the
adapter directory back (config copied unchanged), which
``serving-atr-inference/scripts/merge_loras.py`` can bake into the base.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from wsp.subject import Subject, Prediction


class LoraSubject(Subject):
    def __init__(self, adapter_dir: str | Path, forward: Callable[[Path, Sequence[dict]], list[str]] | None = None):
        from safetensors.numpy import load_file  # lazy

        self.dir = Path(adapter_dir)
        self.config = json.loads((self.dir / "adapter_config.json").read_text())
        self._state = {k: v.astype(np.float32) for k, v in load_file(str(self.dir / "adapter_model.safetensors")).items()}
        self._dtypes = {k: v.dtype for k, v in load_file(str(self.dir / "adapter_model.safetensors")).items()}
        self._forward = forward
        self._work: Path | None = None

    def describe(self) -> dict:
        mods = sorted({k.rsplit(".lora_", 1)[0] for k in self._state})
        return {"kind": "lora", "dir": str(self.dir), "base_model": self.config.get("base_model_name_or_path"),
                "r": self.config.get("r"), "lora_alpha": self.config.get("lora_alpha"),
                "target_modules": self.config.get("target_modules"), "modules": len(mods),
                "parameters": int(sum(np.prod(v.shape) for v in self._state.values()))}

    def tensors(self) -> dict[str, np.ndarray]:
        return self._state

    def set_tensor(self, name: str, value: np.ndarray) -> None:
        if name not in self._state:
            raise KeyError(name)
        if self._state[name].shape != value.shape:
            raise ValueError("adapter tensors keep their shape (rank is fixed by the config)")
        self._state[name] = np.asarray(value, dtype=np.float32)

    def delta(self, module: str) -> np.ndarray:
        """ΔW = B·A for one module — the low-rank update the reading starts from."""
        a = self._state[f"{module}.lora_A.weight"]
        b = self._state[f"{module}.lora_B.weight"]
        return b @ a

    def directions(self, module: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD of ΔW: at most r directions, with their singular values."""
        return np.linalg.svd(self.delta(module), full_matrices=False)

    def forward(self, practice: Sequence[dict]) -> list[Prediction]:
        if self._forward is None:
            raise RuntimeError("LoraSubject needs a forward callable (vLLM runner on asteraix)")
        work = self.save(Path(self._work or (self.dir.parent / (self.dir.name + ".wsp-work"))))
        texts = self._forward(work, practice)
        return [Prediction(key=p["key"], text=t) for p, t in zip(practice, texts)]

    def save(self, path: str | Path) -> Path:
        from safetensors.numpy import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for f in self.dir.iterdir():
            if f.name != "adapter_model.safetensors" and f.is_file():
                shutil.copy2(f, path / f.name)
        save_file({k: v.astype(self._dtypes[k]) for k, v in self._state.items()}, str(path / "adapter_model.safetensors"))
        self._work = path
        return path
