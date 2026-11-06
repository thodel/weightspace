"""The Subject interface: one way to read, edit, run and save any model kind."""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from wsp import ops


@dataclass
class Prediction:
    key: str          # line or page id from the practice set
    text: str
    extra: dict | None = None   # e.g. blank share, finish_reason, tokens


class Subject(ABC):
    """A model artefact the programmer can read, edit, run once per pass, and save."""

    @abstractmethod
    def describe(self) -> dict:
        """Architecture, parameter count, codec/vocab, lineage — no weights."""

    @abstractmethod
    def tensors(self) -> dict[str, np.ndarray]:
        """Editable views as numpy arrays (float32), keyed by canonical name."""

    @abstractmethod
    def set_tensor(self, name: str, value: np.ndarray) -> None:
        """Write a tensor back (shape may change only for ops that say so)."""

    @abstractmethod
    def forward(self, practice: Sequence[dict]) -> list[Prediction]:
        """ONE forward pass over the practice items. The Session counts it."""

    @abstractmethod
    def save(self, path: Path) -> Path:
        """Write the artefact in the format the serving stack reads."""

    def apply(self, op_name: str, names: Sequence[str], **kwargs) -> None:
        """Apply one registered op to each named tensor."""
        for n in names:
            self.set_tensor(n, ops.apply(op_name, self.tensors()[n], **kwargs))

    def digest(self) -> str:
        """sha256 over tensor names, shapes and bytes, in name order."""
        h = hashlib.sha256()
        for name, arr in sorted(self.tensors().items()):
            h.update(name.encode()); h.update(str(arr.shape).encode()); h.update(str(arr.dtype).encode())
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()


class ArraySubject(Subject):
    """A subject over plain arrays with a pluggable forward — for tests and for
    reading a diff without an engine."""

    def __init__(self, tensors: dict[str, np.ndarray], forward=None, description: dict | None = None):
        self._t = {k: np.array(v, dtype=np.float32) for k, v in tensors.items()}
        self._forward = forward
        self._desc = description or {"kind": "array", "tensors": {k: list(v.shape) for k, v in self._t.items()}}

    def describe(self) -> dict:
        return dict(self._desc)

    def tensors(self) -> dict[str, np.ndarray]:
        return self._t

    def set_tensor(self, name: str, value: np.ndarray) -> None:
        self._t[name] = np.asarray(value, dtype=np.float32)

    def forward(self, practice):
        if self._forward is None:
            raise RuntimeError("ArraySubject has no forward")
        return [Prediction(key=item["key"], text=t) for item, t in zip(practice, self._forward(self._t, practice))]

    def save(self, path: Path) -> Path:
        path = Path(path)
        np.savez(path, **self._t)
        return path
