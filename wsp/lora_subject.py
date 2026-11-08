"""LoraSubject — a peft adapter directory as a Subject (phase 5).

Tensors are the adapter's ``lora_A`` / ``lora_B`` weights (safetensors), named
as peft names them. ``forward`` is pluggable: a callable that takes the adapter
directory and the practice items and returns texts — on asteraix a vLLM batch
with the adapter merged or served with ``--enable-lora``. Saving writes the
adapter directory back (config copied unchanged), which
``serving-atr-inference/scripts/merge_loras.py`` can bake into the base.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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

    @property
    def scale(self) -> float:
        """peft's ``lora_alpha / r`` (no rsLoRA here; refused in ``describe`` if set)."""
        return float(self.config["lora_alpha"]) / float(self.config["r"])

    def modules(self) -> list[str]:
        return sorted({k.rsplit(".lora_", 1)[0] for k in self._state})

    def delta(self, module: str) -> np.ndarray:
        """ΔW = (α/r)·B·A for one module — the update the base actually receives."""
        a = self._state[f"{module}.lora_A.weight"]
        b = self._state[f"{module}.lora_B.weight"]
        return self.scale * (b @ a)

    def directions(self, module: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD of ΔW without forming it: QR of B and Aᵀ, then the r×r core.
        Returns (U: out×r, S: r, Vt: r×in), U the directions ΔW writes."""
        a = self._state[f"{module}.lora_A.weight"].astype(np.float64)
        b = self._state[f"{module}.lora_B.weight"].astype(np.float64)
        qb, rb = np.linalg.qr(b)
        qa, ra = np.linalg.qr(a.T)
        u, s, vt = np.linalg.svd(self.scale * rb @ ra.T)
        return (qb @ u).astype(np.float32), s.astype(np.float32), (vt @ qa.T).astype(np.float32)

    def scale_module(self, module: str, alpha: float) -> None:
        """Multiply one module's update by α (written into B, so the saved adapter
        is an ordinary peft directory with the config unchanged)."""
        k = f"{module}.lora_B.weight"
        self.set_tensor(k, self._state[k] * np.float32(alpha))

    def remove_direction(self, module: str, u: np.ndarray) -> None:
        """Project a unit output direction u out of ΔW: B ← (I − uuᵀ)B. Rank stays r."""
        k = f"{module}.lora_B.weight"
        b = self._state[k]
        u = np.asarray(u, dtype=np.float32); u = u / np.linalg.norm(u)
        self.set_tensor(k, b - np.outer(u, u @ b))

    def forward(self, practice: Sequence[dict]) -> list[Prediction]:
        if self._forward is None:
            raise RuntimeError("LoraSubject needs a forward callable (vllm_runner on asteraix)")
        work = self.save(Path(self._work or (self.dir.parent / (self.dir.name + ".wsp-work"))))
        outs = self._forward(work, practice)
        return [Prediction(key=p["key"], text=o["text"] if isinstance(o, dict) else o,
                           extra=o if isinstance(o, dict) else None) for p, o in zip(practice, outs)]

    def save(self, path: str | Path) -> Path:
        from safetensors.numpy import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for f in self.dir.iterdir():
            if f.name != "adapter_model.safetensors" and f.is_file():
                shutil.copy2(f, path / f.name)
                os.chmod(path / f.name, 0o644)  # a read-only source (the cache copy of a trained adapter) must not make the edit read-only
        save_file({k: v.astype(self._dtypes[k]) for k, v in self._state.items()}, str(path / "adapter_model.safetensors"))
        self._work = path
        return path


def vllm_runner(render: str | Path, out_dir: str | Path,
                python: str | Path = Path.home() / ".venvs/vllm/bin/python") -> Callable:
    """A ``forward`` for LoraSubject: one ``wsp.vllm_forward`` process per pass on
    card 0, after ``wsp.card.ready`` (memory, and no trainer job waiting, #396).
    Practice items need only ``key``; the render directory holds images and prompt."""
    from wsp import card

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    count = {"n": 0}

    def run(adapter: Path, practice: Sequence[dict]) -> list[dict]:
        ok, why = card.ready()
        if not ok:
            raise RuntimeError(f"pass not started: {why}")
        count["n"] += 1
        keys = out_dir / f"pass-{count['n']:03d}.keys"
        keys.write_text("".join(p["key"] + "\n" for p in practice))
        out = out_dir / f"pass-{count['n']:03d}.jsonl"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1",
               "PATH": f"{Path(python).parent}:{os.environ.get('PATH', '')}",
               "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
        with open(out.with_suffix(".log"), "w") as log:
            subprocess.run([str(python), "-m", "wsp.vllm_forward", "--render", str(render),
                            "--adapter", str(adapter), "--keys", str(keys), "--out", str(out)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        by = {}
        for line in out.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            by[d["key"]] = {"text": d["text"], "at_cap": d["finish_reason"] == "length", "tokens": d["tokens"]}
        return [by[p["key"]] for p in practice]

    return run


def reference_runner(practice_file: str | Path, data_root: str | Path, out_dir: str | Path) -> Callable:
    """A ``forward`` for LoraSubject through the trainer's own ``evaluate_qlora``
    (transformers, greedy, cap 3,072): the pass engine since gate G1 failed for
    vLLM (docs/PHASE5_PLAN.md §8). Practice items need ``key``; their draw rows
    (``source``, ``text``) come from ``practice_file``. Card 0 after ``wsp.card.ready``."""
    from wsp import card

    rows = {json.loads(l)["key"]: json.loads(l) for l in Path(practice_file).read_text(encoding="utf-8").splitlines() if l.strip()}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    count = {"n": 0}

    def run(adapter: Path, practice: Sequence[dict]) -> list[dict]:
        ok, why = card.ready()
        if not ok:
            raise RuntimeError(f"pass not started: {why}")
        count["n"] += 1
        stem = out_dir / f"pass-{count['n']:03d}"
        draw = stem.with_suffix(".draw.jsonl")
        draw.write_text("".join(json.dumps({"image": rows[p["key"]]["source"], "text": rows[p["key"]]["text"],
                                            "source_type": "page", "bbox": None}, ensure_ascii=False) + "\n"
                                for p in practice), encoding="utf-8")
        report = stem.with_suffix(".json")
        with open(stem.with_suffix(".log"), "w") as log:
            subprocess.run([sys.executable, "-m", "wsp.vlm_reference", "--adapter", str(adapter),
                            "--pages", str(draw), "--data-root", str(data_root), "--report", str(report)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        by = {}
        for line in report.with_suffix(".raw.jsonl").read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            by[Path(d["image"]).stem] = {"text": d["raw"], "at_cap": bool(d["at_cap"])}
        return [by[p["key"]] for p in practice]

    return run


def server_runner(spool: str | Path, out_dir: str | Path, render: str | Path | None = None,
                  timeout: float = 7200.0) -> Callable:
    """A ``forward`` that hands each pass to a running ``wsp.vllm_server`` through its spool
    directory (jobs/ in, done/ out). The server owns card 0 and its checks; this client only
    waits, and raises when the server has yielded (``YIELD``) or stopped."""
    import time

    spool = Path(spool); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    count = {"n": 0}

    def run(adapter: Path | None, practice: Sequence[dict]) -> list[dict]:
        count["n"] += 1
        jid = f"{os.getpid()}-{time.time_ns()}"
        out = out_dir / f"pass-{count['n']:03d}.jsonl"
        job = {"adapter": str(adapter) if adapter else None, "keys": [p["key"] for p in practice], "out": str(out)}
        if render:
            job["render"] = str(render)
        tmp = spool / "jobs" / f"{jid}.tmp"
        tmp.write_text(json.dumps(job)); tmp.rename(spool / "jobs" / f"{jid}.json")
        t0 = time.time()
        while not (spool / "done" / jid).exists():
            if (spool / "YIELD").exists() or not (spool / "READY").exists():
                raise RuntimeError("pass server not running (yielded or stopped): " +
                                   ((spool / "YIELD").read_text() if (spool / "YIELD").exists() else "no READY"))
            if time.time() - t0 > timeout:
                raise TimeoutError(f"pass {out} not answered in {timeout}s")
            time.sleep(5)
        by = {}
        for line in out.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            by[d["key"]] = {"text": d["text"], "at_cap": d["finish_reason"] == "length", "tokens": d["tokens"]}
        return [by[p["key"]] for p in practice]

    return run
