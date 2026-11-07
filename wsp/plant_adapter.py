"""Planted defects in a peft adapter (stage A3): known corruptions, reproducible from the original.

    python -m wsp.plant_adapter make --adapter DIR [--out ~/atr-cache/wsp/planted-adapter]

Each corruption is a pure function of the original adapter and a seed; ``plant.json`` records the
operation, the tensors, the parameters and the digests. The original adapter is never modified:
every corrupted adapter is a full copy directory (config and tokenizer files included) under the
cache. Never exported, registered or uploaded.

| id | what |
|---|---|
| p1-scale3  | ``B`` of one module ×3 (that module's update is three times too strong) |
| p2-zeroB   | ``B`` of one module set to 0 (the module's update is gone) |
| p3-alpha2  | ``lora_alpha`` doubled in the config (every update twice as strong; tensors untouched) |
| p4-randdir | a random matrix of fixed norm added to one module's ``B`` |
| p5-alpha05 | ``lora_alpha`` halved in the config |
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from wsp.lora_subject import LoraSubject

OUT = Path.home() / "atr-cache" / "wsp" / "planted-adapter"
SEED = 20261107
PFX = "base_model.model.model.language_model.layers."
MOD_SCALE = PFX + "27.mlp.down_proj"
MOD_ZERO = PFX + "27.mlp.gate_proj"
MOD_RAND = PFX + "27.mlp.down_proj"


def corrupt(src: Path, cid: str, rel_norm: float = 1.0) -> LoraSubject:
    s = LoraSubject(src)
    if cid == "p1-scale3":
        s.scale_module(MOD_SCALE, 3.0)
    elif cid == "p2-zeroB":
        k = f"{MOD_ZERO}.lora_B.weight"
        s.set_tensor(k, np.zeros_like(s.tensors()[k]))
    elif cid == "p4-randdir":
        k = f"{MOD_RAND}.lora_B.weight"
        b = s.tensors()[k]
        r = np.random.default_rng(SEED).standard_normal(b.shape).astype(np.float32)
        s.set_tensor(k, b + r * (rel_norm * np.linalg.norm(b) / np.linalg.norm(r)))
    elif cid == "p3-alpha2":
        s.config["lora_alpha"] = s.config["lora_alpha"] * 2
    elif cid == "p5-alpha05":
        s.config["lora_alpha"] = s.config["lora_alpha"] // 2
    else:
        raise KeyError(cid)
    return s


def write(s: LoraSubject, path: Path) -> Path:
    path = s.save(path)
    (path / "adapter_config.json").write_text(json.dumps(s.config, indent=2))  # alpha edits live here
    return path


def digest(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update((path / "adapter_config.json").read_bytes())
    h.update((path / "adapter_model.safetensors").read_bytes())
    return h.hexdigest()


def make(src: Path, out: Path = OUT, ids=("p1-scale3", "p2-zeroB", "p3-alpha2", "p4-randdir", "p5-alpha05"),
         rel_norm: float = 1.0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source": str(src), "source_digest": digest(src), "seed": SEED, "rel_norm_randdir": rel_norm}
    for cid in ids:
        s = corrupt(src, cid, rel_norm)
        p = write(s, out / cid)
        meta[cid] = {"module": {"p1-scale3": MOD_SCALE, "p2-zeroB": MOD_ZERO, "p4-randdir": MOD_RAND}.get(cid, "all (config)"),
                     "lora_alpha": s.config["lora_alpha"], "digest": digest(p)}
        print(cid, meta[cid]["digest"][:12])
    (out / "plant.json").write_text(json.dumps(meta, indent=2))
    return meta


SCORES = Path.home() / "atr-cache" / "wsp" / "scores"
SPOOL = Path.home() / "atr-cache" / "wsp" / "spool" / "a3"


def score(adapter: Path | None, set_file: Path, spool: Path = SPOOL) -> dict:
    """Page metrics of an adapter (``None`` = the base) on a page list, through a running pass server,
    cached by (adapter digest, list sha256). A score is a function of the weights and the list."""
    import hashlib
    import time

    from wsp import pages
    from wsp.lora_subject import server_runner

    items = [json.loads(l) for l in set_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    sha = hashlib.sha256(set_file.read_bytes()).hexdigest()
    d = digest(adapter) if adapter else "base"
    cache = SCORES / f"adapter-{set_file.stem}-{sha[:10]}-{d[:16]}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    t = time.time()
    run = server_runner(spool, SCORES / "adapter-passes" / f"{set_file.stem}-{d[:12]}")
    outs = run(adapter, items)
    s = pages.score([o["text"] for o in outs], [x["text"] for x in items], [o["at_cap"] for o in outs])
    res = {"adapter": str(adapter), "digest": d, "set": set_file.stem, "set_sha256": sha, "wall_s": round(time.time() - t, 1),
           "tokens_mean": sum(o.get("tokens", 0) for o in outs) / len(outs),
           "keys": [x["key"] for x in items], "per_page": s.pop("per_page"), "metrics": s}
    SCORES.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.plant_adapter")
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("--adapter", help="peft directory; 'none' = the base alone")
    ap.add_argument("--set", help="page list jsonl (score)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rel-norm", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        make(Path(a.adapter), Path(a.out), rel_norm=a.rel_norm)
    else:
        r = score(None if a.adapter == "none" else Path(a.adapter), Path(a.set))
        print(json.dumps(r["metrics"]))


if __name__ == "__main__":
    main()
