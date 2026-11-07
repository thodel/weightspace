"""A1 readings of adapters of one base: norms, effective rank, layer profile, pairwise agreement.

Budget 0, CPU only. For each adapted module ΔW = (α/r)·B·A is never formed: norms and inner
products come from the r×r Gram matrices of the factors.

    python -m wsp.adapter_compare --out docs/readings/adapter_compare.json  A=dir B=dir ...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from wsp.lora_subject import LoraSubject

KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _ba(s: LoraSubject, m: str):
    return (s._state[f"{m}.lora_B.weight"].astype(np.float64), s._state[f"{m}.lora_A.weight"].astype(np.float64))


def inner(sa: LoraSubject, sb: LoraSubject, m: str) -> float:
    """<ΔW_a, ΔW_b>_F = sa.scale·sb.scale · Σ (Ba'Bb) ∘ (Aa Ab')  — r×r matrices only."""
    ba, aa = _ba(sa, m)
    bb, ab = _ba(sb, m)
    return float(sa.scale * sb.scale * np.sum((ba.T @ bb) * (aa @ ab.T)))


def eff_rank(sv: np.ndarray) -> float:
    """exp of the entropy of the normalised squared singular values (Roy & Vetterli, on energy)."""
    p = sv.astype(np.float64) ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def profile(s: LoraSubject) -> list[dict]:
    rows = []
    for m in s.modules():
        _, sv, _ = s.directions(m)
        e = sv.astype(np.float64) ** 2
        b, a = _ba(s, m)
        rows.append({"module": m, "layer": int(re.search(r"layers\.(\d+)\.", m).group(1)),
                     "kind": m.rsplit(".", 1)[-1], "fro": float(np.sqrt(e.sum())), "sv_max": float(sv[0]),
                     "eff_rank": eff_rank(sv), "stable_rank": float(e.sum() / sv[0] ** 2),
                     "rank90": int(np.searchsorted(np.cumsum(e) / e.sum(), 0.9) + 1),
                     "norm_B": float(np.linalg.norm(b)), "norm_A": float(np.linalg.norm(a))})
    return rows


def pairwise(subs: dict[str, LoraSubject]) -> dict:
    names = list(subs)
    out = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            sx, sy = subs[x], subs[y]
            common = sorted(set(sx.modules()) & set(sy.modules()))
            cos, dot, nx, ny = [], 0.0, 0.0, 0.0
            per_kind: dict[str, list[float]] = {}
            same_A = []
            for m in common:
                d = inner(sx, sy, m)
                a, b = inner(sx, sx, m), inner(sy, sy, m)
                c = d / np.sqrt(a * b)
                cos.append(c)
                dot += d; nx += a; ny += b
                per_kind.setdefault(m.rsplit(".", 1)[-1], []).append(float(c))
                same_A.append(float(np.abs(sx._state[f"{m}.lora_A.weight"] - sy._state[f"{m}.lora_A.weight"]).max()))
            out[f"{x}|{y}"] = {"modules": len(common), "cos_global": float(dot / np.sqrt(nx * ny)),
                               "cos_median": float(np.median(cos)), "cos_min": float(min(cos)), "cos_max": float(max(cos)),
                               "cos_by_kind": {k: round(float(np.median(v)), 4) for k, v in per_kind.items()},
                               "A_identical_modules": int(sum(1 for v in same_A if v == 0.0))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.adapter_compare")
    ap.add_argument("--out", required=True)
    ap.add_argument("adapters", nargs="+", help="label=directory")
    a = ap.parse_args(argv)
    subs = {}
    for spec in a.adapters:
        k, d = spec.split("=", 1)
        subs[k] = LoraSubject(d)
    res = {"adapters": {k: s.describe() | {"dir": str(s.dir)} for k, s in subs.items()},
           "profiles": {k: profile(s) for k, s in subs.items()}, "pairwise": pairwise(subs)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"pairwise": res["pairwise"]}, indent=1))


if __name__ == "__main__":
    main()
