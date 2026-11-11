"""Stage 1 of the TrOCR track: the family of docs/MODELS_TROCR.md read as weights (no forward pass, no data).

    python -m wsp.trocr_family read   --out docs/trocr/readings.json
    python -m wsp.trocr_family tables --readings docs/trocr/readings.json > docs/trocr/readings_tables.md

Tensors are paired by canonical name (``wsp.trocr_subject.canonical_name``); every model of the family has the same shapes,
so there is no alignment step. Distance as in R5 (docs/ZOO_SURVEY.md): d(a, b) = ||a - b|| / mean(||a||, ||b||), pooled over
the weight tensors of a part (biases and norms are reported apart: tiny norms, large ratios); independent random networks
sit at sqrt(2) = 1.41, but here every model starts from ``microsoft/trocr-base-handwritten``, so 1.41 is NOT the null: the
null of stage 2 (same data, different parent; seeds) is. Nothing is written to a model.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

from wsp.trocr_subject import UNUSED, canonical_name, component

REPO = Path(__file__).resolve().parent.parent
CACHE = Path.home() / "atr-cache" / "trocr"
MODELS = {                      # short name -> directory under the cache
    "base": "microsoft__trocr-base-handwritten",
    "kurrent": "dh-unibe__trocr-kurrent",
    "xvi-xvii": "dh-unibe__trocr-kurrent-XVI-XVII",
    "from-kurrent-v2": "dh-unibe__trocr-german-kurrent-xvi-xix-from-kurrent-v2",
    "v2": "dh-unibe__trocr-german-kurrent-xvi-xix-v2",
    "full-v1": "dh-unibe__trocr-german-kurrent-xvi-xix-full-v1",
    "medieval": "dh-unibe__trocr-medieval-escriptmask",
}
#: documented parent of each model (docs/MODELS_TROCR.md section A; the cards, not the weights)
PARENT = {"kurrent": "base", "xvi-xvii": "kurrent", "from-kurrent-v2": "xvi-xvii", "full-v1": "xvi-xvii",
          "v2": "base", "medieval": "base"}
RTOL, ATOL = 1e-4, 1e-6


def _kind(name: str) -> str:
    """norm / bias / embed / weight, for the pooled readings (weights are pooled; the rest is reported apart)."""
    comp = component(name)
    if "norm" in name:
        return "norm"
    if name.endswith(".bias"):
        return "bias"
    if comp in ("enc.embed", "dec.embed", "dec.out"):
        return "embed"
    return "weight"


def _layer(name: str) -> str | None:
    p = name.split(".")
    return p[1] if p[0] in ("enc", "dec") and p[1].isdigit() else None


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    d = a64 - b64
    na, nb = float(np.linalg.norm(a64)), float(np.linalg.norm(b64))
    den = (na + nb) / 2
    cos = float(a64 @ b64 / (na * nb)) if na > 0 and nb > 0 else None
    return {"sq": float(d @ d), "na2": na * na, "nb2": nb * nb,
            "dist": float(np.sqrt(d @ d) / den) if den > 0 else 0.0, "cos": cos,
            "identical": float(np.mean(a.ravel() == b.ravel())),
            "close": float(np.mean(np.isclose(a.ravel(), b.ravel(), rtol=RTOL, atol=ATOL))), "n": int(a.size)}


def read(out: Path) -> dict:
    from safetensors import safe_open

    handles = {k: safe_open(str(CACHE / v / "model.safetensors"), framework="np") for k, v in MODELS.items()}
    names: dict[str, dict[str, str]] = {}               # canonical -> {model: hf name}
    for k, h in handles.items():
        for hf in h.keys():
            names.setdefault(canonical_name(hf), {})[k] = hf
    tensors = {}
    pairs = list(itertools.combinations(MODELS, 2))
    rows = []
    for cn in sorted(names):
        present = names[cn]
        have = [k for k in MODELS if k in present]
        assert len(have) == len(MODELS) or cn.startswith("enc.pooler"), (cn, set(MODELS) - set(present))   # only the unused pooler may be missing (the base has none)
        arr = {k: handles[k].get_tensor(present[k]).astype(np.float32) for k in have}
        shape = {a.shape for a in arr.values()}
        assert len(shape) == 1, (cn, shape)
        comp = component(cn)
        for a, b in pairs:
            if a not in arr or b not in arr:
                continue
            s = pair_stats(arr[a], arr[b])
            rows.append({"t": cn, "a": a, "b": b, **{k: s[k] for k in ("sq", "dist", "cos", "identical", "close", "n")},
                         "na2": s["na2"], "nb2": s["nb2"]})
        tensors[cn] = {"shape": list(next(iter(shape))), "component": comp, "kind": _kind(cn), "layer": _layer(cn),
                       "unused": cn.split(".")[0] + "." + cn.split(".")[1] in UNUSED, "present_in": have}
        if cn == "dec.embed.tokens":                     # row-wise movement, the one place a row means something (a token)
            tokrows = {}
            for a, b in pairs:
                diff = np.linalg.norm(arr[a].astype(np.float64) - arr[b], axis=1)
                den = (np.linalg.norm(arr[a].astype(np.float64), axis=1) + np.linalg.norm(arr[b].astype(np.float64), axis=1)) / 2
                tokrows[f"{a}|{b}"] = {"rows_identical": int((diff == 0).sum()),
                                      "rel_decile_mean": [float(x.mean()) for x in np.array_split(diff / np.where(den > 0, den, 1), 10)],
                                      "abs_decile_mean": [float(x.mean()) for x in np.array_split(diff, 10)]}
            tensors[cn]["token_rows"] = tokrows
    res = {"models": MODELS, "parent": PARENT, "tensors": tensors, "pairs": rows, "rtol": RTOL, "atol": ATOL}
    out.write_text(json.dumps(res, indent=0))
    return res


# -- aggregation ------------------------------------------------------------------------------------------------

def pooled(rows: list[dict], sel) -> dict[tuple[str, str], dict]:
    """Pool ||a-b||^2 and the two squared norms over the selected tensors, per model pair."""
    acc: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if sel(r):
            x = acc.setdefault((r["a"], r["b"]), [0.0, 0.0, 0.0, 0.0, 0.0])
            x[0] += r["sq"]; x[1] += r["na2"]; x[2] += r["nb2"]; x[3] += r["identical"] * r["n"]; x[4] += r["n"]
    return {k: {"dist": float(np.sqrt(v[0]) / ((np.sqrt(v[1]) + np.sqrt(v[2])) / 2)), "identical": v[3] / v[4], "n": int(v[4])}
            for k, v in acc.items()}


def dist_matrix(res: dict, sel) -> dict[tuple[str, str], float]:
    meta = res["tensors"]
    rows = [r for r in res["pairs"] if sel(r, meta[r["t"]])]
    p = pooled(rows, lambda r: True)
    out = {}
    for (a, b), v in p.items():
        out[(a, b)] = out[(b, a)] = v["dist"]
    return out


def mst(models: list[str], d: dict) -> list[tuple[str, str, float]]:
    """Minimum spanning tree (Prim) of the distance matrix: the lineage the weights alone suggest."""
    seen = {models[0]}
    edges = []
    while len(seen) < len(models):
        best = min(((d[(a, b)], a, b) for a in seen for b in models if b not in seen), key=lambda x: x[0])
        edges.append((best[1], best[2], best[0])); seen.add(best[2])
    return edges


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("read"); r.add_argument("--out", default=str(REPO / "docs/trocr/readings.json"))
    a = ap.parse_args(argv)
    if a.cmd == "read":
        read(Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
