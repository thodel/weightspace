"""Planted defects: corrupt a model in a known way, and score any model file on a set (cached by digest).

    python -m wsp.plant make  [--out ~/atr-cache/wsp/planted]       # writes the corrupted .mlmodel files + plant.json
    python -m wsp.plant score MODEL.mlmodel --set blb-eval-hand|--practice PATH [--out FILE]

The corruptions are pure functions of the original tensors and a seed, so ``plant.json`` (op, tensors,
parameters, seed, digest) reproduces every file. Subject: ``kraken-german-xix-blb-v1`` (docs/BRIEF_ARTICLE_EXPERIMENTS.md
item 2). Rows of the output layer are addressed by character through ``KrakenSubject.codec_map``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MODELS = Path.home() / "Repo" / "weightspace" / "models" / "kraken"
SUBJECT = MODELS / "kraken-german-xix-blb-v1.mlmodel"
PARENT = MODELS / "kraken-german-xix-v1.mlmodel"
OUT = Path.home() / "atr-cache" / "wsp" / "planted"
SEED = 20261107
W, B = "O_18.lin.weight", "O_18.lin.bias"


def _row(t, cm, ch, fn):
    r = cm[ch]
    for name in (W, B):
        v = t[name].copy()
        v[r] = fn(v[r])
        t[name] = v


def zero_row(t, cm, ch="e", **_):
    _row(t, cm, ch, lambda x: np.zeros_like(x))


def negate_row(t, cm, ch="n", **_):
    _row(t, cm, ch, lambda x: -x)


def swap_rows(t, cm, a="e", b="n", **_):
    ra, rb = cm[a], cm[b]
    for name in (W, B):
        v = t[name].copy()
        v[[ra, rb]] = v[[rb, ra]]
        t[name] = v


def blank_bias(t, cm, delta=2.0, **_):
    v = t[B].copy(); v[0] += delta; t[B] = v


def scale_tensors(t, cm, names=(), factor=1.0, **_):
    for n in names:
        t[n] = (t[n] * factor).astype(t[n].dtype)


def noise_block(t, cm, names=(), rel_norm=0.25, seed=SEED, **_):
    """Gaussian noise over the weight tensors of one block, total Frobenius norm = rel_norm × the block's norm."""
    rng = np.random.default_rng(seed)
    ws = [n for n in names if t[n].ndim >= 2]
    norm = float(np.sqrt(sum(float((t[n].astype(np.float64) ** 2).sum()) for n in ws)))
    noise = {n: rng.standard_normal(t[n].shape) for n in ws}
    nn = float(np.sqrt(sum(float((x ** 2).sum()) for x in noise.values())))
    for n in ws:
        t[n] = (t[n] + noise[n] * (rel_norm * norm / nn)).astype(t[n].dtype)


def noise_all(t, cm, rel_norm=0.1, seed=SEED, **_):
    """Diffuse: Gaussian noise on every weight tensor (ndim >= 2) of the network, each at rel_norm x its own norm."""
    rng = np.random.default_rng(seed)
    for n in sorted(t):
        if t[n].ndim >= 2:
            z = rng.standard_normal(t[n].shape)
            t[n] = (t[n] + z * (rel_norm * float(np.linalg.norm(t[n].astype(np.float64))) / float(np.linalg.norm(z)))).astype(t[n].dtype)


def zero_tensors(t, cm, names=(), **_):
    for n in names:
        t[n] = np.zeros_like(t[n])


L12 = [f"L_12.layer.{p}{d}" for d in ("", "_reverse") for p in ("weight_ih_l0", "weight_hh_l0", "bias_hh_l0")]
L14 = [f"L_14.layer.{p}{d}" for d in ("", "_reverse") for p in ("weight_ih_l0", "weight_hh_l0", "bias_hh_l0")]
L16 = [f"L_16.layer.{p}{d}" for d in ("", "_reverse") for p in ("weight_ih_l0", "weight_hh_l0", "bias_hh_l0")]

#: id -> (what, op, kwargs)
CORRUPTIONS: dict[str, tuple[str, callable, dict]] = {
    "k1-zero-row-e": ("output row of `e` (weights and bias) set to 0", zero_row, {"ch": "e"}),
    "k2-negate-row-n": ("output row of `n` (weights and bias) negated", negate_row, {"ch": "n"}),
    "k3-swap-rows-e-n": ("output rows of `e` and `n` swapped (weights and bias)", swap_rows, {"a": "e", "b": "n"}),
    "k4-blank-bias-plus2": ("blank bias +2.0", blank_bias, {"delta": 2.0}),
    "k5-output-weight-x2": ("output layer weight matrix x2 (bias unchanged)", scale_tensors, {"names": (W,), "factor": 2.0}),
    "k6-lstm16-x0.5": ("last BiLSTM block (L_16, all weights and biases) x0.5", scale_tensors, {"names": tuple(L16), "factor": 0.5}),
    "k7-noise-lstm14": ("Gaussian noise on the weights of BiLSTM L_14, norm 25 % of the block's", noise_block, {"names": tuple(L14), "rel_norm": 0.25}),
    "k9-noise-all-10pct": ("diffuse: Gaussian noise on every weight tensor, 10 % of each tensor's norm", noise_all, {"rel_norm": 0.10}),
    "k10-noise-lstm14-60pct": ("Gaussian noise on the weights of BiLSTM L_14, norm 60 % of the block's", noise_block, {"names": tuple(L14), "rel_norm": 0.60}),
    "k11-lstm-all-x0.9": ("diffuse: all three BiLSTM blocks (weights and biases) x0.9", scale_tensors, {"names": tuple(L12 + L14 + L16), "factor": 0.9}),
    "k8-zero-conv-c3": ("conv block C_3 (weights and bias) set to 0", zero_tensors, {"names": ("C_3.co.weight", "C_3.co.bias")}),
}


def make(out: Path = OUT, only: tuple = ()) -> dict:
    from wsp.kraken_subject import KrakenSubject
    from wsp.data import sha256_file
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    old = json.loads((out / "plant.json").read_text()) if (out / "plant.json").exists() else {}
    meta = dict(old)
    for cid, (what, fn, kw) in CORRUPTIONS.items():
        if only and cid not in only:
            continue
        s = KrakenSubject(SUBJECT)
        t = {k: v.copy() for k, v in s.tensors().items()}
        orig = s.digest()
        cm = s.codec_map()
        fn(t, cm, **kw)
        changed = [k for k in t if not np.array_equal(t[k], s.tensors()[k])]
        for k in changed:
            s.set_tensor(k, t[k])
        path = s.save(out / f"{cid}.mlmodel")
        meta[cid] = {"what": what, "op": fn.__name__, "args": {k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()},
                     "seed": SEED, "changed_tensors": changed, "original_digest": orig, "digest": s.digest(),
                     "file_sha256": sha256_file(path)}
        print(cid, s.digest()[:12], len(changed), "tensors")
    (out / "plant.json").write_text(json.dumps(meta, indent=2))
    return meta


def score_model(model: Path, items: list[dict], out: Path | None = None) -> dict:
    from wsp.kraken_subject import KrakenSubject
    from wsp.metrics import score, per_line_edits
    s = KrakenSubject(model)
    t = time.time()
    hyps = [p.text for p in s.forward(items)]
    refs = [x["text"] for x in items]
    m = score(hyps, refs)
    res = {"model": str(model), "digest": s.digest(), "lines": len(items), "metrics": m.as_dict(),
           "per_line": per_line_edits(hyps, refs), "hyps": hyps, "keys": [x["key"] for x in items],
           "wall_s": round(time.time() - t, 1)}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, ensure_ascii=False))
    return res


def cache_heldout(res: dict, set_name: str, set_path: Path) -> Path:
    """Store a score where ``wsp.run.heldout`` finds it (same digest, same list)."""
    from wsp.data import sha256_file
    from wsp.run import SCORES
    sha = sha256_file(set_path)
    SCORES.mkdir(parents=True, exist_ok=True)
    p = SCORES / f"heldout-{set_name}-{sha[:12]}-{res['digest'][:16]}.json"
    p.write_text(json.dumps({"digest": res["digest"], "set": set_name, "set_sha256": sha, "keys": res["keys"],
                             "hyps": res["hyps"], "metrics": res["metrics"]}, ensure_ascii=False))
    return p


def main(argv=None):
    from wsp.data import load
    ap = argparse.ArgumentParser(prog="wsp.plant")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--out", default=str(OUT)); m.add_argument("--only", nargs="*", default=[])
    s = sub.add_parser("score"); s.add_argument("model"); s.add_argument("--list", required=True); s.add_argument("--out", required=True)
    s.add_argument("--cache-as", help="held-out set name: also store the score where wsp.run heldout reuses it")
    a = ap.parse_args(argv)
    if a.cmd == "make":
        make(Path(a.out), tuple(a.only))
    else:
        r = score_model(Path(a.model), load(a.list), Path(a.out))
        if a.cache_as:
            cache_heldout(r, a.cache_as, Path(a.list))
        print(f"{Path(a.model).stem}: CER {r['metrics']['cer']:.4f} collapsed {r['metrics']['collapsed']} ({r['lines']} lines, {r['wall_s']} s)")


if __name__ == "__main__":
    main()
