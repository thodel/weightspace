"""Planted defects for the TrOCR positive control (docs/TROCR_STAGE3_PREREG.md): corrupt ``from-kurrent-v2`` in a known way.

    python -m wsp.trocr_plant calibrate --out docs/trocr/plant_calibration.json   # CER cost of every defect and size on the practice set
    python -m wsp.trocr_plant make --out ~/atr-cache/wsp/planted-trocr           # writes the planted models + plant.json

The architecture differs from the kraken subject (wsp/plant.py): the decoder embedding is **tied** to the output projection (one tensor,
``dec.embed.tokens``, read as input and as readout), and the output projection has **no bias** (there is no ``dec.out.bias``), so the kraken
analogues "zero an output row", "blank bias" and "output weight x2" change or do not exist in the same form. Here a token row is edited in the
tied tensor (input and readout at once), scales live on layer norms and on attention/FFN output projections, and a heads is a 64-column slice
of an attention output projection. Every defect is a pure function of the tensors, the context (the frequent token ids of the practice text) and a
seed, so ``plant.json`` (id, size, tensors, digests) reproduces every model.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CACHE = Path.home() / "atr-cache" / "trocr"
SUBJECT = CACHE / "dh-unibe__trocr-german-kurrent-xvi-xix-from-kurrent-v2"
PARENT = CACHE / "dh-unibe__trocr-kurrent-XVI-XVII"             # the parent donor of the subject
UNRELATED = CACHE / "dh-unibe__trocr-medieval-escriptmask"      # same shapes, other lineage (base -> medieval), the unrelated donor
PRACTICE = REPO / "data/practice/trocr-blb-train-300.jsonl"
OUT = Path.home() / "atr-cache" / "wsp" / "planted-trocr"
SEED = 20261118
HEADS, HEAD_DIM = 16, 64


def frequent_token_ids(tokenizer, texts, n: int) -> list[int]:
    """The ``n`` most frequent token ids of the practice text (labels as the trainer builds them, special tokens left out)."""
    c = Counter()
    for t in texts:
        c.update(i for i in tokenizer(t, add_special_tokens=False).input_ids)
    return [i for i, _ in c.most_common(n)]


# ---------------------------------------------------------------------------------------------------- operations on a tensor dict
def zero_rows(t, ctx, ids=None, k=6, **_):
    v = t["dec.embed.tokens"].copy(); v[ctx["freq"][:k]] = 0.0; t["dec.embed.tokens"] = v


def negate_rows(t, ctx, k=6, **_):
    v = t["dec.embed.tokens"].copy(); v[ctx["freq"][:k]] = -v[ctx["freq"][:k]]; t["dec.embed.tokens"] = v


def swap_rows(t, ctx, a=0, b=1, **_):
    v = t["dec.embed.tokens"].copy(); ra, rb = ctx["freq"][a], ctx["freq"][b]; v[[ra, rb]] = v[[rb, ra]]; t["dec.embed.tokens"] = v


def scale(t, ctx, names=(), factor=1.0, **_):
    for n in names:
        t[n] = (t[n] * factor).astype(t[n].dtype)


def zero(t, ctx, names=(), **_):
    for n in names:
        t[n] = np.zeros_like(t[n])


def zero_head(t, ctx, layer=6, head=0, kind="cross", **_):
    n = f"dec.{layer:02d}.{kind}.o.weight"
    v = t[n].copy(); v[:, head * HEAD_DIM:(head + 1) * HEAD_DIM] = 0.0; t[n] = v


def noise(t, ctx, names=(), rel_norm=0.25, seed=SEED, **_):
    """Gaussian noise over the weight tensors named (ndim >= 2), total Frobenius norm = rel_norm x the norm of those tensors."""
    rng = np.random.default_rng(seed)
    ws = [n for n in names if t[n].ndim >= 2]
    norm = float(np.sqrt(sum(float((t[n].astype(np.float64) ** 2).sum()) for n in ws)))
    z = {n: rng.standard_normal(t[n].shape) for n in ws}
    zn = float(np.sqrt(sum(float((x ** 2).sum()) for x in z.values())))
    for n in ws:
        t[n] = (t[n] + z[n] * (rel_norm * norm / zn)).astype(t[n].dtype)


def noise_each(t, ctx, names=(), rel_norm=0.1, seed=SEED, **_):
    """Diffuse: Gaussian noise on every named weight tensor, each at rel_norm x its own norm."""
    rng = np.random.default_rng(seed)
    for n in sorted(names):
        if t[n].ndim >= 2:
            z = rng.standard_normal(t[n].shape)
            t[n] = (t[n] + z * (rel_norm * float(np.linalg.norm(t[n].astype(np.float64))) / float(np.linalg.norm(z)))).astype(t[n].dtype)


def shift(t, ctx, name="dec.embed.norm.bias", rel_norm=1.0, seed=SEED, **_):
    """Add a fixed random direction to a bias vector; its norm = rel_norm x the norm of the paired weight vector."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(t[name].shape)
    w = t[name.replace(".bias", ".weight")]
    t[name] = (t[name] + z * (rel_norm * float(np.linalg.norm(w)) / float(np.linalg.norm(z)))).astype(t[name].dtype)


def dec_layer(i, parts=("self", "cross", "ffn")):
    out = []
    for p in parts:
        if p == "ffn":
            out += [f"dec.{i:02d}.ffn.in.weight", f"dec.{i:02d}.ffn.out.weight"]
        else:
            out += [f"dec.{i:02d}.{p}.{x}.weight" for x in "qkvo"]
    return tuple(out)


ALL_DEC_FFN_OUT = tuple(f"dec.{i:02d}.ffn.out.{k}" for i in range(12) for k in ("weight", "bias"))
ALL_ENC_FFN_OUT = tuple(f"enc.{i:02d}.ffn.out.{k}" for i in range(12) for k in ("weight", "bias"))
ALL_DEC_W = tuple(n for i in range(12) for n in dec_layer(i))
ALL_ENC_W = tuple(f"enc.{i:02d}.{p}" for i in range(12) for p in ("attn.q.weight", "attn.k.weight", "attn.v.weight", "attn.o.weight", "ffn.in.weight", "ffn.out.weight"))
CROSS_O_5 = ("dec.05.cross.o.weight", "dec.05.cross.o.bias")
LN_DEC11 = ("dec.11.norm_ffn.weight", "dec.11.norm_ffn.bias")
ENC6_ATTN_O = ("enc.06.attn.o.weight", "enc.06.attn.o.bias")
ENC3_FFN_OUT = ("enc.03.ffn.out.weight", "enc.03.ffn.out.bias")

#: id -> (what, scope, fn, sizes {label: kwargs}, donor note). The first size is the one calibrated to the target cost (set after the calibration).
DEFECTS: dict[str, dict] = {
    "t1-zero-rows": {"what": "the rows of the k most frequent token ids of the practice text set to 0 in the tied embedding (input and readout)", "scope": "localised: token rows", "fn": zero_rows,
                     "sizes": {"k2": {"k": 2}, "k6": {"k": 6}, "k12": {"k": 12}}, "donor": "parent rows (same ids)"},
    "t2-negate-rows": {"what": "the rows of the k most frequent tokens negated in the tied embedding", "scope": "localised: token rows", "fn": negate_rows,
                       "sizes": {"k2": {"k": 2}, "k6": {"k": 6}}, "donor": "parent rows; also repairable by negating back"},
    "t3-swap-rows": {"what": "the rows of the two most frequent tokens swapped in the tied embedding", "scope": "localised: token rows", "fn": swap_rows,
                     "sizes": {"0-1": {"a": 0, "b": 1}}, "donor": "parent rows; also repairable by swapping back"},
    "t4-cross-o-x0.5": {"what": "decoder layer 5 cross-attention output projection (weight and bias) x0.5", "scope": "localised: one layer", "fn": scale,
                        "sizes": {"x0.5": {"names": CROSS_O_5, "factor": 0.5}, "x0.0": {"names": CROSS_O_5, "factor": 0.0}}, "donor": "parent block or a x2 rescale"},
    "t5-zero-head": {"what": "one cross-attention head (64 columns of the output projection) of decoder layer 6 zeroed", "scope": "localised: one head", "fn": zero_head,
                     "sizes": {"L6h0": {"layer": 6, "head": 0, "kind": "cross"}, "L6h0-3": {"layer": 6, "head": 0, "kind": "cross", "_also": [1, 2, 3]},
                               "L6h0-7": {"layer": 6, "head": 0, "kind": "cross", "_also": [1, 2, 3, 4, 5, 6, 7]},
                               "L6h0-15": {"layer": 6, "head": 0, "kind": "cross", "_also": list(range(1, 16))}},
                     "donor": "parent output projection (columns of that head)"},
    "t6-ln-x0.5": {"what": "decoder layer 11 FFN layer norm (weight and bias) x0.5", "scope": "localised: one norm", "fn": scale,
                   "sizes": {"x0.5": {"names": LN_DEC11, "factor": 0.5}, "x0.2": {"names": LN_DEC11, "factor": 0.2}, "x0.0": {"names": LN_DEC11, "factor": 0.0}}, "donor": "parent norm or a x5 rescale"},
    "t7-noise-dec8": {"what": "Gaussian noise on the weights of decoder layer 8 (self, cross, FFN), total norm a fraction of the layer's", "scope": "localised: one block, no signature", "fn": noise,
                      "sizes": {"25%": {"names": dec_layer(8), "rel_norm": 0.25}, "60%": {"names": dec_layer(8), "rel_norm": 0.60}, "100%": {"names": dec_layer(8), "rel_norm": 1.0}}, "donor": "parent block only"},
    "t8-zero-enc6-attn": {"what": "encoder layer 6 attention output projection (weight and bias) set to 0", "scope": "localised: one layer", "fn": zero,
                          "sizes": {"zero": {"names": ENC6_ATTN_O}}, "donor": "parent block"},
    "t9-enc3-ffn-x0.5": {"what": "encoder layer 3 FFN output projection (weight and bias) x0.5", "scope": "localised: one layer", "fn": scale,
                         "sizes": {"x0.5": {"names": ENC3_FFN_OUT, "factor": 0.5}, "x0.0": {"names": ENC3_FFN_OUT, "factor": 0.0}}, "donor": "parent block or a x2 rescale"},
    "t10-noise-all": {"what": "diffuse: Gaussian noise on every encoder and decoder weight matrix, each at a fraction of its own norm", "scope": "diffuse", "fn": noise_each,
                      "sizes": {"5%": {"names": ALL_ENC_W + ALL_DEC_W, "rel_norm": 0.05}, "10%": {"names": ALL_ENC_W + ALL_DEC_W, "rel_norm": 0.10},
                                "15%": {"names": ALL_ENC_W + ALL_DEC_W, "rel_norm": 0.15}, "20%": {"names": ALL_ENC_W + ALL_DEC_W, "rel_norm": 0.20}}, "donor": "parent only (whole model)"},
    "t11-ffn-x0.9": {"what": "diffuse: every decoder FFN output projection (weights and biases) scaled", "scope": "diffuse", "fn": scale,
                     "sizes": {"x0.9": {"names": ALL_DEC_FFN_OUT, "factor": 0.9}, "x0.7": {"names": ALL_DEC_FFN_OUT, "factor": 0.7}}, "donor": "a global rescale repairs it without a donor"},
    "t15-enc-ffn-x0.5": {"what": "diffuse: every encoder FFN output projection (weights and biases) scaled", "scope": "diffuse", "fn": scale,
                         "sizes": {"x0.7": {"names": ALL_ENC_FFN_OUT, "factor": 0.7}, "x0.5": {"names": ALL_ENC_FFN_OUT, "factor": 0.5}, "x0.3": {"names": ALL_ENC_FFN_OUT, "factor": 0.3}},
                         "donor": "a global rescale repairs it without a donor"},
    "t16-cross-o-all-x0.5": {"what": "diffuse: every decoder cross-attention output projection (weights and biases) scaled", "scope": "diffuse",
                             "fn": scale, "sizes": {"x0.7": {"names": tuple(f"dec.{i:02d}.cross.o.{k}" for i in range(12) for k in ("weight", "bias")), "factor": 0.7},
                                                    "x0.5": {"names": tuple(f"dec.{i:02d}.cross.o.{k}" for i in range(12) for k in ("weight", "bias")), "factor": 0.5}},
                             "donor": "a global rescale repairs it without a donor"},
    "t12-noise-embed": {"what": "diffuse over the vocabulary: Gaussian noise on the tied embedding, a fraction of its norm", "scope": "diffuse: tied embedding", "fn": noise,
                        "sizes": {"20%": {"names": ("dec.embed.tokens",), "rel_norm": 0.20}, "50%": {"names": ("dec.embed.tokens",), "rel_norm": 0.50}}, "donor": "parent embedding only"},
    "t13-shift-embnorm": {"what": "bias shift: a fixed random direction added to the decoder embedding layer-norm bias, a fraction of the norm of its weight", "scope": "diffuse: the bias of the one place a decoder-side bias shift reaches every token", "fn": shift,
                          "sizes": {"1.0": {"rel_norm": 1.0}, "3.0": {"rel_norm": 3.0}}, "donor": "parent bias, or a bias search"},
    "t14-null-out-scale": {"what": "null corruption (analogue of kraken k5): the decoder's final-layer FFN norm bias scaled x1.0001 (changes the digest, not the decoding)", "scope": "null", "fn": scale,
                           "sizes": {"x1.0001": {"names": ("dec.11.norm_ffn.bias",), "factor": 1.0001}}, "donor": "none needed"},
}


def apply_defect(subject, defect: str, size: str, ctx: dict):
    """Apply one defect at one size to a TrocrSubject in place; returns the list of tensors changed."""
    spec = DEFECTS[defect]
    kw = dict(spec["sizes"][size])
    also = kw.pop("_also", None)
    before = {k: v.copy() for k, v in subject.tensors().items()}
    t = {k: v.copy() for k, v in before.items()}
    spec["fn"](t, ctx, **kw)
    for h in (also or []):
        zero_head(t, ctx, layer=kw["layer"], head=h, kind=kw["kind"])
    changed = [k for k in t if not np.array_equal(t[k], before[k])]
    for k in changed:
        subject.set_tensor(k, t[k])
    return changed


def context(subject, practice, hyps=None) -> dict:
    """The tokens a token-row defect hits: the most frequent ids of what the *subject writes* on the practice set (its own readings, ``hyps``),
    not of the reference text: the references are dominated by spaced punctuation (` .`, ` ,`) that the subject almost never emits, and zeroing
    those rows costs nothing (first calibration run, 18.11.2026)."""
    if hyps is None:
        hyps = [p.text for p in subject.forward(practice)]
    return {"freq": frequent_token_ids(subject.tokenizer, hyps, 24), "hyps": hyps}


def calibrate(out: Path, only: tuple = (), device: str = "cuda:0") -> dict:
    import os
    from wsp import card, data
    from wsp.metrics import bootstrap_delta, page_of, per_line_edits, score
    from wsp.trocr_subject import TrocrSubject
    if device.startswith("cuda"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        ok, why = card.ready()
        print("card:", why, flush=True)
        if not ok:
            raise SystemExit(3)
    practice = data.load(PRACTICE)
    refs = [x["text"] for x in practice]
    clusters = [page_of(x["key"]) for x in practice]
    res = json.loads(out.read_text()) if out.exists() else {}
    base_sub = TrocrSubject(SUBJECT, device=device, batch=8)
    if "base" not in res:
        t0 = time.time()
        hy = [p.text for p in base_sub.forward(practice)]
        res["base"] = {"cer": score(hy, refs, blocks=False).cer, "per_line": per_line_edits(hy, refs), "hyps": hy, "wall_s": round(time.time() - t0, 1)}
        out.write_text(json.dumps(res))
    ctx = context(base_sub, practice, res["base"]["hyps"])
    base_lines = [tuple(x) for x in res["base"]["per_line"]]
    orig = {k: v.copy() for k, v in base_sub.tensors().items()}
    res["freq_tokens"] = ctx["freq"][:12]
    res["freq_tokens_text"] = [base_sub.tokenizer.decode([i]) for i in ctx["freq"][:12]]
    for d, spec in DEFECTS.items():
        for sz in spec["sizes"]:
            key = f"{d}|{sz}"
            if (only and d not in only) or key in res:
                continue
            for k, v in orig.items():
                if not np.array_equal(base_sub.tensors()[k], v):
                    base_sub.set_tensor(k, v)
            changed = apply_defect(base_sub, d, sz, ctx)
            t0 = time.time()
            hy = [p.text for p in base_sub.forward(practice)]
            m = score(hy, refs, blocks=False)
            lines = per_line_edits(hy, refs)
            bd = bootstrap_delta(base_lines, lines, resamples=1000, clusters=clusters)
            res[key] = {"what": spec["what"], "scope": spec["scope"], "changed": len(changed), "cer": m.cer, "damage": m.cer - res["base"]["cer"],
                        "ci95_cluster": bd["ci95_cluster"], "collapsed": m.collapsed, "length_ratio": m.length_ratio, "wall_s": round(time.time() - t0, 1)}
            print(f"{key:28s} CER {m.cer:.4f} damage {100 * (m.cer - res['base']['cer']):+.2f} pts  ({res[key]['wall_s']} s)", flush=True)
            out.write_text(json.dumps(res))
    return res


def make(out: Path, chosen: dict, only: tuple = ()) -> dict:
    """Write the planted models (hub format, the subject's tokenizer and processor unchanged) and ``plant.json``.
    ``chosen``: planted id -> {"defect": id, "size": label}; the ids are what the runs are named after."""
    from wsp import data
    from wsp.data import sha256_file
    from wsp.trocr_subject import TrocrSubject
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((out / "plant.json").read_text()) if (out / "plant.json").exists() else {}
    practice = data.load(PRACTICE)
    base = TrocrSubject(SUBJECT, device="cpu", check_card=False)
    hyps = json.loads((REPO / "docs/trocr/plant_calibration.json").read_text())["base"]["hyps"]
    ctx = context(base, practice, hyps)
    orig = base.digest()
    originals = {k: v.copy() for k, v in base.tensors().items()}
    for pid, c in chosen.items():
        if only and pid not in only:
            continue
        for k, v in originals.items():
            if not np.array_equal(base.tensors()[k], v):
                base.set_tensor(k, v)
        changed = apply_defect(base, c["defect"], c["size"], ctx)
        path = base.save(out / pid)
        meta[pid] = {"defect": c["defect"], "size": c["size"], "what": DEFECTS[c["defect"]]["what"], "scope": DEFECTS[c["defect"]]["scope"],
                     "seed": SEED, "changed_tensors": changed, "original_digest": orig, "digest": base.digest(),
                     "file_sha256": sha256_file(path / "model.safetensors"), "frequent_tokens": ctx["freq"][:12]}
        print(pid, base.digest()[:12], len(changed), "tensors", flush=True)
    (out / "plant.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.trocr_plant")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("--out", required=True); c.add_argument("--only", nargs="*", default=[]); c.add_argument("--device", default="cuda:0")
    m = sub.add_parser("make"); m.add_argument("--out", default=str(OUT)); m.add_argument("--chosen", required=True, help="JSON file: planted id -> {defect, size}")
    m.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        calibrate(Path(a.out), tuple(a.only), a.device)
    else:
        make(Path(a.out), json.loads(Path(a.chosen).read_text()), tuple(a.only))


if __name__ == "__main__":
    main()
