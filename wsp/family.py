"""Track C: the served kraken family compared as networks (PLAN.md phase 6, docs/MODEL_FAMILY.md).

    python -m wsp.family registry --gateway <gateway_models.json>   # which models, where on asteraix
    python -m wsp.family weights                                    # RQ1 + RQ2 in weight space
    python -m wsp.family functional                                 # RQ2 across lineages: per-layer linear CKA
    python -m wsp.family report                                     # tables for docs/MODEL_FAMILY.md

Every kraken model the gateway serves has the same body (VGSL ``Cr``×4, ``Lbx200``×3) and an
output layer ``O_18`` whose rows are the alphabet. So:

* **weight space** — per block, for every pair: share of bit-identical parameters, share within
  float tolerance, relative distance, cosine. Body blocks compare element by element; ``O_18``
  compares on the characters both alphabets share (rows paired by character, ``wsp.align``).
  Element-wise numbers only mean something between models with a common initialisation (a
  lineage); between independently trained models hidden units are in arbitrary order, and the
  body cosine is ~0. That body cosine is therefore also the **lineage test**.
* **function** — per block, linear CKA (Kornblith et al. 2019) between the two models' outputs of
  that block on the same probe lines: invariant to the order of units and to isotropic scaling,
  so it compares unrelated models too. Frames are the samples: same architecture and same input
  give the same number of frames per block, so rows are paired across models.

Nothing here writes to a model; results go to ``docs/family/``.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np

from wsp.align import codec_row_map
from wsp.data import load, sha256_file

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "family"
REGISTRY = REPO / "data" / "family" / "models.json"
INHOUSE = Path.home() / "Repo" / "weightspace" / "models" / "kraken"
BASES = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/bases")
DOWNLOADS = Path.home() / "atr-cache" / "wsp" / "family" / "zenodo"
CACHE = Path.home() / "atr-cache" / "wsp" / "family"
BODY = ["C_0", "C_3", "C_6", "C_9", "L_12", "L_14", "L_16"]
BLOCKS = BODY + ["O_18"]
PROBE_SETS = ["blb-train-300", "bern-train-300", "medieval-val-300"]   # train-side practice lines, three domains
PROBE_PER_SET = 40
FRAMES = 2000          # frames sampled per block for CKA
SEED = 20261107
RTOL, ATOL = 1e-4, 1e-6


# ------------------------------------------------------------------ registry
def _local_path(m: dict) -> Path | None:
    if m.get("local_path"):
        p = INHOUSE / Path(m["local_path"]).name
        return p if p.exists() else None
    doi = m.get("zenodo_id")
    if not doi:
        return None
    key = doi.replace("/", "_")
    for root in (BASES / key, DOWNLOADS / key):
        hits = sorted(root.glob("*.mlmodel")) if root.exists() else []
        if hits:
            return hits[0]
    return None


def registry(gateway_json: Path) -> list[dict]:
    """The gateway's kraken models (what agentic_historian can call), with where their weights are on asteraix."""
    d = json.loads(gateway_json.read_text())
    models = d["models"]["models"] if isinstance(d.get("models"), dict) else d["models"]
    rows = []
    for m in models:
        if m.get("engine") != "kraken":
            continue
        p = _local_path(m)
        rows.append({"id": m["id"], "enabled": m.get("enabled"), "task": m.get("task"),
                     "source": "zenodo" if m.get("zenodo_id") else "in-house",
                     "zenodo": m.get("zenodo_id"), "stated_base": m.get("base_model"),
                     "scripts": m.get("scripts"), "centuries": m.get("centuries"), "languages": m.get("languages"),
                     "gateway_digest": (m.get("weights_digest") or {}).get("tensors") if isinstance(m.get("weights_digest"), dict) else None,
                     "asteraix_path": str(p) if p else None})
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps({"source": "gateway_models (ATR gateway, agentic_historian's recogniser)",
                                    "read": time.strftime("%Y-%m-%d"), "models": rows}, indent=1, ensure_ascii=False))
    return rows


def available(reg: list[dict] | None = None) -> list[dict]:
    reg = reg or json.loads(REGISTRY.read_text())["models"]
    for m in reg:                                   # re-resolve: downloads may have arrived since
        p = _local_path({"local_path": m["asteraix_path"] if m["source"] == "in-house" else None, "zenodo_id": m["zenodo"]})
        m["asteraix_path"] = str(p) if p else None
    return [m for m in reg if m["asteraix_path"]]


def _load(path: str):
    from wsp.kraken_subject import KrakenSubject
    return KrakenSubject(path)


def body_spec(spec: str) -> str:
    return re.sub(r"O\{O_18\}1c\d+", "O{O_18}1c*", spec)


def arch(s) -> dict:
    """A model's architecture as far as comparison is concerned: its parameter blocks in order, its
    output block, and a key that is equal exactly when two models have the same input height and the
    same tensor shapes outside the output layer (then they compare element by element; a different
    dropout rate in the VGSL spec, which has no weights, does not split a group)."""
    names = [n for n, m in s.model.nn.named_children() if any(True for _ in m.parameters())]
    out = [n for n in names if n.startswith("O_")][-1]
    shapes = tuple((k, tuple(v.shape)) for k, v in s.tensors().items() if k.split(".")[0] != out)
    key = hashlib.sha256(repr((int(s.model.input[2]), shapes)).encode()).hexdigest()[:10]
    return {"blocks": names, "output": out, "key": key, "height": int(s.model.input[2])}


# ------------------------------------------------------------------ weight space
def _block(t: dict[str, np.ndarray], b: str) -> dict[str, np.ndarray]:
    return {k: v for k, v in t.items() if k.split(".")[0] == b}


def _stats(xs: list[np.ndarray], ys: list[np.ndarray]) -> dict:
    x = np.concatenate([a.astype(np.float64).ravel() for a in xs]); y = np.concatenate([a.astype(np.float64).ravel() for a in ys])
    d = y - x
    nx, ny, nd = float(np.linalg.norm(x)), float(np.linalg.norm(y)), float(np.linalg.norm(d))
    live = ~((x == 0) & (y == 0))           # kraken's LSTM bias_ih is all zero in every model: structural, not shared learning
    n_live = int(live.sum())
    return {"n": int(x.size), "structural_zero": int(x.size - n_live),
            "identical": float((d[live] == 0).mean()) if n_live else None,
            "close": float((np.abs(d[live]) <= ATOL + RTOL * np.abs(x[live])).mean()) if n_live else None,
            "rel_dist": nd / ((nx + ny) / 2) if nx + ny else 0.0,
            "cosine": float((x * y).sum() / (nx * ny)) if nx and ny else None}


def output_rows(sa, sb, out: str = "O_18") -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Output weight rows and biases of the characters both codecs share, plus the blank (row 0)."""
    rm = codec_row_map(sa.codec_map(), sb.codec_map())
    ia = [0] + sorted(rm); ib = [0] + [rm[i] for i in sorted(rm)]
    ta, tb = sa.tensors(), sb.tensors()
    w = [n for n in ta if n.startswith(out + ".") and n.endswith("weight")][0]
    bias = [n for n in ta if n.startswith(out + ".") and n.endswith("bias")][0]
    return [ta[w][ia], ta[bias][ia]], [tb[w][ib], tb[bias][ib]], len(rm)


def compare_pair(sa, sb, blocks: list[str] | None = None, out: str = "O_18") -> dict:
    """Per block statistics for two models of one architecture group; ``body`` = all blocks but the output."""
    blocks = blocks or BLOCKS
    ta, tb = sa.tensors(), sb.tensors()
    res = {}
    for b in blocks:
        if b == out:
            continue
        A, B = _block(ta, b), _block(tb, b)
        names = sorted(set(A) & set(B))
        if set(A) != set(B) or any(A[n].shape != B[n].shape for n in names):
            res[b] = {"comparable": False}
            continue
        res[b] = _stats([A[n] for n in names], [B[n] for n in names])
    xa, xb, shared = output_rows(sa, sb, out)
    res[out] = {**_stats(xa, xb), "shared_chars": shared}
    body = [b for b in blocks if b != out]
    res["body"] = _stats([v for b in body for _, v in sorted(_block(ta, b).items())],
                         [v for b in body for _, v in sorted(_block(tb, b).items())])
    return res


def _key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def weights(models: list[dict]) -> dict:
    subj, info, groups = {}, {}, {}
    for m in models:
        s = _load(m["asteraix_path"]); subj[m["id"]] = s; d = s.describe(); ar = arch(s)
        info[m["id"]] = {"params": d["parameters"], "codec": d["codec_size"], "spec": d["spec"], "arch": ar["key"],
                         "height": ar["height"], "tensor_digest": s.digest(),
                         "file_sha256": sha256_file(Path(m["asteraix_path"])), "path": m["asteraix_path"]}
        g = groups.setdefault(ar["key"], {"key": ar["key"], "blocks": ar["blocks"], "output": ar["output"],
                                          "height": ar["height"], "spec": d["spec"], "models": []})
        g["models"].append(m["id"])
    pairs = {}
    for g in groups.values():
        for a, b in itertools.combinations(sorted(g["models"]), 2):
            pairs[_key(a, b)] = compare_pair(subj[a], subj[b], g["blocks"], g["output"])
    glist = sorted(groups.values(), key=lambda g: (-len(g["models"]), g["key"]))
    for i, g in enumerate(glist):
        g["name"] = "ABCDEFGH"[i]
    return {"models": info, "groups": glist, "pairs": pairs,
            "tolerance": {"rtol": RTOL, "atol": ATOL}, "when": time.strftime("%Y-%m-%d %H:%M")}


# ------------------------------------------------------------------ lineages from weights
LINEAGE_COS = 0.5   # body cosine above which two models share an initialisation (unrelated pairs sit near 0)


def lineages(w: dict) -> list[list[str]]:
    ids = sorted(w["models"]); parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for key, p in w["pairs"].items():
        if p.get("body", {}).get("cosine") is not None and p["body"]["cosine"] > LINEAGE_COS:
            a, b = key.split("|"); parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g))


def stated_bases(w: dict, reg: list[dict]) -> list[dict]:
    """Check every base the gateway registry states (a Zenodo DOI) against the weights."""
    by_doi = {m["zenodo"]: m["id"] for m in reg if m["zenodo"]}
    out = []
    for m in reg:
        base = by_doi.get(m.get("stated_base"))
        if not m.get("stated_base"):
            continue
        row = {"model": m["id"], "stated_base": m["stated_base"], "base_model": base}
        if base is None or m["id"] not in w["models"] or base not in w["models"]:
            row["verdict"] = "base not on the gateway" if base is None else "weights missing"
        elif w["models"][m["id"]]["arch"] != w["models"][base]["arch"]:
            row["verdict"] = "different architecture: cannot be a fine-tune of this file"
        else:
            c = w["pairs"][_key(m["id"], base)]["body"]["cosine"]
            row["body_cosine"] = c
            row["verdict"] = "confirmed" if c > LINEAGE_COS else "contradicted: unrelated weights"
        out.append(row)
    return out


# ------------------------------------------------------------------ function: linear CKA per block
def probe_items() -> list[dict]:
    rng = np.random.default_rng(SEED); items = []
    for s in PROBE_SETS:
        xs = load(REPO / "data" / "practice" / f"{s}.jsonl")
        idx = sorted(rng.choice(len(xs), size=min(PROBE_PER_SET, len(xs)), replace=False))
        items += [dict(xs[i], probe_set=s) for i in idx]
    return items


def _gram(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64); X = X - X.mean(0, keepdims=True)
    return X @ X.T


def cka_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    """Linear CKA = HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L)) with centred features (so centred grams)."""
    hkl = float((K * L).sum()); hkk = float((K * K).sum()); hll = float((L * L).sum())
    return hkl / np.sqrt(hkk * hll) if hkk and hll else float("nan")


def functional_group(models: list[dict], blocks: list[str], items: list[dict], frames: int = FRAMES, log=print) -> dict:
    """CKA between every pair of one architecture group, per conv/recurrent/output block; dead conv filters."""
    sample_idx: dict[str, np.ndarray] = {}
    grams: dict[str, dict[str, np.ndarray]] = {b: {} for b in blocks}
    dead: dict[str, dict] = {}
    frame_counts = None
    for m in models:
        s = _load(m["asteraix_path"])
        acts = s.activations(items, blocks)
        counts = {b: [x.shape[0] for x in acts[b]] for b in blocks}
        if frame_counts is None:
            frame_counts = counts
            rng = np.random.default_rng(SEED)
            for b in blocks:
                total = sum(counts[b]); sample_idx[b] = np.sort(rng.choice(total, size=min(frames, total), replace=False))
        elif counts != frame_counts:
            raise RuntimeError(f"{m['id']}: frame counts differ inside one architecture group")
        dead[m["id"]] = {}
        for b in blocks:
            full = np.concatenate(acts[b], axis=0)
            if b.startswith("C_"):          # Cr = convolution + ReLU: a filter that is 0 on every frame is dead
                C = next(v for k, v in s.tensors().items() if k.split(".")[0] == b and v.ndim == 4).shape[0]
                cmax = full.reshape(full.shape[0], C, -1).max(axis=(0, 2))
                dead[m["id"]][b] = {"dead": int((cmax <= 0).sum()), "filters": int(C),
                                    "dead_index": [int(i) for i in np.flatnonzero(cmax <= 0)]}
            grams[b][m["id"]] = _gram(full[sample_idx[b]]).astype(np.float32)
        log(f"  activations {m['id']}")
    ids = [m["id"] for m in models]
    cka = {b: {_key(x, y): cka_from_grams(grams[b][x].astype(np.float64), grams[b][y].astype(np.float64))
               for x, y in itertools.combinations(ids, 2)} for b in blocks}
    return {"models": ids, "blocks": blocks, "frames": {b: int(len(sample_idx[b])) for b in blocks},
            "cka": cka, "dead_filters": dead}


def functional(models: list[dict], w: dict, frames: int = FRAMES) -> dict:
    items = probe_items()
    by_id = {m["id"]: m for m in models}
    groups = []
    for g in w["groups"]:
        blocks = [b for b in g["blocks"] if b[0] in "CLO"]
        ms = [by_id[i] for i in g["models"] if i in by_id]
        if len(ms) < 2:
            res = {"models": [m["id"] for m in ms], "blocks": blocks, "cka": {}, "dead_filters": {}, "frames": {}}
            if ms:                                              # still count its dead filters
                res = {**functional_group(ms, blocks, items, frames), "cka": {}}
        else:
            res = functional_group(ms, blocks, items, frames)
        res.update({"group": g["name"], "key": g["key"]})
        groups.append(res)
    return {"probe": {"sets": PROBE_SETS, "per_set": PROBE_PER_SET, "lines": len(items), "seed": SEED,
                      "keys": [i["key"] for i in items]}, "groups": groups, "when": time.strftime("%Y-%m-%d %H:%M")}


# ------------------------------------------------------------------ fetch (needs an explicit OK: it downloads)
def fetch(ids: list[str] | None = None, dry_run: bool = True) -> list[dict]:
    """Download the .mlmodel of each Zenodo-published gateway model that has no copy on asteraix, from the
    Zenodo record the gateway names, into ``~/atr-cache/wsp/family/zenodo/<doi>/``; verify Zenodo's checksum.
    ``dry_run`` lists files and sizes without downloading."""
    import hashlib as _h
    import urllib.request
    reg = json.loads(REGISTRY.read_text())["models"]
    todo = [m for m in reg if m["zenodo"] and not m["asteraix_path"] and (not ids or m["id"] in ids)]
    plan = []
    for m in todo:
        rec = m["zenodo"].split("zenodo.")[-1]
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}", timeout=60) as r:
            meta = json.loads(r.read())
        files = [x for x in meta.get("files", []) if x["key"].endswith(".mlmodel")]
        for x in files:
            plan.append({"id": m["id"], "doi": m["zenodo"], "file": x["key"], "bytes": x["size"],
                         "checksum": x.get("checksum"), "url": x["links"]["self"]})
        if not files:
            plan.append({"id": m["id"], "doi": m["zenodo"], "file": None, "note": "no .mlmodel in the record"})
    if dry_run:
        return plan
    for p_ in plan:
        if not p_.get("file"):
            continue
        dest = DOWNLOADS / p_["doi"].replace("/", "_") / p_["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(p_["url"], dest)
        algo, _, want = (p_["checksum"] or "md5:").partition(":")
        got = _h.new(algo, dest.read_bytes()).hexdigest()
        p_["verified"] = got == want
        if not p_["verified"]:
            dest.unlink(); raise RuntimeError(f"{p_['file']}: {algo} {got} != {want}")
    return plan


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.family")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("registry"); r.add_argument("--gateway", required=True)
    sub.add_parser("weights"); sub.add_parser("functional"); sub.add_parser("report")
    fe = sub.add_parser("fetch"); fe.add_argument("--apply", action="store_true", help="download (default: list only)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "registry":
        rows = registry(Path(a.gateway))
        have = [m for m in rows if m["asteraix_path"]]
        print(f"{len(rows)} kraken models on the gateway, {len(have)} with weights on asteraix")
    elif a.cmd == "weights":
        ms = available()
        w = weights(ms)
        w["lineages"] = lineages(w)
        w["stated_bases"] = stated_bases(w, json.loads(REGISTRY.read_text())["models"])
        (OUT / "weights.json").write_text(json.dumps(w, indent=1))
        print(f"{len(ms)} models in {len(w['groups'])} architecture groups, {len(w['pairs'])} comparable pairs; "
              f"lineages of 2+: {[g for g in w['lineages'] if len(g) > 1]}")
    elif a.cmd == "functional":
        ms = available()
        w = json.loads((OUT / "weights.json").read_text())
        f = functional(ms, w)
        (OUT / "functional.json").write_text(json.dumps(f, indent=1))
        print(f"CKA over {len(ms)} models in {len(f['groups'])} groups, {f['probe']['lines']} probe lines")
    elif a.cmd == "report":
        print(report())
    elif a.cmd == "fetch":
        plan = fetch(dry_run=not a.apply)
        for p_ in plan:
            print(f"{p_['id']:42s} {p_['doi']:26s} {p_.get('file')}  {p_.get('bytes', 0)/1e6:.1f} MB  {p_.get('verified', '')}")
        print(f"{sum(1 for p_ in plan if p_.get('file'))} files, {sum(p_.get('bytes', 0) for p_ in plan)/1e6:.0f} MB"
              + ("" if a.apply else " (list only; --apply downloads)"))



# ------------------------------------------------------------------ report
def _s(x: str) -> str:
    return x.replace("kraken-", "")


def _pair(d: dict, a: str, b: str):
    return d.get(f"{a}|{b}", d.get(f"{b}|{a}"))


def _fmt3(x):
    return '' if x is None else f'{x:.3f}'


DISTANT = (0.2, LINEAGE_COS)   # body cosine band of "distant relatives": above unrelated noise, below a recorded fine-tune


def report() -> Path:
    regd = json.loads(REGISTRY.read_text()); reg = regd["models"]
    w = json.loads((OUT / "weights.json").read_text())
    f = json.loads((OUT / "functional.json").read_text()) if (OUT / "functional.json").exists() else None
    lin = [g for g in w["lineages"] if len(g) > 1]
    lin_of = {m: i + 1 for i, g in enumerate(lin) for m in g}
    gname = {m: g["name"] for g in w["groups"] for m in g["models"]}
    cos = lambda a, b: (_pair(w["pairs"], a, b) or {}).get("body", {}).get("cosine")
    rel = lambda a, b: (cos(a, b) or 0) > LINEAGE_COS
    verdict = {r["model"]: r["verdict"] for r in w.get("stated_bases", [])}
    L = ["## Inventory: the gateway's kraken models", "",
         f"{len(reg)} kraken models served for agentic_historian (gateway, {regd['read']}); "
         f"{sum(1 for m in w['models'])} with weights on asteraix. *Arch* is the architecture group (below), *lineage* is "
         f"inferred from the weights (body cosine > {LINEAGE_COS}), *stated base* is the gateway registry's, checked against the weights.", "",
         "| model | source | stated base (check) | scripts / centuries | arch | lineage |", "|---|---|---|---|---|---:|"]
    for m in sorted(reg, key=lambda m: (gname.get(m["id"], "Z"), lin_of.get(m["id"], 99), m["id"])):
        sc = ", ".join(m.get("scripts") or []); ce = ", ".join(str(c) for c in (m.get("centuries") or []))
        sb = f"{m['stated_base']} ({verdict.get(m['id'], '')})" if m.get("stated_base") else ""
        L.append(f"| {_s(m['id'])} | {m['zenodo'] or 'in-house'} | {sb} | {sc}{' / ' if sc and ce else ''}{ce} | "
                 f"{gname.get(m['id'], '—')} | {lin_of.get(m['id'], '')} |")
    L += ["", "## Architecture groups", "",
          "Models compare element by element only inside one group: same input height and the same tensor shapes outside the "
          "output layer (a different dropout rate, which has no weights, does not split a group).", "",
          "| group | models | input height | parameter blocks | VGSL (first model) |", "|---|---:|---:|---|---|"]
    for g in w["groups"]:
        L.append(f"| {g['name']} | {len(g['models'])} | {g['height']} | {' '.join(g['blocks'])} | `{g['spec'][:150]}` |")
    L += ["", "## Lineages and stated bases", "",
          "| lineage | arch | models |", "|---:|---|---|"]
    for i, g in enumerate(lin):
        L.append(f"| {i + 1} | {gname[g[0]]} | {', '.join(_s(m) for m in g)} |")
    dist = sorted(((p["body"]["cosine"], k) for k, p in w["pairs"].items() if DISTANT[0] < p["body"]["cosine"] <= DISTANT[1]), reverse=True)
    if dist:
        L += ["", f"Distant relatives (body cosine {DISTANT[0]}–{DISTANT[1]}; unrelated pairs sit below 0.2, every recorded "
              "fine-tune above 0.6): " + "; ".join(f"{_s(k.split('|')[0])} – {_s(k.split('|')[1])} {c:.2f}" for c, k in dist) + "."]
    L += ["", "| model | stated base | body cosine | verdict |", "|---|---|---:|---|"]
    for r in w.get("stated_bases", []):
        L.append(f"| {_s(r['model'])} | {_s(r['base_model'] or r['stated_base'])} | "
                 f"{_fmt3(r.get('body_cosine'))} | {r['verdict']} |")
    for g in w["groups"]:
        rp = sorted(((k, p) for k, p in w["pairs"].items() if p["body"]["cosine"] > LINEAGE_COS and k.split("|")[0] in g["models"]),
                    key=lambda kv: -kv[1]["body"]["cosine"])
        if not rp:
            continue
        B = g["blocks"]
        L += ["", f"## Group {g['name']}: RQ1 identical parameters and RQ2 distance, related pairs", "",
              "Each cell: share of *live* parameters that are bit-identical (structural zeros excluded) / relative distance "
              "‖B − A‖ / mean(‖A‖, ‖B‖). The output layer on the characters both alphabets share.", "",
              "| pair | body cosine | " + " | ".join(B) + " |", "|---|---:|" + "---:|" * len(B)]
        for k, p in rp:
            a, b = k.split("|")
            L.append(f"| {_s(a)} – {_s(b)} | {p['body']['cosine']:.3f} | " +
                     " | ".join(f"{100*p[x]['identical']:.1f} % / {p[x]['rel_dist']:.2f}" if p.get(x, {}).get("identical") is not None else "—"
                                for x in B) + " |")
    if f:
        L += ["", "## Function: linear CKA per block, by group", "",
              f"Linear CKA between two models' outputs of a block on the same {f['probe']['lines']} probe lines "
              f"({', '.join(f['probe']['sets'])}; {f['probe']['per_set']} each, seed {f['probe']['seed']}). 1 = the same features "
              "up to unit order and scale. Means over pairs; every pair is in docs/family/functional.json."]
        for g in f["groups"]:
            if not g["cka"]:
                continue
            B = g["blocks"]
            L += ["", f"### Group {g['group']} ({len(g['models'])} models, {len(g['cka'][B[0]])} pairs)", "",
                  "| pairs | n | " + " | ".join(B) + " |", "|---|---:|" + "---:|" * len(B)]
            for label, test in (("same lineage", lambda k: rel(*k.split("|"))),
                                ("distant relatives", lambda k: DISTANT[0] < (cos(*k.split("|")) or 0) <= DISTANT[1]),
                                ("unrelated", lambda k: (cos(*k.split("|")) or 0) <= DISTANT[0])):
                ks = [k for k in g["cka"][B[0]] if test(k)]
                if ks:
                    L.append(f"| {label} | {len(ks)} | " + " | ".join(f"{np.mean([g['cka'][x][k] for k in ks]):.2f}" for x in B) + " |")
            L += ["", "Dead (silent) convolution filters per model (output 0 on every probe frame):", "",
                  "| model | " + " | ".join(b for b in B if b.startswith("C_")) + " |", "|---|" + "---:|" * sum(b.startswith("C_") for b in B)]
            for m, d in sorted(g["dead_filters"].items(), key=lambda kv: (lin_of.get(kv[0], 99), kv[0])):
                L.append(f"| {_s(m)} | " + " | ".join(f"{d[b]['dead']}/{d[b]['filters']}" for b in d) + " |")
    ex = sorted(OUT.glob("exchange-*.json"))
    if ex:
        L += ["", "## RQ3 — exchange: S's block moved to G's", ""]
    for p in ex:
        e = json.loads(p.read_text())
        L += [f"### `{_s(e['specific'])}` ← `{_s(e['general'])}` (body cosine {e['body_cosine']:.3f})", "",
              "Δ CER in points against S unedited, per set, paired 95 % interval; **bold** where it excludes 0. "
              f"First set = S's own material or the nearest available. S / G read: " +
              "; ".join(f"`{s}` {100*e['c0'][s]['cer']:.2f} % / {100*e['general_cer'][s]:.2f} %" for s in e["sets"]) + ".", "",
              "| block | β | " + " | ".join(f"`{s}`" for s in e["sets"]) + " |", "|---|---:|" + "---:|" * len(e["sets"])]
        for r in e["rows"]:
            cells = []
            for s in e["sets"]:
                v = r["sets"][s]["vs_S"]; lo, hi = v["ci95"]; bold = "**" if lo > 0 or hi < 0 else ""
                cells.append(f"{bold}{100*v['delta_cer']:+.2f}{bold} [{100*lo:+.2f}, {100*hi:+.2f}]")
            L.append(f"| {r['block']} | {r['beta']} | " + " | ".join(cells) + " |")
        L.append("")
    out = OUT / "tables.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out

if __name__ == "__main__":
    main()
