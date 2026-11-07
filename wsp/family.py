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


def output_rows(sa, sb) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """O_18 weight rows and biases of the characters both codecs share, plus the blank (row 0)."""
    rm = codec_row_map(sa.codec_map(), sb.codec_map())
    ia = [0] + sorted(rm); ib = [0] + [rm[i] for i in sorted(rm)]
    ta, tb = sa.tensors(), sb.tensors()
    w = [n for n in ta if n.startswith("O_18.") and n.endswith("weight")][0]
    bias = [n for n in ta if n.startswith("O_18.") and n.endswith("bias")][0]
    return [ta[w][ia], ta[bias][ia]], [tb[w][ib], tb[bias][ib]], len(rm)


def compare_pair(sa, sb) -> dict:
    ta, tb = sa.tensors(), sb.tensors()
    out = {}
    for b in BODY:
        A, B = _block(ta, b), _block(tb, b)
        names = sorted(set(A) & set(B))
        if any(A[n].shape != B[n].shape for n in names) or set(A) != set(B):
            out[b] = {"comparable": False}
            continue
        out[b] = _stats([A[n] for n in names], [B[n] for n in names])
    xa, xb, shared = output_rows(sa, sb)
    out["O_18"] = {**_stats(xa, xb), "shared_chars": shared}
    body_a = [v for b in BODY for k, v in sorted(_block(ta, b).items())]
    body_b = [v for b in BODY for k, v in sorted(_block(tb, b).items())]
    out["body"] = _stats(body_a, body_b)
    return out


def weights(models: list[dict]) -> dict:
    subj, info = {}, {}
    for m in models:
        s = _load(m["asteraix_path"]); subj[m["id"]] = s; d = s.describe()
        info[m["id"]] = {"params": d["parameters"], "codec": d["codec_size"], "body_spec": body_spec(d["spec"]),
                         "tensor_digest": s.digest(), "file_sha256": sha256_file(Path(m["asteraix_path"])),
                         "path": m["asteraix_path"]}
    specs = {v["body_spec"] for v in info.values()}
    pairs = {}
    for a, b in itertools.combinations(sorted(subj), 2):
        if info[a]["body_spec"] != info[b]["body_spec"]:
            pairs[f"{a}|{b}"] = {"comparable": False}
            continue
        pairs[f"{a}|{b}"] = compare_pair(subj[a], subj[b])
    return {"models": info, "body_specs": sorted(specs), "pairs": pairs,
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
    K = X @ X.T
    return K


def cka_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    """Linear CKA = HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L)) with centred features (so centred grams)."""
    hkl = float((K * L).sum()); hkk = float((K * K).sum()); hll = float((L * L).sum())
    return hkl / np.sqrt(hkk * hll) if hkk and hll else float("nan")


def functional(models: list[dict], frames: int = FRAMES) -> dict:
    items = probe_items()
    CACHE.mkdir(parents=True, exist_ok=True)
    sample_idx: dict[str, np.ndarray] = {}
    grams: dict[str, dict[str, np.ndarray]] = {b: {} for b in BLOCKS}
    dead: dict[str, dict] = {}
    frame_counts = None
    for m in models:
        s = _load(m["asteraix_path"])
        acts = s.activations(items, BLOCKS)
        counts = {b: [x.shape[0] for x in acts[b]] for b in BLOCKS}
        if frame_counts is None:
            frame_counts = counts
            rng = np.random.default_rng(SEED)
            for b in BLOCKS:
                total = sum(counts[b]); sample_idx[b] = np.sort(rng.choice(total, size=min(frames, total), replace=False))
        elif counts != frame_counts:
            raise RuntimeError(f"{m['id']}: frame counts differ from the first model; not the same architecture or input")
        dead[m["id"]] = {}
        for b in BLOCKS:
            full = np.concatenate(acts[b], axis=0)
            if b.startswith("C_"):          # Cr = convolution + ReLU: a filter that is 0 on every frame is dead
                C = s.tensors()[f"{b}.co.weight"].shape[0]
                cmax = full.reshape(full.shape[0], C, -1).max(axis=(0, 2))
                dead[m["id"]][b] = {"dead": int((cmax <= 0).sum()), "filters": int(C),
                                    "dead_index": [int(i) for i in np.flatnonzero(cmax <= 0)]}
            grams[b][m["id"]] = _gram(full[sample_idx[b]]).astype(np.float32)
        print(f"  activations {m['id']}", flush=True)
    ids = [m["id"] for m in models]
    cka = {b: {} for b in BLOCKS}
    for b in BLOCKS:
        for x, y in itertools.combinations(ids, 2):
            cka[b][f"{x}|{y}"] = cka_from_grams(grams[b][x].astype(np.float64), grams[b][y].astype(np.float64))
    return {"probe": {"sets": PROBE_SETS, "per_set": PROBE_PER_SET, "lines": len(items), "seed": SEED,
                      "keys": [i["key"] for i in items]},
            "frames": {b: int(len(sample_idx[b])) for b in BLOCKS}, "models": ids, "cka": cka, "dead_filters": dead,
            "when": time.strftime("%Y-%m-%d %H:%M")}


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
        (OUT / "weights.json").write_text(json.dumps(w, indent=1))
        print(f"{len(ms)} models, {len(w['pairs'])} pairs; lineages: {w['lineages']}")
    elif a.cmd == "functional":
        ms = available()
        f = functional(ms)
        (OUT / "functional.json").write_text(json.dumps(f, indent=1))
        print(f"CKA over {len(ms)} models, {f['probe']['lines']} probe lines")
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


def report() -> Path:
    reg = json.loads(REGISTRY.read_text())["models"]
    w = json.loads((OUT / "weights.json").read_text())
    f = json.loads((OUT / "functional.json").read_text()) if (OUT / "functional.json").exists() else None
    lin = w["lineages"]; lin_of = {m: i + 1 for i, g in enumerate(lin) for m in g}
    rel = lambda a, b: (_pair(w["pairs"], a, b) or {}).get("body", {}).get("cosine", 0) > LINEAGE_COS
    L = ["## Inventory: the gateway's kraken models", "",
         f"{len(reg)} kraken models served for agentic_historian (gateway, {json.loads(REGISTRY.read_text())['read']}); "
         f"{sum(1 for m in reg if m['asteraix_path'])} with weights on asteraix. *Lineage* is inferred from the weights "
         f"(body cosine > {LINEAGE_COS}), *stated base* is the gateway registry's.", "",
         "| model | source | stated base | scripts / centuries | weights here | lineage |", "|---|---|---|---|---|---:|"]
    for m in sorted(reg, key=lambda m: (m["asteraix_path"] is None, lin_of.get(m["id"], 99), m["id"])):
        sc = ", ".join(m.get("scripts") or []); ce = ", ".join(str(c) for c in (m.get("centuries") or []))
        L.append(f"| {_s(m['id'])} | {m['zenodo'] or 'in-house'} | {m['stated_base'] or ''} | {sc}{' / ' if sc and ce else ''}{ce} | "
                 f"{'yes' if m['asteraix_path'] else 'no'} | {lin_of.get(m['id'], '')} |")
    L += ["", "## RQ1 — identical parameters, related pairs", "",
          "Share of *live* parameters (structural zeros excluded: the LSTM input biases, 1,600 per layer, are 0 in every "
          "model) that are bit-identical between the two models, per block. Unrelated pairs are 0.00 everywhere.", "",
          "| pair | body | " + " | ".join(BLOCKS) + " |", "|---|---:|" + "---:|" * len(BLOCKS)]
    rp = sorted(((k, p) for k, p in w["pairs"].items() if p.get("body", {}).get("cosine", 0) > LINEAGE_COS),
                key=lambda kv: -kv[1]["body"]["cosine"])
    for k, p in rp:
        a, b = k.split("|")
        L.append(f"| {_s(a)} – {_s(b)} | {100*p['body']['identical']:.2f} % | " +
                 " | ".join(f"{100*p[x]['identical']:.2f}" for x in BLOCKS) + " |")
    if f:
        L += ["", "Dead (silent) convolution filters per model: output 0 on every frame of the probe lines "
              f"({f['probe']['lines']} lines, {', '.join(f['probe']['sets'])}).", "",
              "| model | " + " | ".join(b for b in BLOCKS if b.startswith('C_')) + " |", "|---|" + "---:|" * 4]
        for m, d in sorted(f["dead_filters"].items(), key=lambda kv: (lin_of.get(kv[0], 99), kv[0])):
            L.append(f"| {_s(m)} | " + " | ".join(f"{d[b]['dead']}/{d[b]['filters']}" for b in d) + " |")
    L += ["", "## RQ2 — how far apart, per block", "", "### Weight space, related pairs", "",
          "Relative distance ‖B − A‖ / mean(‖A‖, ‖B‖) per block; `O_18` on the characters both alphabets share.", "",
          "| pair | body cosine | " + " | ".join(BLOCKS) + " |", "|---|---:|" + "---:|" * len(BLOCKS)]
    for k, p in rp:
        a, b = k.split("|")
        L.append(f"| {_s(a)} – {_s(b)} | {p['body']['cosine']:.3f} | " + " | ".join(f"{p[x]['rel_dist']:.3f}" for x in BLOCKS) + " |")
    if f:
        ids = f["models"]
        L += ["", "### Function, every pair: linear CKA per block", "",
              f"Linear CKA between the two models' outputs of the block on the same {f['probe']['lines']} probe lines "
              f"({', '.join(f['probe']['sets'])}; {f['probe']['per_set']} each, seed {f['probe']['seed']}), "
              f"{f['frames']['C_0']} sampled frames per block. 1 = the same features up to unit order and scale.", "",
              "| pair | lineage | " + " | ".join(BLOCKS) + " |", "|---|---|" + "---:|" * len(BLOCKS)]
        keys = sorted(f["cka"]["C_0"], key=lambda k: (not rel(*k.split("|")), -np.mean([f["cka"][b][k] for b in BLOCKS])))
        for k in keys:
            a, b = sorted(k.split("|"))
            L.append(f"| {_s(a)} – {_s(b)} | {'same' if rel(a, b) else '—'} | " + " | ".join(f"{f['cka'][x][k]:.2f}" for x in BLOCKS) + " |")
        for grp in (True, False):
            ks = [k for k in f["cka"]["C_0"] if rel(*k.split("|")) == grp]
            if ks:
                L.append(f"| **mean, {'same lineage' if grp else 'different lineages'} ({len(ks)})** | | " +
                         " | ".join(f"**{np.mean([f['cka'][x][k] for k in ks]):.2f}**" for x in BLOCKS) + " |")
    ex = sorted(OUT.glob("exchange-*.json"))
    if ex:
        L += ["", "## RQ3 — exchange: S's block moved to G's", ""]
    for p in ex:
        e = json.loads(p.read_text())
        L += [f"### `{_s(e['specific'])}` ← `{_s(e['general'])}` (body cosine {e['body_cosine']:.3f})", "",
              "Δ CER in points against S unedited, per set, paired 95 % interval; **bold** where it excludes 0. "
              f"First set = S's own material. S / G read: " +
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
