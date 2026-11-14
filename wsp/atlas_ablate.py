"""Activation atlas, level L3: what the output depends on, by mean ablation (epic thodel/weightspace#13).

    python -m wsp.atlas_ablate run --model models/kraken/kraken-german-xix-blb-v1.mlmodel \\
        --practice data/practice/blb-val-300.jsonl [--workers 12]
    python -m wsp.atlas_ablate report --out docs/atlas

The rule was fixed before the first run (docs/ATLAS_REPORT.md, "L3 rule"). In short:

- pages are split into selection and evaluation (``split_pages``);
- on selection pages, every one of the 1,392 units (conv channels and BiLSTM units) is mean-ablated alone
  (stage A); from that, the 20 strongest units, the 10 strongest per block and a circuit of up to 10 units
  per target character are chosen (``choose``);
- on evaluation pages, those units and groups are ablated again, together with 50 random groups of equal
  size per group (stage C), and the claims U, G and C are tested (``claims``).

A unit is ablated as the next layer sees it: a forward hook replaces the block's output for that unit (a conv
channel at every height and frame, a BiLSTM unit at every frame) by the unit's mean over the selection lines.
Many ablations of one line run as one batch, one ablation per copy. Caches go to ``~/atr-cache/atlas/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from wsp.atlas import CACHE, _safe

BLOCKS = ["C_0", "C_3", "C_6", "C_9", "L_12", "L_14", "L_16"]
SALT = "20261114"
SEED = 20261114
TOP_UNITS, TOP_PER_BLOCK, CIRCUIT, TARGETS, RANDOM_GROUPS = 20, 10, 10, 10, 50
U_MIN, C_MIN, BEAT = 1.0, 5.0, 48            # points of CER; points of a character's accuracy; of 50 random groups
RESAMPLES = 2000
BATCH = 32
PARENT = {"kraken-german-xix-blb-v1": "kraken-german-xix-v1", "kraken-bern-hertwig-v2": "kraken-bern-hertwig-v1"}


# -- lines and units -----------------------------------------------------------------------------

def split_pages(items: list[dict], salt: str = SALT) -> tuple[list[dict], list[dict]]:
    """Pages in sha256(salt:page) order, alternately to selection and evaluation."""
    from wsp.metrics import page_of
    pages = sorted({page_of(x["key"]) for x in items}, key=lambda p: hashlib.sha256(f"{salt}:{p}".encode()).hexdigest())
    side = {p: i % 2 for i, p in enumerate(pages)}
    return [x for x in items if side[page_of(x["key"])] == 0], [x for x in items if side[page_of(x["key"])] == 1]


def unit_list(widths: dict[str, int]) -> list[tuple[str, int]]:
    return [(b, i) for b in BLOCKS for i in range(widths[b])]


def unit_name(u) -> str:
    b, i = u
    if b.startswith("L"):
        return f"{b} {'fwd' if i < 200 else 'rev'} {i % 200}"
    return f"{b} ch {i}"


# -- forward passes with ablation hooks --------------------------------------------------------------

def block_means(net, imgs: list[np.ndarray], blocks=BLOCKS) -> dict[str, np.ndarray]:
    """Per unit, the mean of its output over every position of the given lines (conv: height × width)."""
    import torch
    from wsp.atlas import forward_blocks
    s: dict = {}; n: dict = {}
    for img in imgs:
        cap = forward_blocks(net, torch.from_numpy(np.ascontiguousarray(img))[None, None], list(blocks))
        for b in blocks:
            o = cap[b][0].double()                       # (C, H, W)
            s[b] = s.get(b, 0) + o.sum(dim=(1, 2)).numpy()
            n[b] = n.get(b, 0) + o.shape[1] * o.shape[2]
    return {b: (s[b] / n[b]).astype(np.float32) for b in blocks}


def read_ablated(net, img: np.ndarray, ablations: list[list[tuple[str, int]]], values: dict[str, np.ndarray],
                 l2c: dict[int, str], batch: int = BATCH) -> list[str]:
    """Greedy reading of one line under each ablation (a list of (block, unit)); ``values`` per block and unit."""
    import torch
    from wsp.layerview import greedy
    from wsp.metrics import normalise
    mods = dict(net.named_children())
    x0 = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None]
    val = {b: torch.from_numpy(np.asarray(v, np.float32)) for b, v in values.items()}
    out: list[str] = []
    cur: list = []

    def hook(name):
        def fn(_m, _i, o):
            t = o[0] if isinstance(o, (list, tuple)) else o
            for k, abl in enumerate(cur):
                idx = [i for b, i in abl if b == name]
                if idx:
                    ii = torch.tensor(idx)
                    t[k, ii] = val[name][ii][:, None, None].to(t.dtype)
            return o
        return fn

    hs = [mods[b].register_forward_hook(hook(b)) for b in values]
    try:
        for k in range(0, len(ablations), batch):
            cur[:] = ablations[k: k + batch]
            X = x0.repeat(len(cur), 1, 1, 1)
            with torch.no_grad():
                o, _ = net(X, torch.full((len(cur),), X.shape[3], dtype=torch.int64))
            lab = o[:, :, 0].argmax(1).numpy()           # (B, W'); argmax of logits = argmax of posteriors
            for row in lab:
                lp = np.full((len(row), int(row.max()) + 1), -1.0); lp[np.arange(len(row)), row] = 0.0
                out.append(normalise("".join(l2c.get(l, "") for l, _, _ in greedy(lp))))
    finally:
        for h in hs:
            h.remove()
    return out


# -- scoring -------------------------------------------------------------------------------------------

class Scorer:
    """Edits and per-character correctness of readings against the references, cached per (line, reading)."""

    def __init__(self, refs: list[str], pages: list[str]):
        self.refs, self.pages = refs, pages
        self._c: dict = {}

    def line(self, i: int, hyp: str) -> tuple[int, list[str]]:
        k = (i, hyp)
        if k not in self._c:
            from wsp.layerview import char_status
            from wsp.metrics import levenshtein
            self._c[k] = (levenshtein(self.refs[i], hyp), char_status(self.refs[i], hyp))
        return self._c[k]

    def edits(self, hyps: list[str]) -> list[tuple[int, int]]:
        return [(self.line(i, h)[0], len(self.refs[i])) for i, h in enumerate(hyps)]

    def char_counts(self, hyps: list[str], chars) -> dict[str, np.ndarray]:
        """Per character: (lines × 2) array of [correct, occurrences]."""
        out = {c: np.zeros((len(hyps), 2)) for c in chars}
        for i, h in enumerate(hyps):
            st = self.line(i, h)[1]
            for ch, s in zip(self.refs[i], st):
                if ch in out:
                    out[ch][i, 0] += s == "ok"; out[ch][i, 1] += 1
        return out

    def other_counts(self, hyps: list[str], exclude: str) -> np.ndarray:
        out = np.zeros((len(hyps), 2))
        for i, h in enumerate(hyps):
            for ch, s in zip(self.refs[i], self.line(i, h)[1]):
                if ch != exclude:
                    out[i, 0] += s == "ok"; out[i, 1] += 1
        return out

    def cer(self, hyps: list[str]) -> float:
        e = self.edits(hyps)
        return 100 * sum(a for a, _ in e) / max(1, sum(b for _, b in e))

    def delta(self, base: list[str], hyps: list[str]) -> dict:
        from wsp.metrics import bootstrap_delta
        d = bootstrap_delta(self.edits(base), self.edits(hyps), resamples=RESAMPLES, clusters=self.pages)
        return {"delta": 100 * d["delta_cer"], "ci95_cluster": [100 * v for v in d["ci95_cluster"]]}


def loss_ci(base: np.ndarray, abl: np.ndarray, pages: list[str], resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Loss of a character's accuracy (points; base − ablated) with a two-stage page-then-line bootstrap.
    ``base``, ``abl``: (lines × 2) [correct, occurrences], paired by line."""
    def acc(a):
        return 100 * a[:, 0].sum() / a[:, 1].sum() if a[:, 1].sum() else np.nan
    point = acc(base) - acc(abl)
    groups: dict = {}
    for i, p in enumerate(pages):
        groups.setdefault(p, []).append(i)
    arrs = [np.asarray(v) for v in groups.values()]
    g = np.random.default_rng(seed)
    ds = []
    for _ in range(resamples):
        idx = np.concatenate([a[g.integers(0, len(a), len(a))] for a in (arrs[k] for k in g.integers(0, len(arrs), len(arrs)))])
        ds.append(acc(base[idx]) - acc(abl[idx]))
    ds = np.array(ds)
    return {"loss": float(point), "ci95_cluster": [float(np.nanquantile(ds, 0.025)), float(np.nanquantile(ds, 0.975))]}


# -- choices on selection pages ------------------------------------------------------------------------

def choose(stage_a: dict, units: list, sc: Scorer, base: list[str]) -> dict:
    """Units, block groups and character circuits from the single-unit ablations of the selection pages."""
    names = [unit_name(u) for u in units]
    cer0 = sc.cer(base)
    dcer = {n: sc.cer(stage_a[n]) - cer0 for n in names}
    cnt = Counter(ch for r in sc.refs for ch in r)
    targets = [" "] + [c for c, _ in cnt.most_common() if c != " "][:TARGETS]
    b0 = sc.char_counts(base, targets)
    acc0 = {c: 100 * b0[c][:, 0].sum() / b0[c][:, 1].sum() for c in targets}
    loss = {c: {} for c in targets}
    for n in names:
        if stage_a[n] == base:
            continue
        a = sc.char_counts(stage_a[n], targets)
        for c in targets:
            l = acc0[c] - 100 * a[c][:, 0].sum() / a[c][:, 1].sum()
            if l > 0:
                loss[c][n] = l
    order = sorted(names, key=lambda n: (-dcer[n], names.index(n)))
    per_block = {b: [n for n in order if n.startswith(b + " ")][:TOP_PER_BLOCK] for b in BLOCKS}
    circuits = {c: [n for n, _ in sorted(loss[c].items(), key=lambda kv: (-kv[1], names.index(kv[0])))][:CIRCUIT]
                for c in targets}
    return {"cer": cer0, "dcer": dcer, "top": order[:TOP_UNITS], "per_block": per_block, "targets": targets,
            "circuits": circuits, "loss": {c: dict(sorted(v.items(), key=lambda kv: -kv[1])[:30]) for c, v in loss.items()}}


def random_groups(pool: list[str], size: int, n: int, seed: int) -> list[list[str]]:
    g = np.random.default_rng(seed)
    return [sorted(g.choice(pool, size=size, replace=False).tolist()) for _ in range(n)] if size else []


def stage_c_plan(ch: dict, names: list[str]) -> dict[str, list[str]]:
    """Every ablation of the evaluation pages, by name: single top units, block groups, circuits, random groups."""
    plan: dict[str, list[str]] = {}
    for n in ch["top"]:
        plan[f"unit:{n}"] = [n]
    for k, (b, grp) in enumerate(ch["per_block"].items()):
        plan[f"group:{b}"] = grp
        pool = [n for n in names if n.startswith(b + " ")]
        for j, r in enumerate(random_groups(pool, len(grp), RANDOM_GROUPS, SEED + 100 * k)):
            plan[f"rgroup:{b}:{j}"] = r
    sizes = sorted({len(c) for c in ch["circuits"].values()} - {0})
    for c, grp in ch["circuits"].items():
        if grp:
            plan[f"circuit:{c}"] = grp
    for s in sizes:
        for j, r in enumerate(random_groups(names, s, RANDOM_GROUPS, SEED + 10_000 + s)):
            plan[f"rcircuit:{s}:{j}"] = r
    for b in BLOCKS:
        plan[f"block-mean:{b}"] = [n for n in names if n.startswith(b + " ")]
    return plan


# -- claims on evaluation pages ---------------------------------------------------------------------

def claims(ch: dict, ev: dict, sc: Scorer, base: list[str]) -> dict:
    out: dict = {"units": {}, "groups": {}, "circuits": {}, "blocks": {}}
    for n in ch["top"]:
        d = sc.delta(base, ev[f"unit:{n}"])
        out["units"][n] = {**d, "selection_dcer": ch["dcer"][n],
                           "holds": bool(d["delta"] >= U_MIN and d["ci95_cluster"][0] > 0)}
    cer0 = sc.cer(base)
    for b in BLOCKS:
        d = sc.delta(base, ev[f"group:{b}"])
        rnd = sorted(sc.cer(ev[f"rgroup:{b}:{j}"]) - cer0 for j in range(RANDOM_GROUPS))
        beaten = sum(d["delta"] > r for r in rnd)
        out["groups"][b] = {**d, "units": ch["per_block"][b], "random_median": float(np.median(rnd)),
                            "random_max": float(max(rnd)), "beats": int(beaten),
                            "holds": bool(d["delta"] >= U_MIN and d["ci95_cluster"][0] > 0 and beaten >= BEAT)}
        out["blocks"][b] = {"mean": sc.cer(ev[f"block-mean:{b}"]) - cer0, "zero": sc.cer(ev[f"block-zero:{b}"]) - cer0}
    targets = ch["targets"]
    b0 = sc.char_counts(base, targets)
    for c, grp in ch["circuits"].items():
        if not grp:
            out["circuits"][c] = {"units": [], "holds": False, "note": "no unit lowers this character on selection pages"}
            continue
        a = sc.char_counts(ev[f"circuit:{c}"], [c])[c]
        lc = loss_ci(b0[c], a, sc.pages)
        acc = lambda m: 100 * m[:, 0].sum() / m[:, 1].sum()
        rnd = sorted(acc(b0[c]) - acc(sc.char_counts(ev[f"rcircuit:{len(grp)}:{j}"], [c])[c]) for j in range(RANDOM_GROUPS))
        o0, o1 = sc.other_counts(base, c), sc.other_counts(ev[f"circuit:{c}"], c)
        other = acc(o0) - acc(o1)
        beaten = sum(lc["loss"] > r for r in rnd)
        out["circuits"][c] = {"units": grp, **lc, "occurrences": int(b0[c][:, 1].sum()), "accuracy": acc(b0[c]),
                              "random_median": float(np.median(rnd)), "random_max": float(max(rnd)), "beats": int(beaten),
                              "other_loss": float(other), "dcer": sc.cer(ev[f"circuit:{c}"]) - cer0,
                              "blocks": dict(Counter(n.split()[0] for n in grp)),
                              "holds": bool(lc["loss"] >= C_MIN and lc["ci95_cluster"][0] > 0 and beaten >= BEAT
                                            and lc["loss"] >= 2 * max(other, 0) and other < lc["loss"])}
    return out


# -- weight change and dead filters (descriptions) -------------------------------------------------------

def weight_change(child: dict[str, np.ndarray], parent: dict[str, np.ndarray]) -> dict[str, float]:
    """Relative change ‖Δ‖/‖parent‖ of each unit's incoming weights and biases in the last fine-tune."""
    out = {}
    for b in BLOCKS:
        if b.startswith("C"):
            w0, w1 = parent[f"{b}.co.weight"], child[f"{b}.co.weight"]
            b0, b1 = parent[f"{b}.co.bias"], child[f"{b}.co.bias"]
            for c in range(w0.shape[0]):
                p = np.concatenate([w0[c].ravel(), b0[c:c + 1]]); q = np.concatenate([w1[c].ravel(), b1[c:c + 1]])
                out[unit_name((b, c))] = float(np.linalg.norm(q - p) / (np.linalg.norm(p) or 1))
        else:
            for d, suf in ((0, ""), (1, "_reverse")):
                keys = [f"{b}.layer.{k}_l0{suf}" for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
                H = parent[keys[1]].shape[1]
                for j in range(H):
                    rows = [j, H + j, 2 * H + j, 3 * H + j]
                    p = np.concatenate([parent[k][rows].ravel() for k in keys])
                    q = np.concatenate([child[k][rows].ravel() for k in keys])
                    out[unit_name((b, d * H + j))] = float(np.linalg.norm(q - p) / (np.linalg.norm(p) or 1))
    return out


def dead_filters(model: str, path: Path = Path("docs/family/functional.json")) -> list[str]:
    d = json.loads(Path(path).read_text())
    for g in d["groups"]:
        if model in g.get("dead_filters", {}):
            return [unit_name((b, i)) for b, v in g["dead_filters"][model].items() for i in v["dead_index"]]
    return []


# -- running (cache) -----------------------------------------------------------------------------------

_W: dict = {}


def _init(model_path: str, threads: int, values: dict, l2c: dict):
    import torch
    from wsp.kraken_subject import KrakenSubject
    torch.set_num_threads(threads)
    _W.update(subject=KrakenSubject(model_path), values=values, l2c=l2c)


def _one(args):
    item, abls = args
    from wsp.layerview import line_image
    img = line_image(_W["subject"], item)
    if img is None:
        return item["key"], None
    return item["key"], read_ablated(_W["subject"].model.nn, img, abls, _W["values"], _W["l2c"])


def _parse(name: str) -> tuple[str, int]:
    parts = name.split()
    if parts[1] == "ch":
        return parts[0], int(parts[2])
    return parts[0], int(parts[2]) + (200 if parts[1] == "rev" else 0)


def run_set(model_path: str, items: list[dict], named: dict[str, list[str]], values: dict, l2c: dict,
            workers: int, threads: int) -> dict[str, list[str]]:
    """Readings of every line under every named ablation (plus 'base'); lines the transform cannot scale are empty."""
    from multiprocessing import get_context
    names = ["base"] + list(named)
    abls = [[]] + [[_parse(n) for n in named[k]] if not k.startswith("block-zero:") else [] for k in named]
    res: dict[str, list[str]] = {k: [] for k in names}
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(model_path, threads, values, l2c)) as pool:
        for key, hyps in pool.imap(_one, [(x, abls) for x in items]):
            for k, h in zip(names, hyps or [""] * len(names)):
                res[k].append(h)
    zero = [k for k in named if k.startswith("block-zero:")]
    if zero:                                               # zero ablation: the same hooks with zeros for that block
        zvals = {b: np.zeros_like(v) for b, v in values.items()}
        named_z = {k: [unit_name((k.split(":")[1], i)) for i in range(len(values[k.split(":")[1]]))] for k in zero}
        abls_z = [[_parse(n) for n in named_z[k]] for k in zero]
        with get_context("spawn").Pool(workers, initializer=_init, initargs=(model_path, threads, zvals, l2c)) as pool:
            out = {k: [] for k in zero}
            for key, hyps in pool.imap(_one, [(x, abls_z) for x in items]):
                for k, h in zip(zero, hyps or [""] * len(zero)):
                    out[k].append(h)
        res.update(out)
    return res


def run(model_path: str, practice: str, workers: int = 12, threads: int = 4, log=print) -> Path:
    import torch
    from wsp.data import load
    from wsp.kraken_subject import KrakenSubject
    from wsp.layerview import line_image
    from wsp.metrics import normalise, page_of
    if "heldout" in Path(practice).parts:
        raise SystemExit("the atlas reads practice lines only")
    model, pset = Path(model_path).stem, Path(practice).stem
    out = CACHE / model / f"{pset}--ablate"; out.mkdir(parents=True, exist_ok=True)
    items = [x for x in load(practice) if model not in (x.get("seen_by") or [])]
    sel, ev = split_pages(items)
    s = KrakenSubject(model_path)
    torch.set_num_threads(workers * threads)
    imgs = [im for im in (line_image(s, x) for x in sel) if im is not None]
    values = block_means(s.model.nn, imgs)
    widths = {b: len(v) for b, v in values.items()}
    units = unit_list(widths)
    names = [unit_name(u) for u in units]
    l2c = {v: k for k, v in s.codec_map().items()}
    log(f"{model}: {len(sel)} selection lines, {len(ev)} evaluation lines, {len(units)} units")
    a = run_set(model_path, sel, {n: [n] for n in names}, values, l2c, workers, threads)
    sc_s = Scorer([normalise(x["text"]) for x in sel], [page_of(x["key"]) for x in sel])
    ch = choose(a, units, sc_s, a["base"])
    log(f"{model}: stage A done, selection CER {ch['cer']:.2f}")
    plan = stage_c_plan(ch, names)
    for b in BLOCKS:
        plan[f"block-zero:{b}"] = []
    c = run_set(model_path, ev, plan, values, l2c, workers, threads)
    meta = {"model": model, "set": pset, "selection": [x["key"] for x in sel], "evaluation": [x["key"] for x in ev],
            "salt": SALT, "seed": SEED, "units": names}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    (out / "stage_a.json").write_text(json.dumps(a, ensure_ascii=False))
    (out / "choices.json").write_text(json.dumps(ch, ensure_ascii=False))
    (out / "stage_c.json").write_text(json.dumps({"plan": plan, "readings": c}, ensure_ascii=False))
    np.savez(out / "means.npz", **values)
    log(f"{model}: stage C done -> {out}")
    return out


# -- report ------------------------------------------------------------------------------------------------

def summarise(model: str, pset: str, models_dir: Path = Path("models/kraken")) -> dict:
    from wsp.data import load
    from wsp.kraken_subject import KrakenSubject
    from wsp.metrics import normalise, page_of
    d = CACHE / model / f"{pset}--ablate"
    meta = json.loads((d / "meta.json").read_text())
    a = json.loads((d / "stage_a.json").read_text())
    ch = json.loads((d / "choices.json").read_text())
    c = json.loads((d / "stage_c.json").read_text())["readings"]
    items = {x["key"]: x for x in load(f"data/practice/{pset}.jsonl")}
    ev = [items[k] for k in meta["evaluation"]]
    sel = [items[k] for k in meta["selection"]]
    sc_e = Scorer([normalise(x["text"]) for x in ev], [page_of(x["key"]) for x in ev])
    cl = claims(ch, c, sc_e, c["base"])
    names = meta["units"]
    # descriptions
    dead = dead_filters(model)
    sc_s = Scorer([normalise(x["text"]) for x in sel], [page_of(x["key"]) for x in sel])
    dead_eff = {n: ch["dcer"][n] for n in dead if n in ch["dcer"]}
    dead_same = {n: a[n] == a["base"] for n in dead if n in a}
    wc = {}
    parent = PARENT.get(model)
    if parent:
        wc = weight_change(KrakenSubject(models_dir / f"{model}.mlmodel").tensors(),
                           KrakenSubject(models_dir / f"{parent}.mlmodel").tensors())
    from wsp.atlas import _spearman
    rho = {b: _spearman(np.array([ch["dcer"][n] for n in names if n.startswith(b + " ")]),
                        np.array([wc[n] for n in names if n.startswith(b + " ")])) for b in BLOCKS} if wc else {}
    rho_all = _spearman(np.array([ch["dcer"][n] for n in names]), np.array([wc[n] for n in names])) if wc else None
    dist = {b: {"changed": int(sum(a[n] != a["base"] for n in names if n.startswith(b + " "))),
                "units": int(sum(n.startswith(b + " ") for n in names)),
                "dcer_max": float(max(ch["dcer"][n] for n in names if n.startswith(b + " "))),
                "dcer_ge_1": int(sum(ch["dcer"][n] >= 1 for n in names if n.startswith(b + " "))),
                "dcer_sum": float(sum(ch["dcer"][n] for n in names if n.startswith(b + " ")))} for b in BLOCKS}
    return {"model": model, "set": pset, "selection_lines": len(sel), "evaluation_lines": len(ev),
            "selection_pages": len(set(sc_s.pages)), "evaluation_pages": len(set(sc_e.pages)),
            "selection_cer": ch["cer"], "evaluation_cer": sc_e.cer(c["base"]), "stage_a": dist,
            "dcer_selection": {n: round(ch["dcer"][n], 4) for n in names},
            "claims": cl, "dead_filters": {"units": dead, "dcer": dead_eff, "reading_unchanged": dead_same},
            "weight_change": {"parent": parent, "spearman_all": rho_all, "spearman_by_block": rho,
                              "top_units": {n: wc.get(n) for n in ch["top"]}},
            "rule": {"top_units": TOP_UNITS, "per_block": TOP_PER_BLOCK, "circuit": CIRCUIT, "targets": TARGETS,
                     "random_groups": RANDOM_GROUPS, "u_min": U_MIN, "c_min": C_MIN, "beat": BEAT, "salt": SALT, "seed": SEED}}


def plot_units(data: dict, th: dict):
    """Single-unit ablation on selection pages: ΔCER of every unit, by block."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 3.8))
    x0 = 0
    for b in BLOCKS:
        v = np.array([d for n, d in data["dcer_selection"].items() if n.startswith(b + " ")])
        col = th["C"] if b.startswith("C") else th["L"]
        ax.bar(np.arange(x0, x0 + len(v)), v, width=1.0, color=col, linewidth=0)
        ax.text(x0 + len(v) / 2, 1.0, b, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        if b.startswith("L"):
            ax.axvline(x0 + len(v) / 2 - 0.5, color=th["grid"], linewidth=0.6)
        ax.axvline(x0 - 0.5, color=th["muted"], linewidth=0.6)
        x0 += len(v)
    ax.axhline(U_MIN, color=th["hi"], linewidth=0.8, linestyle=":")
    ax.set_xlim(-1, x0); ax.set_xticks([])
    ax.set_ylabel("ΔCER (points) when the unit\nis mean-ablated alone", fontsize=8)
    ax.set_xlabel("units (conv channels, then BiLSTM units: forward | reverse)", fontsize=8)
    ax.set_title(f"{data['model']}: selection pages ({data['selection_lines']} lines, CER {data['selection_cer']:.1f} %); "
                 f"dotted: 1 point", loc="left", fontsize=9, pad=16)
    ax.grid(axis="y", linewidth=0.5); ax.set_axisbelow(True)
    return fig


def plot_circuits(data: dict, th: dict):
    """Evaluation pages: loss of each target character under its circuit, against random groups and other characters."""
    import matplotlib.pyplot as plt
    from wsp.viz import glyph
    cs = [c for c, v in data["claims"]["circuits"].items() if v.get("units")]
    fig, ax = plt.subplots(figsize=(1.0 * len(cs) + 2.5, 3.8))
    x = np.arange(len(cs))
    v = data["claims"]["circuits"]
    ax.bar(x, [v[c]["loss"] for c in cs], color=[th["O"] if v[c]["holds"] else th["muted"] for c in cs], width=0.6,
           label="loss of the character under its circuit")
    ax.errorbar(x, [v[c]["loss"] for c in cs], yerr=np.array([[v[c]["loss"] - v[c]["ci95_cluster"][0], v[c]["ci95_cluster"][1] - v[c]["loss"]] for c in cs]).T,
                fmt="none", ecolor=th["fg"], elinewidth=0.8, capsize=2)
    ax.scatter(x, [v[c]["random_max"] for c in cs], marker="_", s=200, color=th["hi"], label="largest of 50 random groups", zorder=3)
    ax.scatter(x, [v[c]["other_loss"] for c in cs], marker="x", s=25, color=th["C"], label="loss of all other characters", zorder=3)
    ax.axhline(C_MIN, color=th["hi"], linewidth=0.8, linestyle=":")
    ax.set_xticks(x); ax.set_xticklabels([glyph(c) for c in cs], fontsize=10)
    ax.set_ylabel("loss of accuracy (points)", fontsize=8)
    ax.set_title(f"{data['model']}: character circuits on evaluation pages (green: claim C holds)", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(axis="y", linewidth=0.5); ax.set_axisbelow(True)
    return fig


def report(pairs, out: Path, fmt: str = "svg") -> list[Path]:
    from wsp.viz import render
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    written = []
    for model, pset in pairs:
        s = summarise(model, pset)
        p = out / f"ablation-{model}__{pset}.json"
        p.write_text(json.dumps(s, ensure_ascii=False, indent=1))
        written.append(p)
        written += render(plot_units, s, out, f"ablation-units-{model}__{pset}", fmt)
        written += render(plot_circuits, s, out, f"ablation-circuits-{model}__{pset}", fmt)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.atlas_ablate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--practice", required=True)
    r.add_argument("--workers", type=int, default=12); r.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("report"); p.add_argument("--out", default="docs/atlas"); p.add_argument("--format", default="svg")
    p.add_argument("--pair", nargs=2, action="append", metavar=("MODEL", "SET"))
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(run(a.model, a.practice, a.workers, a.threads))
    else:
        pairs = a.pair or [("kraken-german-xix-blb-v1", "blb-val-300"), ("kraken-bern-hertwig-v2", "bern-val")]
        for x in report(pairs, Path(a.out), a.format):
            print(x)


if __name__ == "__main__":
    main()
