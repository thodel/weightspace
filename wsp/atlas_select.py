"""Activation atlas, level L1: what each unit is selective for (epic thodel/weightspace#13).

    python -m wsp.atlas_select run --model models/kraken/kraken-german-xix-blb-v1.mlmodel [--workers 12]
    python -m wsp.atlas_select run --model models/kraken/kraken-bern-hertwig-v2.mlmodel
    python -m wsp.atlas_select report --out docs/atlas

The rule was fixed before the first run (docs/ATLAS_REPORT.md, "L1 rule"). In short:

- frames are labelled by the forced CTC alignment of the reference (character, letter class, position in word and line);
  Gate 1 failed, so a label says which character the *alignment* gives a frame, not where the unit looks in the image;
- per unit and label, selectivity s = 2·|AUROC − 0.5| on evaluation pages, the best label chosen on selection pages;
- claim S (unit), claim P (sparse probe per block and direction), prediction D (previous / next character by
  direction), stability against five seed runs (``hertwig-v2`` only), and Gate 2;
- controls: a randomly initialised network, characters shuffled within the line, and blank crops.

``run`` writes per-line activations of every network to ``~/atr-cache/atlas/<model>/<set>--select/`` (never into
the repository); ``report`` writes ``docs/atlas/selectivity-*.json``, the figures and the unit-card pages.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np

from wsp.atlas import CACHE, RANDOM_SEED, _safe
from wsp.atlas_ablate import BLOCKS, SALT, split_pages, unit_name

SEED = 20261114
CHAR_MIN = 50                 # runs on selection pages and on evaluation pages
TOP = 10                      # units tested per block
S_MIN = 0.5                   # claim S: s ≥ 0.5, i.e. AUROC ≥ 0.75 or ≤ 0.25
P_MIN = 0.10                  # claim P: balanced accuracy ≥ random network + 0.10
MAX_TRAIN = 20000
CS = (0.01, 0.1, 1.0)
RESAMPLES = 2000
PERMS = 1000
H = 200                       # BiLSTM units per direction

ASC, DESC, ROUND = set("bdfhkltßſ"), set("gjpqy"), set("aceo")
CLASSES = ["space", "punct", "digit", "upper", "ascender", "descender", "round"]
POSITIONS = ["word-initial", "word-internal", "word-final", "line-start", "line-end"]

SETS = {
    "kraken-german-xix-blb-v1": ("blb-val-300", ["data/practice/blb-val-300.jsonl"]),
    "kraken-bern-hertwig-v2": ("bern-schluss", ["data/practice/bern-val.jsonl", "data/practice/schluss-300.jsonl",
                                                "data/practice/schluss-conf-300.jsonl"]),
}
PARENT = {"kraken-bern-hertwig-v2": "models/kraken/kraken-cursive_inzigkofen.mlmodel"}
SEED_GLOB = {"kraken-bern-hertwig-v2": str(Path.home() / "atr-cache/wsp/seed-null/seed[1-5]/best_*.mlmodel")}


# -- lines and frame labels ------------------------------------------------------------------------------

def load_lines(model: str, practice: list[str]) -> list[dict]:
    """The practice lines the model was not trained on, duplicates counted once, in file order."""
    from wsp.data import load
    seen, out = set(), []
    for p in practice:
        if "heldout" in Path(p).parts:
            raise SystemExit("the atlas reads practice lines only")
        for x in load(p):
            if x["key"] in seen or model in (x.get("seen_by") or []):
                continue
            seen.add(x["key"]); out.append(x)
    return out


def char_class(c: str) -> list[str]:
    out = []
    if c == " ":
        return ["space"]
    if unicodedata.category(c).startswith("P"):
        out.append("punct")
    if c.isdigit():
        out.append("digit")
    if c.isupper():
        out.append("upper")
    lo = c.lower() if c != "ſ" else c
    if lo in ASC:
        out.append("ascender")
    if lo in DESC:
        out.append("descender")
    if lo in ROUND:
        out.append("round")
    return out


def word_position(ref: str, i: int) -> list[str]:
    if ref[i] == " ":
        return []
    ini = i == 0 or ref[i - 1] == " "
    fin = i == len(ref) - 1 or ref[i + 1] == " "
    return (["word-initial"] if ini else []) + (["word-final"] if fin else []) + ([] if ini or fin else ["word-internal"])


def shuffled(ref: str, rng) -> str:
    """The non-space characters permuted among the non-space positions; spaces stay."""
    idx = [i for i, c in enumerate(ref) if c != " "]
    perm = rng.permutation(len(idx))
    out = list(ref)
    for k, i in enumerate(idx):
        out[i] = ref[idx[perm[k]]]
    return "".join(out)


def frame_table(lines: list[dict], seed: int = SEED) -> dict:
    """Per frame of every line: line, page, split, character index (−1 for none), margin (0, 1 start, 2 end), and the
    characters of its run, of the shuffled line, before and after."""
    cols = {k: [] for k in ("line", "page", "split", "char", "shuf", "prev", "next", "margin", "wpos", "frame", "rel")}
    pages = sorted({ln["page"] for ln in lines})
    pid = {p: i for i, p in enumerate(pages)}
    for li, ln in enumerate(lines):
        ref, runs, n = ln["ref"], ln["runs"], ln["frames"]
        shuf = shuffled(ref, np.random.default_rng([seed, li]))
        ch = np.full(n, -1)
        for i, (a, b) in enumerate(runs):
            ch[a: b + 1] = i
        first, last = runs[0][0], runs[-1][1]
        for t in range(n):
            i = int(ch[t])
            cols["line"].append(li); cols["page"].append(pid[ln["page"]]); cols["split"].append(ln["split"])
            cols["frame"].append(t); cols["rel"].append(t / max(n - 1, 1))
            if i >= 0:
                cols["char"].append(ref[i]); cols["shuf"].append(shuf[i])
                cols["prev"].append(ref[i - 1] if i > 0 else ""); cols["next"].append(ref[i + 1] if i + 1 < len(ref) else "")
                cols["margin"].append(0); cols["wpos"].append("|".join(word_position(ref, i)))
            else:
                cols["char"].append(""); cols["shuf"].append(""); cols["prev"].append(""); cols["next"].append("")
                cols["margin"].append(1 if t < first else 2 if t > last else 0); cols["wpos"].append("")
    out = {k: np.array(v) for k, v in cols.items()}
    out["pages"] = pages
    out["run"] = out["char"] != ""
    out["labelled"] = out["run"] | (out["margin"] > 0)
    return out


def label_set(ft: dict) -> list[str]:
    """Characters (not space) and classes with ≥ CHAR_MIN runs on both sides, then the positions."""
    def runs(mask):
        # a run is counted once: frames of the same line and character index form one run; count run starts
        start = mask & np.r_[True, (ft["line"][1:] != ft["line"][:-1]) | (ft["char"][1:] != ft["char"][:-1])
                             | ~ft["run"][:-1]]
        return [int((start & (ft["split"] == s)).sum()) for s in (0, 1)]

    labels = []
    for c in sorted(set(ft["char"][ft["run"]]) - {" "}):
        if min(runs(ft["char"] == c)) >= CHAR_MIN:
            labels.append(f"char:{c}")
    for k in CLASSES:
        if min(runs(class_mask(ft, k, "char"))) >= CHAR_MIN:
            labels.append(f"class:{k}")
    return labels + [f"pos:{p}" for p in POSITIONS]


def class_mask(ft: dict, k: str, col: str) -> np.ndarray:
    chars = ft[col]
    keep = {c for c in set(chars) if c and k in char_class(c)}
    return np.isin(chars, list(keep))


def label_masks(ft: dict, labels: list[str], col: str = "char") -> tuple[np.ndarray, np.ndarray]:
    """(positives L×n, universe L×n). ``col='shuf'`` gives the shuffled-character control (positions unchanged)."""
    n = len(ft["line"])
    P = np.zeros((len(labels), n), bool); U = np.zeros((len(labels), n), bool)
    for i, lab in enumerate(labels):
        kind, v = lab.split(":", 1)
        if kind == "char":
            P[i] = ft[col] == v; U[i] = ft["run"]
        elif kind == "class":
            P[i] = class_mask(ft, v, col); U[i] = ft["run"]
        else:
            if v == "line-start":
                P[i] = ft["margin"] == 1
            elif v == "line-end":
                P[i] = ft["margin"] == 2
            else:
                P[i] = np.char.find(ft["wpos"].astype(str), v) >= 0
            U[i] = ft["labelled"]
    return P, U


# -- AUROC -----------------------------------------------------------------------------------------------

def _ranks(X: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    R = np.empty(X.shape, np.float64)
    for j in range(X.shape[1]):
        R[:, j] = rankdata(X[:, j])
    return R


def auroc(X: np.ndarray, P: np.ndarray, U: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """AUROC (labels × units) of every unit for every label, label against the rest of its universe.
    ``rows`` restricts to a subset of frames (a split). Ties count one half; a constant unit gives 0.5."""
    rows = np.ones(X.shape[0], bool) if rows is None else rows
    out = np.full((P.shape[0], X.shape[1]), 0.5)
    groups: dict[bytes, list[int]] = {}
    for li in range(P.shape[0]):
        groups.setdefault(np.packbits(U[li] & rows).tobytes(), []).append(li)
    for lis in groups.values():
        idx = np.flatnonzero(U[lis[0]] & rows)
        if not len(idx):
            continue
        R = _ranks(X[idx].astype(np.float32))
        for li in lis:
            pos = P[li, idx]
            npos, nneg = pos.sum(), (~pos).sum()
            if npos and nneg:
                out[li] = (R[pos].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)
    return out


def sel(a: np.ndarray) -> np.ndarray:
    return 2 * np.abs(np.asarray(a) - 0.5)


def page_auc(x: np.ndarray, pos: np.ndarray, pages: np.ndarray, resamples: int = RESAMPLES,
             seed: int = SEED) -> dict:
    """AUROC with a page-clustered bootstrap: pairs (positive in page p, negative in page q) are counted once per page pair,
    so a resample of pages re-weights the pair matrix without ranking again."""
    ps = np.unique(pages); k = {p: i for i, p in enumerate(ps)}
    pi = np.array([k[p] for p in pages])
    npos = np.bincount(pi[pos], minlength=len(ps)).astype(float)
    nneg = np.bincount(pi[~pos], minlength=len(ps)).astype(float)
    M = np.zeros((len(ps), len(ps)))
    xp, pp = x[pos], pi[pos]
    for q in range(len(ps)):
        neg = np.sort(x[(~pos) & (pi == q)])
        if not len(neg):
            continue
        lt = np.searchsorted(neg, xp, "left"); le = np.searchsorted(neg, xp, "right")
        M[:, q] = np.bincount(pp, weights=lt + 0.5 * (le - lt), minlength=len(ps))
    point = M.sum() / (npos.sum() * nneg.sum())
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(ps), np.full(len(ps), 1 / len(ps)), size=resamples).astype(float)
    den = (W @ npos) * (W @ nneg)
    ok = den > 0
    boot = np.einsum("bp,pq,bq->b", W[ok], M, W[ok]) / den[ok]
    s = sel(boot)
    return {"auroc": float(point), "s": float(sel(point)), "s_ci95": [float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))]}


# -- activations -------------------------------------------------------------------------------------------

def unit_values(net, img: np.ndarray) -> np.ndarray:
    """(frames, 1392): conv channels averaged over the height, BiLSTM units as they are, pooled to the output frames."""
    import torch
    from wsp.atlas import forward_blocks
    cap = forward_blocks(net, torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None],
                         BLOCKS + ["O_18"])
    frames = cap["O_18"].shape[3]
    cols = []
    for b in BLOCKS:
        o = cap[b][0].mean(1)                                          # (C, W')
        r = o.shape[1] // frames
        cols.append(o[:, : frames * r].reshape(o.shape[0], frames, r).mean(-1).T)
    return torch.cat(cols, 1).numpy()


_W: dict = {}


def _init(model_path: str, others: dict[str, str], threads: int):
    import torch
    from wsp.atlas import random_net
    from wsp.kraken_subject import KrakenSubject
    torch.set_num_threads(threads)
    s = KrakenSubject(model_path)
    nets = {"model": s.model.nn, "random": random_net(s.model.spec, RANDOM_SEED)}
    for name, p in others.items():
        nets[name] = KrakenSubject(p).model.nn
    _W.update(subject=s, nets=nets)


def _one(args):
    item, out = args
    import torch
    from wsp.atlas import _logp, forward_blocks
    from wsp.layerview import line_image, read_line
    from wsp.metrics import page_of
    s, nets = _W["subject"], _W["nets"]
    img = line_image(s, item)
    if img is None:
        return None
    o = forward_blocks(nets["model"], torch.from_numpy(np.ascontiguousarray(img))[None, None], ["O_18"])["O_18"]
    lp = _logp(o)[0].numpy()
    ln = read_line(s, item, {"logp": lp, "frames": lp.shape[0]})
    if ln is None or not ln["labels"]:
        return None
    arrays = {name: unit_values(net, img).astype(np.float16) for name, net in nets.items()}
    blank = np.full_like(img, np.median(img))
    arrays["blank"] = unit_values(nets["model"], blank).astype(np.float16)
    meta = {"key": item["key"], "page": page_of(item["key"]), "ref": ln["ref"], "hyp": ln["hyp"],
            "runs": [list(map(int, r)) for r in ln["runs"]], "frames": int(lp.shape[0])}
    np.savez(out / f"{_safe(item['key'])}.npz", meta=json.dumps(meta, ensure_ascii=False), **arrays)
    return item["key"]


def cache_dir(model: str, pset: str) -> Path:
    return CACHE / model / f"{pset}--select"


def run(model_path: str, practice: list[str] | None = None, workers: int = 12, threads: int = 4, log=print) -> Path:
    from multiprocessing import get_context
    model = Path(model_path).stem
    if model in SETS:
        pset, default = SETS[model]
    elif practice:
        pset, default = Path(practice[0]).stem, practice
    else:
        raise SystemExit(f"no default line set for {model}: pass --practice")
    items = load_lines(model, practice or default)
    sel_, ev = split_pages(items)
    side = {x["key"]: 0 for x in sel_} | {x["key"]: 1 for x in ev}
    others = {}
    if model in PARENT:
        others["parent"] = PARENT[model]
    for p in sorted(glob.glob(SEED_GLOB.get(model, "/nonexistent"))):
        others[Path(p).parent.name] = p
    out = cache_dir(model, pset); (out / "lines").mkdir(parents=True, exist_ok=True)
    log(f"{model}: {len(items)} lines ({len(sel_)} selection, {len(ev)} evaluation); networks: model, random, "
        f"{', '.join(others)}, blank crops")
    with get_context("spawn").Pool(workers, _init, (model_path, others, threads)) as pool:
        done = [k for k in pool.imap(_one, [(x, out / "lines") for x in items], chunksize=4) if k]
    meta = {"model": model, "set": pset, "practice": practice or default, "salt": SALT, "seed": SEED,
            "random_seed": RANDOM_SEED, "networks": ["model", "random", "blank", *others],
            "others": {k: Path(v).name for k, v in others.items()},
            "lines": [{"key": k, "split": side[k]} for k in done]}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    log(f"{model}: {len(done)} of {len(items)} lines aligned -> {out}")
    return out


def load(model: str, pset: str) -> tuple[dict, list[dict], dict[str, np.ndarray]]:
    """meta, lines (with split) and per network the stacked frame values (n × 1392, float16)."""
    out = cache_dir(model, pset)
    meta = json.loads((out / "meta.json").read_text())
    lines, acts = [], {k: [] for k in meta["networks"]}
    for row in meta["lines"]:
        z = np.load(out / "lines" / f"{_safe(row['key'])}.npz")
        ln = json.loads(str(z["meta"])); ln["split"] = row["split"]
        lines.append(ln)
        for k in acts:
            acts[k].append(z[k])
    return meta, lines, {k: np.concatenate(v) for k, v in acts.items()}


# -- probes ------------------------------------------------------------------------------------------------

def _probe_fit(Xtr, ytr, C, seed):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=300, tol=1e-3,
                                  random_state=seed).fit(Xtr, ytr)


def _balanced(pred, y, classes) -> float:
    acc = [np.mean(pred[y == c] == c) for c in classes if (y == c).any()]
    return float(np.mean(acc)) if acc else float("nan")


def probe(X: np.ndarray, y: np.ndarray, split: np.ndarray, page: np.ndarray, tune_half: np.ndarray,
          seed: int = SEED) -> dict:
    """Sparse probe: C chosen on alternate selection pages, refit on all selection frames, scored on evaluation frames.
    Returns predictions on the evaluation frames for the page bootstrap, the chosen C and the units used."""
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(split == 0)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd < 1e-6] = 1.0
    Z = (X - mu) / sd

    def sub(idx):
        return idx if len(idx) <= MAX_TRAIN else np.sort(rng.choice(idx, MAX_TRAIN, replace=False))

    classes = np.unique(y[tr])
    fit, tune = sub(tr[~tune_half[tr]]), tr[tune_half[tr]]
    scores = {}
    for C in CS:
        m = _probe_fit(Z[fit], y[fit], C, seed)
        scores[C] = _balanced(m.predict(Z[tune]), y[tune], classes)
    C = max(CS, key=lambda c: (scores[c], -c))
    fit = sub(tr)                                       # one draw: frames and labels must come from the same sample
    m = _probe_fit(Z[fit], y[fit], C, seed)
    ev = np.flatnonzero(split == 1)
    pred = m.predict(Z[ev])
    used = int((np.abs(m.coef_).max(0) > 0).sum())
    return {"C": C, "tune": {str(k): v for k, v in scores.items()}, "units_used": used, "units": int(X.shape[1]),
            "pred": pred, "y": y[ev], "page": page[ev], "bal": _balanced(pred, y[ev], classes)}


def bal_boot(rs: list[dict], resamples: int = RESAMPLES, seed: int = SEED) -> list[np.ndarray]:
    """Page-clustered bootstrap of balanced accuracy for one or more probes scored on the same frames
    (the same resampled pages for all of them, so differences are paired)."""
    pages = np.unique(rs[0]["page"]); pk = {p: i for i, p in enumerate(pages)}
    classes = np.unique(rs[0]["y"]); ck = {c: i for i, c in enumerate(classes)}
    pi = np.array([pk[p] for p in rs[0]["page"]]); ci = np.array([ck[c] for c in rs[0]["y"]])
    tot = np.zeros((len(pages), len(classes)))
    np.add.at(tot, (pi, ci), 1)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(pages), np.full(len(pages), 1 / len(pages)), size=resamples).astype(float)
    out = []
    for r in rs:
        cor = np.zeros_like(tot)
        np.add.at(cor, (pi, ci), (r["pred"] == r["y"]).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.nanmean((W @ cor) / (W @ tot), axis=1))
    return out


def _ci(b: np.ndarray) -> list[float]:
    return [float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))]


# -- analysis ----------------------------------------------------------------------------------------------

def block_cols(names: list[str]) -> dict[str, np.ndarray]:
    """Columns of each block, and of each direction of the BiLSTMs (``L_12 fwd``, ``L_12 rev``)."""
    out = {}
    for b in BLOCKS:
        idx = np.array([i for i, n in enumerate(names) if n.split()[0] == b])
        out[b] = idx
        if b.startswith("L"):
            out[f"{b} fwd"] = idx[:H]; out[f"{b} rev"] = idx[H:]
    return out


def unit_names() -> list[str]:
    widths = {"C_0": 32, "C_3": 32, "C_6": 64, "C_9": 64, "L_12": 2 * H, "L_14": 2 * H, "L_16": 2 * H}
    return [unit_name((b, i)) for b in BLOCKS for i in range(widths[b])]


def analyse(model: str, pset: str, workers: int = 24, log=print) -> dict:
    from joblib import Parallel, delayed
    meta, lines, acts = load(model, pset)
    names = unit_names()
    ft = frame_table(lines)
    labels = label_set(ft)
    P, U = label_masks(ft, labels)
    Ps, _ = label_masks(ft, labels, "shuf")
    s0, s1 = ft["split"] == 0, ft["split"] == 1
    log(f"{model}: {len(lines)} lines, {len(ft['line'])} frames, {len(labels)} labels")

    A = {"model_sel": auroc(acts["model"], P, U, s0), "model_ev": auroc(acts["model"], P, U, s1),
         "random_sel": auroc(acts["random"], P, U, s0), "random_ev": auroc(acts["random"], P, U, s1),
         "blank_ev": auroc(acts["blank"], P, U, s1), "shuffle_ev": auroc(acts["model"], Ps, U, s1)}
    others = [k for k in meta["networks"] if k not in ("model", "random", "blank")]
    for k in others:
        A[f"{k}_ev"] = auroc(acts[k], P, U, s1)
    log(f"{model}: AUROC done")

    cols = block_cols(names)
    # claim S
    tested, rand_top = {}, {}
    for b in BLOCKS:
        for who in ("model", "random"):
            ss = sel(A[f"{who}_sel"][:, cols[b]])
            best = ss.argmax(0)
            top = cols[b][np.argsort(-ss.max(0), kind="stable")[:TOP]]
            if who == "random":
                rand_top[b] = float(max(sel(A["random_ev"][best[list(cols[b]).index(u)], u]) for u in top))
            else:
                tested[b] = [(int(u), int(best[list(cols[b]).index(u)])) for u in top]
    claims_s = {}
    pages = ft["page"]
    for b, pairs in tested.items():
        for u, li in pairs:
            m = s1 & U[li]
            r = page_auc(acts["model"][m, u].astype(np.float64), P[li, m], pages[m])
            lo = r["s_ci95"][0]
            blank_s = float(sel(A["blank_ev"][li, u]))
            kind = labels[li].split(":")[0]
            shuf_s = float(sel(A["shuffle_ev"][li, u])) if kind != "pos" else None
            b_ok = lo > rand_top[b]
            c_ok = lo > blank_s and (shuf_s is None or lo > shuf_s)
            a_ok = r["s"] >= S_MIN
            claims_s[names[u]] = {"block": b, "label": labels[li], "sign": "on" if r["auroc"] > 0.5 else "off",
                                  **r, "selection_s": float(sel(A["model_sel"][li, u])), "random_top": rand_top[b],
                                  "blank_s": blank_s, "shuffle_s": shuf_s,
                                  "holds": bool(a_ok and b_ok and c_ok),
                                  "verdict": "holds" if a_ok and b_ok and c_ok else
                                  "inconclusive" if b_ok and c_ok else "no"}
    log(f"{model}: claim S done ({sum(v['holds'] for v in claims_s.values())} hold)")

    # probes
    keep = {lab.split(":", 1)[1] for lab in labels if lab.startswith("char:")} | ({" "} if "class:space" in labels else set())
    order = {p: i for i, p in enumerate(sorted(set(pages[s0]), key=lambda p: hashlib.sha256(
        f"{SALT}:{ft['pages'][p]}".encode()).hexdigest()))}
    tune_half = np.array([order.get(p, 0) % 2 == 1 for p in pages])
    idf = ft["run"] & np.isin(ft["char"], list(keep))
    jobs = []
    for g, c in cols.items():
        for who, col in (("model", "char"), ("random", "char"), ("shuffle", "shuf")):
            net = "random" if who == "random" else "model"
            m = idf & np.isin(ft[col], list(keep))
            jobs.append(("P", g, who, net, m, col))
    for b in ("L_12", "L_14", "L_16"):
        for d in ("fwd", "rev"):
            for tgt in ("prev", "next"):
                m = ft["run"] & np.isin(ft[tgt], list(keep))
                jobs.append(("D", f"{b} {d}", tgt, "model", m, tgt))

    calls = (delayed(probe)(acts[net][m][:, cols[g]].astype(np.float32), ft[col][m], ft["split"][m], pages[m],
                            tune_half[m]) for _, g, _, net, m, col in jobs)
    res = Parallel(n_jobs=workers, backend="loky", pre_dispatch="n_jobs")(calls)
    log(f"{model}: {len(res)} probes done")
    pr = {(k, g, w): r for (k, g, w, *_), r in zip(jobs, res)}
    claims_p = {}
    for g in cols:
        rm, rr, rs = pr[("P", g, "model")], pr[("P", g, "random")], pr[("P", g, "shuffle")]
        bm = bal_boot([rm])[0]
        lo = _ci(bm)[0]
        ok = rm["bal"] >= rr["bal"] + P_MIN and lo > rr["bal"] and lo > rs["bal"]
        claims_p[g] = {"bal": rm["bal"], "ci95_cluster": _ci(bm), "random": rr["bal"], "shuffle": rs["bal"],
                       "C": rm["C"], "units_used": rm["units_used"], "units": rm["units"],
                       "random_units_used": rr["units_used"], "classes": int(len(np.unique(rm["y"]))),
                       "holds": bool(ok) if " " not in g else None}
    pred_d = {}
    for b in ("L_12", "L_14", "L_16"):
        for tgt, better in (("prev", "fwd"), ("next", "rev")):
            worse = "rev" if better == "fwd" else "fwd"
            r1, r2 = pr[("D", f"{b} {better}", tgt)], pr[("D", f"{b} {worse}", tgt)]
            b1, b2 = bal_boot([r1, r2])
            d = b1 - b2
            pred_d[f"{b} {tgt}"] = {"better": better, "bal_better": r1["bal"], "bal_other": r2["bal"],
                                    "diff": r1["bal"] - r2["bal"], "ci95_cluster": _ci(d), "holds": bool(_ci(d)[0] > 0)}
    log(f"{model}: claim P and prediction D done")

    # stability
    stab = None
    seeds = [k for k in others if k.startswith("seed")]
    if seeds:
        prof = {k: A[f"{k}_ev"] - 0.5 for k in ["model", *others, "random"]}
        stab = {}
        rng = np.random.default_rng(SEED)
        for b in BLOCKS:
            idx = cols[b]
            row = {}
            for k in [*others, "random"]:
                C = _corr(prof["model"][:, idx], prof[k][:, idx])
                diag = np.diag(C)
                obs = float(np.nanmedian(diag))
                null = np.array([np.nanmedian(C[np.arange(len(idx)), rng.permutation(len(idx))]) for _ in range(PERMS)])
                row[k] = {"median_r": obs, "null_q99": float(np.nanquantile(null, 0.99)),
                          "above": bool(obs > np.nanquantile(null, 0.99))}
            row["holds"] = all(row[k]["above"] for k in seeds)
            stab[b] = row
        log(f"{model}: stability done")

    s_blocks = {v["block"] for v in claims_s.values() if v["holds"]}
    gate = {"P": any(v["holds"] for v in claims_p.values() if v["holds"] is not None),
            "S": bool(s_blocks),
            "stability": (any(stab[b]["holds"] for b in s_blocks) if stab else None)}
    gate["passes"] = bool(gate["P"] and gate["S"] and gate["stability"])

    cards = make_cards(model, names, labels, A, acts, ft, claims_s, stab, others)
    return {"model": model, "set": pset, "lines": len(lines),
            "selection_lines": sum(ln["split"] == 0 for ln in lines),
            "evaluation_lines": sum(ln["split"] == 1 for ln in lines),
            "selection_pages": len(set(pages[s0])), "evaluation_pages": len(set(pages[s1])),
            "frames": int(len(ft["line"])), "labels": labels, "networks": meta["networks"], "others": meta["others"],
            "claims_s": claims_s, "random_top": rand_top, "claims_p": claims_p, "prediction_d": pred_d,
            "stability": stab, "gate2": gate, "blocks": block_summary(names, labels, A, cols),
            "l3": l3_links(model, pset, names, A, labels, cols), "cards": cards}


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of a with every column of b (labels are rows); NaN for constant columns."""
    a = a - a.mean(0); b = b - b.mean(0)
    na, nb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a.T @ b) / np.outer(na, nb)


def block_summary(names, labels, A, cols) -> dict:
    """Per block: the distribution of best evaluation s (best label chosen on selection pages), trained and random,
    and the best s per label kind (character, class, position)."""
    out = {}
    for b in BLOCKS:
        idx = cols[b]
        row = {}
        for who in ("model", "random"):
            best = sel(A[f"{who}_sel"][:, idx]).argmax(0)
            ev = sel(A[f"{who}_ev"][best, idx])
            row[who] = {"median": float(np.median(ev)), "q90": float(np.quantile(ev, 0.9)), "max": float(ev.max()),
                        "ge_0_5": int((ev >= S_MIN).sum()), "values": [round(float(v), 4) for v in ev]}
            for kind in ("char", "class", "pos"):
                li = [i for i, l in enumerate(labels) if l.startswith(kind + ":")]
                row[who][f"max_{kind}"] = float(sel(A[f"{who}_ev"][li][:, idx]).max()) if li else None
        row["kinds"] = {}
        best = sel(A["model_sel"][:, idx]).argmax(0)
        for kind in ("char", "class", "pos"):
            row["kinds"][kind] = int(sum(labels[i].startswith(kind + ":") for i in best))
        out[b] = row
    return out


def l3_links(model, pset, names, A, labels, cols) -> dict | None:
    """Spearman of evaluation s (best selection label) against L3 stage-A ΔCER; selectivity of circuit units for their
    own character."""
    from wsp.atlas import _spearman
    hits = sorted(Path("docs/atlas").glob(f"ablation-{model}__*.json"))
    if not hits:
        return None
    path = hits[0]
    d = json.loads(path.read_text())
    best = sel(A["model_sel"]).argmax(0)
    s_ev = sel(A["model_ev"][best, np.arange(len(names))])
    dc = np.array([d["dcer_selection"].get(n, np.nan) for n in names])
    ok = np.isfinite(dc)
    out = {"source": path.name, "spearman_all": _spearman(s_ev[ok], dc[ok]), "by_block": {}}
    for b in BLOCKS:
        m = np.zeros(len(names), bool); m[cols[b]] = True; m &= ok
        out["by_block"][b] = _spearman(s_ev[m], dc[m])
    circ = {}
    for ch, c in d["claims"]["circuits"].items():
        lab = "class:space" if ch == " " else f"char:{ch}"
        if lab not in labels:
            continue
        li = labels.index(lab)
        own = [float(A["model_ev"][li, names.index(u)]) for u in c["units"]]
        circ[ch] = {"holds": c["holds"], "own_auroc": own, "own_s_median": float(np.median(sel(own))),
                    "all_units_s_median": float(np.median(sel(A["model_ev"][li])))}
    out["circuits"] = circ
    return out


def make_cards(model, names, labels, A, acts, ft, claims_s, stab, others) -> list[dict]:
    """One card per unit (description): best labels on evaluation pages, controls, mean activation per label kind,
    activity over the line (20 bins of relative frame position), L3 role where an L3 claim holds, seed correlations."""
    from wsp.atlas_ablate import dead_filters
    ev = ft["split"] == 1
    X = acts["model"][ev].astype(np.float32)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0
    Z = (X - mu) / sd
    bins = np.minimum((ft["rel"][ev] * 20).astype(int), 19)
    cnt = np.bincount(bins, minlength=20)
    prof = np.stack([np.bincount(bins, weights=Z[:, j], minlength=20) / np.maximum(cnt, 1) for j in range(Z.shape[1])])
    P, _ = label_masks({k: (v[ev] if isinstance(v, np.ndarray) else v) for k, v in ft.items()}, labels)
    means = (P.astype(np.float32) @ Z) / np.maximum(P.sum(1, keepdims=True), 1)        # labels × units
    role = l3_roles(model)
    dead = set(dead_filters(model))
    best_sel = sel(A["model_sel"]).argmax(0)
    corr = {}
    if stab is not None:
        for k in others:
            a, c = A["model_ev"] - 0.5, A[f"{k}_ev"] - 0.5
            a = a - a.mean(0); c = c - c.mean(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr[k] = (a * c).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(c, axis=0))
    cards = []
    for j, n in enumerate(names):
        a = A["model_ev"][:, j]
        top = np.argsort(-sel(a), kind="stable")[:5]
        li = int(best_sel[j])
        card = {"unit": n, "dead": n in dead,
                "top": [[labels[i], round(float(a[i]), 3)] for i in top],
                "best_selection": labels[li], "auroc": round(float(a[li]), 3),
                "random": round(float(A["random_ev"][li, j]), 3), "blank": round(float(A["blank_ev"][li, j]), 3),
                "shuffle": (round(float(A["shuffle_ev"][li, j]), 3) if not labels[li].startswith("pos:") else None),
                "means": [round(float(v), 2) for v in means[:, j]], "line": [round(float(v), 2) for v in prof[j]]}
        if n in claims_s:
            c = claims_s[n]
            card["tested"] = {"s": round(c["s"], 3), "ci": [round(v, 3) for v in c["s_ci95"]], "verdict": c["verdict"]}
        if n in role:
            card["l3"] = role[n]
        if corr:
            card["seeds"] = {k: (None if not np.isfinite(v[j]) else round(float(v[j]), 2)) for k, v in corr.items()}
        cards.append(card)
    return cards


def l3_roles(model: str) -> dict[str, list[str]]:
    """Units whose L3 role is a claim that holds: U, or membership of a circuit for which C holds."""
    hits = sorted(Path("docs/atlas").glob(f"ablation-{model}__*.json"))
    if not hits:
        return {}
    d = json.loads(hits[0].read_text())
    out: dict[str, list[str]] = {}
    for u, v in d["claims"]["units"].items():
        if v["holds"]:
            out.setdefault(u, []).append(f"U: ΔCER {v['delta']:+.2f} [{v['ci95_cluster'][0]:+.2f}, {v['ci95_cluster'][1]:+.2f}]")
    for ch, c in d["claims"]["circuits"].items():
        if c["holds"]:
            for u in c["units"]:
                out.setdefault(u, []).append(f"circuit of '{'␣' if ch == ' ' else ch}' (C holds)")
    return out


# -- figures and pages --------------------------------------------------------------------------------------

def plot_selectivity(data: dict, th: dict):
    """Per block: evaluation s of every unit's best label (chosen on selection pages), trained against random."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.6))
    rng = np.random.default_rng(0)
    for k, b in enumerate(BLOCKS):
        for who, dx, col in (("random", 0.2, th["muted"]), ("model", -0.2, th["C"] if b.startswith("C") else th["L"])):
            v = np.array(data["blocks"][b][who]["values"])
            ax.scatter(k + dx + rng.uniform(-0.12, 0.12, len(v)), v, s=3, color=col, alpha=0.5, linewidths=0)
        held = [c["s"] for c in data["claims_s"].values() if c["block"] == b and c["holds"]]
        ax.scatter([k - 0.2] * len(held), held, s=16, facecolors="none", edgecolors=th["hi"], linewidths=0.9)
        ax.hlines(data["random_top"][b], k + 0.05, k + 0.35, color=th["fg"], linewidth=1)
    ax.axhline(S_MIN, color=th["hi"], linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(BLOCKS)), BLOCKS)
    ax.set_ylabel("selectivity s on evaluation pages\n(best label from selection pages)", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(f"{data['model']} · {data['set']}: trained units (left), random network (right, grey); "
                 f"rings: claim S holds; bar: random network's largest tested s; dotted: s = 0.5", loc="left", fontsize=8.5)
    ax.grid(axis="y", linewidth=0.5); ax.set_axisbelow(True)
    return fig


def plot_probes(data: dict, th: dict):
    """Claim P: balanced accuracy of the sparse character probe per block and direction, with the controls;
    prediction D: previous and next character by direction."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.45})
    gs = list(data["claims_p"])
    x = np.arange(len(gs))
    p = data["claims_p"]
    m = np.array([p[g]["bal"] for g in gs]); lo = np.array([p[g]["ci95_cluster"][0] for g in gs])
    hi = np.array([p[g]["ci95_cluster"][1] for g in gs])
    cols = [th["alt"] if p[g]["holds"] is None else th["hi"] if p[g]["holds"] else th["muted"] for g in gs]
    a1.bar(x, m, color=cols, width=0.65)
    a1.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor=th["fg"], linewidth=0.8, capsize=2)
    a1.scatter(x, [p[g]["random"] for g in gs], marker="_", s=120, color=th["muted"], label="random network", zorder=3)
    a1.scatter(x, [p[g]["shuffle"] for g in gs], marker="x", s=14, color=th["fg"], label="shuffled characters", zorder=3)
    a1.set_xticks(x, [g.replace(" ", "\n") for g in gs], fontsize=7)
    a1.set_ylabel("balanced accuracy, evaluation pages", fontsize=8); a1.set_ylim(0, 1)
    h, _ = a1.get_legend_handles_labels()
    a1.legend(h + [Patch(color=th["hi"]), Patch(color=th["muted"]), Patch(color=th["alt"])],
              ["random network", "shuffled characters", "block: P holds", "block: P does not hold",
               "one direction (described, not a claim)"], fontsize=7, frameon=False, loc="upper left")
    a1.set_title(f"{data['model']}: sparse character probe, claim P", loc="left", fontsize=8.5)
    d = data["prediction_d"]
    ks = list(d)
    y = np.array([d[k]["diff"] for k in ks])
    lo = np.array([d[k]["ci95_cluster"][0] for k in ks]); hi = np.array([d[k]["ci95_cluster"][1] for k in ks])
    a2.barh(np.arange(len(ks)), y, color=[th["hi"] if d[k]["holds"] else th["muted"] for k in ks])
    a2.errorbar(y, np.arange(len(ks)), xerr=[y - lo, hi - y], fmt="none", ecolor=th["fg"], linewidth=0.8, capsize=2)
    a2.axvline(0, color=th["fg"], linewidth=0.8)
    a2.set_yticks(np.arange(len(ks)), [f"{k.split()[0]} {'previous' if k.endswith('prev') else 'next'}: "
                                       f"{d[k]['better']} − {'rev' if d[k]['better'] == 'fwd' else 'fwd'}"
                                       for k in ks], fontsize=7)
    a2.set_xlabel("difference in balanced accuracy", fontsize=8)
    a2.set_title("prediction D (highlighted: holds)", loc="left", fontsize=8.5)
    for ax in (a1, a2):
        ax.grid(axis="y" if ax is a1 else "x", linewidth=0.5); ax.set_axisbelow(True)
    return fig


def plot_stability(data: dict, th: dict):
    """Median correlation of a unit's label profile in the model with the same unit in each seed run, the parent and the
    random network, against the 99th percentile of the unit-permutation null."""
    import matplotlib.pyplot as plt
    st = data["stability"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    others = [k for k in st[BLOCKS[0]] if k != "holds"]
    for k, b in enumerate(BLOCKS):
        for o in others:
            v = st[b][o]
            col = th["L"] if o.startswith("seed") else th["hi"] if o == "parent" else th["muted"]
            mk = "o" if o.startswith("seed") else "D" if o == "parent" else "s"
            ax.scatter(k + (0.15 if o == "parent" else -0.15 if o == "random" else 0), v["median_r"], color=col,
                       marker=mk, s=18, zorder=3)
        q = max(st[b][o]["null_q99"] for o in others)
        ax.hlines(q, k - 0.3, k + 0.3, color=th["fg"], linewidth=1)
    ax.axhline(0, color=th["muted"], linewidth=0.6)
    ax.set_xticks(range(len(BLOCKS)), [f"{b}\n{'holds' if st[b]['holds'] else 'no'}" for b in BLOCKS], fontsize=8)
    ax.set_ylabel("median per-unit profile correlation", fontsize=8)
    ax.set_title(f"{data['model']}: same unit, other network (circles: five seed runs; diamond: parent; square: random); "
                 "bar: largest null 99th percentile", loc="left", fontsize=8.5)
    ax.grid(axis="y", linewidth=0.5); ax.set_axisbelow(True)
    return fig


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Unit cards</title><style>
:root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#6b6b66;--line:#d9d8d2;--card:#ffffff;--on:#1f6fb2;--off:#b2541f;--hi:#0b7a5a;--warn:#8a5a00}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#17181a;--fg:#e8e6e1;--muted:#9a988f;--line:#34353a;--card:#1f2023;--on:#6aaee8;--off:#e89a6a;--hi:#4cc59b;--warn:#e0b04a}}
:root[data-theme=dark]{--bg:#17181a;--fg:#e8e6e1;--muted:#9a988f;--line:#34353a;--card:#1f2023;--on:#6aaee8;--off:#e89a6a;--hi:#4cc59b;--warn:#e0b04a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 4px}
.banner{border-left:3px solid var(--warn);padding:6px 10px;margin:10px 0;color:var(--fg);background:var(--card)}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}select,input{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:4px;padding:3px 6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
.c{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 10px}
.c h3{margin:0;font-size:14px;display:flex;justify-content:space-between;gap:6px}.m{color:var(--muted);font-size:12px}
.tag{font-size:11px;border:1px solid var(--line);border-radius:3px;padding:0 4px;margin-left:4px}.h{color:var(--hi);border-color:var(--hi)}
table{border-collapse:collapse;width:100%;font-size:12px}td{padding:1px 2px}td.n{text-align:right;font-variant-numeric:tabular-nums}
svg{display:block;width:100%;height:34px}.l3{color:var(--hi);font-size:12px}
</style></head><body><main>
<h1 id=t></h1><div class=m id=sub></div>
<div class=banner>Activation, not location. Gate 1 failed for this model, so a label names the character that the CTC alignment gives a frame, not what the unit sees in the image. AUROC on evaluation pages; s = 2·|AUROC − 0.5|.</div>
<div class=bar><label>block <select id=blk></select></label><label>sort <select id=srt><option value=s>selectivity</option><option value=u>unit</option></select></label>
<label><input type=checkbox id=tst> tested units only</label><label><input type=checkbox id=role> L3 role only</label><input id=q placeholder="label, e.g. char:e" size=14><span class=m id=cnt></span></div>
<div class=grid id=g></div></main>
<script type=application/json id=data>__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
document.getElementById('t').textContent='Unit cards: '+D.model;
document.getElementById('sub').textContent=D.set+' · '+D.lines+' lines · '+D.cards.length+' units · Gate 2 '+(D.gate2.passes?'passes':'does not pass');
const blk=document.getElementById('blk');['all',...D.blocks].forEach(b=>{const o=document.createElement('option');o.value=o.textContent=b;blk.append(o)});
const s=a=>2*Math.abs(a-0.5),f=x=>x==null?'–':x.toFixed(2);
function spark(v){const w=200,h=34,mx=Math.max(1e-6,...v.map(Math.abs));const y=x=>h/2-x/mx*(h/2-2);
let p=v.map((x,i)=>(i?'L':'M')+(i*w/(v.length-1)).toFixed(1)+','+y(x).toFixed(1)).join('');
return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio=none aria-label="activity over the line"><line x1=0 x2=${w} y1=${h/2} y2=${h/2} stroke="var(--line)"/><path d="${p}" fill=none stroke="var(--fg)" stroke-width=1.2 vector-effect=non-scaling-stroke /></svg>`}
function card(c){const e=document.createElement('div');e.className='c';
const t=c.tested?`<span class="tag ${c.tested.verdict=='holds'?'h':''}">S ${c.tested.verdict}</span>`:'';
const rows=c.top.map(([l,a])=>`<tr><td>${l.replace(/ /g,'␣')}</td><td class=n style="color:var(${a>0.5?'--on':'--off'})">${a.toFixed(3)}</td><td class=n>s ${s(a).toFixed(2)}</td></tr>`).join('');
const seeds=c.seeds?`<div class=m>profile r with ${Object.entries(c.seeds).map(([k,v])=>k+' '+f(v)).join(', ')}</div>`:'';
e.innerHTML=`<h3><span>${c.unit}${c.dead?' <span class=tag>dead</span>':''}</span><span>${t}</span></h3>
<div class=m>best label (selection): ${c.best_selection.replace(/ /g,'␣')} · AUROC ${c.auroc.toFixed(3)}${c.tested?` · s ${c.tested.s.toFixed(2)} [${c.tested.ci[0].toFixed(2)}, ${c.tested.ci[1].toFixed(2)}]`:''}</div>
<div class=m>controls, AUROC for that label: random ${f(c.random)} · blank crop ${f(c.blank)} · shuffled ${f(c.shuffle)}</div>
<table>${rows}</table><div class=m>activity over the line (start → end, z-score)</div>${spark(c.line)}
${c.l3?`<div class=l3>L3: ${c.l3.join('; ')}</div>`:`<div class=m>L3: no single-unit effect of 1 point (inconclusive)</div>`}${seeds}`;return e}
function draw(){const b=blk.value,q=document.getElementById('q').value.trim(),tst=document.getElementById('tst').checked,rl=document.getElementById('role').checked;
let cs=D.cards.filter(c=>(b=='all'||c.unit.split(' ')[0]==b)&&(!tst||c.tested)&&(!rl||c.l3)&&(!q||c.top.some(([l])=>l.includes(q))));
if(document.getElementById('srt').value=='s')cs=cs.slice().sort((x,y)=>s(y.top[0][1])-s(x.top[0][1]));
const g=document.getElementById('g');g.replaceChildren(...cs.slice(0,300).map(card));
document.getElementById('cnt').textContent=cs.length+' units'+(cs.length>300?' (first 300 shown)':'')}
['blk','srt','tst','role','q'].forEach(i=>document.getElementById(i).addEventListener('input',draw));draw();
</script></body></html>"""


def render_cards(data: dict) -> str:
    blob = json.dumps({"model": data["model"], "set": data["set"], "lines": data["lines"], "blocks": BLOCKS,
                       "gate2": data["gate2"], "cards": data["cards"]}, ensure_ascii=False, separators=(",", ":"))
    return PAGE.replace("__DATA__", blob.replace("</", "<\\/"))


def report(pairs, out: Path, fmt: str = "svg", workers: int = 24) -> list[Path]:
    from wsp.viz import render
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    paths = []
    for model, pset in pairs:
        d = analyse(model, pset, workers)
        stem = f"{model}__{pset}"
        (out / f"units-{stem}.html").write_text(render_cards(d))
        summary = json.loads(json.dumps({k: v for k, v in d.items() if k != "cards"}))
        for row in summary["blocks"].values():
            for w in ("model", "random"):
                row[w].pop("values")
        summary["rule"] = "docs/ATLAS_REPORT.md, 'L1 rule' (fixed 14.11.2026 before any selectivity run)"
        p = out / f"selectivity-{stem}.json"
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        paths += [p, out / f"units-{stem}.html"]
        paths += render(plot_selectivity, d, out, f"selectivity-units-{stem}", fmt)
        paths += render(plot_probes, d, out, f"selectivity-probes-{stem}", fmt)
        if d["stability"]:
            paths += render(plot_stability, d, out, f"selectivity-seeds-{stem}", fmt)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.atlas_select")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--practice", nargs="*")
    r.add_argument("--workers", type=int, default=12); r.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("report"); p.add_argument("--out", default="docs/atlas"); p.add_argument("--format", default="svg")
    p.add_argument("--pair", nargs=2, action="append", metavar=("MODEL", "SET")); p.add_argument("--workers", type=int, default=24)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.model, a.practice or None, a.workers, a.threads)
    else:
        pairs = a.pair or [(m, s) for m, (s, _) in SETS.items()]
        for path in report(pairs, Path(a.out), a.format, a.workers):
            print(path)


if __name__ == "__main__":
    main()
