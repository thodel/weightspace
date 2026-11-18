"""Activation atlas of a kraken recogniser, levels L0 and L2 (epic thodel/weightspace#13).

    python -m wsp.atlas run --model models/kraken/kraken-german-xix-blb-v1.mlmodel \\
        --practice data/practice/blb-val-300.jsonl [--n 80] [--workers 12] [--random]
    python -m wsp.atlas report --out docs/atlas

**L0, the activation map.** Every block's output per line on the frame grid of ``wsp.layerview``
(one frame = ``px_per_frame`` = 8 px of the line): conv blocks as channels × frames (mean over the
remaining height), BiLSTMs as 400 units × frames (forward 0–199, reverse 200–399), the output layer
as the emission track. A viewer, not a claim.

**L2, dependence on the image.** A window of ``WINDOW`` px (full line height) is set to the line's
background (its median pixel) and slid over the line in steps of ``STEP`` px. Recorded per window:

- per block and frame, the relative change of the frame's activation
  (``‖a_occluded − a‖ / mean over frames of ‖a‖``): the *measured receptive field* of the block, against
  the one computed from the VGSL spec (``spec_rf``) and against a randomly initialised network;
- per reference character, the drop of its score ``s_i = max over its forced CTC run ± TOL frames of
  log p(label)``: the character's *occlusion profile*; its peak is where in the image the model's evidence
  for that character lies;
- gradient × (input − background) of ``s_i``, summed per window: the first-order estimate of the same drop,
  as the one gradient method the epic asks to check against occlusion.

Gate 1 (rule fixed before the first run, docs/ATLAS_REPORT.md): a character is a *hit* when its occlusion
peak is nearer to its own CTC run centre than to any other character's. Pass: hit share ≥ 0.80 with the
lower end of the page-clustered 95 % interval ≥ 0.75.

Caches (activations, profiles, the line as the network sees it) go to ``~/atr-cache/atlas/`` and never
into the repository; ``report`` writes summaries, figures and the viewer to ``docs/atlas``. CPU only.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

import numpy as np

CACHE = Path.home() / "atr-cache" / "atlas"
BLOCKS = ["C_0", "C_3", "C_6", "C_9", "L_12", "L_14", "L_16", "O_18"]
WINDOW, STEP = 16, 4          # px
TOL = 2                       # frames either side of the CTC run that count for a character's score
MIN_DROP = 0.1                # nat; a character whose largest drop is smaller is a miss
SALT = "20261113"
GATE_SHARE, GATE_LOW = 0.80, 0.75
RESAMPLES = 2000
VIEW_LINES = 6
RANDOM_SEED = 20261113


# -- receptive field from the spec ---------------------------------------------------------

_TOK = re.compile(r"(Cr|Mp|Lbx|Lfx|Lrx|O|S|Do)\{(\w+)\}([\d.,]*)")


def spec_rf(spec: str) -> dict[str, dict]:
    """Horizontal receptive field (px) and jump (px between neighbouring columns) of every named block,
    computed from the VGSL spec: a conv of kernel width k adds (k − 1)·jump, a max-pool of width k adds
    (k − 1)·jump and multiplies the jump by its stride. Recurrent and output blocks see the whole line (rf None)."""
    rf, jump, out = 1, 1, {}
    for kind, name, args in _TOK.findall(spec):
        nums = [float(a) for a in args.split(",") if a]
        if kind == "Cr":
            rf += (int(nums[1]) - 1) * jump
            out[name] = {"rf": rf, "jump": jump}
        elif kind == "Mp":
            kw = int(nums[1]); sw = int(nums[3]) if len(nums) >= 4 else kw
            rf += (kw - 1) * jump; jump *= sw
        elif kind in ("Lbx", "Lfx", "Lrx", "O"):
            out[name] = {"rf": None, "jump": jump}
    return out


def rf_bound(rf: int, jump: int, px_per_frame: int, window: int = WINDOW) -> float:
    """Largest |window centre − frame centre| (px) at which a window can change a conv block's frame:
    a frame pools px_per_frame / jump columns, whose fields together span rf + px_per_frame − jump px."""
    return (rf + px_per_frame - jump + window) / 2


# -- lines ----------------------------------------------------------------------------------

def sample_lines(items: list[dict], model: str, n: int, min_chars: int = 15, salt: str = SALT) -> list[dict]:
    """Practice lines the model was not trained on (``seen_by``), at least ``min_chars`` long, in
    sha256(salt:key) order; the caller takes the first n the transform can scale."""
    keep = [x for x in items if model not in (x.get("seen_by") or []) and len(x["text"].strip()) >= min_chars]
    return sorted(keep, key=lambda x: hashlib.sha256(f"{salt}:{x['key']}".encode()).hexdigest())


def windows(width: int, window: int = WINDOW, step: int = STEP) -> list[tuple[int, int, float]]:
    """(first column, end column, centre px) of every window position; clipped at the line's ends."""
    out = []
    for c in range(0, width + 1, step):
        lo, hi = max(0, c - window // 2), min(width, c + window // 2)
        if hi > lo:
            out.append((lo, hi, (lo + hi) / 2))
    return out


# -- forward passes (any torch module called as net(x, lens) -> (out, lens), with named blocks) ----

def _frames(o, frames: int):
    """(B, C, H, W') block output → (B, frames, C·H), mean over the columns of each frame (as layerview.to_frames)."""
    B, C, H, W = o.shape
    r = W // frames
    o = o[..., : frames * r].reshape(B, C, H, frames, r).mean(-1)
    return o.permute(0, 3, 1, 2).reshape(B, frames, C * H)


def forward_blocks(net, x, blocks: list[str]) -> dict:
    """One forward pass of a batch (B, 1, H, W); returns every named block's output (B, C, H, W')."""
    import torch
    mods = dict(net.named_children())
    cap: dict = {}

    def hook(name):
        def fn(_m, _i, out):
            cap[name] = (out[0] if isinstance(out, (list, tuple)) else out).detach()
        return fn

    hs = [mods[b].register_forward_hook(hook(b)) for b in blocks]
    try:
        with torch.no_grad():
            net(x, torch.full((x.shape[0],), x.shape[3], dtype=torch.int64))
    finally:
        for h in hs:
            h.remove()
    return cap


def _logp(o):
    """Output block (B, C, 1, W') → log posteriors (B, W', C)."""
    import torch
    return torch.log_softmax(o[:, :, 0].transpose(1, 2).double(), dim=-1)


def char_windows(runs, frames: int, tol: int = TOL) -> list[tuple[int, int]]:
    return [(max(0, a - tol), min(frames - 1, b + tol)) for a, b in runs]


def scores(logp, labels: list[int], spans: list[tuple[int, int]]):
    """(B, frames, C) → (B, characters): the best log p of each character's label over its span."""
    import torch
    if not labels:
        return torch.zeros((logp.shape[0], 0), dtype=logp.dtype)
    return torch.stack([logp[:, lo: hi + 1, l].max(dim=1).values for l, (lo, hi) in zip(labels, spans)], dim=1)


def occlude(net, img: np.ndarray, blocks: list[str], labels=None, runs=None, window: int = WINDOW,
            step: int = STEP, batch: int = 16) -> dict:
    """The L2 sweep of one line. ``img`` is (H, W) as the network sees it. With ``labels`` and ``runs``
    (forced CTC alignment) also the per-character score drops. Returns numpy arrays."""
    import torch
    H, W = img.shape
    fill = float(np.median(img))
    x0 = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None]
    base = forward_blocks(net, x0, blocks)
    frames = base[blocks[-1]].shape[-1]
    bf = {b: _frames(base[b], frames)[0] for b in blocks}
    norm = {b: float(bf[b].norm(dim=1).mean()) or 1.0 for b in blocks}
    wins = windows(W, window, step)
    delta = {b: np.zeros((len(wins), frames), np.float32) for b in blocks}
    out = {"frames": frames, "px_per_frame": W // frames, "width": W, "fill": fill,
           "centres": np.array([c for _, _, c in wins], np.float32), "delta": delta,
           "heat": {b: base[b].mean(2)[0, :, : frames * (base[b].shape[-1] // frames)]
                    .reshape(base[b].shape[1], frames, -1).mean(-1).T.numpy().astype(np.float32) for b in blocks}}
    with_chars = labels is not None and runs is not None
    if with_chars:
        spans = char_windows(runs, frames)
        lp0 = _logp(base[blocks[-1]])
        out["logp"] = lp0[0].numpy().astype(np.float32)
        s0 = scores(lp0, labels, spans)[0].numpy()
        drop = np.zeros((len(labels), len(wins)), np.float32)
    for k in range(0, len(wins), batch):
        chunk = wins[k: k + batch]
        X = x0.repeat(len(chunk), 1, 1, 1)
        for j, (lo, hi, _) in enumerate(chunk):
            X[j, 0, :, lo:hi] = fill
        cap = forward_blocks(net, X, blocks)
        for b in blocks:
            d = (_frames(cap[b], frames) - bf[b][None]).norm(dim=2) / norm[b]
            delta[b][k: k + len(chunk)] = d.numpy()
        if with_chars:
            drop[:, k: k + len(chunk)] = (s0[None] - scores(_logp(cap[blocks[-1]]), labels, spans).numpy()).T
    if with_chars:
        out["score"] = s0.astype(np.float32)
        out["drop"] = drop
    return out


def gradient(net, img: np.ndarray, labels: list[int], runs, out_block: str = "O_18",
             window: int = WINDOW, step: int = STEP) -> np.ndarray:
    """Gradient × (input − background) of each character's score, summed over each occlusion window:
    the first-order estimate of the occlusion drop. Returns (characters, windows)."""
    import torch
    H, W = img.shape
    fill = float(np.median(img))
    x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None].requires_grad_(True)
    mods = dict(net.named_children())
    cap: dict = {}
    h = mods[out_block].register_forward_hook(lambda _m, _i, o: cap.__setitem__("o", o[0] if isinstance(o, (list, tuple)) else o))
    try:
        net(x, torch.tensor([W], dtype=torch.int64))
    finally:
        h.remove()
    lp = _logp(cap["o"])
    frames = lp.shape[1]
    s = scores(lp, labels, char_windows(runs, frames))[0]
    wins = windows(W, window, step)
    cs = np.zeros((len(labels), W + 1))
    for i in range(len(labels)):
        g, = torch.autograd.grad(s[i], x, retain_graph=i < len(labels) - 1)
        col = (g[0, 0] * (x.detach()[0, 0] - fill)).sum(0).double().numpy()
        cs[i, 1:] = np.cumsum(col)
    return np.stack([cs[:, hi] - cs[:, lo] for lo, hi, _ in wins], axis=1).astype(np.float32)


def random_net(spec: str, seed: int = RANDOM_SEED):
    """A randomly initialised network of the same VGSL spec (the control of the epic)."""
    import torch
    from kraken.lib.vgsl import TorchVGSLModel
    torch.manual_seed(seed)
    m = TorchVGSLModel(vgsl=spec)
    m.init_weights()
    m.nn.eval()
    return m.nn


# -- analysis -------------------------------------------------------------------------------------

def centres_px(runs, px_per_frame: int) -> np.ndarray:
    return np.array([(a + b + 1) / 2 * px_per_frame for a, b in runs], np.float64)


def char_rows(rec: dict) -> list[dict]:
    """Per reference character: CTC centre, occlusion peak, gradient peak, hit (Gate 1 rule)."""
    ctr = centres_px(rec["runs"], rec["px_per_frame"])
    xs = np.asarray(rec["centres"], np.float64)
    rows = []
    for i, (ch, st) in enumerate(zip(rec["ref"], rec["status"])):
        d = np.asarray(rec["drop"][i], np.float64)
        p = int(d.argmax()); peak = float(xs[p])
        own = abs(peak - ctr[i]); others = np.delete(np.abs(peak - ctr), i)
        near = int(np.abs(peak - ctr).argmin())
        row = {"char": ch, "status": st, "centre": float(ctr[i]), "peak": peak, "max_drop": float(d[p]),
               "offset": peak - float(ctr[i]), "nearest": near,
               "hit": bool(d[p] >= MIN_DROP and (others.size == 0 or own < others.min()))}
        if "grad" in rec:
            g = np.asarray(rec["grad"][i], np.float64)
            gp = float(xs[int(g.argmax())])
            row.update(grad_peak=gp, grad_same=int(np.abs(gp - ctr).argmin()) == near,
                       grad_r=_spearman(d, g))
        rows.append(row)
    return rows


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def scored(row: dict) -> bool:
    return row["status"] == "ok" and row["char"] != " "


def cluster_share(hits: list[tuple[str, str, bool]], resamples: int = RESAMPLES, seed: int = RANDOM_SEED) -> dict:
    """Share of hits with a two-stage bootstrap: pages with replacement, then the lines within each drawn page
    (the cluster scheme of ``wsp.metrics.bootstrap_delta``). ``hits``: (page, line key, hit)."""
    pages: dict = {}
    for pg, key, h in hits:
        pages.setdefault(pg, {}).setdefault(key, [0, 0])
        pages[pg][key][0] += int(h); pages[pg][key][1] += 1
    arr = [np.array(list(v.values()), np.float64) for v in pages.values()]
    tot = np.concatenate(arr) if arr else np.zeros((0, 2))
    point = float(tot[:, 0].sum() / tot[:, 1].sum()) if len(tot) else float("nan")
    g = np.random.default_rng(seed)
    sh = []
    for _ in range(resamples):
        h = n = 0.0
        for k in g.integers(0, len(arr), len(arr)):
            a = arr[k][g.integers(0, len(arr[k]), len(arr[k]))]
            h += a[:, 0].sum(); n += a[:, 1].sum()
        sh.append(h / n if n else np.nan)
    sh = np.sort(np.array(sh))
    return {"share": point, "ci95_cluster": [float(np.nanquantile(sh, 0.025)), float(np.nanquantile(sh, 0.975))],
            "n": int(tot[:, 1].sum()), "lines": int(len(tot)), "pages": len(arr), "resamples": resamples}


def gate(recs: list[dict]) -> dict:
    """Gate 1 for one model's lines, with the reported-alongside numbers."""
    from wsp.metrics import page_of
    rows = [(rec["key"], r) for rec in recs for r in char_rows(rec)]
    sc = [(k, r) for k, r in rows if scored(r)]
    res = cluster_share([(page_of(k), k, r["hit"]) for k, r in sc])
    off = np.array([r["offset"] for _, r in sc]) if sc else np.zeros(0)
    res.update(passed=bool(res["share"] >= GATE_SHARE and res["ci95_cluster"][0] >= GATE_LOW),
               rule={"share": GATE_SHARE, "ci_low": GATE_LOW, "min_drop": MIN_DROP, "window": WINDOW, "step": STEP, "tol": TOL},
               below_min_drop=int(sum(r["max_drop"] < MIN_DROP for _, r in sc)),
               offset_median=float(np.median(off)) if len(off) else None,
               abs_offset_median=float(np.median(np.abs(off))) if len(off) else None,
               offset_q=[float(q) for q in np.quantile(off, [0.1, 0.25, 0.75, 0.9])] if len(off) else None)
    other = {}
    for name, f in (("space_ok", lambda r: r["status"] == "ok" and r["char"] == " "),
                    ("letter_sub", lambda r: r["status"] == "sub" and r["char"] != " "),
                    ("letter_del", lambda r: r["status"] == "del" and r["char"] != " "),
                    ("space_del", lambda r: r["status"] == "del" and r["char"] == " ")):
        sel = [r for _, r in rows if f(r)]
        other[name] = {"n": len(sel), "hit_share": float(np.mean([r["hit"] for r in sel])) if sel else None}
    res["not_gating"] = other
    if sc and "grad_same" in sc[0][1]:
        res["gradient"] = {"same_character": float(np.mean([r["grad_same"] for _, r in sc])),
                           "spearman_median": float(np.nanmedian([r["grad_r"] for _, r in sc]))}
    res["chars_per_line_median"] = float(np.median([len(r["ref"]) for r in recs])) if recs else None
    return res


def rf_profile(recs: list[dict], block: str, step: int = STEP) -> dict:
    """Mean relative change of a block's frame activation by window offset (window centre − frame centre, px)."""
    acc: dict[int, list[float]] = {}
    for rec in recs:
        d = np.asarray(rec["delta"][block], np.float64)
        xs = np.asarray(rec["centres"], np.float64)
        fc = (np.arange(rec["frames"]) + 0.5) * rec["px_per_frame"]
        off = np.rint((xs[:, None] - fc[None, :]) / step).astype(int)
        for o in np.unique(off):
            m = off == o
            a = acc.setdefault(int(o), [0.0, 0])
            a[0] += float(d[m].sum()); a[1] += int(m.sum())
    ks = sorted(acc)
    prof = np.array([acc[k][0] / acc[k][1] for k in ks])
    offs = np.array(ks, float) * step
    pos = prof.max() if len(prof) else 0.0
    nz = offs[prof > 1e-5 * pos] if pos > 0 else np.zeros(0)
    return {"offset": offs.tolist(), "mean": [float(f"{v:.4g}") for v in prof],
            "extent": float(np.abs(nz).max()) if len(nz) else 0.0,
            "half_width_50": _mass_width(offs, prof, 0.5), "half_width_90": _mass_width(offs, prof, 0.9)}


def _mass_width(offs: np.ndarray, prof: np.ndarray, q: float) -> float:
    """Smallest h such that offsets within ±h hold share q of the profile's mass."""
    if not len(prof) or prof.sum() <= 0:
        return 0.0
    order = np.argsort(np.abs(offs))
    c = np.cumsum(prof[order]) / prof.sum()
    return float(np.abs(offs[order])[np.searchsorted(c, q)])


# -- running on kraken models (cache) -----------------------------------------------------------------

def cache_dir(model: str, pset: str, random: bool = False) -> Path:
    return CACHE / (f"{model}--random{RANDOM_SEED}" if random else model) / pset


def _safe(key: str) -> str:
    return key.replace("/", "_")


def _encode(rec: dict) -> dict:
    arrays = {"img": (np.clip(rec.pop("img"), 0, 1) * 255).astype(np.uint8), "centres": rec["centres"]}
    for name in ("drop", "grad", "score", "logp"):
        if name in rec:
            arrays[name] = rec.pop(name)
    for b, a in rec.pop("delta").items():
        arrays[f"delta_{b}"] = a.astype(np.float16)
    for b, a in rec.pop("heat").items():
        arrays[f"heat_{b}"] = a.astype(np.float16)
    rec.pop("centres")
    return arrays


def load_rec(path: Path) -> dict:
    z = np.load(path)
    rec = json.loads(str(z["meta"]))
    rec["centres"] = z["centres"]
    rec["delta"] = {k[6:]: z[k].astype(np.float32) for k in z.files if k.startswith("delta_")}
    rec["heat"] = {k[5:]: z[k] for k in z.files if k.startswith("heat_")}
    for name in ("drop", "grad", "score", "logp", "img"):
        if name in z.files:
            rec[name] = z[name]
    return rec


_W: dict = {}


def _init(model_path: str, random: bool, threads: int):
    import torch
    from wsp.kraken_subject import KrakenSubject
    torch.set_num_threads(threads)
    s = KrakenSubject(model_path)
    _W.update(subject=s, net=random_net(s.model.spec) if random else s.model.nn, random=random)


def _one(args):
    item, out = args
    from wsp.layerview import line_image, read_line
    s, net = _W["subject"], _W["net"]
    img = line_image(s, item)
    if img is None:
        return None
    rec = {"key": item["key"], "doc": item.get("doc"), "page": item.get("page")}
    if _W["random"]:
        r = occlude(net, img, BLOCKS)
    else:
        import torch
        o = forward_blocks(net, torch.from_numpy(np.ascontiguousarray(img))[None, None], ["O_18"])["O_18"]
        lp = _logp(o)[0].numpy()
        ln = read_line(s, item, {"logp": lp, "frames": lp.shape[0]})
        if ln is None or not ln["labels"]:
            return None
        r = occlude(net, img, BLOCKS, ln["labels"], ln["runs"])
        r["grad"] = gradient(net, img, ln["labels"], ln["runs"])
        rec.update(ref=ln["ref"], hyp=ln["hyp"], status=ln["status"], labels=ln["labels"],
                   runs=[list(map(int, x)) for x in ln["runs"]],
                   hyp_runs=[list(map(int, x)) for x in ln["hyp_runs"]])
    rec.update(frames=r["frames"], px_per_frame=r["px_per_frame"], width=r["width"], fill=r["fill"])
    arrays = _encode({**r, "img": img})
    np.savez_compressed(out / f"{_safe(item['key'])}.npz", meta=json.dumps(rec, ensure_ascii=False), **arrays)
    return item["key"]


def run(model_path: str, practice: str, n: int = 80, workers: int = 12, threads: int = 4, random: bool = False,
        log=print) -> Path:
    from multiprocessing import get_context
    from wsp.data import load
    if "heldout" in Path(practice).parts:
        raise SystemExit("the atlas reads practice lines only")
    model, pset = Path(model_path).stem, Path(practice).stem
    out = cache_dir(model, pset, random); out.mkdir(parents=True, exist_ok=True)
    cand = sample_lines(load(practice), model, n=n)
    done: list[str] = []
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(model_path, random, threads)) as pool:
        it = iter(cand)
        while len(done) < n:                          # take the first n the transform can scale
            batch = [x for _, x in zip(range(n - len(done)), it)]
            if not batch:
                break
            for k in pool.imap(_one, [(x, out) for x in batch]):
                if k:
                    done.append(k)
            log(f"{model} {pset}{' random' if random else ''}: {len(done)}/{n}")
    (out / "lines.json").write_text(json.dumps({"model": model, "set": pset, "random": random, "keys": done,
                                                "salt": SALT, "window": WINDOW, "step": STEP}, indent=0))
    return out


def load_set(model: str, pset: str, random: bool = False) -> list[dict]:
    d = cache_dir(model, pset, random)
    keys = json.loads((d / "lines.json").read_text())["keys"]
    return [load_rec(d / f"{_safe(k)}.npz") for k in keys]


# -- report: summaries, figures, viewer ---------------------------------------------------------------

def summary(model: str, pset: str, spec: str) -> dict:
    recs = load_set(model, pset)
    rnd = load_set(model, pset, random=True) if (cache_dir(model, pset, True) / "lines.json").exists() else []
    th = spec_rf(spec)
    ppf = recs[0]["px_per_frame"]
    rf = {}
    for b in BLOCKS:
        t = th.get(b, {})
        rf[b] = {"spec_rf": t.get("rf"), "jump": t.get("jump"),
                 "bound": rf_bound(t["rf"], t["jump"], ppf) if t.get("rf") else None,
                 "trained": rf_profile(recs, b), "random": rf_profile(rnd, b) if rnd else None}
    return {"model": model, "set": pset, "lines": len(recs), "random_lines": len(rnd), "px_per_frame": ppf,
            "gate1": gate(recs), "rf": rf}


def plot_rf(data: dict, th: dict):
    """Measured receptive field per block: relative change of a frame's activation by how far the occluding
    window is from the frame, trained against random weights, with the bound from the spec for conv blocks."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True, gridspec_kw={"hspace": 0.35, "wspace": 0.25})
    for ax, b in zip(axes.flat, BLOCKS):
        r = data["rf"][b]
        for key, col, lab, ls in (("trained", th["C"] if b.startswith("C") else th["L"] if b.startswith("L") else th["O"], "trained", "-"),
                                  ("random", th["muted"], "random weights", "--")):
            p = r.get(key)
            if not p:
                continue
            y = np.array(p["mean"]); y = y / y.max() if y.max() > 0 else y
            ax.plot(p["offset"], y, color=col, linestyle=ls, linewidth=1.3, label=lab)
        if r["bound"]:
            for s in (-1, 1):
                ax.axvline(s * r["bound"], color=th["hi"], linewidth=0.8, linestyle=":")
        t = r["trained"]
        ax.set_title(f"{b}" + (f"  (spec rf {r['spec_rf']} px)" if r["spec_rf"] else "  (whole line)"), loc="left", fontsize=9)
        ax.text(0.98, 0.95, f"90 % within ±{t['half_width_90']:.0f} px\n50 % within ±{t['half_width_50']:.0f} px",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=th["muted"])
        ax.set_xlim(-400, 400); ax.set_ylim(0, 1.05)
        ax.grid(linewidth=0.5); ax.set_axisbelow(True)
    axes[0, 0].legend(loc="upper left", fontsize=7, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("window centre − frame centre (px)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("change of the frame's activation\n(relative to the block's peak)", fontsize=8)
    fig.suptitle(f"{data['model']} on {data['set']}: what part of the line each block's frame depends on "
                 f"({data['lines']} lines; dotted: bound from the spec)", x=0.01, ha="left", fontsize=10)
    return fig


def plot_gate(data: list[dict], th: dict):
    """Gate 1: where occlusion locates each correctly read letter, against the centre of its CTC run."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(data), figsize=(6.2 * len(data), 3.8), squeeze=False)
    for ax, d in zip(axes[0], data):
        g = d["gate1"]
        off = np.array(d["offsets"])
        bins = np.arange(-120, 124, 8)
        ax.hist(np.clip(off, -120, 120), bins=bins, color=th["C"], alpha=0.85)
        ax.axvline(0, color=th["fg"], linewidth=0.8)
        ax.axvline(g["offset_median"], color=th["hi"], linewidth=1, linestyle="--")
        ax.set_xlabel("occlusion peak − CTC run centre (px; one frame = 8 px)", fontsize=8)
        ax.set_ylabel("letters read correctly", fontsize=8)
        lo, hi = g["ci95_cluster"]
        ax.set_title(f"{d['model']} · {d['set']}\nhit share {g['share']:.2f} [{lo:.2f}, {hi:.2f}] "
                     f"({'passes' if g['passed'] else 'fails'} Gate 1), median offset {g['offset_median']:+.0f} px",
                     loc="left", fontsize=8.5)
        ax.grid(axis="y", linewidth=0.5); ax.set_axisbelow(True)
    return fig


def _q8(a: np.ndarray) -> dict:
    """(frames, units) → per-unit min and max and the unit-normalised values as uint8 (base64, units × frames)."""
    a = np.asarray(a, np.float32)
    lo, hi = a.min(0), a.max(0)
    q = np.round((a - lo) / np.where(hi > lo, hi - lo, 1) * 255).astype(np.uint8).T
    return {"units": int(a.shape[1]), "lo": np.round(lo, 4).tolist(), "hi": np.round(hi, 4).tolist(),
            "q": base64.b64encode(q.tobytes()).decode()}


def view_lines(recs: list[dict]) -> list[dict]:
    """The lines the viewer shows: the first VIEW_LINES of the sample read at CER ≤ 20 % (a line read worse is
    mostly a crop defect, docs/viz/README.md); display only, the numbers use every line."""
    from wsp.metrics import levenshtein
    return [r for r in recs if levenshtein(r["ref"], r["hyp"]) <= 0.2 * len(r["ref"])][:VIEW_LINES]


def view_data(model: str, pset: str, recs: list[dict], l2c: dict[int, str], located: bool) -> dict:
    lines = []
    for rec in view_lines(recs):
        lp = np.asarray(rec["logp"], np.float64)
        p = np.exp(lp)
        nb = p[:, 1:].argmax(1) + 1
        rows = char_rows(rec)
        lines.append({
            "key": rec["key"], "img": f"img/{_safe(rec['key'])}.png", "frames": rec["frames"], "ppf": rec["px_per_frame"],
            "width": rec["width"], "ref": rec["ref"], "hyp": rec["hyp"], "status": rec["status"], "runs": rec["runs"],
            "centres": np.asarray(rec["centres"]).round(1).tolist(),
            "drop": np.round(np.asarray(rec["drop"], np.float64), 2).tolist(),
            "grad": np.round(np.asarray(rec["grad"], np.float64), 2).tolist(),
            "hit": [r["hit"] for r in rows], "peak": [r["peak"] for r in rows],
            "blank": np.round(p[:, 0], 3).tolist(), "best": np.round(p[np.arange(len(nb)), nb], 3).tolist(),
            "bestc": "".join(l2c.get(int(l), "?")[:1] for l in nb),
            "heat": {b: _q8(rec["heat"][b]) for b in BLOCKS if b != "O_18"}})
    return {"model": model, "set": pset, "located": located, "lines": lines}


def write_view_images(recs: list[dict], out: Path) -> None:
    from PIL import Image
    (out / "img").mkdir(parents=True, exist_ok=True)
    for rec in view_lines(recs):
        Image.fromarray(rec["img"]).save(out / "img" / f"{_safe(rec['key'])}.png")


def report(pairs: list[tuple[str, str]], out: Path, models_dir: Path = Path("models/kraken"), fmt: str = "svg") -> list[Path]:
    from wsp.kraken_subject import KrakenSubject
    from wsp.viz import render
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    written, gates = [], []
    for model, pset in pairs:
        s = KrakenSubject(models_dir / f"{model}.mlmodel")
        summ = summary(model, pset, s.model.spec)
        recs = load_set(model, pset)
        summ["offsets"] = [r["offset"] for rec in recs for r in char_rows(rec) if scored(r)]
        (out / f"atlas-{model}__{pset}.json").write_text(
            json.dumps({k: v for k, v in summ.items() if k != "offsets"}, ensure_ascii=False, indent=1))
        written.append(out / f"atlas-{model}__{pset}.json")
        written += render(plot_rf, summ, out, f"rf-{model}__{pset}", fmt)
        gates.append(summ)
        l2c = {v: k for k, v in s.codec_map().items()}
        vd = view_data(model, pset, recs, l2c, located=summ["gate1"]["passed"])
        write_view_images(recs, out)
        page = out / f"{model}__{pset}.html"
        page.write_text(render_page(vd))
        written.append(page)
    written += render(plot_gate, gates, out, "gate1", fmt)
    (out / "index.html").write_text(render_index(gates))
    written.append(out / "index.html")
    return written


# -- HTML ---------------------------------------------------------------------------------------------

_CSS = """
:root{--bg:#fff;--fg:#1b1b1b;--muted:#6e6e6e;--grid:#e2e2e2;--C:#0072B2;--L:#E69F00;--O:#009E73;--hi:#D55E00;--alt:#CC79A7}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16181d;--fg:#e6e6e6;--muted:#a3a3a3;--grid:#33363d;--C:#56B4E9;--hi:#E8743B}}
:root[data-theme=dark]{--bg:#16181d;--fg:#e6e6e6;--muted:#a3a3a3;--grid:#33363d;--C:#56B4E9;--hi:#E8743B}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}
h1{font-size:20px;margin:.2em 0}h2{font-size:15px;margin:1.2em 0 .3em}.muted{color:var(--muted)}
table{border-collapse:collapse}td,th{padding:3px 10px;border-bottom:1px solid var(--grid);text-align:left}
a{color:var(--C)}img.fig{max-width:100%}
"""


def render_index(gates: list[dict]) -> str:
    rows = "".join(
        f"<tr><td><a href='{g['model']}__{g['set']}.html'>{g['model']}</a></td><td>{g['set']}</td><td>{g['lines']}</td>"
        f"<td>{g['gate1']['share']:.2f} [{g['gate1']['ci95_cluster'][0]:.2f}, {g['gate1']['ci95_cluster'][1]:.2f}]</td>"
        f"<td>{g['gate1']['offset_median']:+.0f} px</td><td><b>{'passes' if g['gate1']['passed'] else 'fails'}</b></td></tr>"
        for g in gates)
    figs = "".join(f"<h2>{g['model']}: receptive field per block</h2>"
                   f"<picture><source srcset='rf-{g['model']}__{g['set']}-dark.svg' media='(prefers-color-scheme: dark)'>"
                   f"<img class=fig src='rf-{g['model']}__{g['set']}.svg' alt='measured receptive field per block'></picture>"
                   for g in gates)
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Activation atlas</title><style>{_CSS}</style></head><body>
<h1>Activation atlas: L0 and L2</h1>
<p><a href="explorer.html">Atlas explorer</a>: move along a line and see every block, the output, and the results of L1-L3 with their controls.</p>
<p class=muted>Epic thodel/weightspace#13. Rule and readings: docs/ATLAS_REPORT.md. The images of the six viewer lines per model are in
<code>img/</code>; they are practice lines, not held-out data.</p>
<h2>Gate 1: does occlusion find each letter where the CTC alignment puts it?</h2>
<table><tr><th>model</th><th>practice set</th><th>lines</th><th>hit share [page-clustered 95 %]</th><th>median offset</th><th>Gate 1</th></tr>{rows}</table>
<picture><source srcset='gate1-dark.svg' media='(prefers-color-scheme: dark)'><img class=fig src='gate1.svg' alt='occlusion peak against CTC centre'></picture>
{figs}
</body></html>"""


def render_page(vd: dict) -> str:
    blob = json.dumps(vd, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return _PAGE.replace("__CSS__", _CSS).replace("__TITLE__", f"Atlas {vd['model']}").replace("__DATA__", blob)


_PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>__CSS__
#wrap{overflow-x:auto;border:1px solid var(--grid);padding:4px 0}
.track{position:relative}.lab{font-size:11px;color:var(--muted);margin:6px 0 2px}
canvas{display:block;image-rendering:pixelated}
#chars span{position:absolute;font:13px monospace;transform:translateX(-50%);cursor:pointer;padding:0 1px}
#chars span.del{color:var(--hi)}#chars span.sub{color:var(--alt)}#chars span.on{background:var(--grid);outline:1px solid var(--fg)}
#tip{position:fixed;pointer-events:none;background:var(--bg);border:1px solid var(--grid);padding:2px 6px;font-size:12px;display:none}
.ctl{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:6px 0}
.banner{border-left:4px solid var(--hi);padding:4px 10px;margin:8px 0}
</style></head><body>
<h1 id=h></h1>
<p class=muted>L0: every block's output under the line (one column = one frame = 8 px of the line). L2: hover or tap a
reference character to see its occlusion profile (solid: drop of its score when a 16 px window over the line is
blanked) and gradient × input (dashed), with the CTC run centre (tick) and the occlusion peak (dot). Shown: the first six sampled lines read at CER ≤ 20 %.</p>
<div id=banner></div>
<div class=ctl><label>line <select id=sel></select></label>
<label>scale <select id=norm><option value=unit>per unit</option><option value=block>per block</option></select></label>
<label>order <select id=ord><option value=nat>unit index</option><option value=peak>by peak frame</option></select></label></div>
<div class=muted id=txt></div>
<div id=wrap><div id=inner class=track></div></div><div id=tip></div>
<script type="application/json" id=data>__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),PX=6;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
document.getElementById('h').textContent=`Activation atlas: ${D.model} on ${D.set}`;
if(!D.located)document.getElementById('banner').innerHTML='<div class=banner>Gate 1 failed for this model: the character row is placed by the CTC alignment for reading only; the page claims activation, not location.</div>';
const sel=document.getElementById('sel');D.lines.forEach((l,i)=>sel.add(new Option(`${i+1}: ${l.ref.slice(0,40)}`,i)));
const tip=document.getElementById('tip');
function dec(h){const b=atob(h.q),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a}
function ramp(t){const c=[[255,255,255],[86,180,233],[0,114,178],[0,40,90]];if(css('--bg')!=='#fff'&&css('--bg')!=='#ffffff'){c[0]=[22,24,29];c[3]=[230,230,230]}
 const x=Math.min(2.999,Math.max(0,t)*3),i=Math.floor(x),f=x-i;return c[i].map((v,k)=>Math.round(v+(c[i+1][k]-v)*f))}
function draw(){
 const L=D.lines[+sel.value],W=L.frames*PX,inner=document.getElementById('inner');inner.innerHTML='';inner.style.width=W+'px';
 document.getElementById('txt').innerHTML=`reference: <code>${L.ref.replace(/</g,'&lt;')}</code><br>reading: <code>${L.hyp.replace(/</g,'&lt;')}</code>`;
 const add=(lab,el)=>{const d=document.createElement('div');d.className='lab';d.textContent=lab;inner.append(d,el);return el};
 const im=new Image();im.src=L.img;im.width=W;im.height=90;im.alt='line image (built locally)';im.style.display='block';add('line as the network sees it',im);
 const sx=L.frames*PX/L.width;
 const chars=document.createElement('div');chars.id='chars';chars.className='track';chars.style.height='18px';
 const prof=document.createElement('canvas');prof.width=W;prof.height=70;
 L.ref.split('').forEach((c,i)=>{const s=document.createElement('span');s.textContent=c===' '?'␣':c;s.className=L.status[i];
  const [a,b]=L.runs[i];s.style.left=((a+b+1)/2*PX)+'px';
  const show=()=>{chars.querySelectorAll('.on').forEach(e=>e.classList.remove('on'));s.classList.add('on');drawProf(prof,L,i,sx)};
  s.onmouseenter=show;s.onclick=show;chars.append(s)});
 add('reference characters at their CTC runs (orange: deleted, pink: replaced)',chars);
 add('occlusion profile of the selected character',prof);drawProf(prof,L,-1,sx);
 const em=document.createElement('canvas');em.width=W;em.height=40;const g=em.getContext('2d');
 L.best.forEach((p,t)=>{g.fillStyle=css('--O');g.fillRect(t*PX,40-36*p,PX-1,36*p);g.fillStyle=css('--muted');g.globalAlpha=.35;g.fillRect(t*PX,0,PX-1,4*L.blank[t]);g.globalAlpha=1});
 em.onmousemove=e=>{const t=Math.floor(e.offsetX/PX);showTip(e,`frame ${t}: best '${L.bestc[t]}' ${L.best[t]}, blank ${L.blank[t]}`)};em.onmouseleave=()=>tip.style.display='none';
 add('O_18: best non-blank posterior per frame (grey band: blank)',em);
 for(const [b,h] of Object.entries(L.heat))add(b+(b.startsWith('L')?' (forward 0–199 | reverse 200–399)':' (channels)'),heat(b,h,L));
}
function drawProf(cv,L,i,sx){const g=cv.getContext('2d');g.clearRect(0,0,cv.width,cv.height);if(i<0){g.fillStyle=css('--muted');g.font='12px sans-serif';g.fillText('hover a character',4,14);return}
 const d=L.drop[i],gr=L.grad[i],m=Math.max(1e-6,...d.map(Math.abs),...gr.map(Math.abs)),y=v=>60-55*v/m;
 g.strokeStyle=css('--grid');g.beginPath();g.moveTo(0,60);g.lineTo(cv.width,60);g.stroke();
 for(const [arr,dash] of [[d,[]],[gr,[4,3]]]){g.setLineDash(dash);g.strokeStyle=css('--C');g.beginPath();arr.forEach((v,k)=>{const x=L.centres[k]*sx;k?g.lineTo(x,y(v)):g.moveTo(x,y(v))});g.stroke()}
 g.setLineDash([]);const [a,b]=L.runs[i];g.strokeStyle=css('--fg');g.beginPath();g.moveTo((a+b+1)/2*PX,62);g.lineTo((a+b+1)/2*PX,70);g.stroke();
 g.fillStyle=L.hit[i]?css('--O'):css('--hi');g.beginPath();g.arc(L.peak[i]*sx,y(Math.max(...d)),3.5,0,7);g.fill();
 g.fillStyle=css('--muted');g.font='11px sans-serif';g.fillText(`'${L.ref[i]}' ${L.status[i]}, ${L.hit[i]?'hit':'miss'}, peak drop ${Math.max(...d).toFixed(2)} nat`,4,12)}
function heat(b,h,L){const U=h.units,T=L.frames,q=dec(h),rh=U>100?1:3,cv=document.createElement('canvas');cv.width=T*PX;cv.height=U*rh+(U>100?3:0);
 const block=document.getElementById('norm').value==='block',glo=Math.min(...h.lo),ghi=Math.max(...h.hi);
 let order=[...Array(U).keys()];if(document.getElementById('ord').value==='peak'){const pk=order.map(u=>{let best=0,bt=0;for(let t=0;t<T;t++)if(q[u*T+t]>best){best=q[u*T+t];bt=t}return bt});
  const half=U>100?U/2:U;order=[...order.slice(0,half).sort((a,c)=>pk[a]-pk[c]),...order.slice(half).sort((a,c)=>pk[a]-pk[c])]}
 const g=cv.getContext('2d'),img=g.createImageData(cv.width,cv.height);
 order.forEach((u,r)=>{const y0=r*rh+(U>100&&r>=U/2?3:0);for(let t=0;t<T;t++){let v=q[u*T+t]/255;if(block)v=(h.lo[u]+v*(h.hi[u]-h.lo[u])-glo)/((ghi-glo)||1);const c=ramp(v);
  for(let dy=0;dy<rh;dy++)for(let dx=0;dx<PX;dx++){const p=((y0+dy)*cv.width+t*PX+dx)*4;img.data[p]=c[0];img.data[p+1]=c[1];img.data[p+2]=c[2];img.data[p+3]=255}}});
 g.putImageData(img,0,0);
 cv.onmousemove=e=>{const r=Math.min(U-1,Math.floor((e.offsetY-(U>100&&e.offsetY>U/2*rh?3:0))/rh)),u=order[Math.max(0,r)],t=Math.floor(e.offsetX/PX),v=h.lo[u]+q[u*T+t]/255*(h.hi[u]-h.lo[u]);
  showTip(e,`${b} ${U>100?(u<U/2?'fwd '+u:'rev '+(u-U/2)):'channel '+u}, frame ${t}: ${v.toFixed(3)}`)};cv.onmouseleave=()=>tip.style.display='none';return cv}
function showTip(e,s){tip.textContent=s;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px'}
['sel','norm','ord'].forEach(id=>document.getElementById(id).onchange=draw);draw();
</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.atlas")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--practice", required=True)
    r.add_argument("--n", type=int, default=80); r.add_argument("--workers", type=int, default=12)
    r.add_argument("--threads", type=int, default=4); r.add_argument("--random", action="store_true")
    p = sub.add_parser("report"); p.add_argument("--out", default="docs/atlas"); p.add_argument("--format", default="svg")
    p.add_argument("--pair", nargs=2, action="append", metavar=("MODEL", "SET"),
                   help="default: kraken-german-xix-blb-v1 blb-val-300 and kraken-bern-hertwig-v2 bern-val")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(run(a.model, a.practice, a.n, a.workers, a.threads, a.random))
    else:
        pairs = a.pair or [("kraken-german-xix-blb-v1", "blb-val-300"), ("kraken-bern-hertwig-v2", "bern-val")]
        for p_ in report(pairs, Path(a.out), fmt=a.format):
            print(p_)


if __name__ == "__main__":
    main()
