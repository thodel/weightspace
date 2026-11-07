"""Where in the network a character is found and where it is lost (epic thodel/weightspace#3).

    python -m wsp.layerview --model models/kraken/kraken-german-xix-blb-v1.mlmodel \\
        --practice data/practice/blb-train-300.jsonl --out docs/viz [--lines KEY ...] [--views 4]

Three pieces, all on practice lines, none counted against a budget (like ``wsp.stats``):

1. **Frame map.** The conv stack pools width by 2 three times, so one output frame of
   ``O_18`` covers ``px_per_frame`` (= 8) input columns. Every block's output
   (``KrakenSubject.activations``) is mean-pooled onto that frame grid: ``C_0`` over 8
   columns, ``C_3`` over 4, ``C_6`` over 2, ``C_9`` and the BiLSTMs already sit on it.
   The ``input`` pseudo-layer is the 8 × 120 pixels of a frame.
2. **Forced CTC alignment.** The reference is aligned to the model's own log-posteriors
   (Viterbi over the blank-extended label sequence), so every reference character gets
   the run of frames on which the model would have to emit it, also the characters the
   model deletes. Each character is marked ok / sub / del from the edit alignment of
   reference and greedy hypothesis (``wsp.metrics.align``).
3. **Layer probes.** A ridge readout per layer (closed form, features standardised,
   λ chosen on validation lines) from the frame's activations: an *identity* probe (which
   character, trained on character frames) and an *emission* probe (character vs blank).
   A character is *decodable* at a layer when the probe's argmax equals it on at least
   one frame of its run. Cross-fitted over five document folds, so every line is scored
   by probes that never saw its document. The probe is linear and sees one frame: a character
   that needs context is not decodable from early layers by construction, which is
   the point — it shows from which layer on the network has the character.

Output: ``<out>/<model>__<set>.json`` (folds, λ, per-layer metrics, per line the
aligned characters and a decodability string per layer), the probe figure
(``probe-<model>.svg``, no manuscript image in it) and per-line decision views
(``lines/<model>__<key>.svg``, with the line image: git-ignored, see docs/viz/README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from wsp.metrics import align as edit_align, normalise

BLOCKS = ["C_0", "C_3", "C_6", "C_9", "L_12", "L_14", "L_16", "O_18"]
LAYERS = ["input"] + BLOCKS
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]       # × mean feature variance, after standardisation
FOLDS = 5                                     # cross-fitting by document: test fold f, λ chosen on fold f+1, train on the rest
SEED = 20261107


# -- frame map ---------------------------------------------------------------------------

def to_frames(act: np.ndarray, frames: int) -> np.ndarray:
    """(width, features) → (frames, features), mean over the columns each frame covers (floor-pooled like Mp)."""
    r = act.shape[0] // frames
    if r < 1:
        raise ValueError(f"{act.shape[0]} columns for {frames} frames")
    return act[: frames * r].reshape(frames, r, -1).mean(axis=1)


def input_frames(image: np.ndarray, frames: int) -> np.ndarray:
    """(height, width) line tensor → (frames, px_per_frame × height): the raw pixels a frame covers."""
    h, w = image.shape
    r = w // frames
    return image[:, : frames * r].T.reshape(frames, r * h)


def line_image(subject, item) -> np.ndarray | None:
    """The line as the network sees it, (120, width), with ``KrakenSubject.forward``'s transforms."""
    from kraken.lib.dataset import ImageInputTransforms
    from PIL import Image
    batch, channels, height, width = subject.model.input
    tf = ImageInputTransforms(batch, height, width, channels, (16, 0), valid_norm=False)
    try:
        return tf(Image.open(item["image"]).convert("L")).numpy()[0]
    except ValueError:
        return None


def capture(subject, item) -> dict | None:
    """Every layer of one line on the frame grid, plus log-posteriors and the image."""
    acts = subject.activations([item], BLOCKS)
    if not acts["O_18"]:
        return None
    img = line_image(subject, item)
    logits = acts["O_18"][0].astype(np.float64)
    frames = logits.shape[0]
    out = {"image": img, "frames": frames, "px_per_frame": img.shape[1] // frames,
           "logp": logits - np.logaddexp.reduce(logits, axis=1, keepdims=True),
           "input": input_frames(img, frames)}
    for b in BLOCKS:
        out[b] = to_frames(acts[b][0], frames)
    return out


# -- CTC alignment ---------------------------------------------------------------------------

def forced_align(logp: np.ndarray, labels: list[int]) -> list[tuple[int, int]] | None:
    """Viterbi CTC alignment of ``labels`` (blank = 0) to (frames, classes) log-posteriors.

    Returns, per label, its run of frames as (first, last); None if the line has fewer frames than the
    labels need."""
    T, L = logp.shape[0], len(labels)
    if L == 0:
        return []
    ext = [0]
    for l in labels:
        ext += [l, 0]
    S = len(ext)
    ext_a = np.array(ext)
    skip = np.zeros(S, bool)                              # s may come from s-2: a label differing from s-2's
    skip[3::2] = ext_a[3::2] != ext_a[1:-2:2]
    neg = -np.inf
    dp = np.full(S, neg); dp[0] = logp[0, 0]; dp[1] = logp[0, ext[1]]
    back = np.zeros((T, S), np.int8)
    for t in range(1, T):
        stay, step = dp, np.concatenate([[neg], dp[:-1]])
        jump = np.where(skip, np.concatenate([[neg, neg], dp[:-2]]), neg)
        cand = np.stack([stay, step, jump])
        k = cand.argmax(axis=0)
        back[t] = k
        dp = cand[k, np.arange(S)] + logp[t, ext_a]
    end = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if not np.isfinite(dp[end]):
        return None
    path = np.empty(T, int); s = end
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    runs: list[list[int]] = [[] for _ in labels]
    for t, s in enumerate(path):
        if s % 2:
            runs[s // 2].append(t)
    return [(r[0], r[-1]) for r in runs]


def greedy(logp: np.ndarray) -> list[tuple[int, int, int]]:
    """Best path: (label, first frame, last frame) per emitted character."""
    best = logp.argmax(axis=1)
    out, prev = [], 0
    for t, l in enumerate(best):
        if l != 0 and l == prev:
            out[-1] = (l, out[-1][1], t)
        elif l != 0:
            out.append((int(l), t, t))
        prev = l
    return out


def char_status(ref: str, hyp: str) -> list[str]:
    """Per reference character: ok, sub (replaced) or del (not emitted)."""
    st = []
    for r, h in edit_align(ref, hyp):
        if r:
            st.append("ok" if r == h else ("sub" if h else "del"))
    return st


def frame_labels(frames: int, runs: list[tuple[int, int]], labels: list[int]) -> np.ndarray:
    y = np.zeros(frames, int)
    for (a, b), l in zip(runs, labels):
        y[a: b + 1] = l
    return y


def read_line(subject, item, cap: dict) -> dict | None:
    """Reference, hypothesis and alignment of one captured line."""
    c2l = subject.codec_map()
    l2c = {v: k for k, v in c2l.items()}
    ref = normalise(item["text"])
    labels = [c2l[c] for c in ref if c in c2l]
    ref = "".join(c for c in ref if c in c2l)                 # characters outside the codec cannot be aligned
    runs = forced_align(cap["logp"], labels)
    if runs is None:
        return None
    hyp_runs = greedy(cap["logp"])
    hyp = normalise("".join(l2c.get(l, "") for l, _, _ in hyp_runs))
    return {"key": item["key"], "ref": ref, "hyp": hyp, "labels": labels, "runs": runs,
            "status": char_status(ref, hyp), "hyp_runs": hyp_runs,
            "y": frame_labels(cap["frames"], runs, labels)}


# -- probes ------------------------------------------------------------------------------------

class Ridge:
    """Streaming ridge regression to one-hot classes: sums of X, XᵀX and XᵀY, solved in closed form."""

    def __init__(self, dim: int, classes: int):
        self.n = 0
        self.s = np.zeros(dim); self.G = np.zeros((dim, dim))
        self.B = np.zeros((dim, classes)); self.ys = np.zeros(classes)

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        X = X.astype(np.float64)
        self.n += len(X); self.s += X.sum(0); self.G += X.T @ X
        np.add.at(self.B.T, y, X); np.add.at(self.ys, y, 1)

    @classmethod
    def total(cls, parts: list["Ridge"]) -> "Ridge":
        """The sums of several folds: ridge sums add, so a fold's training probe needs no second pass."""
        r = cls(len(parts[0].s), len(parts[0].ys))
        for q in parts:
            r.n += q.n; r.s += q.s; r.G += q.G; r.B += q.B; r.ys += q.ys
        return r

    def solve(self, lam: float) -> tuple[np.ndarray, np.ndarray]:
        """Weights and offset for scores = X @ W + c; λ is relative to the mean standardised variance (1)."""
        mu = self.s / self.n
        Gc = self.G - np.outer(self.s, mu)
        Bc = self.B - np.outer(self.s, self.ys / self.n)
        sd = np.sqrt(np.maximum(np.diag(Gc) / self.n, 1e-12))
        A = Gc / np.outer(sd, sd) + lam * self.n * np.eye(len(sd))
        Ws = np.linalg.solve(A, Bc / sd[:, None])
        W = Ws / sd[:, None]
        return W, self.ys / self.n - mu @ W


def fold_docs(items: list[dict], k: int = FOLDS, seed: int = SEED) -> dict[str, int]:
    """key → fold 0…k−1, whole documents at a time (documents in sha256 order, each to the fold with fewest lines)."""
    doc = lambda x: x.get("doc") or x["key"].rsplit("/", 1)[0]
    lines = Counter(doc(x) for x in items)
    order = sorted(lines, key=lambda d: hashlib.sha256(f"{seed}:{d}".encode()).hexdigest())
    size, fold = [0] * k, {}
    for d in order:
        f = min(range(k), key=lambda i: (size[i], i))
        fold[d] = f; size[f] += lines[d]
    return {x["key"]: fold[doc(x)] for x in items}


def decodable(hit: np.ndarray, runs) -> str:
    """'1' per character for which ``hit`` (a per-frame bool for that character) holds on at least one frame of its run."""
    return "".join("1" if hit[i][a: b + 1].any() else "0" for i, (a, b) in enumerate(runs))


def identity_hits(pred: np.ndarray, labels) -> list[np.ndarray]:
    return [pred == l for l in labels]


def lost_at(ln: dict, i: int) -> str:
    """For one character: the deepest layer whose probes still have it — identity and emission both on its
    aligned frames ('never' if no layer does). The identity probe nearly always says "space" on a space
    (the commonest class), so for spaces this is in effect the emission probe."""
    hits = [layer for layer in LAYERS if ln["identity"][layer][i] == "1" and ln["emission"][layer][i] == "1"]
    return hits[-1] if hits else "never"


def summarise(lines: list[dict]) -> dict:
    """Per-layer identity and emission decodability on test lines, by edit status and for spaces;
    for deleted characters the deepest layer that still has their identity."""
    out = {}
    for layer in LAYERS + ["model"]:
        cnt: dict[str, list[int]] = {}
        for ln in lines:
            for i, st in enumerate(ln["status"]):
                for g in ("all", st, ("space_" if ln["ref"][i] == " " else "char_") + st):
                    c = cnt.setdefault(g, [0, 0, 0])
                    c[0] += ln["identity"][layer][i] == "1"; c[1] += ln["emission"][layer][i] == "1"; c[2] += 1
        out[layer] = {g: {"identity": a / n, "emission": e / n, "n": n} for g, (a, e, n) in sorted(cnt.items())}
        for k in ("identity_frame_acc", "emission_balanced_acc"):
            v = [ln["frame"][layer][k] for ln in lines if layer in ln["frame"]]
            if v:
                out[layer][k] = float(np.average([x for x, _ in v], weights=[n for _, n in v]))
    lost = Counter(); lost_space = Counter()
    for ln in lines:
        for i, st in enumerate(ln["status"]):
            if st == "del":
                where = lost_at(ln, i)
                lost[where] += 1
                if ln["ref"][i] == " ":
                    lost_space[where] += 1
    letters = Counter(c for ln in lines for c in ln["ref"] if c != " ")
    chance = letters.most_common(1)[0][1] / sum(letters.values()) if letters else 0.0
    return {"layers": out, "deleted_lost_at": dict(lost), "deleted_space_lost_at": dict(lost_space),
            "identity_chance": chance}


def _balanced(pred: np.ndarray, y: np.ndarray) -> tuple[int, int, int, int]:
    """hits on character frames, character frames, hits on blank frames, blank frames"""
    ch = y != 0
    return int((pred[ch] == 1).sum()), int(ch.sum()), int((pred[~ch] == 0).sum()), int((~ch).sum())


def run_probes(subject, items: list[dict], log=print) -> tuple[dict, dict]:
    """Two probes per layer, both ridge readouts of one frame:

    * **identity** — trained on the frames the forced alignment gives a character, to that character;
      read as the argmax over characters. Answers *which* character the layer encodes there.
    * **emission** — trained on every frame, character vs blank. Answers *whether* the layer already
      says "emit here". λ by balanced accuracy, because four frames in five are blank.

    Cross-fitted over ``FOLDS`` document folds: for test fold f the probes are trained on the other folds
    but f+1, λ is chosen on f+1, so every line is scored once by probes that never saw its document.
    Pass 1 accumulates ridge sums per fold; pass 2 scores every line under its own and its predecessor's split."""
    fold = fold_docs(items)
    K = FOLDS
    classes = dict(subject.model.nn.named_children())["O_18"].lin.out_features
    ident = [dict() for _ in range(K)]; emit = [dict() for _ in range(K)]
    skipped = []
    t0 = time.time()
    for it in items:
        cap = capture(subject, it)
        rd = read_line(subject, it, cap) if cap else None
        if rd is None:
            skipped.append(it["key"]); continue
        ch = rd["y"] != 0
        f = fold[it["key"]]
        for layer in LAYERS:
            X = cap[layer]
            ident[f].setdefault(layer, Ridge(X.shape[1], classes)).add(X[ch], rd["y"][ch])
            emit[f].setdefault(layer, Ridge(X.shape[1], 2)).add(X, ch.astype(int))
    log(f"fold sums {time.time() - t0:.0f}s")
    sol = []                                        # sol[f][layer][lam] for test fold f
    for f in range(K):
        train = [g for g in range(K) if g not in (f, (f + 1) % K)]
        sol.append({layer: {lam: (Ridge.total([ident[g][layer] for g in train]).solve(lam),
                                  Ridge.total([emit[g][layer] for g in train]).solve(lam)) for lam in LAMBDAS}
                    for layer in LAYERS})
    del ident, emit
    log(f"solved {time.time() - t0:.0f}s")

    def predict(cap, f, layer, lam):
        (Wi, ci), (We, ce) = sol[f][layer][lam]
        X = cap[layer]
        return (X @ Wi + ci)[:, 1:].argmax(1) + 1, (X @ We + ce).argmax(1)

    vid = [{layer: {lam: [0, 0] for lam in LAMBDAS} for layer in LAYERS} for _ in range(K)]
    vem = [{layer: {lam: [0, 0, 0, 0] for lam in LAMBDAS} for layer in LAYERS} for _ in range(K)]
    test_lines = []
    for it in items:
        if it["key"] in skipped:
            continue
        cap = capture(subject, it)
        rd = read_line(subject, it, cap)
        f = fold[it["key"]]
        pv = (f - 1) % K                            # this line is the validation fold of split pv
        ch = rd["y"] != 0
        for layer in LAYERS:
            for lam in LAMBDAS:
                pi, pe = predict(cap, pv, layer, lam)
                vid[pv][layer][lam][0] += int((pi[ch] == rd["y"][ch]).sum()); vid[pv][layer][lam][1] += int(ch.sum())
                vem[pv][layer][lam] = [u + v for u, v in zip(vem[pv][layer][lam], _balanced(pe, rd["y"]))]
        rd["preds"] = {layer: {lam: predict(cap, f, layer, lam) for lam in LAMBDAS} for layer in LAYERS}
        rd["own"] = cap["logp"]; rd["fold"] = f
        test_lines.append(rd)
    log(f"predicted {time.time() - t0:.0f}s")
    bal = lambda v: (v[0] / max(v[1], 1) + v[2] / max(v[3], 1)) / 2
    lam_id = [{layer: max(LAMBDAS, key=lambda lam: vid[f][layer][lam][0] / max(vid[f][layer][lam][1], 1)) for layer in LAYERS}
              for f in range(K)]
    lam_em = [{layer: max(LAMBDAS, key=lambda lam: bal(vem[f][layer][lam])) for layer in LAYERS} for f in range(K)]
    rows = []
    for rd in test_lines:
        y, ch, f = rd["y"], rd["y"] != 0, rd["fold"]
        idn, emi, fr = {}, {}, {}
        for layer in LAYERS:
            pi = rd["preds"][layer][lam_id[f][layer]][0]
            pe = rd["preds"][layer][lam_em[f][layer]][1]
            idn[layer] = decodable(identity_hits(pi, rd["labels"]), rd["runs"])
            emi[layer] = decodable([pe == 1] * len(rd["labels"]), rd["runs"])
            fr[layer] = {"identity_frame_acc": (float((pi[ch] == y[ch]).mean()) if ch.any() else 0.0, int(ch.sum())),
                         "emission_balanced_acc": (bal(_balanced(pe, y)), len(y))}
        own = rd["own"]
        idn["model"] = decodable(identity_hits(own[:, 1:].argmax(1) + 1, rd["labels"]), rd["runs"])
        emi["model"] = decodable([own.argmax(1) != 0] * len(rd["labels"]), rd["runs"])
        rows.append({"key": rd["key"], "fold": f, "ref": rd["ref"], "hyp": rd["hyp"], "status": rd["status"],
                     "runs": rd["runs"], "identity": idn, "emission": emi, "frame": fr})
    meta = {"folds": {str(f): sum(v == f for v in fold.values()) for f in range(K)},
            "lines_scored": len(rows), "skipped": skipped,
            "lambda_identity": lam_id, "lambda_emission": lam_em}
    return meta, {"lines": rows, **summarise(rows)}


# -- figures -------------------------------------------------------------------------------------

LAYER_LABEL = {"input": "pixels", "model": "model's own output"}
GROUPS = [("char_ok", "letter, emitted correctly"), ("char_sub", "letter, replaced"), ("char_del", "letter, deleted"),
          ("space_ok", "space, emitted"), ("space_del", "space, deleted")]


def plot_probe(data: dict, th: dict):
    """Per layer, what a one-frame linear probe can read: which character (identity) and whether to emit
    (emission), split by what the model did with the character; and where deleted characters are lost."""
    import matplotlib.pyplot as plt
    lay = data["layers"]
    fig, (a, e, b) = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.25, 1.25, 1], "wspace": 0.3})
    x = np.arange(len(LAYERS))
    style = {"char_ok": (th["C"], "o", "-"), "char_sub": (th["alt"], "D", "-"), "char_del": (th["hi"], "s", "-"),
             "space_ok": (th["C"], "o", ":"), "space_del": (th["hi"], "s", ":")}
    for ax, key, title, frame_key, frame_lab in (
            (a, "identity", "which letter: identity probe (spaces left out)", "identity_frame_acc", "frame accuracy on character frames"),
            (e, "emission", "whether to emit: character-vs-blank probe", "emission_balanced_acc", "balanced frame accuracy")):
        for g, lab in GROUPS:
            if g not in lay["O_18"] or (key == "identity" and g.startswith("space")):
                continue
            col, m, ls = style[g]
            ax.plot(x, [lay[layer][g][key] for layer in LAYERS], color=col, marker=m, linestyle=ls, markersize=5,
                    linewidth=1.4, markerfacecolor=th["bg"] if g.startswith("space") else col,
                    label=f"{lab} (n = {lay['O_18'][g]['n']:,})")
            ax.plot(len(LAYERS) - 1 + 0.45, lay["model"][g][key], marker=m, color=col, markerfacecolor=th["bg"],
                    markersize=5, linestyle="")
        ax.plot(x, [lay[layer].get(frame_key, np.nan) for layer in LAYERS], color=th["muted"], linewidth=1,
                linestyle="--", label=frame_lab)
        ax.set_xticks(x); ax.set_xticklabels([LAYER_LABEL.get(l, l) for l in LAYERS], fontsize=8, rotation=45)
        ax.set_xlim(-0.4, len(LAYERS) - 0.3)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", linewidth=0.5); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.text(len(LAYERS) - 1 + 0.45, 1.0, "model\noutput", ha="center", va="bottom", fontsize=6.5, color=th["muted"],
                transform=ax.get_xaxis_transform())
    a.axhline(data.get("identity_chance", 0), color=th["muted"], linewidth=0.8, linestyle=":")
    a.text(0, data.get("identity_chance", 0), " always the commonest letter", va="bottom", fontsize=7, color=th["muted"])
    a.set_ylabel("share of reference characters the probe has\non at least one of their aligned frames")
    a.legend(loc="upper left", fontsize=7, frameon=False)
    e.legend(loc="upper left", fontsize=7, frameon=False)

    order = LAYERS + ["never"]
    lost = data["deleted_lost_at"]; sp = data.get("deleted_space_lost_at", {})
    tot = sum(lost.values()) or 1
    yy = np.arange(len(order))
    b.barh(yy, [lost.get(o, 0) for o in order], color=th["hi"], height=0.7, label="deleted characters")
    b.barh(yy, [sp.get(o, 0) for o in order], color=th["bg"], edgecolor=th["fg"], hatch="////", height=0.7,
           linewidth=0.6, label="of which spaces")
    for y, o in zip(yy, order):
        v = lost.get(o, 0)
        if v:
            b.text(v, y, f" {v} ({100 * v / tot:.0f} %)", va="center", fontsize=7.5)
    b.set_yticks(yy); b.set_yticklabels([LAYER_LABEL.get(o, "nowhere" if o == "never" else o) for o in order], fontsize=8)
    b.invert_yaxis()
    b.set_xlabel("deleted reference characters")
    b.grid(axis="x", linewidth=0.5); b.set_axisbelow(True)
    b.legend(loc="center right", fontsize=7, frameon=False)
    b.set_title("deleted: deepest layer whose probes\nstill have identity and emission", loc="left", fontsize=9.5)
    b.set_xlim(0, max(lost.values() or [1]) * 1.4)
    m = data.get("meta", {})
    fig.suptitle(f"{data.get('model', '')} on {data.get('set', '')}: {m.get('lines_scored', '?')} lines, each scored by probes "
                 f"trained on other documents ({FOLDS}-fold); ridge readout of one frame per layer, reference placed by forced CTC alignment",
                 x=0.0, ha="left", y=1.03, fontsize=10)
    return fig


def decision_data(subject, item, probes_row: dict | None = None) -> dict:
    """Everything the decision view draws for one line, as plain arrays (the figure needs no model)."""
    cap = capture(subject, item)
    rd = read_line(subject, item, cap)
    l2c = {v: k for k, v in subject.codec_map().items()}
    conv = {}
    for b in BLOCKS[:4]:
        a = cap[b]                                            # (frames, C·H), feature order c·H + h
        C = dict(subject.model.nn.named_children())[b].output_shape[1]
        conv[b] = np.sqrt((a.reshape(a.shape[0], C, -1) ** 2).sum(-1)).T      # (channels, frames)
    half = lambda b: cap[b].shape[1] // 2                     # bidirectional output: forward | reverse
    lstm = {b: (np.linalg.norm(cap[b][:, :half(b)], axis=1), np.linalg.norm(cap[b][:, half(b):], axis=1)) for b in BLOCKS[4:7]}
    p = np.exp(cap["logp"])
    return {"key": item["key"], "image": cap["image"], "px_per_frame": cap["px_per_frame"], "frames": cap["frames"],
            "conv": conv, "lstm": lstm, "blank": p[:, 0], "top": p[:, 1:].max(1),
            "ref": rd["ref"], "hyp": rd["hyp"], "status": rd["status"], "runs": rd["runs"],
            "hyp_chars": [(l2c.get(l, "?"), a, b) for l, a, b in rd["hyp_runs"]],
            "identity": (probes_row or {}).get("identity"), "emission": (probes_row or {}).get("emission")}


def _g(ch: str) -> str:
    from wsp.viz import glyph
    return glyph(ch)


def plot_decision(d: dict, th: dict):
    """One line, frame by frame: image, conv channel energy, BiLSTM state norms, CTC posteriors,
    reference (forced-aligned) against hypothesis, and per layer whether a probe still has each character."""
    import matplotlib.pyplot as plt
    T = d["frames"]
    layers = [l for l in LAYERS if d.get("identity") and l in d["identity"]]
    hr = [1.6, 0.55, 0.55, 0.55, 0.55, 0.6, 0.6, 0.6, 1.0, 0.9] + ([0.18 * len(layers) + 0.2] if layers else [])
    H = sum(hr) * 0.95 + 0.6
    fig, axes = plt.subplots(len(hr), 1, figsize=(max(9, min(T / 14, 24)), H), sharex=True,
                             gridspec_kw={"height_ratios": hr, "hspace": 0.12})
    fig.subplots_adjust(top=1 - 0.45 / H)
    ax = iter(axes)
    a = next(ax)
    img = d["image"]
    a.imshow(img, cmap="gray", aspect="auto", extent=(0, img.shape[1] / d["px_per_frame"], img.shape[0], 0), vmin=0, vmax=1)
    a.set_yticks([]); a.set_ylabel("line", rotation=0, ha="right", va="center")
    for b, e in d["conv"].items():
        a = next(ax)
        a.imshow(e / (e.max(1, keepdims=True) + 1e-9), aspect="auto", cmap="cividis", extent=(0, T, e.shape[0], 0),
                 interpolation="nearest")
        a.set_yticks([]); a.set_ylabel(f"{b}\n{e.shape[0]} ch", rotation=0, ha="right", va="center", fontsize=8)
    for b, (f, r) in d["lstm"].items():
        a = next(ax)
        t = np.arange(T) + 0.5
        a.plot(t, f, color=th["C"], linewidth=0.9, label="forward")
        a.plot(t, r, color=th["L"], linewidth=0.9, label="reverse")
        a.set_yticks([]); a.set_ylabel(f"{b}\n‖h‖", rotation=0, ha="right", va="center", fontsize=8)
        if b == "L_12":
            a.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    a = next(ax)
    t = np.arange(T) + 0.5
    a.fill_between(t, 0, d["blank"], color=th["muted"], alpha=0.35, linewidth=0, step="mid", label="P(blank)")
    a.step(t, d["top"], where="mid", color=th["O"], linewidth=1, label="max P(character)")
    for ch, s, e in d["hyp_chars"]:
        a.text((s + e + 1) / 2, 1.02, _g(ch), ha="center", va="bottom", fontsize=7.5, color=th["O"])
    a.set_ylim(0, 1.25); a.set_yticks([0, 1]); a.set_ylabel("CTC", rotation=0, ha="right", va="center", fontsize=8)
    a.legend(loc="upper right", fontsize=7, frameon=False, ncol=2, bbox_to_anchor=(1, 0.85))
    a = next(ax)
    col = {"ok": th["fg"], "sub": th["alt"], "del": th["hi"]}
    for ch, st, (s, e) in zip(d["ref"], d["status"], d["runs"]):
        x = (s + e + 1) / 2
        a.axvspan(s, e + 1, ymin=0.55, ymax=0.95, color=col[st], alpha=0.12 if st == "ok" else 0.35, linewidth=0)
        a.text(x, 0.75, _g(ch), ha="center", va="center", fontsize=8, color=col[st],
               fontweight="bold" if st != "ok" else "normal")
    for ch, s, e in d["hyp_chars"]:
        a.text((s + e + 1) / 2, 0.25, _g(ch), ha="center", va="center", fontsize=8, color=th["O"])
    a.set_ylim(0, 1); a.set_yticks([0.25, 0.75]); a.set_yticklabels(["hypothesis", "reference"], fontsize=8)
    if layers:
        a = next(ax)
        for j, layer in enumerate(layers):
            for i, ((s, e), st) in enumerate(zip(d["runs"], d["status"])):
                ident, emit = d["identity"][layer][i] == "1", d["emission"][layer][i] == "1"
                a.plot((s + e + 1) / 2, j, marker="s", markersize=3.8, linestyle="",
                       color=col[st] if ident and emit else th["bg"], markeredgecolor=col[st] if ident else th["grid"],
                       markeredgewidth=0.9)
        a.set_ylim(len(layers) - 0.5, -0.5)
        a.set_yticks(range(len(layers))); a.set_yticklabels([LAYER_LABEL.get(l, l) for l in layers], fontsize=7)
        a.set_ylabel("probes\n■ identity + emit\n□ identity only", rotation=0, ha="right", va="center", fontsize=7)
    axes[-1].set_xlim(0, T)
    axes[-1].set_xlabel(f"output frame ({d['px_per_frame']} px of the line each)")
    nd = d["status"].count("del"); ns = d["status"].count("sub")
    fig.suptitle(f"{d.get('model', '')}  {d['key']}: {len(d['ref'])} reference characters, {nd} deleted (vermillion), "
                 f"{ns} replaced (purple); reference placed by forced CTC alignment, hypothesis by best path",
                 x=0.0, ha="left", y=1.0, fontsize=9.5)
    return fig


# -- CLI -------------------------------------------------------------------------------------

def pick_lines(rows: list[dict], n: int) -> list[str]:
    """Lines with the most deleted spaces, then the most deletions. Skipped: short lines (< 15 characters),
    collapsed lines and lines read at CER > 20 % — mostly crop or reference
    defects (rotated, degenerate boxes, OCR junk; docs/PHASE4_REPORT.md), not decisions of the network."""
    from wsp.metrics import is_collapsed, levenshtein
    cand = [r for r in rows if len(r["ref"]) >= 15 and "del" in r["status"] and not is_collapsed(r["hyp"], r["ref"])
            and levenshtein(r["ref"], r["hyp"]) <= 0.2 * len(r["ref"])]
    score = lambda r: (sum(st == "del" and c == " " for c, st in zip(r["ref"], r["status"])), r["status"].count("del"))
    return [r["key"] for r in sorted(cand, key=score, reverse=True)[:n]]


def main(argv=None):
    from wsp.data import load, sha256_file
    from wsp.kraken_subject import KrakenSubject
    from wsp.viz import render
    ap = argparse.ArgumentParser(prog="wsp.layerview")
    ap.add_argument("--model", required=True); ap.add_argument("--practice", required=True)
    ap.add_argument("--out", default="docs/viz")
    ap.add_argument("--lines", nargs="*", default=None, help="keys for decision views (default: picked from test lines)")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--format", default="svg", choices=["svg", "png"])
    a = ap.parse_args(argv)
    if "heldout" in Path(a.practice).parts:
        raise SystemExit("layer probes read practice lines only")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    subject = KrakenSubject(a.model)
    items = load(a.practice)
    model, pset = Path(a.model).stem, Path(a.practice).stem
    meta, res = run_probes(subject, items)
    res.update(model=model, set=pset, meta={**meta, "model_sha256": sha256_file(Path(a.model)),
                                            "practice_sha256": sha256_file(Path(a.practice)), "lambdas": LAMBDAS,
                                            "fold_rule": f"{FOLDS} folds by document, sha256('{SEED}:'+doc) order, each to the smallest fold"})
    jp = out / f"{model}__{pset}.json"
    jp.write_text(json.dumps(res, ensure_ascii=False, indent=0))
    print(jp)
    for p in render(plot_probe, res, out, f"probe-{model}__{pset}", a.format):
        print(p)
    rows = {r["key"]: r for r in res["lines"]}
    by_key = {x["key"]: x for x in items}
    for key in a.lines if a.lines is not None else pick_lines(res["lines"], a.views):
        d = decision_data(subject, by_key[key], rows.get(key))
        d["model"] = model
        safe = key.replace("/", "_")
        for p in render(plot_decision, d, out / "lines", f"{model}__{safe}", a.format):
            print(p)


if __name__ == "__main__":
    main()
