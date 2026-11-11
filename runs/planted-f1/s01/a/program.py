"""planted-f1 / s01 / task (a), budget 10 — the programmer's script (claude-fable-5-1).

Written in full before the first pass; see preregistration_hypotheses.md next to it.
Reading of the pack: C_3.co.weight and C_3.co.bias are identically zero, so the model emits
only blanks (CER 1.0, 300/300 collapsed). No parent exists; the only donor is an unrelated
model of the same architecture. Plan: graft a C_3 (donor filters, or an identity pass-through),
keep whichever the rule accepts, then tune its amplitude, optionally try the donor's whole conv
stack, and recalibrate the blank bias. Only counted passes (s.measure) are used.
"""
from __future__ import annotations

import json

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel"   # the only donor
C3W, C3B = "C_3.co.weight", "C_3.co.bias"
CONV = ["C_0.co.weight", "C_0.co.bias", C3W, C3B,
        "C_6.co.weight", "C_6.co.bias", "C_9.co.weight", "C_9.co.bias"]
LSTM_W = [f"L_{l}.layer.weight_{k}_l0{sfx}" for l in (12, 14, 16) for k in ("ih", "hh") for sfx in ("", "_reverse")]
NOTHING = {"", "∅", "None", "none", "<none>", "<nothing>", "ø", "<eps>", "ε"}

s = open_session(".")


class OutOfPasses(RuntimeError):
    pass


# ---------------------------------------------------------------- helpers
def note(txt: str) -> None:
    print(txt, flush=True)
    try:
        s.note(txt)
    except Exception:
        pass


def conf_items(conf) -> list[tuple[str, str, int]]:
    """Normalise m.confusions (dict keyed by (ref, hyp), nested dict, or list of triples) to (ref, hyp, count)."""
    items: list[tuple[str, str, int]] = []
    if conf is None:
        return items
    try:
        if isinstance(conf, dict):
            for k, v in conf.items():
                if isinstance(k, (tuple, list)) and len(k) == 2:
                    items.append((str(k[0]), str(k[1]), int(v)))
                elif isinstance(v, dict):
                    for h, c in v.items():
                        items.append((str(k), str(h), int(c)))
                elif isinstance(k, str) and "→" in k:
                    r, h = k.split("→", 1)
                    items.append((r.strip(), h.strip(), int(v)))
        else:
            for it in conf:
                if isinstance(it, dict):
                    r = it.get("ref", it.get("reference", ""))
                    h = it.get("hyp", it.get("hypothesis", ""))
                    c = it.get("count", it.get("n", 0))
                    items.append((str(r), str(h), int(c)))
                elif isinstance(it, (tuple, list)) and len(it) == 3:
                    items.append((str(it[0]), str(it[1]), int(it[2])))
                elif isinstance(it, (tuple, list)) and len(it) == 2 and isinstance(it[0], (tuple, list)):
                    items.append((str(it[0][0]), str(it[0][1]), int(it[1])))
    except Exception:
        pass
    return items


def diag(m) -> dict:
    d = {"cer": float(m.cer), "collapsed": int(m.collapsed)}
    try:
        d["wer"] = float(m.wer)
    except Exception:
        pass
    try:
        ad = m.as_dict()
        for k in ("length_ratio", "edits", "ref_chars", "chars"):
            if k in ad:
                d[k] = ad[k]
    except Exception:
        pass
    dels = ins = subs = 0
    for r, h, c in conf_items(getattr(m, "confusions", None)):
        if h in NOTHING and r not in NOTHING:
            dels += c
        elif r in NOTHING and h not in NOTHING:
            ins += c
        else:
            subs += c
    d.update(dels=dels, ins=ins, subs=subs)
    return d


def direction(d: dict):
    """'up' = output too sparse (collapse / deletions), 'down' = too many insertions, None = balanced/unknown."""
    if d.get("collapsed", 0) >= 30:
        return "up"
    lr = d.get("length_ratio")
    if isinstance(lr, (int, float)) and np.isfinite(lr):
        if lr < 0.8:
            return "up"
        if lr > 1.2:
            return "down"
        return None
    dels, ins = d.get("dels", 0), d.get("ins", 0)
    if dels > 0 and dels > 3 * max(ins, 1):
        return "up"
    if ins > 0 and ins > 3 * max(dels, 1):
        return "down"
    return None


def measure(label: str):
    if s.budget_left <= 0:
        raise OutOfPasses(label)
    m = s.measure(label)
    d = diag(m)
    note(f"[pass {s.budget - s.budget_left}] {label}: {json.dumps(d, default=str)}")
    return m, d


def try_edit(label, hyp_text, prediction, edit_class, op, names, keep_reason, **kw):
    """hypothesis -> edit -> one pass -> s.keep (the rule decides). Returns (kept, diag); (None, None) if no pass is left."""
    if s.budget_left <= 0:
        note(f"skip {label}: no pass left")
        return None, None
    h = s.hypothesis(hyp_text, prediction, edit_class)
    s.edit(op, names, hypothesis=h, **kw)
    try:
        m, d = measure(label)
    except Exception as e:
        s.revert(f"pass failed: {e!r}")
        raise
    kept = s.keep(keep_reason)
    note(f"{label}: kept={kept} cer={d['cer']:.4f} collapsed={d['collapsed']}")
    return kept, d


def finish() -> None:
    try:
        if getattr(s, "_last_edit", None) is not None:
            s.revert("program ended with an undecided edit")
    except Exception as e:
        print("revert at exit failed:", repr(e))
    try:
        summary = s.freeze()
        print(json.dumps({k: summary.get(k) for k in ("used", "edits", "kept", "final_digest", "last_pass")},
                         default=str, indent=1))
    except Exception as e:
        print("freeze failed:", repr(e))


# ---------------------------------------------------------------- main
def main() -> None:
    T = s.subject.tensors()
    note(f"start: {s.budget_left} passes; tensors {len(T)}")

    # weight-space readings that cost no pass
    c3_zero = float(np.abs(T[C3W]).max()) == 0.0 and float(np.abs(T[C3B]).max()) == 0.0
    note(f"C_3 all-zero: {c3_zero} (|w|max {float(np.abs(T[C3W]).max()):.4g}, |b|max {float(np.abs(T[C3B]).max()):.4g})")
    for n in LSTM_W:
        if n in T:
            big = int((np.abs(T[n]) > 3.0).sum())
            if big:
                note(f"{n}: {big} elements with |w| > 3 (abs max {float(np.abs(T[n]).max()):.3f}); noted, not edited")

    donor = None
    try:
        D = KrakenSubject(PARENT).tensors()
        ok = all(n in D and tuple(D[n].shape) == tuple(T[n].shape) for n in CONV)
        if ok:
            donor = D
        note(f"donor loaded: conv shapes match={ok}; donor C_3 weight rms {float(np.sqrt((D[C3W] ** 2).mean())):.4f}, "
             f"bias rms {float(np.sqrt((D[C3B] ** 2).mean())):.4f}, |w|max {float(np.abs(D[C3W]).max()):.3f}")
    except Exception as e:
        note(f"donor unavailable: {e!r}")

    # pass 1: baseline
    m0, d0 = measure("C0 baseline")
    best = d0
    if not c3_zero:
        note("C_3 is not zero at run time: the pack's reading does not hold; no edit is applied (nothing else is "
             "diagnosable from the readings). Freezing unchanged.")
        return
    if d0["cer"] < 0.5:
        note("baseline CER is far from the pack's 1.0 although C_3 is zero; the practice set differs from the pack. "
             "Proceeding with the C_3 repair anyway (a zero conv cannot be intended).")

    # the two grafts
    kh, kw = T[C3W].shape[2], T[C3W].shape[3]
    ident_w = np.zeros(T[C3W].shape, dtype=np.float32)
    for i in range(min(T[C3W].shape[0], T[C3W].shape[1])):
        ident_w[i, i, kh // 2, kw // 2] = 1.0
    ident_b = np.zeros(T[C3B].shape, dtype=np.float32)

    def graft_args(w, b):
        return {C3W: {"donor": np.asarray(w, dtype=np.float32)}, C3B: {"donor": np.asarray(b, dtype=np.float32)}}

    current = None            # which graft is in the model
    results = {}

    if donor is not None:
        kept, d = try_edit(
            "H1 donor C_3",
            "C_3 (weight and bias) is identically zero, so the model emits only blanks. Grafting the unrelated donor's "
            "trained C_3 restores an image-dependent signal through C_6, in mismatched channel slots.",
            "CER 1.00 -> ~0.85 (range 0.60-1.10); collapsed 300 -> < 50",
            "transplant (one conv block from the unrelated donor)",
            "transplant", [C3W, C3B], "H1: donor C_3 graft beats the all-blank model", per_tensor=graft_args(donor[C3W], donor[C3B]))
        results["H1"] = d
        if kept:
            best, current = d, "H1"

    kept, d = try_edit(
        "H2 identity C_3",
        "Alternative graft for the zero C_3: identity pass-through (centre tap, channel i -> i, bias 0), so C_6 "
        "receives the pooled C_0 features directly.",
        "CER 1.00 -> ~0.85 (range 0.60-1.10); kept only if significantly better than the current state",
        "set (constructed identity conv)",
        "transplant", [C3W, C3B], "H2: identity pass-through beats the current state", per_tensor=graft_args(ident_w, ident_b))
    results["H2"] = d
    if kept:
        best, current = d, "H2"

    # rescue: scaled grafts if neither was accepted
    if current is None and s.budget_left > 0:
        measured = [(k, v) for k, v in results.items() if v is not None]
        measured.sort(key=lambda kv: kv[1]["cer"])
        for k, dk in measured:
            if s.budget_left <= 0:
                break
            dirk = direction(dk)
            if dirk == "down" or (dirk is None and dk["cer"] >= 1.0 and dk["collapsed"] < 150):
                f = 0.25
            else:
                f = 4.0
            if k == "H1" and donor is not None:
                w, b = donor[C3W] * f, donor[C3B] * f
            else:
                w, b = ident_w * f, ident_b
            kept, d = try_edit(
                f"rescue {k} x{f}",
                f"{k} was refused (cer {dk['cer']:.3f}, collapsed {dk['collapsed']}): the graft's amplitude may be "
                f"wrong for the subject's C_6; same graft scaled by {f}.",
                "CER 1.00 -> ~0.85 (range 0.60-1.10)",
                "transplant (scaled)",
                "transplant", [C3W, C3B], f"rescue: scaled {k} beats the all-blank model", per_tensor=graft_args(w, b))
            if kept:
                best, current = d, f"{k}x{f}"
                break

    # H3: amplitude probes on the kept graft
    if current is not None and s.budget_left > 0:
        dir0 = direction(best)
        f = 2.0 if dir0 == "up" else 0.5
        kept, d = try_edit(
            f"H3 scale C_3 x{f}",
            f"The grafted C_3 ({current}) drives the subject's C_6/LSTM at an arbitrary amplitude (diagnostic: {dir0}); "
            f"the stack is nearly homogeneous, so scaling C_3 rescales the LSTM input.",
            "-0.02 to -0.10 CER relative to the kept state",
            "scale",
            "scale", [C3W, C3B], "H3: amplitude probe improves CER", factor=f)
        if kept:
            best = d
            f2 = f
        else:
            f2 = 1.0 / f
        if s.budget_left > 0:
            kept, d = try_edit(
                f"H3b scale C_3 x{f2}",
                f"Second amplitude probe ({'same direction' if f2 == f else 'opposite direction'}).",
                "-0.02 to -0.10 CER relative to the kept state",
                "scale",
                "scale", [C3W, C3B], "H3b: amplitude probe improves CER", factor=f2)
            if kept:
                best = d

    # H4: the donor's whole conv stack, only if the repair is still poor
    if donor is not None and best["cer"] > 0.60 and s.budget_left > 0:
        kept, d = try_edit(
            "H4 donor conv stack",
            "If a single grafted C_3 leaves the features incoherent for C_6/C_9, the donor's whole conv stack "
            "(C_0, C_3, C_6, C_9) gives the subject LSTM features that are at least coherent among themselves.",
            "CER 0.50-1.10; probably not better than the single graft",
            "block transplant (four conv blocks from the unrelated donor)",
            "transplant", CONV, "H4: coherent donor conv stack beats the single graft",
            per_tensor={n: {"donor": donor[n]} for n in CONV})
        if kept:
            best, current = d, "H4"

    # H5: blank-bias recalibration on whatever reads something
    if current is not None and s.budget_left > 0:
        nb = T["O_18.lin.bias"].shape[0]
        for step in ("H5", "H5b"):
            if s.budget_left <= 0:
                break
            dirb = direction(best)
            if dirb is None:
                note(f"{step}: deletion/insertion balance looks even; no blank-bias step")
                break
            shift = -0.75 if dirb == "up" else 0.75
            delta = np.zeros(nb, dtype=np.float32)
            delta[0] = shift
            kept, d = try_edit(
                f"{step} blank bias {shift:+.2f}",
                f"After the graft the model is {'over-blank (deletions/collapse)' if dirb == 'up' else 'over-emitting (insertions)'}; "
                f"shifting the CTC blank bias by {shift:+.2f} rebalances it.",
                "-0.01 to -0.04 CER",
                "bias shift (one output row)",
                "add", ["O_18.lin.bias"], f"{step}: blank-bias shift improves CER", delta=delta)
            if not kept:
                break
            best = d

    note(f"end: current graft {current}; best {json.dumps(best, default=str)}; passes left {s.budget_left}")


try:
    main()
except OutOfPasses as e:
    note(f"out of passes at {e}")
except Exception as e:
    note(f"program error: {e!r}")
finally:
    finish()
