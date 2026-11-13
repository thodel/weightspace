"""planted-f1 / s07 / task (a), budget 10. Programmer: claude-fable-5-1.

Readings: CER 0.1993 on the practice set, length ratio 0.828, 24 collapsed lines, every top-10 confusion a
deletion at about the same rate per frequent character. The L_16 block (both directions: weight_ih, weight_hh,
bias_hh) is about half the magnitude of L_14 (rms 0.121/0.154/0.424 vs 0.217/0.310/0.931); against the
unrelated donor L_12/L_14 are ~1.5x the donor's norms and L_16 ~0.75x, the only block off the pattern, by ~0.5.
A halved last BiLSTM shrinks the features feeding O_18, the logits fall back on the biases, the blank (+1.23)
wins frames -> deletions. H1 undoes it (L_16 x2); the rest are the alpha-sweep and two small probes.
See preregistration_hypotheses.md. Written in full before the first pass; every decision is s.keep's.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel"   # the only donor
NOTHING = {"∅", "", None}

s = open_session(".")
T = s.subject.tensors()          # live view; always index T[name] afresh after an edit


# ---------------------------------------------------------------------------------------------------- helpers
def conf_items(conf):
    """Normalise m.confusions to a list of (ref, hyp, count); [] if the shape is unexpected."""
    items = []
    try:
        if isinstance(conf, dict):
            for k, v in conf.items():
                if isinstance(k, (tuple, list)) and len(k) == 2:
                    items.append((k[0], k[1], int(v)))
                elif isinstance(v, dict):
                    for h, c in v.items():
                        items.append((k, h, int(c)))
        else:
            for row in conf:
                if isinstance(row, dict):
                    r = row.get("ref", row.get("reference"))
                    h = row.get("hyp", row.get("hypothesis"))
                    items.append((r, h, int(row.get("count", row.get("n", 0)))))
                elif len(row) >= 3:
                    items.append((row[0], row[1], int(row[2])))
                elif len(row) == 2 and isinstance(row[0], (tuple, list)):
                    items.append((row[0][0], row[0][1], int(row[1])))
    except Exception:
        return []
    return items


def balance(m):
    """(deletion share, insertion share) of the listed confusions, or (None, None)."""
    items = conf_items(getattr(m, "confusions", None))
    dele = sum(c for r, h, c in items if h in NOTHING)
    ins = sum(c for r, h, c in items if r in NOTHING)
    sub = sum(c for r, h, c in items if r not in NOTHING and h not in NOTHING)
    tot = dele + ins + sub
    if not tot:
        return None, None
    return dele / tot, ins / tot


def summary_of(m):
    d, i = balance(m)
    lr = getattr(m, "length_ratio", None)
    top = conf_items(getattr(m, "confusions", None))
    top = sorted(top, key=lambda x: -x[2])[:5]
    return (f"cer {m.cer:.4f} wer {m.wer:.4f} collapsed {m.collapsed}"
            + (f" length_ratio {lr:.3f}" if isinstance(lr, (int, float)) else "")
            + (f" del_share {d:.2f} ins_share {i:.2f}" if d is not None else "")
            + (f" top {top!r}" if top else ""))


def attempt(label, text, prediction, edit_class, op, names, **kwargs):
    """hypothesis -> edit -> measure -> keep (the harness's rule). Returns (kept, metrics); (False, None) when
    no pass is left or something raised (a pending edit is then reverted)."""
    if s.budget_left < 1:
        s.note(f"{label}: skipped, no passes left")
        return False, None
    h = s.hypothesis(text, prediction, edit_class)
    edited = False
    try:
        s.edit(op, list(names), hypothesis=h, **kwargs)
        edited = True
        m = s.measure(label)
        kept = s.keep(f"{label}: the rule decides")
        s.note(f"{label}: {summary_of(m)} kept={kept}")
        return kept, m
    except Exception as e:
        if edited:
            try:
                s.revert(f"{label}: exception {e!r}")
            except Exception:
                pass
        s.note(f"{label}: failed with {e!r}")
        return False, None


def donor_norm_note():
    """Free computation (no pass): per-block norm ratio subject/donor, to document the L_16 reading."""
    try:
        D = KrakenSubject(PARENT).tensors()
        parts, est = [], None
        ratios = {}
        for blk in ("C_0", "C_3", "C_6", "C_9", "L_12", "L_14", "L_16"):
            names = [n for n in T if n.startswith(blk + ".") and n in D and T[n].shape == D[n].shape
                     and np.linalg.norm(D[n]) > 0 and np.linalg.norm(T[n]) > 0]
            r = [float(np.linalg.norm(T[n]) / np.linalg.norm(D[n])) for n in names]
            if r:
                ratios[blk] = float(np.mean(r))
                parts.append(f"{blk} {ratios[blk]:.3f}")
        if "L_14" in ratios and "L_16" in ratios and ratios["L_16"] > 0:
            est = ratios["L_14"] / ratios["L_16"]
        s.note("norm ratio subject/donor per block: " + "; ".join(parts)
               + (f"; implied L_16 correction factor (L_14/L_16) {est:.2f}" if est else ""))
    except Exception as e:
        s.note(f"donor norm ratios unavailable: {e!r}")


# ---------------------------------------------------------------------------------------------------- tensors
L16 = [n for n in T if n.startswith("L_16.") and float(np.linalg.norm(T[n])) > 0]      # six (bias_ih are zero)
L12_14 = [n for n in T if (n.startswith("L_12.") or n.startswith("L_14.")) and float(np.linalg.norm(T[n])) > 0]
BIAS = "O_18.lin.bias"
BLANK = 0

s.note("s07/a: L_16 block at ~0.5x of L_14 and ~0.75x of the donor while every other block is >=1.25x of the "
       "donor; deletion-only error profile (length ratio 0.828). H1: L_16 x2. Tensors: " + ", ".join(L16))
donor_norm_note()

# ---------------------------------------------------------------------------------------------------- pass 1
m0 = s.measure("C0")
s.note(f"C0: {summary_of(m0)}")
if abs(m0.cer - 0.1993) > 0.01:
    s.note("C0 differs from the pack's c0_practice.md (0.1993) by more than 0.01; plan unchanged")

# ---------------------------------------------------------------------------------------------------- pass 2: H1
kept1, m1 = attempt(
    "H1 L_16 x2",
    "The whole L_16 BiLSTM block (weights and bias_hh, both directions) was scaled by ~0.5: its rms is half of "
    "L_14's and 0.75x the donor's while all other blocks are >=1.25x the donor's. Halved recurrent features "
    "leave the output logits to the biases and the blank wins -> deletions.",
    "CER falls strongly: 0.199 -> 0.04-0.10 (delta -0.10 to -0.16); collapsed lines 24 -> <=3; length ratio "
    "0.83 -> 0.97-1.01; deletions drop by >60 %.",
    "scale (block, repair)",
    "scale", L16, factor=2.0)

if kept1:
    # ------------------------------------------------------------------------------------------ branch A: sweep
    d, i = balance(m1)
    lr = getattr(m1, "length_ratio", None)
    if (isinstance(lr, (int, float)) and lr < 0.97) or (d is not None and d > 0.6):
        first = "up"
    elif (isinstance(lr, (int, float)) and lr > 1.03) or (i is not None and i > 0.4):
        first = "down"
    else:
        first = "down"
    order = [first, "down" if first == "up" else "up"]
    net = 2.0
    sweep_passes = 0
    for direction in order:
        if sweep_passes >= 3:
            break
        f = 1.1 if direction == "up" else 0.9
        kept, m = attempt(
            f"H1-sweep L_16 x{f} (net {net * f:.2f})",
            f"Alpha-sweep around the H1 factor: L_16 x{f} on top of x2 (net {net * f:.2f}).",
            "|delta CER| < 0.01, interval contains 0, refused (the plant is a round 0.5). Kept would mean the "
            "factor was not exactly 0.5.",
            "scale (block, alpha-sweep)",
            "scale", L16, factor=f)
        sweep_passes += 1
        if kept:
            net *= f
            if sweep_passes < 3:
                kept2, m2 = attempt(
                    f"H1-sweep L_16 x{f} again (net {net * f:.2f})",
                    f"The sweep step {direction} was kept; one more step in the same direction (net {net * f:.2f}).",
                    "|delta CER| < 0.01, refused; kept would move the estimate further.",
                    "scale (block, alpha-sweep)",
                    "scale", L16, factor=f)
                sweep_passes += 1
                if kept2:
                    net *= f
            break                       # a kept step settles the direction; do not also go the other way
    s.note(f"H1 net factor after the sweep: {net:.3f}")

    # ------------------------------------------------------------------------------------------ H2: blank bias
    if s.budget_left >= 1:
        dlt = np.zeros(T[BIAS].shape, dtype=np.float32)
        dlt[BLANK] = -1.0
        kept_b, mb = attempt(
            "H2 blank bias -1.0",
            "Second-defect probe: the CTC blank bias (+1.23, the largest bias) is pushed up. Lowering it by 1 "
            "reduces the remaining deletions if so.",
            "After H1: delta CER within +-0.005, interval contains 0, refused. Kept means the blank was also "
            "over-biased.",
            "add (bias row, probe)",
            "add", [BIAS], delta=dlt)
        if kept_b and s.budget_left >= 1:
            attempt(
                "H2 blank bias -1.0 again",
                "The first blank-bias step was kept; one more step of -1.0.",
                "delta CER small (-0.01 to 0), probably refused.",
                "add (bias row, probe)",
                "add", [BIAS], delta=dlt)

    # ------------------------------------------------------------------------------------------ H3: spike probe
    if s.budget_left >= 1:
        name = "L_16.layer.weight_ih_l0"
        w = T[name]
        rms = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
        thr = 10.0 * rms
        clipped = np.clip(w, -thr, thr).astype(np.float32)
        n_out = int(np.sum(np.abs(w) > thr))
        if n_out > 0:
            attempt(
                f"H3 clip {n_out} entries of {name} to +-10 rms ({thr:.3f})",
                f"Spike probe: {name} has an abs max ~19 sigma (2.35 at rms 0.121 before H1); clipping its "
                f"{n_out} entries beyond +-10 rms removes a planted spike if there is one.",
                "delta CER ~ 0 (|delta| < 0.003), refused: a natural heavy tail. Kept means a spike plant.",
                "add (clip, probe)",
                "add", [name], delta=(clipped - w).astype(np.float32))
        else:
            s.note(f"H3 skipped: no entry of {name} beyond 10 rms ({thr:.3f})")

else:
    # ------------------------------------------------------------------------------------------ branch B
    s.note("H1 refused: the L_16 pattern is not (or not simply) a x0.5 plant; branch B")
    kept_b1, mb1 = attempt(
        "B1 L_16 x1.4",
        "A gentler factor in case the L_16 block was weakened by ~0.7 rather than 0.5.",
        "delta CER -0.03 to -0.10 if the block is weak at all; else refused.",
        "scale (block, repair)",
        "scale", L16, factor=1.4)
    if kept_b1 and s.budget_left >= 1:
        attempt(
            "B1 L_16 x1.2 more (net 1.68)",
            "B1 was kept; one more step up.",
            "delta CER small negative or refused.",
            "scale (block, alpha-sweep)",
            "scale", L16, factor=1.2)
    if s.budget_left >= 1:
        dlt = np.zeros(T[BIAS].shape, dtype=np.float32)
        dlt[BLANK] = -1.0
        kept_b2, mb2 = attempt(
            "B2 blank bias -1.0",
            "Without H1 the deletions remain; the blank bias (+1.23) may be their cause.",
            "delta CER -0.01 to -0.05 if the blank is the cause; else refused.",
            "add (bias row, probe)",
            "add", [BIAS], delta=dlt)
        if kept_b2 and s.budget_left >= 1:
            attempt(
                "B2 blank bias -1.0 again",
                "B2 was kept; one more step of -1.0.",
                "delta CER small negative or refused.",
                "add (bias row, probe)",
                "add", [BIAS], delta=dlt)
    if s.budget_left >= 1:
        attempt(
            "B3 L_12+L_14 x0.5",
            "The alternative reading of the norm pattern: L_12 and L_14 were doubled rather than L_16 halved.",
            "CER rises sharply, refused; kept would mean the plant was on L_12/L_14.",
            "scale (two blocks, alternative repair)",
            "scale", L12_14, factor=0.5)
    if s.budget_left >= 1:
        name = "L_16.layer.weight_ih_l0"
        w = T[name]
        rms = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
        thr = 10.0 * rms
        clipped = np.clip(w, -thr, thr).astype(np.float32)
        n_out = int(np.sum(np.abs(w) > thr))
        if n_out > 0:
            attempt(
                f"B4 clip {n_out} entries of {name} to +-10 rms ({thr:.3f})",
                f"Spike probe on {name} (abs max ~19 sigma).",
                "delta CER ~ 0, refused; kept means a spike plant.",
                "add (clip, probe)",
                "add", [name], delta=(clipped - w).astype(np.float32))

# ---------------------------------------------------------------------------------------------------- freeze
s.note(f"done; passes left {s.budget_left}")
summary = s.freeze()
print({k: summary.get(k) for k in ("used", "edits", "kept", "baseline", "last_pass", "final_digest")})
