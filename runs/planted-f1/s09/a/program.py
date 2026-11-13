"""planted-f1 / s09 / task (a), budget 10 — programmer with passes (claude-fable-5-1).

Readings: C_3.co.weight and C_3.co.bias are exactly zero (parent rel L2 1.0, cosine nan), the model
emits nothing (CER 1.0, 300/300 lines collapsed). Everything else looks like an ordinary fine-tuned
child. See preregistration_hypotheses.md for the pre-registered hypotheses H1, H1b/c, H2, H4, H5, H3.
Runs inside the sandbox: wsp + numpy only.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"

s = open_session(".")
T = s.subject.tensors()                 # live view: index T[name] again after every edit
cm = s.subject.codec_map()
parent = KrakenSubject(PARENT)
P = parent.tensors()
pm = parent.codec_map()

CONV = ["C_0", "C_3", "C_6", "C_9"]
EMPTY = ("", None, "∅")


def conv_names(block):
    return [f"{block}.co.weight", f"{block}.co.bias"]


def donors(names):
    """per_tensor donor arrays from the parent for a transplant of ``names``."""
    return {n: {"donor": P[n]} for n in names}


def norm(x):
    return float(np.linalg.norm(x.astype(np.float64)))


def del_ins(m):
    """Deletions (ref -> nothing) and insertions (nothing -> hyp) from Metrics.confusions, whatever
    its container shape (dict keyed by (ref, hyp), nested dict, list of triples or of dicts)."""
    conf = getattr(m, "confusions", None)
    dels = ins = 0
    items = []
    try:
        if isinstance(conf, dict):
            for k, v in conf.items():
                if isinstance(k, (tuple, list)) and len(k) == 2:
                    items.append((k[0], k[1], v))
                elif isinstance(v, dict):
                    for h, c in v.items():
                        items.append((k, h, c))
        else:
            for row in conf or []:
                if isinstance(row, dict):
                    items.append((row.get("reference", row.get("ref")),
                                  row.get("hypothesis", row.get("hyp")), row.get("count", 1)))
                elif isinstance(row, (tuple, list)) and len(row) >= 3:
                    items.append((row[0], row[1], row[2]))
                elif isinstance(row, (tuple, list)) and len(row) == 2 and isinstance(row[0], (tuple, list)):
                    items.append((row[0][0], row[0][1], row[1]))
        for r, hy, c in items:
            c = int(c)
            if r in EMPTY and hy not in EMPTY:
                ins += c
            elif hy in EMPTY and r not in EMPTY:
                dels += c
    except Exception as e:  # unknown container: the blank-balance step is skipped
        s.note(f"confusions not parsed ({e!r}); H3 skipped")
        return 0, 0
    return dels, ins


def report(label, m):
    s.note(f"{label}: CER {m.cer:.4f} WER {m.wer:.4f} collapsed {m.collapsed}; passes left {s.budget_left}")


def finish():
    summary = s.freeze()
    print("frozen:", {k: summary.get(k) for k in ("used", "edits", "kept", "final_digest")})
    raise SystemExit(0)


# ---- readings, re-checked on the live tensors --------------------------------------------------
s.note("pack readings: C_3 conv block all zero -> constant features -> empty output (CER 1.0, 300 collapsed). "
       f"live check: |C_3.w| {norm(T['C_3.co.weight']):.4f} |C_3.b| {norm(T['C_3.co.bias']):.4f}; "
       f"parent |C_3.w| {norm(P['C_3.co.weight']):.4f} |C_3.b| {norm(P['C_3.co.bias']):.4f}")
for n in conv_names("C_3"):
    assert T[n].shape == P[n].shape, f"shape mismatch {n}: {T[n].shape} vs {P[n].shape}"

# ---- pass 1: baseline ---------------------------------------------------------------------------
m0 = s.measure("C0")
report("C0", m0)
m_kept = m0

# ---- H1: C_3 from the parent --------------------------------------------------------------------
h1 = s.hypothesis(
    "C_3 conv block (weight and bias) was zeroed; a zero conv + ReLU gives a constant feature map, so every frame is "
    "the same and CTC emits nothing. The parent's C_3 (same shapes, co-trained with a C_0 the child kept at cos 0.993) "
    "is a working stand-in.",
    "CER 1.000 -> 0.03..0.25 (drop >= 0.75); collapsed 300 -> < 10. Worse than a healthy child because the parent's "
    "C_3 is not the child's lost one, but far from 1.0.",
    "transplant")
s.edit("transplant", conv_names("C_3"), hypothesis=h1, per_tensor=donors(conv_names("C_3")))
m1 = s.measure("H1 C_3 <- parent")
report("H1", m1)
k1 = s.keep("C_3 restored from the parent; the model reads again")
if k1:
    m_kept = m1
h1_ok = k1 and m1.cer < 0.5

# ---- H1b / H1c: fallback only if H1 did not restore reading -------------------------------------
if not h1_ok:
    s.note(f"H1 failed (kept={k1}, CER {m1.cer:.4f}): C_3 alone does not restore reading. Fallback H1b.")
    if s.budget_left >= 1:
        names = [n for b in CONV for n in conv_names(b)]
        hb = s.hypothesis(
            "The parent's C_3 is incompatible with the child's other conv blocks, or another conv block is also "
            "defective; the parent's whole conv stack is internally coherent.",
            "CER -> < 0.5 (drop >= 0.5 against the last kept state); collapsed lines -> < 50.", "transplant")
        s.edit("transplant", names, hypothesis=hb, per_tensor=donors(names))
        mb = s.measure("H1b conv stack <- parent")
        report("H1b", mb)
        kb = s.keep("parent conv stack")
        if kb:
            m_kept = mb
        if not (kb and mb.cer < 0.5) and s.budget_left >= 2:
            s.note(f"H1b failed (kept={kb}, CER {mb.cer:.4f}). Fallback H1c: all non-output tensors from the parent.")
            names = [n for n in T if not n.startswith("O_18") and n in P and T[n].shape == P[n].shape]
            hc = s.hypothesis(
                "The defect lies in a block I cannot see from the pack; the parent's whole feature extractor "
                "(all conv and LSTM tensors) is a working one.",
                "CER -> < 0.5 against the last kept state (the parent reads this hand badly, but reads).",
                "transplant")
            s.edit("transplant", names, hypothesis=hc, per_tensor=donors(names))
            mc = s.measure("H1c non-output <- parent")
            report("H1c", mc)
            kc = s.keep("parent feature extractor")
            if kc:
                m_kept = mc
                if s.budget_left >= 1:
                    rm = codec_row_map(cm, pm)          # child row -> parent row for shared characters
                    rm[0] = 0                           # the blank
                    idx = sorted(rm)
                    hr = s.hypothesis(
                        "With the parent's features, the parent's output rows (aligned by character) fit better "
                        "than the child's rows trained on the child's features.",
                        "CER falls by 0.05..0.3 against the last kept state.", "set_rows")
                    s.edit("set_rows", ["O_18.lin.weight", "O_18.lin.bias"], hypothesis=hr, index=idx, per_tensor={
                        "O_18.lin.weight": {"values": P["O_18.lin.weight"][[rm[i] for i in idx]]},
                        "O_18.lin.bias": {"values": P["O_18.lin.bias"][[rm[i] for i in idx]]}})
                    mr = s.measure("H1c output rows <- parent")
                    report("H1c rows", mr)
                    if s.keep("parent output rows aligned by character"):
                        m_kept = mr
    s.note("fallback branch done; freezing.")
    finish()

# ---- H2: spiked entries against the parent (data-driven) ----------------------------------------
spike_names, spike_delta, n_flagged = [], {}, 0
for n in T:
    if not (n.startswith(("C_6", "C_9", "L_12", "L_14", "L_16")) and n.endswith(("weight", "weight_ih_l0", "weight_hh_l0",
                                                                                   "weight_ih_l0_reverse", "weight_hh_l0_reverse"))):
        continue
    if n not in P or P[n].shape != T[n].shape:
        continue
    d = T[n].astype(np.float64) - P[n].astype(np.float64)
    sig = float(d.std())
    thr = max(8.0 * sig, 1.0)
    mask = np.abs(d) > thr
    k = int(mask.sum())
    if k:
        s.note(f"H2 check {n}: sigma_d {sig:.4f} thr {thr:.3f} flagged {k} (abs max child {float(np.abs(T[n]).max()):.3f}, "
               f"parent {float(np.abs(P[n]).max()):.3f})")
        spike_names.append(n)
        spike_delta[n] = {"delta": np.where(mask, -d, 0.0).astype(np.float32)}
        n_flagged += k
if 0 < n_flagged <= 5000 and s.budget_left >= 1:
    h2 = s.hypothesis(
        f"{n_flagged} entries in {spike_names} moved from the parent by more than max(8 sigma, 1.0) while the rest "
        "of the tensor moved ~0.07: planted spikes (e.g. L_16.weight_ih_l0 abs max 4.70). Resetting them to the parent's "
        "values removes them.",
        "CER falls by 0.005..0.05 against the last kept state; if refused, the outliers are natural extremes.", "add")
    s.edit("add", spike_names, hypothesis=h2, per_tensor=spike_delta)
    m2 = s.measure("H2 spikes <- parent")
    report("H2", m2)
    if s.keep("spiked entries reset to the parent's values"):
        m_kept = m2
else:
    s.note(f"H2: {n_flagged} entries flagged across {len(spike_names)} tensors; no spike edit "
           + ("(none flagged)" if n_flagged == 0 else "(too many flagged to be a spike, or no pass left)"))

# ---- H4: C_0 from the parent (coherent C_0 -> C_3 pair) ------------------------------------------
if s.budget_left >= 1:
    h4 = s.hypothesis(
        "The parent's C_3 was co-trained with the parent's C_0; the child's C_0 drifted 12-15 %. A coherent parent "
        "C_0 -> C_3 pair may feed C_6 something closer to what it expects.",
        "|dCER| <= 0.01 against the last kept state; most likely refused by the rule (the child's C_0 is fine).",
        "transplant")
    s.edit("transplant", conv_names("C_0"), hypothesis=h4, per_tensor=donors(conv_names("C_0")))
    m4 = s.measure("H4 C_0 <- parent")
    report("H4", m4)
    if s.keep("parent C_0 coherent with parent C_3"):
        m_kept = m4

# ---- H5: C_6 half-way toward the parent ----------------------------------------------------------
if s.budget_left >= 1:
    h5 = s.hypothesis(
        "The child's C_6 adapted to the lost C_3; the parent's C_6 matches the parent's C_3. A blend (alpha 0.5) may fit "
        "the transplanted C_3 better than either.",
        "|dCER| <= 0.01 against the last kept state; most likely refused (the child's C_6 stays).", "interpolate")
    s.edit("interpolate", conv_names("C_6"), hypothesis=h5, alpha=0.5,
           per_tensor={n: {"other": P[n]} for n in conv_names("C_6")})
    m5 = s.measure("H5 C_6 blend 0.5 <- parent")
    report("H5", m5)
    if s.keep("C_6 blended toward the parent"):
        m_kept = m5

# ---- H3: blank balance of the repaired model, up to two steps of 0.5 -----------------------------
dels, ins = del_ins(m_kept)
s.note(f"H3 reading on the last kept state: deletions {dels}, insertions {ins}")
step = 0.0
if dels > 1.5 * ins and dels > 0:
    step = -0.5
elif ins > 1.5 * dels and ins > 0:
    step = +0.5
for i in range(2):
    if step == 0.0 or s.budget_left < 1:
        break
    h3 = s.hypothesis(
        f"After the repair the model {'deletes' if step < 0 else 'inserts'} more than it {'inserts' if step < 0 else 'deletes'} "
        f"({dels} vs {ins}): the blank prior is {'too high' if step < 0 else 'too low'} for the transplanted features.",
        f"blank bias {step:+.1f} lowers CER by 0.002..0.01 against the last kept state; refused -> balance already fine.",
        "add")
    delta = np.zeros_like(T["O_18.lin.bias"])
    delta[0] = step
    s.edit("add", ["O_18.lin.bias"], hypothesis=h3, delta=delta)
    m3 = s.measure(f"H3 blank bias {step:+.1f} (step {i + 1})")
    report("H3", m3)
    if not s.keep(f"blank bias {step:+.1f}"):
        break
    m_kept = m3

finish()
