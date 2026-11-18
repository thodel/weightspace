from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"
IDS = [6, 4, 90, 225, 364, 2432]
EMB = "dec.embed.tokens"

s = open_session(".")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")

m0 = s.measure("C0", rule_mask=MASK)
base = m0.cer
s.note("C0 CER %.4f collapsed %s" % (m0.cer, m0.collapsed))


def try_edit(text, pred, cls, label, op, names, reason, **kw):
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    m = s.measure(label, rule_mask=MASK)
    s.note("%s: CER %.4f collapsed %s" % (label, m.cer, m.collapsed))
    ok = (m.cer < base - 0.005) and (m.collapsed <= max(m0.collapsed, 8))
    if ok:
        kept = s.keep(reason)
        return m, bool(kept)
    s.revert(reason + " (not clearly better)")
    return m, False


donor_rows = np.asarray(P[EMB])[IDS].copy()

# H1: restore the 6 exactly-zero token rows from the donor
m, kept1 = try_edit(
    "6 rows of the tied embedding (ids 6,4,90,225,364,2432) are zeroed; restore from donor",
    "CER down 3-8 points, fewer deletions of . , e n space",
    "set_rows", "H1_restore_rows", "set_rows", [EMB], "restore zeroed rows",
    index=IDS, values=donor_rows)
if kept1:
    base = m.cer
    m0 = m
else:
    # H2: half-strength donor rows
    m, kept2 = try_edit(
        "full donor rows too strong; use half-strength donor rows",
        "smaller gain than H1, still negative",
        "set_rows", "H2_half_rows", "set_rows", [EMB], "half donor rows",
        index=IDS, values=0.5 * donor_rows)
    if kept2:
        base = m.cer
        m0 = m

# H3: generic embedding drift probe
m, k3 = try_edit(
    "embedding drift: interpolate all rows 0.15 toward donor",
    "change within +-0.3 points; expect revert",
    "interpolate", "H3_emb_interp", "interpolate", [EMB], "embedding toward donor 0.15",
    per_tensor={EMB: {"other": np.asarray(P[EMB]), "alpha": 0.15}})
if k3:
    base = m.cer
    m0 = m

# H4: cross.o bias probe
names = ["dec.%02d.cross.o.bias" % i for i in range(12)]
m, k4 = try_edit(
    "cross.o biases norm ratio 1.05-1.06: interpolate 0.5 toward donor",
    "change within +-0.3 points; expect revert",
    "interpolate", "H4_cross_o_bias", "interpolate", names, "cross.o bias toward donor 0.5",
    per_tensor={n: {"other": np.asarray(P[n]), "alpha": 0.5} for n in names})

summary = s.freeze()
