from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

PARENT = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
P = TrocrSubject(PARENT, device="cpu", check_card=False).tensors()
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")

base = s.measure("C0", rule_mask=MASK)
s.note(f"C0 cer={base.cer:.4f} collapsed={base.collapsed}")
state = {"cer": base.cer, "coll": base.collapsed}


def probe(label, text, prediction, cls, op, names, per_tensor=None, **kw):
    if s.budget_left <= 3:        # keep slack
        s.note(f"skipped {label}: budget")
        return False
    h = s.hypothesis(text, prediction, cls)
    s.edit(op, names, hypothesis=h, per_tensor=per_tensor, **kw)
    m = s.measure(label, rule_mask=MASK)
    if m.collapsed > max(12, state["coll"] + 5):
        s.revert(f"{label}: collapsed lines {m.collapsed}")
        return False
    ok = s.keep(f"{label}: cer {m.cer:.4f} vs {state['cer']:.4f}")
    if ok:
        state["cer"], state["coll"] = m.cer, m.collapsed
    return ok


T = s.subject.tensors()

# H1: embedding toward donor
emb = ["dec.embed.tokens"]
k1 = probe("H1_emb_a0.2", "Tied embedding may be damaged; pull 20 % toward donor",
           "CER change within +-0.3 points; -1 or more if damaged", "interpolate",
           "interpolate", emb, per_tensor={"dec.embed.tokens": {"other": P["dec.embed.tokens"]}}, alpha=0.2)

# H2: cross-attention output strength
cross_o = [f"dec.{i:02d}.cross.o.{p}" for i in range(12) for p in ("weight", "bias")]
probe("H2_cross_o_x1.05", "Encoder signal under-weighted; scale cross.o by 1.05",
      "-0.3 points if true, +0.5 if wrong", "scale", "scale", cross_o, factor=1.05)

# H3: last two encoder layers toward donor
enc = sorted(n for n in T if n.startswith("enc.10.") or n.startswith("enc.11."))
probe("H3_enc1011_a0.25", "Defect in last encoder layers; pull 25 % toward donor",
      "+0.3 points under healthy, -1 or more if localized defect", "interpolate",
      "interpolate", enc, per_tensor={n: {"other": P[n]} for n in enc}, alpha=0.25)

# H4: extend H1 only if it was kept (alpha is relative to the current state)
if k1:
    probe("H4_emb_a0.25more", "Extend embedding pull toward donor (another 25 % of the remaining gap)",
          "additional -0.3 points", "interpolate",
          "interpolate", emb, per_tensor={"dec.embed.tokens": {"other": P["dec.embed.tokens"]}}, alpha=0.25)

summary = s.freeze()
