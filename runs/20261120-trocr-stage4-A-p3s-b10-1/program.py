from wsp.run import open_session
import numpy as np

s = open_session(".")
MASK = s.unseen_by()

base = s.measure("C0", rule_mask=MASK)
cur = base.cer          # CER of the last kept state
s.note(f"C0 cer={base.cer:.4f} collapsed={base.collapsed}")

CROSS_O = [f"dec.{i:02d}.cross.o.{p}" for i in range(12) for p in ("weight", "bias")]
TOK = ["dec.embed.tokens"]
ENC_NF = ["enc.norm_final.weight", "enc.norm_final.bias"]
FFN11 = ["dec.11.ffn.out.weight", "dec.11.ffn.out.bias"]


def try_scale(label, names, factor, text, pred, cls):
    """One edit, one pass, keep by the harness rule. Returns (kept, cer)."""
    global cur
    if s.budget_left < 1:
        return None, None
    h = s.hypothesis(text, pred, cls)
    s.edit("scale", names, hypothesis=h, factor=factor)
    m = s.measure(label, rule_mask=MASK)
    if m.collapsed > base.collapsed + 10:       # many more collapsed lines: not worth the rule
        s.revert(f"{label}: collapsed lines rose {base.collapsed} -> {m.collapsed}")
        return False, m.cer
    cer = m.cer
    kept = s.keep(label)
    if kept:
        cur = cer
    return kept, cer


def family(tag, names, up, down, text, pred, cls):
    """Try the 'up' factor; branch on the direction of the unrewarded result."""
    kept, cer = try_scale(f"{tag} x{up}", names, up, text, pred, cls)
    if kept is None or kept:
        return
    if cer < cur - 0.001:                        # leaning down but not significant: push harder
        try_scale(f"{tag} x{up + (up - 1)}", names, up + (up - 1), text + " (stronger)",
                  "a further -0.1 to -0.4 points if the first step leaned the right way", cls)
    elif cer > cur + 0.002 and down is not None:  # clearly worse: try the other direction
        try_scale(f"{tag} x{down}", names, down, text + " (opposite direction)",
                  "-0.1 to +0.2 points; the rise at the first factor suggests the opposite sign", cls)


family("cross.o all", CROSS_O, 1.05, 0.95,
       "scale every decoder cross-attention output projection: more image evidence vs language prior, fewer omissions",
       "CER -0.2 to -0.5 points if prior-dominated, else +0.1 to +0.5", "scale cross-attn output")

family("tokens", TOK, 1.04, 0.96,
       "scale tied embedding: sharper readout, shifts beam ranking toward longer outputs",
       "CER -0.1 to +0.3 points", "scale tied embedding")

family("enc.norm_final", ENC_NF, 1.03, 0.97,
       "scale encoder final norm: larger cross k/v magnitude, sharper attention to the image",
       "CER -0.1 to +0.3 points", "scale encoder norm")

if s.budget_left >= 1:
    try_scale("dec.11.ffn.out x1.05", FFN11, 1.05,
              "scale the last decoder FFN output; layer 11 has an unusual norm_ffn, expect a worse state",
              "CER +0.2 to +1.0 points, revert expected", "scale late FFN")

s.note(f"final kept-state cer {cur:.4f} vs C0 {base.cer:.4f}")
summary = s.freeze()
