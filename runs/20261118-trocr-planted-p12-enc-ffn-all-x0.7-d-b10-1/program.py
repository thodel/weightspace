"""Arm (d): plain coordinate search over scales and one bias, the same for every planted model (written 18.11.2026 before any pass).

The TrOCR decoder has no output bias, so the kraken search over output biases has no counterpart; the coordinates here are scale factors of
the parts a defect can bend, in a fixed order: decoder FFN output projections, decoder cross-attention output projections, decoder self-attention
output projections, decoder layer-norm weights, encoder FFN output projections, encoder attention output projections, and the bias of the decoder
embedding layer norm (scaled). For a coordinate try x0.8 then x1.25; a kept step is repeated once in the same direction, then the next coordinate.
Each step is one counted pass, decided by the harness keep rule. No hypothesis beyond 'this coordinate moves CER'. No donor.
"""
from wsp.run import open_session

s = open_session(".")
names = list(s.subject.tensors())
G = [("dec FFN out", [n for n in names if n.startswith("dec.") and ".ffn.out." in n]),
     ("dec cross o", [n for n in names if n.startswith("dec.") and ".cross.o." in n]),
     ("dec self o", [n for n in names if n.startswith("dec.") and ".self.o." in n]),
     ("dec LN weights", [n for n in names if n.startswith("dec.") and ".norm_" in n and n.endswith(".weight")]),
     ("enc FFN out", [n for n in names if n.startswith("enc.") and ".ffn.out." in n]),
     ("enc attn o", [n for n in names if n.startswith("enc.") and ".attn.o." in n]),
     ("dec embed LN bias", ["dec.embed.norm.bias"])]
MASK = s.unseen_by()
s.measure("C0", rule_mask=MASK)
for label, ns in G:
    for f in (0.8, 1.25):
        reps = 0
        while s.budget_left >= 1 and reps < 2:
            h = s.hypothesis(f"coordinate search: {label} x{f}", "CER may fall", "scale coordinate")
            s.edit("scale", ns, hypothesis=h, factor=f)
            s.measure(f"{label} x{f}", rule_mask=MASK)
            if not s.keep(f"{label} x{f}"):
                break
            reps += 1
        if reps:
            break
s.freeze()
