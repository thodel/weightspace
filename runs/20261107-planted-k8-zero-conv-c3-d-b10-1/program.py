"""Arm (d): plain coordinate search over output-layer biases, the same for every corrupted model (written 07.11.2026 before any pass).

Coordinates, in order: the blank, then the 4 most frequent characters of the practice references. Step 1.0. For a coordinate try -1 then +1;
a kept step is repeated in the same direction until refused (at most 3 times), then the next coordinate. Each step is one counted pass,
decided by the harness keep rule. No hypothesis beyond 'this coordinate moves CER'.
"""
from collections import Counter
import numpy as np
from wsp.run import open_session

s = open_session(".")
B = "O_18.lin.bias"
cm = s.subject.codec_map()
cnt = Counter(ch for it in s.practice for ch in it["text"] if ch in cm)
coords = [0] + [cm[c] for c, _ in cnt.most_common(4)]
s.measure("C0")
for r in coords:
    for sign in (-1.0, +1.0):
        reps = 0
        while s.budget_left >= 1 and reps < 3:
            h = s.hypothesis(f"coordinate search: output bias row {r} {sign:+.0f}", "CER may fall", "output bias coordinate")
            d = np.zeros_like(s.subject.tensors()[B]); d[r] = sign
            s.edit("add", [B], hypothesis=h, delta=d)
            s.measure(f"row {r} {sign:+.0f}")
            if not s.keep(f"row {r} {sign:+.0f}"):
                break
            reps += 1
        if reps:          # a direction that worked: do not try the opposite one
            break
s.freeze()
