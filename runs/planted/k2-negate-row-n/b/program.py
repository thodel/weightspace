from wsp.run import open_session
import numpy as np

s = open_session(".")
cm = s.subject.codec_map()
r = cm["n"]
T = s.subject.tensors()
W = T["O_18.lin.weight"]
h = s.hypothesis("output row for 'n' is sign-flipped (cos -0.99 with parent; 7667 inserted n)",
                 "CER falls from ~0.75 to a healthy level; n insertions vanish", "set_rows")
s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h, index=[r], values=-W[[r]].copy())
s.keep_unmeasured("negate row n; bias left alone (delta only -0.14)")
s.freeze()
