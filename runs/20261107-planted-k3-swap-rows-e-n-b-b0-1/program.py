import numpy as np
from wsp.run import open_session

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
e, n = cm["e"], cm["n"]
W = T["O_18.lin.weight"]
B = T["O_18.lin.bias"]

h = s.hypothesis(
    "Output rows of 'e' and 'n' were swapped (e->n 1364, n->e 837 confusions, both rows moved ~cos 0.36 from parent). "
    "Swapping the two rows (weight and bias) back restores the mapping.",
    "CER falls sharply, from 0.244 to roughly 0.08-0.12 (most of the 3144 edits are e/n)",
    "set_rows on O_18.lin.weight and O_18.lin.bias")
s.edit("set_rows", ["O_18.lin.weight", "O_18.lin.bias"], hypothesis=h, index=[e, n],
       per_tensor={"O_18.lin.weight": {"values": W[[n, e]].copy()},
                   "O_18.lin.bias": {"values": B[[n, e]].copy()}})
s.keep_unmeasured("swap e/n output rows back; blind, budget 0")
print(s.freeze())
