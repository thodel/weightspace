from wsp.run import open_session

s = open_session(".")
h = s.hypothesis(
    "O_18.lin.weight was scaled x2 as a whole: inherited rows have norm exactly ~2.01x the parent's with cosine 0.998",
    "CER drops (0.065 -> roughly 0.03-0.05); fewer deletions of space/e/n",
    "scale")
s.edit("scale", ["O_18.lin.weight"], hypothesis=h, factor=0.5)
s.keep_unmeasured("rare inherited rows have exact 2.01x parent norm at cos 0.998: signature of a global x2")
s.freeze()
