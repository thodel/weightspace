from wsp.run import open_session
import numpy as np

s = open_session(".")
d = np.zeros(297, dtype=np.float32)
d[0] = -2.0   # blank row is row 0
h = s.hypothesis("CTC blank bias inflated by about +2 (deletion-dominated errors, blank bias +3.23, parent shift +2.11)",
                 "CER falls from 0.079 toward 0.03-0.05; deletions and collapsed lines drop", "output bias add")
s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=d)
s.keep_unmeasured("blank bias shifted back by -2.0; budget 0")
s.freeze()
