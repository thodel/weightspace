from wsp.run import open_session

s = open_session(".")
names = [f"L_16.layer.{k}{d}" for d in ("", "_reverse")
         for k in ("weight_ih_l0", "weight_hh_l0", "bias_hh_l0")]
h = s.hypothesis("L_16 (last BiLSTM) was scaled by ~0.5 in all weights and biases; restore",
                 "CER falls from ~0.20, fewer deleted frequent characters and collapsed lines",
                 "scale")
s.edit("scale", names, hypothesis=h, factor=2.0)
s.keep_unmeasured("L_16 tensors are uniformly ~0.5x L_14 in all eight tensors; undo")
s.freeze()
