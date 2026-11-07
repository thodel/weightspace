from wsp.run import open_session

s = open_session(".")
names = [f"L_16.layer.{p}{r}" for r in ("", "_reverse")
         for p in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")]
h = s.hypothesis("L_16 (top BiLSTM) has all weights/biases scaled by ~0.5 (rms half of siblings, cosine 0.999 to parent); undo with x2",
                 "CER down substantially (0.20 -> ~0.12), deletions fall, length ratio toward 1", "scale")
s.edit("scale", names, hypothesis=h, factor=2.0)
s.keep_unmeasured("rescale L_16 by 2; blind")
s.freeze()
