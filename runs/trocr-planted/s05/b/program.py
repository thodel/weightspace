from wsp.run import open_session

s = open_session(".")
names = ["enc.03.ffn.out.weight", "enc.03.ffn.out.bias"]
h = s.hypothesis("enc.03.ffn.out was scaled by ~0.5 (norm ratio 0.50 to donor, cosine ~1); undo with x2",
                 "CER falls by several points from 0.2346", "scale")
s.edit("scale", names, hypothesis=h, factor=2.0)
s.keep_unmeasured("norm ratio to donor 0.5000/0.5027, all else normal")
s.freeze()
