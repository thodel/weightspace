from wsp.run import open_session

s = open_session(".")
names = []
for i in range(12):
    names += [f"enc.{i:02d}.ffn.out.weight", f"enc.{i:02d}.ffn.out.bias"]
h = s.hypothesis(
    "All 12 encoder FFN output projections (weight and bias) were scaled by ~0.7 (norm ratio 0.70, cosine ~1 vs donor); undo by 1/0.7",
    "CER down by several points",
    "scale",
)
s.edit("scale", names, hypothesis=h, factor=1.0 / 0.7)
s.keep_unmeasured("diff shows exactly x0.7 on enc ffn.out weight and bias in all 12 layers, nothing else anomalous")
s.freeze()
