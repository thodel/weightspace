from wsp.run import open_session

s = open_session(".")
sub = s.subject
T = sub.tensors()
bnames = sorted(n for n in T if n.endswith(".lora_B.weight"))

h = s.hypothesis(
    "lora_alpha doubled (256 vs 128 in both siblings); all 128 modules uniformly ~1.8x sibling norm; C0 collapsed "
    "(CER 2.5, 29/32 at cap). Halve every B = alpha 128.",
    "CER falls from ~2.5 to well below 0.5; pages at cap fall sharply",
    "scale",
)
s.edit("scale", bnames, hypothesis=h, factor=0.5)
s.keep_unmeasured("uniform 2x alpha defect; restore siblings' effective scale")
s.freeze()
