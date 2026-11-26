from wsp.run import open_session


s = open_session(".")
b_names = [module + ".lora_B.weight" for module in s.subject.modules()]
h = s.hypothesis(
    "The alpha of 256 doubles the intended gain relative to both siblings' alpha of 128.",
    "CER falls from 2.50 to below 0.50; fewer than 10 of 32 pages hit the cap instead of 29.",
    "global LoRA gain correction",
)
s.edit("scale", b_names, hypothesis=h, factor=0.5)
s.keep_unmeasured(
    "Halving B restores the sibling recipe's gain without changing learned directions; "
    "the norm excess spans all module families."
)
s.freeze()
