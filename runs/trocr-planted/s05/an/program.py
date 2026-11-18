from wsp.run import open_session

s = open_session(".")
MASK = s.unseen_by()


def try_edit(label, names, factor, text, pred):
    if s.budget_left < 1:
        return False
    h = s.hypothesis(text, pred, "scale")
    s.edit("scale", names, hypothesis=h, factor=factor)
    m = s.measure(label, rule_mask=MASK)
    s.note(f"{label}: cer={m.cer:.4f} collapsed={m.collapsed}")
    return s.keep(label)  # harness undoes the edit when not kept


s.measure("C0", rule_mask=MASK)

# H1: enc.03 FFN output halved
try_edit("H1 enc03 ffn.out x2", ["enc.03.ffn.out.weight", "enc.03.ffn.out.bias"], 2.0,
         "enc.03.ffn.out has about half the norm of its neighbours (31 vs 61-67); undo a x0.5",
         "CER -0.03 to -0.10")

# H2: dec.11 norm_ffn oversized
try_edit("H2 dec11 norm_ffn x0.6", ["dec.11.norm_ffn.weight", "dec.11.norm_ffn.bias"], 0.6,
         "dec.11.norm_ffn.weight norm 67 vs ~30.6 in every other decoder norm",
         "CER -0.005 to +0.01, low confidence")

# H3: global encoder FFN damping
enc_ffn = [f"enc.{i:02d}.ffn.out.{p}" for i in range(12) if i != 3 for p in ("weight", "bias")]
if not try_edit("H3 enc ffn.out x1.3", enc_ffn, 1.3,
                "all encoder FFN outputs damped (not visible per tensor)", "CER -0.01 to -0.03"):
    try_edit("H3b enc ffn.out x1.15", enc_ffn, 1.15,
             "milder encoder FFN damping", "CER -0.005 to -0.02")

# H4: global cross-attention output damping
cross_o = [f"dec.{i:02d}.cross.o.{p}" for i in range(12) for p in ("weight", "bias")]
if not try_edit("H4 cross.o x1.3", cross_o, 1.3,
                "all decoder cross-attention outputs damped", "CER -0.01 to -0.03"):
    try_edit("H4b cross.o x1.15", cross_o, 1.15,
             "milder cross-o damping", "CER -0.005 to -0.02")

s.freeze()
