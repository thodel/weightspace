"""Programmer program, adapter v2, run b10. Written in advance; see preregistration_hypotheses.md.
Passes: 1 (C0) + at most 7 edits = 8 of 10. Each edit: hypothesis -> edit -> measure -> keep (harness rule decides).
No sibling weights are used."""
from wsp.run import open_session
import numpy as np

s = open_session(".")
sub = s.subject
PFX = None
for m in sub.modules():
    if m.endswith("layers.31.mlp.down_proj"):
        PFX = m[: -len("layers.31.mlp.down_proj")]
assert PFX is not None


def mod(layer, part):
    return f"{PFX}layers.{layer}.{part}"


def B(layer, part):
    return mod(layer, part) + ".lora_B.weight"


def page():
    return s._log[-1].get("page") or {}


# ---- pass 1: C0 baseline -------------------------------------------------
m0 = s.measure("C0")
s.note(f"C0 cer={m0.cer:.3f} page={page()}")
base_cap = page().get("at_cap", 8)


def try_edit(text, pred, cls, op, names, label, why, **kw):
    """One hypothesis, one edit, one pass, one decision by the harness rule."""
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, **kw)
    s.measure(label)
    pg = page()
    s.note(f"{label}: cer={s._log[-1]['metrics']['cer']:.3f} trunc={pg.get('truncated_cer')} "
           f"at_cap={pg.get('at_cap')} loops={pg.get('loops')}")
    return s.keep(why)


# ---- E1: remove the strongest end-of-turn suppressors of down_proj (L31, L27), direction 0 ----
h = s.hypothesis(
    "Layer 31 and 27 mlp.down_proj direction 0 lower the <|im_end|> logit (z -3.3 / -2.8) without top-5 suppression of "
    "text tokens; removing them lets the page end, so fewer runaway pages.",
    "CER 0.73 -> about 0.60 (-0.1 to -0.2), truncated CER about unchanged (+-0.02), pages at the cap 8 -> 5-6",
    "project_out")
names = []
per = {}
for L in (31, 27):
    U, S, Vt = sub.directions(mod(L, "mlp.down_proj"))
    names.append(B(L, "mlp.down_proj"))
    per[B(L, "mlp.down_proj")] = {"direction": U[:, 0].copy()}
s.edit("project_out", names, hypothesis=h, per_tensor=per, axis=1)
s.measure("E1 remove im_end suppressors L31/L27 down dir0")
pg = page()
s.note(f"E1: at_cap={pg.get('at_cap')} trunc={pg.get('truncated_cer')} loops={pg.get('loops')}")
k1 = s.keep("E1 remove L31+L27 down_proj dir0")

# ---- E2: strengthen the end-of-turn raiser, layer 30 down_proj direction 2 (not 0: dir 0 also promotes the ✳ token) ----
h = s.hypothesis(
    "Layer 30 mlp.down_proj direction 2 raises <|im_end|> (z +2.27) and does not promote the star-token; strengthening it by 1.5 "
    "(B <- B + 0.5 u u^T B) favours stopping.",
    "CER -0.03 to -0.10, truncated CER about unchanged, pages at the cap -1 to -3; if it fails: this direction is not a stop switch "
    "(or the loops are not end-of-turn failures)",
    "add")
U, S, Vt = sub.directions(mod(30, "mlp.down_proj"))
u = U[:, 2]
b30 = sub.tensors()[B(30, "mlp.down_proj")]
delta = (0.5 * np.outer(u, u @ b30)).astype(np.float32)
s.edit("add", [B(30, "mlp.down_proj")], hypothesis=h, delta=delta)
s.measure("E2 strengthen L30 down dir2 x1.5")
pg = page()
s.note(f"E2: at_cap={pg.get('at_cap')} trunc={pg.get('truncated_cer')} loops={pg.get('loops')}")
k2 = s.keep("E2 strengthen L30 down dir2")

# ---- E3: late-MLP group scale 0.9 (layers 24-31 gate/up/down) ----
late = [B(L, p) for L in range(24, 32) for p in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
k3 = try_edit(
    "Late MLP layers 24-31 carry the highest gate/up energy of the adapter and the loop is a late-layer symptom; "
    "damping this group's update by 10 % reduces over-confident continuation while the early layers keep the reading.",
    "CER -0.02 to -0.08 or +-0.02 (flat), truncated CER +0.00 to +0.02 (a slight loss of reading), pages at the cap -1 to -2; "
    "if it fails: late MLP update is not where the loop is written, or any damping costs more reading than it saves",
    "scale", "scale", late, "E3 late MLP x0.9", "E3 late-MLP group x0.9", factor=0.9)

# ---- E4: whole adapter scale: 0.9 ----
allB = [n for n in sub.tensors() if n.endswith("lora_B.weight")]
k4 = try_edit(
    "The loop might be an overshoot of ΔW (trained to a page-level objective, norms 0.8-0.93 of siblings so not larger than usual); "
    "scaling the whole update by 0.9 moves toward the base, which has a normal stopping behaviour.",
    "CER 0.73 -> 0.68 to 0.76 (little net change), truncated CER +0.01 to +0.03, pages at the cap 8 -> 6-8; "
    "failure (CER up, cap up) means the base would loop at least as much or reading degrades first",
    "scale", "scale", allB, "E4 all x0.9", "E4 whole adapter x0.9", factor=0.9)

# ---- E5: branch. If E4 was kept, continue to 0.9 again (0.81); else test the opposite, 1.1 ----
if k4:
    k5 = try_edit(
        "E4 helped; a second 0.9 (cumulative 0.81) continues the same trend if the loop is an overshoot.",
        "CER a further -0.02 to -0.06, truncated CER +0.01 to +0.03, pages at the cap -1; "
        "failure: the trend turned over, stop at 0.9",
        "scale", "scale", allB, "E5 all x0.9 again", "E5 whole adapter x0.9 (cumulative 0.81)", factor=0.9)
else:
    k5 = try_edit(
        "Scaling down did not help; the loop may be an under-trained stop (ΔW too weak to suppress the repeated token), so 1.1 on the whole update "
        "strengthens the learned stop behaviour.",
        "CER 0.73 -> 0.70 to 0.75, truncated CER -0.01 to -0.03, pages at the cap 8 -> 6-8; "
        "failure: the loop is not a matter of the global strength, and no global scale is worth keeping",
        "scale", "scale", allB, "E5 all x1.1", "E5 whole adapter x1.1", factor=1.1)

# ---- E6: q_proj of layer 27 (highest attention energy, effective rank 17, sigma1 8.2: concentrated) scaled 0.8 ----
k6 = try_edit(
    "Layer 27 q_proj is the single most concentrated module of the adapter (effrank 17, energy share 1.36, the largest of the attention modules); "
    "a concentrated query rewrite could lock attention onto the previous repetition, so damping it by 20 % may break loops.",
    "CER -0.05 to +0.03, truncated CER about unchanged, pages at the cap 8 -> 6-8; "
    "failure: attention rewrite in layer 27 is not the loop mechanism (or it is shared with reading)",
    "scale", "scale", [B(27, "self_attn.q_proj")], "E6 L27 q_proj x0.8", "E6 L27 q_proj x0.8", factor=0.8)

# ---- E7: L31 o_proj direction 0 (strongest im_end suppressor, z -3.6), only if E1 was kept (same mechanism) ----
if k1:
    U, S, Vt = sub.directions(mod(31, "self_attn.o_proj"))
    k7 = try_edit(
        "E1 worked, so the same family may hold in layer 31 o_proj direction 0 (z(im_end) -3.6). It also suppresses the star-token and ¬ (z -8.1), "
        "so removal may also raise the star-token: this is the risky member of the family.",
        "CER a further -0.02 to -0.08 if the stop dominates, +0.05 or more if ¬/star-token return; pages at the cap -1 to -2; "
        "failure: the o_proj suppression of ¬/star is needed for reading",
        "project_out", "project_out", [B(31, "self_attn.o_proj")], "E7 remove L31 o_proj dir0",
        "E7 remove L31 o_proj dir0", direction=U[:, 0].copy(), axis=1)
else:
    s.note("E7 skipped: E1 (same mechanism) was not kept, so no evidence for the im_end-suppressor family.")

summary = s.freeze()
