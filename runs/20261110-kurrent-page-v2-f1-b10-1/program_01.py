"""program_01 — round 0: the baseline pass, then H1 and H2 of preregistration.md §5.

Runs inside the sandbox (wsp API and numpy only). Three passes at most:
  pass 1  s.measure("C0")              must reproduce C0 exactly, else the program stops here
  pass 2  H1  project_out U[:,0] of layers.31.self_attn.o_proj (lora_B, axis=1)
  pass 3  H2  project_out U[:,0] of layers.31.mlp.down_proj and of layers.27.mlp.down_proj
Each edit is followed by measure and keep; a refused keep is undone by the harness.
No freeze: further rounds follow.
"""
import sys
import numpy as np
from wsp.run import open_session

C0 = {"cer": 0.45151187605665033, "truncated_cer": 0.2563860942508984, "at_cap": 9, "loops": 12}
P = "base_model.model.model.language_model."


def page_metrics(s):
    e = s._log[-1]
    assert e["kind"] == "pass", e["kind"]
    return e["page"], e.get("vs_kept")


def report(s, label):
    page, vs = page_metrics(s)
    print(f"[{label}] cer {page['cer']:.4f}  truncated {page['truncated_cer']:.4f}  "
          f"length_ratio {page['length_ratio']:.3f}  at_cap {page['at_cap']}  loops {page['loops']}", flush=True)
    if vs is not None:
        print(f"[{label}] vs last kept: delta {vs['delta_cer']:+.4f}  ci95 [{vs['ci95'][0]:+.4f}, {vs['ci95'][1]:+.4f}]", flush=True)
    return page


def top_direction(sub, module):
    U, S, Vt = sub.directions(P + module)
    print(f"  {module}: sigma[:4] {np.round(S[:4], 3)}  share of module energy in dir 0 {S[0]**2 / float((S**2).sum()):.3f}", flush=True)
    return U[:, 0].astype(np.float32), float(S[0])


s = open_session(".")
sub = s.subject
print(sub.describe()["parameters"], "parameters;", s.budget_left, "passes left", flush=True)
s.note("program_01 (round 0, Claude Fable 5.1): baseline, then H1 and H2 of preregistration §5")

# ---- pass 1: baseline -------------------------------------------------------
m = s.measure("C0")
page = report(s, "C0")
dev = {k: abs(page[k] - C0[k]) for k in ("cer", "truncated_cer")}
if dev["cer"] > 1e-4 or dev["truncated_cer"] > 1e-4 or page["at_cap"] != C0["at_cap"]:
    s.note(f"baseline does not reproduce C0 (deviation {dev}, at_cap {page['at_cap']} vs {C0['at_cap']}); "
           "stopping before any edit, as the brief requires")
    print("BASELINE MISMATCH against C0; no edit run. Stop and report.", flush=True)
    sys.exit(0)
if page["loops"] != C0["loops"]:
    s.note(f"baseline loops {page['loops']} vs C0's {C0['loops']} while CER, truncated CER and cap agree; continuing")
print("baseline reproduces C0; continuing with H1", flush=True)

# ---- pass 2: H1 --------------------------------------------------------------
mod1 = "layers.31.self_attn.o_proj"
u1, s1 = top_direction(sub, mod1)
h1 = s.hypothesis(
    f"H1: the top singular direction of {mod1} (sigma {s1:.2f}; lens z -3.6 on <|im_end|>, -5.5 on newline, "
    "-8.1 on the hyphen sign) carries the adapter's stop decision; removing it from lora_B lets the page end.",
    "pages at the cap 9 -> 7 or fewer; CER -0.03 (range -0.08 to +0.01); truncated_cer within +-0.005; loops 12 -> 10",
    "project_out")
s.edit("project_out", [P + mod1 + ".lora_B.weight"], hypothesis=h1, direction=u1, axis=1)
s.measure("H1 project_out o_proj.31 dir0")
report(s, "H1")
kept = s.keep("H1: top direction of layers.31.self_attn.o_proj removed")
print("H1 kept" if kept else "H1 refused (undone)", flush=True)

# ---- pass 3: H2 --------------------------------------------------------------
mod2a, mod2b = "layers.31.mlp.down_proj", "layers.27.mlp.down_proj"
u2a, s2a = top_direction(sub, mod2a)
u2b, s2b = top_direction(sub, mod2b)
h2 = s.hypothesis(
    f"H2: the top singular directions of {mod2a} (sigma {s2a:.2f}, lens z -3.3 on <|im_end|>) and "
    f"{mod2b} (sigma {s2b:.2f}, z -2.8) are the late MLP writes that keep the model from stopping; "
    "removing both from lora_B lets the page end.",
    "pages at the cap -2 against the last kept state; CER -0.03 (range -0.08 to +0.01); truncated_cer within +-0.005; loops -2",
    "project_out")
nA, nB = P + mod2a + ".lora_B.weight", P + mod2b + ".lora_B.weight"
s.edit("project_out", [nA, nB], hypothesis=h2, axis=1,
       per_tensor={nA: {"direction": u2a}, nB: {"direction": u2b}})
s.measure("H2 project_out down_proj.31 dir0 + down_proj.27 dir0")
report(s, "H2")
kept = s.keep("H2: top directions of layers.31 and layers.27 mlp.down_proj removed")
print("H2 kept" if kept else "H2 refused (undone)", flush=True)

print("program_01 done;", s.budget_left, "passes left; no freeze", flush=True)
