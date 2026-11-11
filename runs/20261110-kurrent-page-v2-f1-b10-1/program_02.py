"""program_02 — round 1: follow-ups on the kept H1 state (preregistration §5, rules 1 and 3).

Two passes, both on top of the kept H1 state (dir 0 of layers.31.self_attn.o_proj removed):
  pass 4  H5  remove the next two directions (old dirs 1 and 2, the module's remaining lens rows)
              of layers.31.self_attn.o_proj from lora_B, in one edit (op "add" with
              delta = -(U2 U2^T) B, U2 orthonormal: an exact two-direction project_out)
  pass 5  H3  scale 0.5 on lora_B of q/k/v/o_proj in the last three full-attention layers (23, 27, 31)
Each edit is followed by measure and keep; a refused keep is undone by the harness.
No freeze: a third round follows with 5 passes left.
"""
import numpy as np
from wsp.run import open_session

P = "base_model.model.model.language_model."
KEPT = {"cer": 0.3470339883410475, "at_cap": 4}


def report(s, label):
    e = s._log[-1]
    assert e["kind"] == "pass", e["kind"]
    page, vs = e["page"], e.get("vs_kept")
    print(f"[{label}] cer {page['cer']:.4f}  truncated {page['truncated_cer']:.4f}  "
          f"length_ratio {page['length_ratio']:.3f}  at_cap {page['at_cap']}  loops {page['loops']}", flush=True)
    if vs is not None:
        print(f"[{label}] vs last kept: delta {vs['delta_cer']:+.4f}  ci95 [{vs['ci95'][0]:+.4f}, {vs['ci95'][1]:+.4f}]", flush=True)
    return page


s = open_session(".")
sub = s.subject
T = sub.tensors()
print(sub.describe()["parameters"], "parameters;", s.budget_left, "passes left", flush=True)
s.note("program_02 (round 1, Claude Fable 5.1): H5 and H3 on the kept H1 state; H2 was refused and is undone")

# ---- sanity: the restored state is the H1 state ----------------------------
mod = "layers.31.self_attn.o_proj"
U, S, Vt = sub.directions(P + mod)
print(f"  {mod} in the restored state: sigma[:3] {np.round(S[:3], 3)}  smallest {S[-1]:.2e} (dir 0 of C0 removed)", flush=True)
assert S[0] < 1.6 and S[-1] < 1e-4, "restored state is not the H1 state"

# ---- pass 4: H5 --------------------------------------------------------------
h5 = s.hypothesis(
    f"H5 (rule 1 follow-up of H1): the next two singular directions of {mod} (sigma {S[0]:.2f} and {S[1]:.2f}; "
    "the module's lens rows 1 and 2, whose lens z on <|im_end|> is +0.5 and -0.2) also carry part of the stop "
    "suppression; removing them from lora_B as well lets more pages end. If it fails, the H1 effect is specific to dir 0.",
    "pages at the cap 4 -> 3 or fewer; CER -0.01 (range -0.05 to +0.04); truncated_cer within +-0.01; loops 8 -> 7",
    "add (two-direction project_out: delta = -(U2 U2^T) B)")
name = P + mod + ".lora_B.weight"
B = T[name]
U2 = U[:, :2].astype(np.float64)
delta = -(U2 @ (U2.T @ B.astype(np.float64))).astype(np.float32)
print(f"  H5 delta norm {np.linalg.norm(delta):.4f} on ||B|| {np.linalg.norm(B):.4f}", flush=True)
s.edit("add", [name], hypothesis=h5, delta=delta)
U_, S_, _ = sub.directions(P + mod)
print(f"  after H5: sigma[:3] {np.round(S_[:3], 3)}", flush=True)
s.measure("H5 project_out o_proj.31 dirs 1-2 (on H1)")
report(s, "H5")
kept = s.keep("H5: next two directions of layers.31.self_attn.o_proj removed")
print("H5 kept" if kept else "H5 refused (undone)", flush=True)

# ---- pass 5: H3 --------------------------------------------------------------
names = [P + f"layers.{l}.self_attn.{k}.lora_B.weight" for l in (23, 27, 31) for k in ("q_proj", "k_proj", "v_proj", "o_proj")]
assert all(n in T for n in names)
h3 = s.hypothesis(
    "H3 (preregistered): the adapter's attention rewrite in the last three full-attention layers (23, 27, 31; "
    "q/k/v/o_proj, 12 tensors, 3.9 % of the adapter's energy) stops the model from seeing what it has already "
    "written; halving it moves those layers half-way back to the base's attention. Tested on top of the kept state.",
    "loops -2 against the last kept state; pages at the cap 4 -> 3 or fewer; CER -0.02 (range -0.06 to +0.04); "
    "truncated_cer +0.01 (reading slightly worse)",
    "scale")
s.edit("scale", names, hypothesis=h3, factor=0.5)
s.measure("H3 scale 0.5 attention layers 23,27,31 (on kept)")
report(s, "H3")
kept = s.keep("H3: late full-attention adapters at 0.5")
print("H3 kept" if kept else "H3 refused (undone)", flush=True)

print("program_02 done;", s.budget_left, "passes left; no freeze", flush=True)
