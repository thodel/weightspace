"""program_03 — round 2: freeze only (preregistration §5, rule 4). No pass.

Kept state: H1 alone (dir 0 of layers.31.self_attn.o_proj removed from lora_B).
H2, H5 and H3 were refused and are undone. 5 passes are left unused on purpose:
H5 showed the H1 effect is specific to dir 0 (one cap page stopped, one returned: a swap
inside the interval), H3 showed the late attention adapters carry the reading (42 of 64
pages worse), and H4 is the coarse form of H2, which re-broke two pages. Nothing further
is indicated that would not be a variant hunted for a gain.
"""
import numpy as np
from wsp.run import open_session

P = "base_model.model.model.language_model."

s = open_session(".")
sub = s.subject
print(sub.describe()["parameters"], "parameters;", s.budget_left, "passes left", flush=True)

mod = "layers.31.self_attn.o_proj"
U, S, Vt = sub.directions(P + mod)
print(f"  {mod} in the restored state: sigma[:3] {np.round(S[:3], 3)}  smallest {S[-1]:.2e}", flush=True)
assert S[0] < 1.6 and S[-1] < 1e-4, "restored state is not the H1 state"

s.note("program_03 (round 2, Claude Fable 5.1): freeze with H1 as the only kept edit; H2, H5, H3 refused and undone; "
       "5 passes left unused by preregistration rule 4 (no further hypothesis indicated; H4 is the coarse form of the "
       "refused H2). Final state: C0 with the top singular direction of layers.31.self_attn.o_proj removed from lora_B.")
summary = s.freeze()
print("frozen:", {k: summary[k] for k in ("used", "edits", "kept", "final_digest")}, flush=True)
print("changed tensors:", [k.split("language_model.")[-1] for k in summary["changed_tensors"]], flush=True)
print("last pass (not the kept state; the kept state is pass 2, H1):", summary["last_pass"]["cer"], flush=True)
