"""Round 5 program (`program_06.py`):
Pass 7 (H6): On top of the kept H1 state (where `layers.31.self_attn.o_proj` dir 0 and
`layers.31.mlp.down_proj` dir 0 were projected out, corresponding to coefficient c=0 along those
two singular directions), test adding `-0.5 * u_0 (u_0^T B_orig)` to `lora_B.weight` of both
layer-31 residual writers (`add` edit class), so that their rank-1 `<|im_end|>`-suppressing
singular directions are inverted at half strength (`c = -0.5`).
Whenever the layer-31 input activations trigger these two rank-1 right singular vectors at page end
 or inside repetition loops, they will now actively write `-0.5 * sigma_0 * u_0` into the final
residual stream (promoting `<|im_end|>`, `\n`, and `¬` and reinforcing `layers.30.mlp.down_proj`
dir 0), while all other 63 singular directions of layer 31 and all upstream layers 0..30 remain
completely untouched.
"""
import json
from pathlib import Path
import numpy as np
from wsp.run import open_session
from wsp.lora_subject import LoraSubject

s = open_session(".")

meta = json.loads(Path("session.json").read_text())
orig_sub = LoraSubject(meta["subject"])

mod_o31 = "base_model.model.model.language_model.layers.31.self_attn.o_proj"
mod_d31 = "base_model.model.model.language_model.layers.31.mlp.down_proj"

b_o31_name = f"{mod_o31}.lora_B.weight"
b_d31_name = f"{mod_d31}.lora_B.weight"

u_o31_0 = orig_sub.directions(mod_o31)[0][:, 0]
u_d31_0 = orig_sub.directions(mod_d31)[0][:, 0]

b_o31_orig = orig_sub.tensors()[b_o31_name]
b_d31_orig = orig_sub.tensors()[b_d31_name]

delta_o31 = (-0.5 * np.outer(u_o31_0, u_o31_0 @ b_o31_orig)).astype(np.float32)
delta_d31 = (-0.5 * np.outer(u_d31_0, u_d31_0 @ b_d31_orig)).astype(np.float32)

h6 = s.hypothesis(
    "H6: In H1 (kept), we projected out dir 0 of layers.31.self_attn.o_proj (sigma=2.226, "
    "z(<|im_end|>)=-3.60) and dir 0 of layers.31.mlp.down_proj (sigma=2.612, z(<|im_end|>)=-3.32), "
    "setting their rank-1 coefficient from c=+1 to c=0. Here, on top of H1, we add "
    "-0.5 * u_0 (u_0^T B_orig) to both layer-31 lora_B.weight tensors (setting c=-0.5), so that "
    "whenever the final layer's right singular vector v_0 fires at page end or in a loop, it "
    "actively writes -0.5 * sigma_0 * u_0 into the final residual stream (promoting <|im_end|> "
    "and reinforcing layer 30 dir 0), without touching any upstream layer or any of the other 63 "
    "directions in layer 31.",
    "CER drops by 0.04 to 0.11 against H1 (to ~0.26-0.33), truncated_cer stays within +-0.004 "
    "of H1 (~0.237-0.245), and pages at the cap drop by 2 to 4 pages (from 5 to 1-3).",
    "add",
)

s.edit(
    "add",
    [b_o31_name, b_d31_name],
    hypothesis=h6,
    per_tensor={
        b_o31_name: {"delta": delta_o31},
        b_d31_name: {"delta": delta_d31},
    },
)

m6 = s.measure("H6_invert05_L31_o_down_dir0")
p6 = s._log[-1]["page"]
vs6 = s._log[-1]["vs_kept"]
kept6 = s.keep("H6: invert dir 0 of layers.31.self_attn.o_proj and layers.31.mlp.down_proj at c=-0.5")

print(
    f"Pass 7 (H6): CER={m6.cer:.6f} (delta_vs_kept={vs6['delta_cer']:+.6f}, ci95={vs6['ci95']}), "
    f"truncated_cer={p6['truncated_cer']:.6f}, length_ratio={p6['length_ratio']:.6f}, "
    f"at_cap={p6['at_cap']}, loops={p6['loops']}, kept={kept6}",
    flush=True,
)
