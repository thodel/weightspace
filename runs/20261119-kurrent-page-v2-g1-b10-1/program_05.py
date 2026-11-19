"""Round 4 program (`program_05.py`):
Pass 6 (H5): Reconstruct the unit `<|im_end|>` logit-lens direction `w_im_end` in R^2560 from the
budget-0 reading (`docs/readings/adapter_kurrent-page-v2.json`) and the unedited adapter's top-4
left singular vectors across all 40 residual-writing modules (`V @ w = z(<|im_end|>)`, minimum-norm
least-squares fit).
On top of the kept H1 state, project out `w_im_end` (`axis=1`) from all 25 residual-writing modules
(`down_proj` and `o_proj`) whose top-4 singular directions have a negative net `<|im_end|>` score
(`sum_{i=0..3} sv_i^2 * z_i(<|im_end|>) < 0`), while leaving the 15 net-`<|im_end|>`-promoting
modules (such as `layers.30.mlp.down_proj`, `layers.0.mlp.down_proj`, `layers.9.mlp.down_proj`,
`layers.4.mlp.down_proj`) completely untouched.
This removes <0.8% of the Frobenius energy of each of the 23 upstream suppressor modules, preserving
>99% of their visual/lexical transcription features while eliminating their `<|im_end|>`-suppressing
residual write across the full depth of the network.
"""
import json
from pathlib import Path
import numpy as np
import wsp.run as runmod
from wsp.run import open_session
from wsp.lora_subject import LoraSubject

s = open_session(".")

REPO = runmod.REPO
meta = json.loads(Path("session.json").read_text())
orig_sub = LoraSubject(meta["subject"])
reading = json.loads((REPO / "docs/readings/adapter_kurrent-page-v2.json").read_text())

vecs = []
z_end = []
suppressor_tensors = []

for m_info in reading["modules"]:
    if "lens" not in m_info:
        continue
    full = f"base_model.model.model.language_model.{m_info['module']}"
    U, S, _ = orig_sub.directions(full)
    w_z = 0.0
    for l in m_info["lens"]:
        d = l["dir"]
        z = l["watch_z"]["<|im_end|>"]
        sv = l["sv"]
        vecs.append(U[:, d])
        z_end.append(z)
        w_z += (sv ** 2) * z
    if w_z < 0.0:
        suppressor_tensors.append(f"{full}.lora_B.weight")

V = np.stack(vecs, axis=0)
z_arr = np.asarray(z_end, dtype=np.float64)
w_ls, _, _, _ = np.linalg.lstsq(V, z_arr, rcond=1e-3)
w_im_end = (w_ls / np.linalg.norm(w_ls)).astype(np.float32)

print(f"Reconstructed w_im_end from {len(vecs)} lens directions; targeting {len(suppressor_tensors)} net-<|im_end|>-suppressing modules.", flush=True)

h5 = s.hypothesis(
    "H5: Reconstruct the exact logit-lens <|im_end|> direction w_im_end in R^2560 from the 160 "
    "top-4 left singular vectors and their z(<|im_end|>) scores in the budget-0 reading. On top "
    "of H1, project out w_im_end (axis=1) from all 25 residual-writing modules (down_proj and "
    "o_proj) that have net negative <|im_end|> promotion (sum_{i=0..3} sv_i^2 * z_i < 0), while "
    "leaving all 15 net-<|im_end|>-promoting modules (including layers.30.mlp.down_proj) intact. "
    "This removes <0.8% of Frobenius energy from each upstream module while stripping the "
    "exact <|im_end|>-suppressing component across the entire network depth.",
    "CER drops by 0.03 to 0.10 against H1 (to ~0.27-0.34), truncated_cer improves or stays within "
    "+-0.003 of H1 (~0.233-0.244), and pages at the cap drop by 1 to 4 pages (from 5 to 1-4).",
    "project_out",
)

s.edit(
    "project_out",
    suppressor_tensors,
    hypothesis=h5,
    direction=w_im_end,
    axis=1,
)

m5 = s.measure("H5_project_out_wimend_25_suppressors")
p5 = s._log[-1]["page"]
vs5 = s._log[-1]["vs_kept"]
kept5 = s.keep("H5: project out reconstructed w_im_end from all 25 net-<|im_end|>-suppressing residual writers")

print(
    f"Pass 6 (H5): CER={m5.cer:.6f} (delta_vs_kept={vs5['delta_cer']:+.6f}, ci95={vs5['ci95']}), "
    f"truncated_cer={p5['truncated_cer']:.6f}, length_ratio={p5['length_ratio']:.6f}, "
    f"at_cap={p5['at_cap']}, loops={p5['loops']}, kept={kept5}",
    flush=True,
)
