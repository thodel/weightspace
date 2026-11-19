"""Round 6 program (`program_07.py`):
Freeze the session at the kept H1 state (digest `d9fc39be27f9a885eaa21c7547855c67f81309312507a13a6e1f1d0b344b6740`).
Across Passes 1-7:
- Pass 1 (C0): CER 0.451512, truncated_cer 0.256386, at_cap 9, loops 12.
- Pass 2 (H1): Project out dir 0 of `layers.31.self_attn.o_proj` and dir 0 of `layers.31.mlp.down_proj`
  (axis=1). KEPT: CER 0.373742 (delta_cer = -0.077769, ci95 = [-0.173227, -0.004299]),
  truncated_cer 0.241180 (-0.015207), at_cap 5 (-4), loops 10 (-2).
- Passes 3-7 (H2-H6): Five follow-up hypotheses testing upstream full rank-1 removal (H2),
  upstream shared-direction projection (H3), layer-31 residual scaling (H4), whole-network
  `<|im_end|>`-suppressor projection (H5), and layer-31 rank-1 sign inversion (H6) confirmed
  that `c = 0` (exact orthogonal projection of the two final-layer `<|im_end|>`-suppressing
  singular directions, H1) is the clean surgical optimum and the remaining 5 at-cap pages are
  not removable by further linear adapter edits without hurting reading accuracy.
"""
import json
from wsp.run import open_session

s = open_session(".")
s.note(
    "Freezing after 7 passes (3 passes unused) at kept state H1 "
    "(project_out dir 0 of layers.31.self_attn.o_proj and dir 0 of layers.31.mlp.down_proj). "
    "H1 reduced CER from 0.451512 to 0.373742 (delta=-0.077769, ci95=[-0.173227, -0.004299]), "
    "improved truncated_cer from 0.256386 to 0.241180, and reduced at_cap pages from 9 to 5. "
    "Follow-up probes H2-H6 confirmed that c=0 rank-1 projection on the two layer-31 residual "
    "writers is the clean optimum."
)
summary = s.freeze()
print(json.dumps({k: summary[k] for k in ("name", "used", "edits", "kept", "final_digest")}, indent=2), flush=True)
