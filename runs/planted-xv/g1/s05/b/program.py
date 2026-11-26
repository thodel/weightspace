"""Weight-space repair program for pack s05, task (b) (budget 0, blind programmer).

Diagnosis:
In `describe.md` and `diff_parent_to_subject.md`, every non-zero tensor of the
final BiLSTM block `L_16` has high cosine similarity with the parent model
(`~0.999` on `bias_hh`, `~0.978` on `weight_hh`, `~0.95` on `weight_ih`), yet
its L2 norm and RMS are halved compared to `L_12`/`L_14` and the parent
(`rel L2` against parent is `0.5054` to `0.5499`, matching `sqrt(1.25 - cos)`
for a `0.5x` scaling of the fine-tuned weights). This `0.5x` attenuation of
`L_16` shrinks its output activations into `O_18`, causing the CTC blank
(bias +1.232) to dominate over character rows (mean bias -1.921), resulting in
24 collapsed lines, length ratio 0.828, and CER 0.1993 dominated by deletions.

Repair:
Scale all `L_16` tensors by `factor=2.0` to exactly invert the `0.5x` scaling.
"""
from wsp.run import open_session

s = open_session(".")

l16_tensors = [
    "L_16.layer.weight_ih_l0",
    "L_16.layer.weight_hh_l0",
    "L_16.layer.bias_hh_l0",
    "L_16.layer.weight_ih_l0_reverse",
    "L_16.layer.weight_hh_l0_reverse",
    "L_16.layer.bias_hh_l0_reverse",
]

h1 = s.hypothesis(
    "All non-zero tensors of L_16 were scaled by 0.5x (norms/RMS are half of L_14 and parent while cosines are 0.947-0.999, with rel L2 = sqrt(1.25 - cos) ~ 0.505-0.550); scaling L_16 by 2.0 restores the true fine-tuned weights.",
    "CER drops by ~0.13-0.15 (from 0.1993 to ~0.05-0.07), collapsed lines drop from 24 to ~0, length ratio recovers from 0.828 to ~1.0.",
    "scale",
)
s.edit("scale", l16_tensors, hypothesis=h1, factor=2.0)
s.keep_unmeasured("Invert exact 0.5x scaling defect on L_16 BiLSTM tensors by scaling by 2.0")

s.freeze()
