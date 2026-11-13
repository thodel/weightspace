"""planted-f1 / s14 / task (b), budget 0 — the blind programmer's script.

Programmer: claude-fable-5-1. Decision (see hypotheses.md): no reading in the pack is a defect I am fairly
sure about, so no edit is applied. The session is opened, the reasoning is logged, and the model is frozen
unchanged. Runs inside the sandbox: wsp API + numpy only; no pass is taken (budget 0).
"""
from wsp.run import open_session

s = open_session(".")
assert s.budget == 0, "this program is written for a budget-0 (blind) session"

s.note("blind programmer (budget 0), claude-fable-5-1: readings reviewed — describe.md, diff_parent_to_subject.md, "
       "c0_practice.md. Decision: no edit. See hypotheses.md for the seven candidates and why each fell short of "
       "'fairly sure'.")
s.note("H1 blank over-prediction (deletion-heavy confusions, 8 collapsed lines, length ratio 0.979): not applied — "
       "blank bias +1.23 vs parent +1.12 (natural); the deleted letters e/n/i/r/c/s/o are the Kurrent minim letters, "
       "a/t/d/h/u are absent: the natural error profile of this hand.")
s.note("H3 accent rows at ~2x parent norm (cos 0.998, bias more negative): natural CTC growth of never-seen classes, "
       "not a planted scale; no insertions of those characters in the confusions.")
s.note("H4 L_16.layer.weight_ih_l0 abs max 4.70 (19x rms): forward/reverse norm asymmetry is inherited from the parent "
       "(back-solved from rel L2 and cosine); rel L2 vs parent in line with siblings. Left alone.")
s.note("H5 space bias -7.85 (most negative row): 78 space deletions of ~2000 is ~4 %, inconsistent with a planted "
       "multi-logit shift; normal for a CTC line model. Left alone.")
s.note("H2 blank row rel 1.27, H6 C_3.co.bias rel 1.09 (tiny tensor), H7 no swapped/zeroed/exploded output row: "
       "all consistent with ordinary fine-tuning. Expected result: final CER = C0 CER.")

summary = s.freeze()
print("frozen unchanged:", summary["edits"], "edits,", summary["kept"], "kept, digest", summary["final_digest"][:12])
