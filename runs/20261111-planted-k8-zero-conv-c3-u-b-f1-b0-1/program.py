"""planted-f1 / s01 / task (b), budget 0 — blind programmer (claude-fable-5-1).

Diagnosis from the pack (see hypotheses.md): `C_3.co.weight` and `C_3.co.bias` are exactly zero, so the
second convolution block emits ReLU(0) = 0 everywhere, every downstream layer sees a constant input, the
output layer emits the same logit vector at every time step, the blank wins, and greedy CTC decoding gives
the empty string: CER 1.0000, 300/300 lines collapsed, length ratio 0, deletions only. That is the whole
of the C0 reading.

The only repair is a transplant of the `C_3` block from the model's own parent (kraken-german-xix-v1),
which is not available. The permitted donor (kraken-bern-hertwig-v2) is from another lineage: its C_0 and
C_6 have cosine 0.09 / 0.06 with ours, so its C_3 maps features this model does not produce onto a basis
this model's C_6 never saw. Transplanting it (or an identity pass-through) would turn empty output into
garbage of roughly the time-step count, and with insertions the CER can exceed the dead model's 1.0.
At budget 0 nobody checks before scoring, so nothing I am not fairly sure about is applied: this program
applies NO edit. It records the diagnosis and freezes the subject unchanged.
"""
from wsp.run import open_session

s = open_session(".")

T = s.subject.tensors()
w, b = T["C_3.co.weight"], T["C_3.co.bias"]
c3_dead = bool((w == 0).all() and (b == 0).all())

s.note(
    "blind programmer (budget 0, claude-fable-5-1). Diagnosis from the pack: C_3.co.weight and C_3.co.bias are "
    f"exactly zero (confirmed in the loaded subject: {c3_dead}). A zero Cr3,13,32 block emits ReLU(0)=0 for every "
    "pixel; C_6/C_9/LSTMs then see constant input, the output layer emits one fixed logit vector per step, the blank "
    "(bias +1.23) wins, and greedy CTC decodes to the empty string. This matches the C0 reading exactly: CER 1.0, "
    "300/300 collapsed, length ratio 0, 12,867 deletions and no insertion or substitution."
)
s.note(
    "No edit applied. The repair would be transplant of C_3.co.weight/bias from the model's own parent "
    "(kraken-german-xix-v1), which is not available. The permitted donor kraken-bern-hertwig-v2 is from another "
    "lineage (C_0 cosine 0.09, C_6 cosine 0.06 with the subject): its C_3 maps donor-C_0 features onto the donor-C_6 "
    "basis, neither of which this model shares. A transplanted or identity C_3 would feed the intact LSTM stack a "
    "foreign feature stream; predicted result is garbage output of roughly the time-step count with CER >= 1.0 and "
    "plausibly 1.5-3 from insertions, i.e. worse than the dead baseline's exact 1.0. Not 'fairly sure', so not shipped."
)
s.note(
    "Secondary readings judged not actionable: space bias -7.85 (lowest row bias; plausibly natural for the "
    "'no ink' class, no healthy reference available, and irrelevant while C_3 is dead); L_16.weight_ih_l0 abs max 4.7 "
    "with ordinary rms (a single large LSTM weight is not evidence of damage). Predicted practice CER after this "
    "program: 1.0000, unchanged. Final model is identical to the subject."
)

summary = s.freeze()
print("edits:", summary["edits"], "kept:", summary["kept"], "final digest:", summary["final_digest"][:12])
