"""Programmer h1 for kraken-bern-combined-v1 on the bern-val practice set (513 lines).

Pre-registered in preregistration_hypotheses.md. Budget 10 passes: baseline plus at most 8 edit passes.
Each edit gets exactly one measure and one keep/refuse decision (the harness reverts a refused edit).
"""
import numpy as np

from wsp.run import open_session
from wsp.align import codec_row_map, gather_rows

HERTWIG = "kraken-bern-hertwig-v2"
SCHLUSS = "kraken-bern-schlussredaktion-v1"
SCHLUSS_PATH = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-schlussredaktion-v1.mlmodel"
BIAS = "O_18.lin.bias"
OUT_W = "O_18.lin.weight"
MIN_DONOR_LINES = 60

s = open_session(".")
cm = s.subject.codec_map()

# The subject was initialised from hertwig-v2, so every keep decision is judged on lines hertwig did not train on.
mask_h = s.unseen_by(HERTWIG)
mask_hs = s.unseen_by(HERTWIG, SCHLUSS)
s.note(f"rule masks: unseen by hertwig {sum(mask_h)} lines; unseen by hertwig and schlussredaktion {sum(mask_hs)} lines")


def bias_step(row, value):
    d = np.zeros(s.subject.tensors()[BIAS].shape, dtype=np.float32)
    d[row] = value
    return d


def trial(label, text, prediction, edit_class, mask, op, names, per_tensor=None, **kwargs):
    h = s.hypothesis(text, prediction, edit_class)
    s.edit(op, names, hypothesis=h, per_tensor=per_tensor, **kwargs)
    m = s.measure(label, rule_mask=mask)
    ok = s.keep(label)
    print(f"{label}: CER {m.cer:.4f} collapsed {m.collapsed} -> {'kept' if ok else 'refused'}")
    return ok


# Pass 1: C0 baseline on the rule lines.
base = s.measure("C0 baseline", rule_mask=mask_h)
print(f"C0 baseline: CER {base.cer:.4f} collapsed {base.collapsed}")

# H1: lower the CTC blank bias (row 0). The step after it depends on the first result.
ok1 = trial("H1 blank -0.25",
            "Lower the blank bias by 0.25 so marginal frames emit characters; deletions dominate and length ratio is 0.949.",
            "CER down 0.005 to 0.020", "output bias (blank row)", mask_h,
            "add", [BIAS], delta=bias_step(0, -0.25))

step = -0.25 if ok1 else -0.10
trial("H1b blank step %+.2f" % step,
      "Second blank step, from the kept state if H1 was kept, otherwise a smaller step from C0.",
      "CER down 0.000 to 0.008, or refused as the optimum", "output bias (blank row)", mask_h,
      "add", [BIAS], delta=bias_step(0, step))

# H2: space row bias up (reference spaces missing 195 times vs spurious 99).
if " " in cm:
    trial("H2 space +0.15",
          "Raise the space bias by 0.15; missing spaces outnumber spurious spaces.",
          "CER down 0.001 to 0.004", "output bias (space row)", mask_h,
          "add", [BIAS], delta=bias_step(cm[" "], 0.15))

# H3: slash row bias up (98 missing vs 57 spurious).
if "/" in cm:
    trial("H3 slash +0.15",
          "Raise the slash bias by 0.15; missing slashes outnumber spurious ones.",
          "CER down 0.000 to 0.002", "output bias (slash row)", mask_h,
          "add", [BIAS], delta=bias_step(cm["/"], 0.15))

# Donor edits: schlussredaktion-v1 is a sibling of the subject. Run only if enough lines are unseen by both.
if sum(mask_hs) >= MIN_DONOR_LINES:
    from wsp.kraken_subject import KrakenSubject

    donor = KrakenSubject(SCHLUSS_PATH)
    dT = donor.tensors()
    rmap = codec_row_map(cm, donor.codec_map())
    s.note(f"donor {SCHLUSS}: {len(rmap)} characters shared with the subject")

    # H4: output layer interpolated toward the donor, rows aligned by character, alpha 0.5.
    out_names = [OUT_W, BIAS]
    T = s.subject.tensors()
    pt_out = {n: {"other": gather_rows(dT[n], rmap, T[n].shape[0], fill=T[n])} for n in out_names}
    ok4 = trial("H4 output layer -> schlussredaktion a=0.5",
                "Move the output layer halfway toward the sibling's character-aligned rows.",
                "CER down 0.000 to 0.010", "interpolate with donor (output layer)", mask_hs,
                "interpolate", out_names, per_tensor=pt_out, alpha=0.5)

    # H5: all body tensors (C_ and L_) interpolated toward the donor, alpha 0.25. Shapes match, no gather needed.
    T = s.subject.tensors()
    body = [n for n in T if n.startswith(("C_", "L_")) and dT[n].shape == T[n].shape]
    pt_body = {n: {"other": dT[n]} for n in body}
    ok5 = trial("H5 body -> schlussredaktion a=0.25",
                "Move the convolutional and recurrent weights a quarter of the way toward the sibling.",
                "CER -0.005 to +0.002", "interpolate with donor (body)", mask_hs,
                "interpolate", body, per_tensor=pt_body, alpha=0.25)

    # H6: the same body interpolation at alpha 0.10, only if H5 was refused.
    if not ok5:
        trial("H6 body -> schlussredaktion a=0.10",
              "A smaller body interpolation after H5 was refused.",
              "CER within +/-0.002", "interpolate with donor (body)", mask_hs,
              "interpolate", body, per_tensor=pt_body, alpha=0.10)
else:
    s.note(f"donor edits skipped: only {sum(mask_hs)} lines unseen by both donors (threshold {MIN_DONOR_LINES})")

summary = s.freeze()
print("used", summary["used"], "edits", summary["edits"], "kept", summary["kept"])
