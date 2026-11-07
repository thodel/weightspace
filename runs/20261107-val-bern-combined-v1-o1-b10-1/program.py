"""o1 program for kraken-bern-combined-v1 (programmer: claude-opus-5-5).

Written in advance from the pack; see preregistration_hypotheses.md. At most 9 of the 10 passes are used.
"""
import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map, gather_rows, common_names

MODELS = "/home/tobias/Repo/weightspace/models/kraken"
SCHLUSS = "kraken-bern-schlussredaktion-v1"
HERTWIG = "kraken-bern-hertwig-v2"
BIAS, WEIGHT = "O_18.lin.bias", "O_18.lin.weight"
MAX_PASSES = 10

s = open_session(".")
T = s.subject.tensors
cm = s.subject.codec_map()
passes = 0


def measure(label, mask=None):
    global passes
    if passes >= MAX_PASSES or s.budget_left <= 0:
        return None
    passes += 1
    return s.measure(label, rule_mask=mask) if mask is not None else s.measure(label)


m0 = measure("C0 baseline")
kept_cer = m0.cer
s.note(f"baseline CER {m0.cer:.4f}, collapsed {m0.collapsed}")


def bias_trial(text, prediction, offsets, label):
    """offsets: {row index: additive change of the output bias}. No donor, rule on all lines."""
    global kept_cer
    if passes >= MAX_PASSES or s.budget_left <= 0:
        return None
    h = s.hypothesis(text, prediction, "bias")
    d = np.zeros_like(T()[BIAS])
    for r, v in offsets.items():
        d[r] += v
    s.edit("add", [BIAS], hypothesis=h, delta=d)
    m = measure(label)
    k = s.keep(label)
    s.note(f"{label}: CER {m.cer:.4f} (last kept {kept_cer:.4f}), collapsed {m.collapsed}, kept={k}")
    if k:
        kept_cer = m.cer
    return k, m


def donor_targets(donor):
    """Per-tensor donor arrays in the subject's layout: shared-shape tensors as they are, output rows by character."""
    dT = donor.tensors()
    cur = T()
    targets = {n: dT[n] for n in common_names(cur, dT) if not n.startswith("O_18")}
    rmap = codec_row_map(cm, donor.codec_map())
    rmap[0] = 0                                    # the CTC blank
    targets[WEIGHT] = gather_rows(dT[WEIGHT], rmap, cur[WEIGHT].shape[0], fill=cur[WEIGHT])
    targets[BIAS] = gather_rows(dT[BIAS], rmap, cur[BIAS].shape[0], fill=cur[BIAS])
    # skip tensors identical in both (the zero bias_ih) to keep the log readable
    return {n: a for n, a in targets.items() if not np.array_equal(a, cur[n])}


def donor_trial(donor_id, donor, alpha, text, prediction, label):
    """Interpolate every tensor toward the donor (alpha<0 extrapolates away). Donor rule mask + own full-set guard."""
    global kept_cer
    if passes >= MAX_PASSES or s.budget_left <= 0:
        return None
    tg = donor_targets(donor)
    names = sorted(tg)
    h = s.hypothesis(text, prediction, "interpolate")
    s.edit("interpolate", names, hypothesis=h, alpha=alpha, per_tensor={n: {"other": tg[n]} for n in names})
    m = measure(label, mask=s.unseen_by(donor_id))
    if m.cer > kept_cer:
        s.revert(f"{label}: own guard, CER on all lines {m.cer:.4f} > last kept {kept_cer:.4f}")
        s.note(f"{label}: reverted by the full-set guard")
        return False, m
    k = s.keep(label)
    s.note(f"{label}: CER {m.cer:.4f} (last kept {kept_cer:.4f}), collapsed {m.collapsed}, kept={k}")
    if k:
        kept_cer = m.cer
    return k, m


# H1: blank bias -0.5
r = bias_trial("Deletion surplus (length ratio 0.949): the blank wins marginal frames; lower the blank bias by 0.5.",
               "CER down 0.5-1.0 pp", {0: -0.5}, "H1 blank -0.5")
h1_kept = bool(r and r[0])

# H1b: further step if kept, half step if not
if h1_kept:
    bias_trial("Blank offset not yet at its optimum after -0.5; a further -0.5 (total -1.0).",
               "CER down a further 0.2-0.5 pp", {0: -0.5}, "H1b blank further -0.5")
else:
    bias_trial("The -0.5 step overshot or was too noisy; try blank -0.25.",
               "CER 0 to -0.3 pp", {0: -0.25}, "H1b blank -0.25")

# H2: space bias +0.3
if " " in cm:
    bias_trial("Space deletions exceed insertions 195:99; raise the space prior by 0.3.",
               "CER down 0.1-0.3 pp", {cm[" "]: 0.3}, "H2 space +0.3")

# H3: '/' bias +0.3
if "/" in cm:
    bias_trial("Virgule deletions exceed insertions 98:57; raise the '/' prior by 0.3.",
               "CER down 0.05-0.15 pp", {cm["/"]: 0.3}, "H3 slash +0.3")

# H4: soup with the sibling schlussredaktion-v1
schluss = KrakenSubject(f"{MODELS}/{SCHLUSS}.mlmodel")
r = donor_trial(SCHLUSS, schluss, 0.25,
                "Subject and sibling are fine-tunes of hertwig-v2 in one basin; a 0.25 soup averages out noise.",
                "CER on the 297 masked lines down 0.3-1.0 pp", "H4 soup schluss a=0.25")
if r is not None:
    if r[0]:
        donor_trial(SCHLUSS, schluss, 0.2,
                    "The 0.25 soup helped; go a further 0.2 of the remaining way toward the sibling.",
                    "CER down up to 0.3 pp more", "H4b soup schluss further a=0.2")
    else:
        donor_trial(SCHLUSS, schluss, 0.1,
                    "0.25 was too far; a light 0.1 soup.",
                    "CER 0 +/- 0.3 pp", "H4b soup schluss a=0.1")
del schluss

# H5: re-tune the blank after whatever was kept
bias_trial("Re-tune the blank offset after the kept edits: blank -0.25.",
           "CER 0 to -0.2 pp", {0: -0.25}, "H5 blank -0.25 re-tune")

# H6: task-vector extrapolation away from the parent hertwig-v2
hertwig = KrakenSubject(f"{MODELS}/{HERTWIG}.mlmodel")
donor_trial(HERTWIG, hertwig, -0.15,
            "Extrapolate 15 % further along the parent->subject fine-tuning direction (the added Schlussredaktion hand).",
            "CER on the 301 masked lines down 0.2-0.6 pp; fails as CER up >= 1 pp", "H6 extrapolate from hertwig a=-0.15")
del hertwig

s.note(f"end: {passes} passes used, last kept CER {kept_cer:.4f} (baseline {m0.cer:.4f})")
summary = s.freeze()
print({"passes": passes, "baseline": m0.cer, "last_kept_cer": kept_cer, "kept": summary.get("kept")})
