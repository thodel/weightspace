"""program_03.py — round 2 (the last) of session 20261111-bern-combined-v1-f1-b10-1.

Programmer: Claude Fable 5.1 (claude-fable-5-1). Runs inside the sandbox (wsp + numpy only).
Restores the last kept state (blank bias -0.40) and runs (preregistration.md section 5, round 2):
    P8  = H1c third blank step (-0.20, total -0.60);
    P9  = H1d fourth blank step, only if H1c was kept with delta <= -0.0025;
    P9/10 = H4b interpolate alpha 0.125 toward the parent, rows aligned, blank bias held;
then s.freeze().
Each edit is measured, then kept by the harness's rule or reverted; an edit that raises the collapsed
count against the last kept state is reverted even if the rule would keep it.
"""
import json
from pathlib import Path

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map, gather_rows
from wsp.metrics import score, align, normalise

MODELS = "/home/tobias/Repo/weightspace/models/kraken/"
PARENT = MODELS + "kraken-bern-hertwig-v2.mlmodel"

s = open_session(".")
s.note("program_03 (round 2, last), programmer Claude Fable 5.1 (claude-fable-5-1): H1c blank -0.20 more, "
       "H1d only if H1c kept and not saturating, H4b interpolate alpha 0.125 toward hertwig-v2, freeze")
print(f"budget {s.budget}, used {s._used}, left {s.budget_left}")

refs = [it["text"] for it in s.practice]
hand = ["H" if it["key"].startswith("A_V_1443") else "S" for it in s.practice]


def last_kept_collapsed():
    """Collapsed count of the last kept state, from log.jsonl (the baseline if nothing was kept)."""
    events = [json.loads(l) for l in Path("log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = None
    last_pass = None
    for e in events:
        if e["kind"] == "pass":
            last_pass = e
            if kept is None:
                kept = e["metrics"]["collapsed"]
        elif e["kind"] == "decision" and e.get("kept") and last_pass is not None:
            kept = last_pass["metrics"]["collapsed"]
    return kept


state = {"kept_collapsed": last_kept_collapsed()}
print(f"last kept state: collapsed {state['kept_collapsed']}, blank bias "
      f"{float(s.subject.tensors()['O_18.lin.bias'][0]):+.4f}")


def report(label, m):
    rd = Path("readings") / f"pass_{s._used:03d}.jsonl"
    hyps = {}
    for line in rd.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            hyps[r["key"]] = r["text"]
    hy = [hyps[it["key"]] for it in s.practice]
    per = {}
    for h in "HS":
        idx = [i for i in range(len(hand)) if hand[i] == h]
        mh = score([hy[i] for i in idx], [refs[i] for i in idx], blocks=False)
        per[h] = (mh.cer, mh.char_edits, mh.ref_chars, mh.collapsed, mh.length_ratio)
    dels = ins = subs = 0
    for h_, r_ in zip(hy, refs):
        for a, b in align(normalise(r_), normalise(h_)):
            if a == b:
                continue
            if b == "":
                dels += 1
            elif a == "":
                ins += 1
            else:
                subs += 1
    e = s._log[-1]
    vk = e.get("vs_kept") or {}
    vb = e.get("vs_baseline") or {}
    print(f"[{label}] pass {s._used}: CER {m.cer:.4f} ({m.char_edits}/{m.ref_chars})"
          f"  H {per['H'][0]:.4f} ({per['H'][1]}/{per['H'][2]})  S {per['S'][0]:.4f} ({per['S'][1]}/{per['S'][2]})"
          f"  collapsed {m.collapsed} (H {per['H'][3]}, S {per['S'][3]})  length_ratio {m.length_ratio:.3f}"
          f" (H {per['H'][4]:.3f}, S {per['S'][4]:.3f})  WER {m.wer:.4f}")
    print(f"   del {dels} ins {ins} sub {subs}")
    print(f"   vs_kept delta {vk.get('delta_cer', float('nan')):+.4f} ci95 {vk.get('ci95')}"
          f"   vs_baseline delta {vb.get('delta_cer', float('nan')):+.4f} ci95 {vb.get('ci95')}")
    print("   confusions:", ", ".join(f"{a or '∅'}→{b or '∅'} {n}" for a, b, n in m.confusions))
    return vk.get("delta_cer", 0.0)


def decide(m, reason):
    kc = state["kept_collapsed"]
    if kc is not None and m.collapsed > kc:
        s.revert(f"collapsed rose {kc} -> {m.collapsed}; {reason}")
        print(f"   -> REVERTED by the programmer: collapsed rose {kc} -> {m.collapsed}")
        return False
    ok = s.keep(reason)
    if ok:
        state["kept_collapsed"] = m.collapsed
        print("   -> KEPT (rule met)")
    else:
        print(f"   -> REFUSED by the rule: {s._log[-1].get('reason')}")
    return ok


def blank_step(tag, total_after, extra_text, prediction):
    T = s.subject.tensors()
    d = np.zeros_like(T["O_18.lin.bias"])
    d[0] = -0.20
    h = s.hypothesis(
        f"{tag} blank bias step: lower O_18.lin.bias[0] by another 0.20 (total {total_after:+.2f} from the "
        f"subject's +0.998). {extra_text} No donor weights.",
        prediction, "bias shift (output, blank row)")
    s.edit("add", ["O_18.lin.bias"], hypothesis=h, delta=d)
    print(f"{tag}: blank bias -> {float(s.subject.tensors()['O_18.lin.bias'][0]):+.4f}")
    m = s.measure(f"{tag} blank bias -0.20 more (total {total_after:+.2f})")
    delta = report(tag, m)
    kept = decide(m, f"{tag} blank step: {prediction[:60]}")
    return kept, delta


# ---------------------------------------------------------------- P8: H1c
h1c_kept, h1c_delta = blank_step(
    "H1c", -0.60,
    "The two earlier steps were kept with consistent per-line gains (16/7 and 30/9 lines better/worse), "
    "deletions still 941 against 173 insertions.",
    "CER -0.1 to -0.4 points; deletions down by 30-60, insertions up by at most 25; collapsed not up. "
    "If refused: the knob is exhausted at -0.40; no fourth step.")

# ---------------------------------------------------------------- P9: H1d (conditional)
h1d_kept = None
if h1c_kept and h1c_delta <= -0.0025:
    h1d_kept, _ = blank_step(
        "H1d", -0.80,
        "H1c was kept with delta <= -0.0025 (not saturating).",
        "CER -0.1 to -0.3 points; insertions rise faster than before; collapsed not up. "
        "If refused: exhausted at -0.60.")
else:
    print(f"H1d skipped (H1c kept={h1c_kept}, delta {h1c_delta:+.4f})")

# ---------------------------------------------------------------- P9/P10: H4b alpha 0.125 toward the parent
par = KrakenSubject(PARENT)
Tp = par.tensors()
cs = s.subject.codec_map()
cp = par.codec_map()
row_map = codec_row_map(cs, cp)
row_map[0] = 0
only_subject = [c for c in cs if c not in cp]
T = s.subject.tensors()
names = [n for n in T if "bias_ih" not in n]
assert len(names) == 28, names
per_tensor = {}
for n in names:
    if n.startswith("O_18"):
        other = gather_rows(Tp[n], row_map, T[n].shape[0], fill=T[n])
        if n == "O_18.lin.bias":
            other[0] = T[n][0]                        # hold this session's blank bias
        per_tensor[n] = {"other": other}
    else:
        assert Tp[n].shape == T[n].shape, (n, Tp[n].shape, T[n].shape)
        per_tensor[n] = {"other": Tp[n]}
print(f"H4b: {len(row_map)} rows aligned; subject-only rows kept: {only_subject}; blank bias held at "
      f"{float(T['O_18.lin.bias'][0]):+.4f}")
h4b = s.hypothesis(
    "H4b toward the parent at alpha 0.125 (follow-up of H4, which was refused at alpha 0.25 with Hertwig "
    "56/7 lines better/worse and Schlussredaktion 24/70, delta -0.0053 [-0.0116, +0.0002]). Interpolate "
    "every trained tensor an eighth of the way to kraken-bern-hertwig-v2, output rows aligned by character, "
    "the subject-only rows (½ ƒ ̊) and the blank bias held. The parent was trained on the 103 Hertwig "
    "practice lines with correct images (a Hertwig gain is partly memorisation) and never saw "
    "Schlussredaktion.",
    "Hertwig -0.5 to -1.5 points (confounded), Schlussredaktion +0.1 to +0.5, net -0.1 to -0.4 points; "
    "collapsed not up. If kept: a practice-set result with a memorisation confound; held-out and C5 decide. "
    "If refused: the parent direction is not kept at any alpha.",
    "interpolate toward the parent (whole network, rows aligned)")
s.edit("interpolate", names, hypothesis=h4b, alpha=0.125, per_tensor=per_tensor)
m_h4b = s.measure("H4b interpolate alpha 0.125 toward hertwig-v2")
report("H4b", m_h4b)
h4b_kept = decide(m_h4b, "H4b toward the parent alpha 0.125: predicted net -0.1 to -0.4 points (Hertwig part confounded)")
if h4b_kept:
    s.note("H4b kept: its Hertwig gain is measured on the parent's own training lines and is partly "
           "memorisation; the held-out set and the C5 sham decide whether anything transfers.")

# ---------------------------------------------------------------- freeze
summary = s.freeze()
print(f"program_03 done: H1c kept={h1c_kept}, H1d kept={h1d_kept}, H4b kept={h4b_kept}; "
      f"passes used {summary['used']} of {summary['budget']}, edits {summary['edits']}, kept {summary['kept']}; "
      f"baseline CER {summary['baseline']['cer']:.4f}, last pass CER {summary['last_pass']['cer']:.4f}; "
      f"final digest {summary['final_digest'][:12]}. FROZEN.")
print("changed tensors:", sorted(summary["changed_tensors"]))
