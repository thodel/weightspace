"""planted-f1 s12, task (b), budget 0: blind programmer (claude-fable-5-1).

One hypothesis (hypotheses.md H1): the `e` and `n` rows of the output layer are swapped.
Evidence: 70 % of all C0 edits are e<->n substitutions; against the parent exactly these two rows
have cosine 0.36 with unchanged norm, every other aligned row has cosine >= 0.88.

No pass exists, so the edit is gated by a check that needs no forward: the child's `e` row must be
closer (cosine) to the parent's `n` row than to the parent's `e` row, and vice versa. Only then the
two rows (weight and bias) are swapped back; otherwise nothing is edited.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
W, B = "O_18.lin.weight", "O_18.lin.bias"


def cos(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
s.note("budget 0: blind programmer; readings point to swapped e/n output rows (hypotheses.md H1)")

parent = KrakenSubject(PARENT)
PT = parent.tensors()
pm = parent.codec_map()

re_, rn = cm["e"], cm["n"]
pe, pn = pm["e"], pm["n"]
w, pw = T[W], PT[W]

c_ee, c_en = cos(w[re_], pw[pe]), cos(w[re_], pw[pn])   # child e against parent e / parent n
c_nn, c_ne = cos(w[rn], pw[pn]), cos(w[rn], pw[pe])     # child n against parent n / parent e
s.note(f"child e row: cos to parent e {c_ee:.3f}, to parent n {c_en:.3f}; "
       f"child n row: cos to parent n {c_nn:.3f}, to parent e {c_ne:.3f}; "
       f"norms child e {np.linalg.norm(w[re_]):.2f} n {np.linalg.norm(w[rn]):.2f}, "
       f"parent e {np.linalg.norm(pw[pe]):.2f} n {np.linalg.norm(pw[pn]):.2f}; "
       f"bias child e {T[B][re_]:+.3f} n {T[B][rn]:+.3f}, parent e {PT[B][pe]:+.3f} n {PT[B][pn]:+.3f}")

# the gate: both rows sit on the other character's parent row, by a clear margin
MARGIN = 0.15
swapped = (c_en > c_ee + MARGIN) and (c_ne > c_nn + MARGIN)

if swapped:
    h = s.hypothesis(
        "H1: the e and n rows of O_18 (weight and bias) have been exchanged; swapping them back restores "
        "the child's own rows. Gate passed: child e row is closer to parent n than to parent e and vice versa.",
        "CER 0.244 -> about 0.07-0.09 (the 2201 e<->n substitutions and most e/n deletions-insertions vanish); "
        "WER 0.63 -> about 0.3; collapsed lines not up",
        "set_rows (output rows e and n swapped, weight + bias)")
    new_w = np.stack([w[rn].copy(), w[re_].copy()])          # row e takes the current n row and vice versa
    new_b = np.array([T[B][rn], T[B][re_]], dtype=np.float32)
    s.edit("set_rows", [W, B], hypothesis=h, index=[re_, rn],
           per_tensor={W: {"values": new_w}, B: {"values": new_b}})
    T = s.subject.tensors()
    s.note(f"after swap: child e row cos to parent e {cos(T[W][re_], pw[pe]):.3f}, "
           f"child n row cos to parent n {cos(T[W][rn], pw[pn]):.3f}")
    s.keep_unmeasured("budget 0; readings (70 % of edits are e<->n, rows e/n cosine 0.36 to the parent with "
                      "unchanged norms) and the in-program cross-cosine gate both say the two rows were exchanged")
else:
    s.note(f"gate failed (cross cosines {c_en:.3f}/{c_ne:.3f} not above straight cosines {c_ee:.3f}/{c_nn:.3f} "
           f"by {MARGIN}): the e/n rows are not a simple swap; no edit applied, model shipped unchanged")

summary = s.freeze()
print("edits", summary["edits"], "kept", summary["kept"], "final", summary["final_digest"][:12])
