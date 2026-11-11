"""planted-f1 / s13 / task (b), budget 0: blind repair of the subject.

Readings (pack): CER 0.7456 with 7,667 insertions of `n`; the output row of `n` has cosine -0.991 against the
parent's `n` row with a norm in the healthy range -> the child's own `n` row was sign-flipped. Repair: negate it back.
No pass exists; every edit is kept with keep_unmeasured. See hypotheses.md.
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
import numpy as np

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
CHAR = "n"

s = open_session(".")
s.note("task (b), budget 0, programmer claude-fable-5-1. Readings: CER 0.7456, 7667 insertions of 'n', "
       "O_18 row 'n' cos -0.991 to parent (rel L2 2.04, norm 7.96 -> 8.33). Diagnosis: the 'n' output row is negated.")

T = s.subject.tensors()
cm = s.subject.codec_map()

parent = KrakenSubject(PARENT)
PT = parent.tensors()
pm = parent.codec_map()

W = T["O_18.lin.weight"]
B = T["O_18.lin.bias"]
PW = PT["O_18.lin.weight"]
PB = PT["O_18.lin.bias"]

if CHAR not in cm or CHAR not in pm:
    s.note(f"character {CHAR!r} missing from a codec (child: {CHAR in cm}, parent: {CHAR in pm}); nothing applied")
else:
    row = int(cm[CHAR])
    prow = int(pm[CHAR])
    w_child = W[row].astype(np.float64)
    w_parent = PW[prow].astype(np.float64)
    cos = float(w_child @ w_parent / (np.linalg.norm(w_child) * np.linalg.norm(w_parent)))
    s.note(f"row {row} ('{CHAR}') vs parent row {prow}: cosine {cos:.4f}, norm child {np.linalg.norm(w_child):.3f}, "
           f"parent {np.linalg.norm(w_parent):.3f}; bias child {float(B[row]):+.4f}, parent {float(PB[prow]):+.4f}")

    # Sanity check of the whole output layer: which shared rows are anti-aligned with the parent? (expected: only 'n')
    anti = []
    for ch, r in cm.items():
        if ch in pm:
            a = W[r].astype(np.float64); b = PW[pm[ch]].astype(np.float64)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0 and float(a @ b / (na * nb)) < -0.5:
                anti.append(ch)
    s.note(f"shared rows with cosine < -0.5 to the parent: {anti!r}")

    if cos < -0.9:
        # H1: the row was sign-flipped. Negating it restores the child's own trained row.
        h1 = s.hypothesis(
            f"O_18.lin.weight row of '{CHAR}' is the negation of the healthy child row (cos {cos:.3f} to the parent, "
            "norm in the healthy range); it makes 'n' win on almost every frame (7667 insertions). Negate it back.",
            "CER falls from 0.746 to roughly 0.05-0.15 (>= 0.55 absolute); length ratio 1.61 -> ~1.0; "
            "the insertions of 'n' vanish and the missed 'n' return.",
            "set_rows (sign flip of one output row)")
        s.edit("set_rows", ["O_18.lin.weight"], hypothesis=h1, index=[row], values=(-W[row]).astype(np.float32))
        s.keep_unmeasured("budget 0: a cosine of -0.991 to the parent on a frequent character is a sign flip, "
                          "and the confusion table (7667 insertions of 'n') is exactly its symptom")

        # H2: was the bias of 'n' flipped with the row? The pack cannot tell (bias delta -0.142 fits either story).
        # Flip it only if the flipped value is far closer to the parent's bias than the current one.
        T = s.subject.tensors()
        B = T["O_18.lin.bias"]
        bc = float(B[row]); bp = float(PB[prow])
        d_keep = abs(bc - bp)
        d_flip = abs(-bc - bp)
        s.note(f"bias of '{CHAR}': child {bc:+.4f}, parent {bp:+.4f}; |keep-parent| {d_keep:.4f}, |flip-parent| {d_flip:.4f}")
        if d_flip < 0.25 * d_keep:
            h2 = s.hypothesis(
                f"O_18.lin.bias of '{CHAR}' was negated with the row: the flipped value ({-bc:+.3f}) is much closer to the "
                f"parent's ({bp:+.3f}) than the current one ({bc:+.3f}).",
                "small: CER change below 0.01 absolute, in the same direction as H1",
                "set_rows (sign flip of one bias entry)")
            s.edit("set_rows", ["O_18.lin.bias"], hypothesis=h2, index=[row], values=np.array([-bc], dtype=np.float32))
            s.keep_unmeasured("budget 0: the flipped bias matches the parent far better than the current one; the stake is small")
        else:
            s.note("bias of 'n' left as it is: the current value is not clearly the negation of a healthy one")
    else:
        s.note(f"guard failed: cosine {cos:.4f} is not < -0.9; the pack's reading does not reproduce, nothing applied")

summary = s.freeze()
print("frozen:", {k: summary.get(k) for k in ("edits", "kept", "final_digest")})
