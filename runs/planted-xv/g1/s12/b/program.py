"""Blind weight-space repair program for pack s12, task (b) (budget 0).

Repairs the swapped output classifier rows for characters 'e' and 'n'
in O_18.lin.weight and O_18.lin.bias.
"""
from wsp.run import open_session


def main() -> None:
    s = open_session(".")
    T = s.subject.tensors()
    cm = s.subject.codec_map()

    ie = cm["e"]
    in_ = cm["n"]

    w_out = T["O_18.lin.weight"]
    b_out = T["O_18.lin.bias"]

    h1 = s.hypothesis(
        "Output layer O_18.lin rows for 'e' and 'n' are swapped (cos 0.356/0.361 vs parent, "
        "norm swap 7.64->8.33 and 7.96->7.57, and 1364 e->n + 837 n->e confusions in C0).",
        "CER decreases by ~0.171 on the practice set (from 0.2443 to ~0.073) by resolving ~2201 e<->n substitutions.",
        "set_rows",
    )
    s.edit(
        "set_rows",
        ["O_18.lin.weight", "O_18.lin.bias"],
        hypothesis=h1,
        index=[ie, in_],
        per_tensor={
            "O_18.lin.weight": {"values": w_out[[in_, ie]].copy()},
            "O_18.lin.bias": {"values": b_out[[in_, ie]].copy()},
        },
    )
    s.keep_unmeasured("Swap O_18.lin.weight and O_18.lin.bias rows for 'e' and 'n' back to their true positions.")

    s.freeze()


if __name__ == "__main__":
    main()
