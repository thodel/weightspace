"""Blind repair program for pack s03, task (b) (budget 0)."""
import numpy as np
from wsp.run import open_session


def main() -> None:
    s = open_session(".")
    T = s.subject.tensors()

    h1 = s.hypothesis(
        "CTC blank bias O_18.lin.bias[0] has a planted +2.0 offset (+3.232 vs parent +1.119, "
        "bias delta +2.113), causing 14 collapsed lines, length ratio 0.941, and dominant deletion confusions",
        "CER decreases by ~0.030-0.045 (from 0.0789 to ~0.035-0.048), collapsed lines drop from 14 to ~0, "
        "and length ratio returns to ~1.00",
        "add",
    )

    delta = np.zeros_like(T["O_18.lin.bias"])
    delta[0] = -2.0
    s.edit("add", ["O_18.lin.bias"], hypothesis=h1, delta=delta)
    s.keep_unmeasured(
        "Subtract 2.0 from CTC blank bias O_18.lin.bias[0] (+3.232 -> +1.232) to undo planted blank-bias defect"
    )

    s.freeze()


if __name__ == "__main__":
    main()
