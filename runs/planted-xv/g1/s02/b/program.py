"""Blind programmer script for s02 task (b) (budget 0).

Pack readings (describe.md, diff_parent_to_subject.md, c0_practice.md) show a
healthy fine-tuned model with no planted defect. At budget 0, no edits are
applied and the uncorrupted model is frozen directly.
"""
from wsp.run import open_session


def main() -> None:
    s = open_session(".")
    s.note(
        "No defect detected in pack readings (all tensor norms, parent-to-child cosines, "
        "output row alignments, and C0 practice CER 0.0726 / length ratio 0.971 are normal). "
        "Applying no edits at budget 0."
    )
    s.freeze()


if __name__ == "__main__":
    main()
