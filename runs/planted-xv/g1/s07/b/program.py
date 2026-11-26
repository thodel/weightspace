from wsp.run import open_session


def main() -> None:
    s = open_session(".")
    s.note(
        "No defect detected in s07 (negative control): all conv, BiLSTM, and output "
        "layer tensor statistics in describe.md and diff_donor_to_subject.md are normal, "
        "and c0_practice.md (CER 0.1993, 24 collapsed lines, deletions proportional to "
        "character frequency) matches the uncorrupted baseline model. Applying no edits "
        "at budget 0."
    )
    s.freeze()


if __name__ == "__main__":
    main()
