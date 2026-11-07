"""wsp.collapse: the stricter repeat test and the line classes (engine-free)."""
from wsp.collapse import classify, degenerate_repeat


def test_correct_long_line_is_not_a_repeat():
    ref = "Act . 1276 . ind . IV . sexto non . Maji . et dimidia denar . Constant ."
    assert not degenerate_repeat(ref, ref)                      # " . " six times in correct text
    assert classify(ref, ref, 1500, 50) is None or classify(ref, ref, 1500, 50) == "detector_false_positive"


def test_real_loop_is_a_repeat():
    assert degenerate_repeat("ababababababababababab", "a short line")


def test_classes_in_order():
    long_ref = "der deutschen sprache , theils wegen den in denselben"
    assert classify(long_ref, "", 120, 1300) == "vertical"
    assert classify(long_ref, "", 1700, 2) == "degenerate_box"
    assert classify("ab", "", 50, 40) == "tiny_reference"
    assert classify(long_ref, "der", 900, 80, siblings=[long_ref]) == "model_specific"
    assert classify(long_ref, "der", 900, 80, siblings=["x"]) == "short_all_models"
