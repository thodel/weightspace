import random

from wsp import metrics, pages


def test_levenshtein_matches_reference_implementation():
    rng = random.Random(3)
    for _ in range(200):
        a = "".join(rng.choice("abc ¬\nä") for _ in range(rng.randrange(0, 30)))
        b = "".join(rng.choice("abc ¬\nä") for _ in range(rng.randrange(0, 30)))
        assert pages.levenshtein(a, b) == metrics.levenshtein(a, b)


def test_page_loop_needs_more_repeats_than_the_reference():
    line = "und hat der Rath beschlossen, "
    assert pages.page_loop("Anfang. " + line * 8, "Anfang. " + line)
    assert not pages.page_loop("Anfang. " + line * 8, "Anfang. " + line * 8)
    assert not pages.page_loop("kurz", "kurz")


def test_score_counts_cap_truncation_and_length():
    s = pages.score(["abc", "abcdefXYZ"], ["abc", "abcdef"], [False, True])
    assert s["cer"] == 3 / 9
    assert s["truncated_cer"] == 0.0
    assert s["at_cap"] == 1
    assert abs(s["length_ratio"] - 12 / 9) < 1e-12


def test_agreement_flags_different_cap_sets():
    a = {"p1": {"text": "eins", "at_cap": False}, "p2": {"text": "zwei", "at_cap": True}}
    b = {"p1": {"text": "eins", "at_cap": False}, "p2": {"text": "zwei", "at_cap": False}}
    g = pages.agreement(a, b)
    assert g["exact"] == 2 and g["cer_between"] == 0.0 and not g["at_cap_same"]
