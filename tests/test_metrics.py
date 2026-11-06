from wsp.metrics import levenshtein, score, is_collapsed, bootstrap_delta, per_line_edits, align


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein(["a", "b"], ["a", "c"]) == 1


def test_score_exact_and_nfd():
    m = score(["Laßberg", "über"], ["Laßberg", "über"])
    assert m.cer == 0 and m.wer == 0 and m.collapsed == 0


def test_score_counts_corpus_level():
    m = score(["abcd", ""], ["abce", "xyz"])
    assert m.ref_chars == 7 and m.char_edits == 4
    assert abs(m.cer - 4 / 7) < 1e-9
    assert m.collapsed == 1  # the empty line against xyz


def test_confusions_listed():
    m = score(["ab"], ["ac"])
    assert m.confusions[0][:2] == ("c", "b")


def test_collapse_detection():
    assert is_collapsed("eeeeeeeeeeeeeeeeeeeeeeee", "ein kurzer satz hier")
    assert is_collapsed("ab", "ein kurzer satz hier")
    assert not is_collapsed("ein kurzer satz hier", "ein kurzer satz hier")


def test_align_lengths():
    pairs = align("abc", "axbd")
    assert "".join(a for a, _ in pairs) == "abc"
    assert "".join(b for _, b in pairs) == "axbd"


def test_bootstrap_delta_sign():
    refs = ["ein satz"] * 20
    before = per_line_edits(["ein sxtz"] * 20, refs)
    after = per_line_edits(["ein satz"] * 20, refs)
    r = bootstrap_delta(before, after, resamples=200)
    assert r["delta_cer"] < 0 and r["ci95"][1] <= 0
