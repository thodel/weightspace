from wsp.vizpage import confusion_index, insertions, replacements


def test_replacements_and_insertions_follow_the_edit_alignment():
    assert replacements("abc", "abc") == ["", "", ""]
    assert replacements("abc", "axc") == ["", "x", ""]
    assert insertions("ab", "aXb") == [[0, "X"]]
    assert insertions("ab", "Xab") == [[-1, "X"]]


def test_confusion_index_counts_deletions_substitutions_and_insertions():
    lines = [
        {"k": "a/1", "r": "ein", "s": "odo", "x": [1, 2, 3], "lost": ["", "never", ""], "rp": ["", "", ""], "ins": []},
        {"k": "a/2", "r": "ein", "s": "oso", "x": [1, 2, 3], "lost": ["", "", ""], "rp": ["", "l", ""], "ins": [[2, " "]]},
        {"k": "a/3", "r": "in", "s": "do", "x": [1, 2], "lost": ["L_12", ""], "rp": ["", ""], "ins": []},
    ]
    idx = {(e["a"], e["b"]): e for e in confusion_index(lines)}
    assert idx[("i", "")]["n"] == 2 and idx[("i", "")]["lines"] == 2
    assert idx[("i", "")]["lost_at"] == {"never": 1, "L_12": 1}
    assert idx[("i", "l")]["n"] == 1 and idx[("", " ")]["occ"][0]["ins"] is True
