import random

from wsp.blocks import (MUFI, SPACE, LOWER, UPPER, block_of, bootstrap_blocks, errors, line_counts,
                        per_line_counts, table)
from wsp.metrics import bootstrap_delta, levenshtein, normalise, per_line_edits, score


def test_partition_is_total():
    for ch in "a Z 9 ; ä ſ ß ꝛ  ⁊ ✳ ½ ̈ ͤ α ד 𝔄".replace(" ", ""):
        assert block_of(ch)
    assert block_of(" ") == SPACE and block_of("q") == LOWER and block_of("Q") == UPPER
    assert block_of("̈") == "Combining Diacritical Marks"
    assert block_of("ſ") == "Latin Extended-A"
    assert block_of("ꝛ") == "Latin Extended-D"
    assert block_of("") == "Private Use Area"
    assert block_of("𝔄").startswith("Other (U+1D500")


def test_errors_sum_to_levenshtein_and_insertions_are_charged():
    pairs = [("Laßberg über", "Lasberg uber"), ("ein satz", "ein  satzx"), ("abc", ""), ("", "xyz"), ("ſ ꝛ", "s r")]
    for ref, hyp in pairs:
        c = line_counts(ref, hyp)
        total = sum(errors(v) for g, v in c.items() if g != MUFI)
        assert total == levenshtein(normalise(ref), normalise(hyp)), (ref, hyp, c)
    c = line_counts("ab", "abx")
    assert c[LOWER]["ins"] == 1 and c[LOWER]["ref"] == 2 and c[LOWER]["correct"] == 2


def test_umlaut_loss_is_a_combining_mark_error():
    c = line_counts("über", "uber")        # NFD: u + U+0308
    assert c["Combining Diacritical Marks"]["del"] == 1
    assert c[LOWER]["correct"] == 4


def test_shares_add_up_to_cer_and_mufi_is_an_overlay():
    hyps, refs = ["Lasberg uber", "ein  satzx", "ꝑ"], ["Laßberg über", "ein satz", "ꝑro"]
    m = score(hyps, refs)
    t = m.blocks
    assert abs(t["cer"] - m.cer) < 1e-12
    assert abs(sum(r["share_of_cer"] for g, r in t["blocks"].items() if not r["overlay"]) - m.cer) < 1e-12
    assert t["blocks"][MUFI]["overlay"] and t["blocks"][MUFI]["ref"] == 1


def test_block_deltas_add_up_to_the_corpus_delta():
    rng = random.Random(1)
    refs = ["".join(rng.choice("abcdefg äöüſ.,") for _ in range(30)) for _ in range(60)]
    before = ["".join(ch for ch in r if rng.random() > 0.05) for r in refs]
    after = ["".join(ch for ch in r if rng.random() > 0.08) for r in refs]
    bb = bootstrap_blocks(per_line_counts(before, refs), per_line_counts(after, refs), resamples=300)
    corpus = bootstrap_delta(per_line_edits(before, refs), per_line_edits(after, refs), resamples=300)
    total = sum(v["delta_share"] for g, v in bb.items() if not v["overlay"])
    assert abs(total - corpus["delta_cer"]) < 1e-12


def test_table_without_errors():
    t = table(per_line_counts(["abc"], ["abc"]))
    assert t["cer"] == 0 and t["blocks"][LOWER]["block_cer"] == 0
