"""The first thing to run on asteraix: a no-op round-trip through kraken 7.0.2.

Skipped wherever kraken is not installed. Needs WSP_MODEL (a .mlmodel) and
WSP_PRACTICE (a jsonl of {"key","text","image"} line crops, train side only).
"""
import json
import os
from importlib.metadata import version
from pathlib import Path

import pytest

kraken = pytest.importorskip("kraken")

from wsp.kraken_subject import KrakenSubject  # noqa: E402
from wsp.metrics import score  # noqa: E402


@pytest.fixture
def practice():
    p = os.environ.get("WSP_PRACTICE")
    if not p:
        pytest.skip("WSP_PRACTICE not set")
    return [json.loads(l) for l in Path(p).read_text().splitlines()][:20]


def test_kraken_version_is_what_the_serving_loader_reads():
    assert version("kraken").startswith("7.0."), version("kraken")  # kraken 7 has no __version__ attribute


def test_noop_roundtrip(tmp_path, practice):
    path = os.environ.get("WSP_MODEL") or pytest.skip("WSP_MODEL not set")
    s = KrakenSubject(path)
    d0 = s.digest()
    before = [p.text for p in s.forward(practice)]
    out = s.save(tmp_path / "copy.mlmodel")
    s2 = KrakenSubject(out)
    assert s2.digest() == d0
    after = [p.text for p in s2.forward(practice)]
    assert before == after
    refs = [p["text"] for p in practice]
    assert score(before, refs).cer == score(after, refs).cer
