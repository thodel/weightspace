import json

import numpy as np
import pytest

safetensors = pytest.importorskip("safetensors.numpy")

from wsp import run


def test_adapter_session_opens_in_page_mode_and_persists_a_kept_state(tmp_path, monkeypatch):
    ad = tmp_path / "ad"; ad.mkdir()
    m = "base_model.model.layers.0.mlp.down_proj"
    safetensors.save_file({f"{m}.lora_A.weight": np.ones((2, 3), np.float32),
                           f"{m}.lora_B.weight": np.ones((4, 2), np.float32)}, str(ad / "adapter_model.safetensors"))
    (ad / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2}))
    prac = tmp_path / "p.jsonl"
    prac.write_text(json.dumps({"key": "p1", "text": "a", "source": "data/pages/p1.jpg"}) + "\n")
    monkeypatch.setattr(run, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(run, "_unshare_works", lambda: False)
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "", "returncode": 0})())
    d = run.new(str(ad), str(prac), 2, run_id="t", data_root=str(tmp_path))
    meta = json.loads((d / "session.json").read_text())
    assert meta["unit"] == "page" and meta["data_root"] == str(tmp_path)
    s = run.open_session(d)
    assert s.unit == "page"
    s.subject.scale_module(m, 0.5)
    s.subject.save(d / "current.adapter")
    again = run.open_session(d)
    assert np.allclose(again.subject.delta(m), s.subject.delta(m))
