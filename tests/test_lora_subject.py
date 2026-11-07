import json

import numpy as np
import pytest

safetensors = pytest.importorskip("safetensors.numpy")

from wsp.lora_subject import LoraSubject


@pytest.fixture
def adapter(tmp_path):
    rng = np.random.default_rng(0)
    t = {}
    for m in ("base_model.model.layers.0.mlp.down_proj", "base_model.model.layers.1.mlp.down_proj"):
        t[f"{m}.lora_A.weight"] = rng.normal(size=(4, 12)).astype(np.float32)
        t[f"{m}.lora_B.weight"] = rng.normal(size=(10, 4)).astype(np.float32)
    safetensors.save_file(t, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8, "target_modules": ["down_proj"]}))
    return tmp_path


def test_directions_match_full_svd_of_scaled_delta(adapter):
    s = LoraSubject(adapter)
    m = s.modules()[0]
    u, sv, vt = s.directions(m)
    assert s.scale == 2.0
    full = np.linalg.svd(s.delta(m), compute_uv=False)[:4]
    assert np.allclose(sv, full, rtol=1e-4)
    assert np.allclose((u * sv) @ vt, s.delta(m), atol=1e-4)


def test_remove_direction_kills_it_and_keeps_rank(adapter):
    s = LoraSubject(adapter)
    m = s.modules()[0]
    u = s.directions(m)[0][:, 0]
    s.remove_direction(m, u)
    assert np.abs(u @ s.delta(m)).max() < 1e-4
    assert s.tensors()[f"{m}.lora_B.weight"].shape == (10, 4)


def test_scale_module_and_save_roundtrip(adapter, tmp_path):
    s = LoraSubject(adapter)
    m = s.modules()[1]
    before = s.delta(m)
    s.scale_module(m, 0.5)
    assert np.allclose(s.delta(m), 0.5 * before, atol=1e-5)
    out = s.save(tmp_path / "out")
    back = LoraSubject(out)
    assert np.allclose(back.delta(m), s.delta(m), atol=1e-5)
    assert json.loads((out / "adapter_config.json").read_text())["r"] == 4
