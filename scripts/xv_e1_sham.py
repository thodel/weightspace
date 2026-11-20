"""E1 cross-vendor (docs/XVENDOR_PLAN.md §5): C5 shams for the edit Gemini and GPT both kept.

Both sessions (runs/20261119-kurrent-page-v2-{g1,o1}-b10-1) froze on the same H1: direction 0 of
the reading projected out of layers.31.self_attn.o_proj.lora_B and layers.31.mlp.down_proj.lora_B.
The sham removes one seeded random unit direction from each of the same two B tensors (one rng per
seed, o_proj first), the way the phase-5 shams did for one tensor (scripts/p5f_queue.py). Seeds are
those of the earlier E1 shams, 20261108, 20261110, 20261111. Each sham is scored on the 64 practice
pages with the session engine (wsp.run.score_adapter_on_draw, card 0), resumable per chunk.

    python scripts/xv_e1_sham.py            # writes docs/baselines/xv-e1-sham-<seed>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.lora_subject import LoraSubject  # noqa: E402
from wsp.run import KURRENT, load, score_adapter_on_draw  # noqa: E402

RUN = Path("runs/20261119-kurrent-page-v2-g1-b10-1")
V2 = Path(json.loads((RUN / "session.json").read_text())["subject"])
VAL = KURRENT / "vlm-pages-val"
WORK = Path.home() / "atr-cache/wsp/xv/e1-sham"
P = "base_model.model.model.language_model.layers.31."
MODULES = (P + "self_attn.o_proj", P + "mlp.down_proj")
SEEDS = (20261108, 20261110, 20261111)

practice_keys = [json.loads(l)["key"] for l in Path("data/practice/kurrent-val-64.jsonl").read_text(encoding="utf-8").splitlines()]
val_rows = {Path(r["image"]).stem: r for r in load(VAL / "pages.jsonl")}
PRACTICE = [val_rows[k] for k in practice_keys]


def adapter(seed: int) -> Path:
    d = WORK / "adapters" / f"sham-{seed}"
    if not (d / "adapter_model.safetensors").exists():
        s = LoraSubject(V2)
        rng = np.random.default_rng(seed)
        for m in MODULES:
            v = rng.normal(size=s.tensors()[f"{m}.lora_B.weight"].shape[0])
            s.remove_direction(m, v / np.linalg.norm(v))
        s.save(d)
    return d


for seed in SEEDS:
    a = adapter(seed)
    res = score_adapter_on_draw(a, VAL, WORK / f"sham-{seed}", rows=PRACTICE)
    res["per_page"] = dict(zip(res.pop("keys"), res["per_page"]))
    res["adapter"] = str(a)
    res["edit"] = {"modules": list(MODULES), "seed": seed, "kind": "remove one random unit direction per B"}
    Path(f"docs/baselines/xv-e1-sham-{seed}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(seed, {k: round(v, 4) if isinstance(v, float) else v for k, v in res["wsp"].items()}, flush=True)
