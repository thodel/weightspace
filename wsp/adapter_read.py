"""Budget-0 reading of a Qwen3.5 page adapter (phase 5, docs/PHASE5_PLAN.md P5.4).

No forward pass. For every adapted module: ΔW's singular values, how much of
its energy the top directions hold, and its share of the adapter's energy.
For the modules that write into the residual stream (``down_proj``,
``o_proj``), a logit lens on the top output directions: each direction u, put
through the base's final RMSNorm gain (zero-centred: 1 + w) and the tied
unembedding, gives a vector of token scores. The scores are listed for the
stop tokens, the newline and the hyphen sign ``¬``, and the tokens each
direction promotes and suppresses most. This ignores everything downstream of
the module (later layers, the norm's division), so it ranks hypotheses; it
does not prove one.

    python -m wsp.adapter_read <adapter dir> --out docs/readings/adapter_<name>.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from wsp.lora_subject import LoraSubject

BASE = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub/models--Qwen--Qwen3.5-4B/snapshots")
WATCH = {"<|im_end|>": 248046, "<|endoftext|>": 248044, "\\n": 198, "¬": 79259}


def base_head():
    """(gain, unembedding) of the base: (1 + norm.weight) and the tied embedding, float32."""
    from safetensors.torch import load_file
    snap = next(BASE.iterdir())
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for name in ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight"):
        out[name] = load_file(str(snap / idx[name]))[name].float().numpy()
    vocab = json.loads((snap / "tokenizer.json").read_text())
    tokens = {i: t for t, i in vocab["model"]["vocab"].items()}
    tokens.update({a["id"]: a["content"] for a in vocab["added_tokens"]})
    return 1.0 + out["model.language_model.norm.weight"], out["model.language_model.embed_tokens.weight"], tokens


def _show(tok: str) -> str:
    """Byte-level BPE as readable text (Ġ space, Ċ newline)."""
    try:
        raw = bytes(_BYTE_DECODER[c] for c in tok).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return tok
    return raw.replace("\n", "\\n")


def _byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _byte_decoder()


def read(adapter: Path, top_dirs: int = 4, top_tokens: int = 8) -> dict:
    s = LoraSubject(adapter)
    gain, unembed, tokens = base_head()
    mods = []
    for m in s.modules():
        u, sv, _ = s.directions(m)
        e = sv.astype(np.float64) ** 2
        row = {"module": re.sub(r"^base_model\.model\.model\.language_model\.", "", m),
               "layer": int(re.search(r"layers\.(\d+)\.", m).group(1)),
               "kind": m.rsplit(".", 1)[-1], "energy": float(e.sum()),
               "sv_top": [round(float(x), 4) for x in sv[:8]],
               "top4_share": float(e[:4].sum() / e.sum()),
               "rank90": int(np.searchsorted(np.cumsum(e) / e.sum(), 0.9) + 1)}
        if row["kind"] in ("down_proj", "o_proj"):
            lens = []
            for i in range(top_dirs):
                logits = unembed @ (gain * u[:, i]) * float(sv[i])
                z = (logits - logits.mean()) / logits.std()
                order = np.argsort(z)
                lens.append({"dir": i, "sv": float(sv[i]),
                             "watch_z": {k: round(float(z[t]), 2) for k, t in WATCH.items()},
                             "promotes": [_show(tokens.get(int(t), "?")) for t in order[::-1][:top_tokens]],
                             "suppresses": [_show(tokens.get(int(t), "?")) for t in order[:top_tokens]]})
            row["lens"] = lens
        mods.append(row)
    total = sum(r["energy"] for r in mods)
    for r in mods:
        r["energy_share"] = r["energy"] / total
    return {"adapter": str(adapter), "describe": s.describe(), "scale": s.scale, "modules": mods}


def md(d: dict, title: str, top: int = 12) -> str:
    rows = d["modules"]
    out = [f"# {title}", "", f"`{d['adapter']}` — {d['describe']['modules']} modules, "
           f"r = {d['describe']['r']}, scale α/r = {d['scale']}. Budget 0: no forward pass.", "",
           "## Energy by layer and kind (share of Σ‖ΔW‖²_F)", "",
           "| layer | " + " | ".join(k for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")) + " |",
           "|---:|" + "---:|" * 7]
    for layer in sorted({r["layer"] for r in rows}):
        cell = {r["kind"]: r for r in rows if r["layer"] == layer}
        out.append(f"| {layer} | " + " | ".join(f"{cell[k]['energy_share']*100:.2f}" if k in cell else "" for k in
                                                 ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")) + " |")
    lens = [(l["watch_z"]["<|im_end|>"], r["module"], l) for r in rows for l in r.get("lens", [])]
    for label, key, rev in (("lower the end-of-turn token most", 0, False), ("raise it most", 0, True)):
        out += ["", f"## Residual-writing directions that {label} (logit lens)", "",
                "| module | dir | σ | z(<\\|im_end\\|>) | z(\\n) | z(¬) | promotes | suppresses |",
                "|---|---:|---:|---:|---:|---:|---|---|"]
        for z, mod, l in sorted(lens, key=lambda x: x[0], reverse=rev)[:top]:
            w = l["watch_z"]
            out.append(f"| {mod} | {l['dir']} | {l['sv']:.3f} | {z:+.2f} | {w['\\n']:+.2f} | {w['¬']:+.2f} | "
                       + " ".join(f"`{t}`" for t in l["promotes"][:5]) + " | "
                       + " ".join(f"`{t}`" for t in l["suppresses"][:5]) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.adapter_read")
    ap.add_argument("adapter")
    ap.add_argument("--out", required=True, help="JSON; a .md beside it")
    ap.add_argument("--title", default="Adapter reading")
    a = ap.parse_args(argv)
    d = read(Path(a.adapter))
    out = Path(a.out)
    out.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    out.with_suffix(".md").write_text(md(d, a.title), encoding="utf-8")
    print(out, out.with_suffix(".md"))


if __name__ == "__main__":
    main()
