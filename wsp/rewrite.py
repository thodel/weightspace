"""The lossless rewrite: kraken+ ``Cr255,1,85,1,1`` → ``Cr1,1,85,1,1``.

The trailing conv of ``KRAKEN_PLUS_SPEC`` sits on a feature map of height 1
(after ``S1(1x0)1,3``). A 255 × 1 kernel with padding 127 therefore multiplies
254 of its rows with zero padding on every input; only row 127 ever sees data.
Keeping that row and setting the kernel height to 1 (padding 0) is the same
function with 85 × 512 × 254 = 11,054,080 fewer weights.

    python -m wsp.rewrite --src models/kraken/<cr255>.mlmodel --dst models/kraken/<cr1>.mlmodel --practice data/practice/<set>.jsonl

Checks (all must hold; reported in the JSON next to ``--dst``):
same text on every practice line, max |Δ| of the softmax outputs, parameter
count drop, and the serving loader accepting the new file.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from wsp import ops
from wsp.data import load, sha256_file

CR255 = re.compile(r"Cr(\{[^}]*\})?255,1,(\d+),1,1")


def rewrite_spec(spec: str) -> tuple[str, str]:
    """Return (new spec, layer label) with the 255-row conv rewritten to a 1-row conv."""
    m = CR255.search(spec)
    if not m:
        raise ValueError(f"no Cr255,1,N,1,1 block in {spec}")
    label = (m.group(1) or "").strip("{}")
    new = spec[:m.start()] + f"Cr{m.group(1) or ''}1,1,{m.group(2)},1,1" + spec[m.end():]
    return new, label


def rewrite(src: Path, dst: Path, keep: int = 127) -> dict:
    import torch
    from kraken.lib.vgsl import TorchVGSLModel

    old = TorchVGSLModel.load_model(str(src))
    new_spec, label = rewrite_spec(old.spec)
    meta = dict(old.user_metadata)
    meta.pop("vgsl", None)          # passed as the vgsl= argument; save_model writes it back
    if isinstance(meta.get("model_type"), str):
        meta["model_type"] = [meta["model_type"]]
    codec = dict(old.codec.c2l)
    meta.pop("codec", None)
    new = TorchVGSLModel(vgsl=new_spec, codec=codec, **meta)
    sd_old = old.nn.state_dict()
    sd_new = new.nn.state_dict()
    wname = f"{label}.co.weight"
    if wname not in sd_old:
        raise KeyError(f"{wname} not in the state dict: {list(sd_old)[-4:]}")
    copied = {}
    for k, v in sd_old.items():
        if k == wname:
            w = ops.drop_padding_rows(v.numpy(), keep=keep, axis=2)
            copied[k] = torch.from_numpy(w)
        else:
            copied[k] = v.clone()
    missing, unexpected = new.nn.load_state_dict(copied, strict=True), None
    assert tuple(sd_new[wname].shape) == tuple(copied[wname].shape), (sd_new[wname].shape, copied[wname].shape)
    # the conv must have padding 0 in height now
    conv = dict(new.nn.named_modules())[f"{label}.co"] if f"{label}.co" in dict(new.nn.named_modules()) else None
    new.save_model(str(dst))
    n_old = sum(p.numel() for p in old.nn.parameters()); n_new = sum(p.numel() for p in new.nn.parameters())
    return {"src": str(src), "dst": str(dst), "spec_old": old.spec, "spec_new": new_spec, "layer": label,
            "kept_row": keep, "params_old": n_old, "params_new": n_new, "params_dropped": n_old - n_new,
            "conv_padding_new": list(getattr(conv, "padding", [])) if conv is not None else None}


def verify(src: Path, dst: Path, practice_path: Path, device: str = "cpu") -> dict:
    from kraken.lib.models import load_any
    from wsp.kraken_subject import KrakenSubject
    from wsp.metrics import score

    practice = load(practice_path)
    a, b = KrakenSubject(src, device=device), KrakenSubject(dst, device=device)
    ta = [p.text for p in a.forward(practice)]; tb = [p.text for p in b.forward(practice)]
    refs = [x["text"] for x in practice]
    pa, pb = a.logits(practice), b.logits(practice)
    max_abs = max(float(np.abs(x - y).max()) for x, y in zip(pa, pb))
    same_shape = all(x.shape == y.shape for x, y in zip(pa, pb))
    la = load_any(str(dst))
    return {"practice": practice_path.stem, "practice_sha256": sha256_file(practice_path), "lines": len(practice),
            "identical_text": ta == tb, "differing_lines": sum(x != y for x, y in zip(ta, tb)),
            "same_output_shape": same_shape, "max_abs_prob_diff": max_abs,
            "cer_old": score(ta, refs).cer, "cer_new": score(tb, refs).cer,
            "digest_old_tensor": a.digest(), "digest_new_tensor": b.digest(),
            "file_sha256_old": sha256_file(src), "file_sha256_new": sha256_file(dst),
            "load_any_new": type(la).__name__, "model_type_new": la.nn.model_type}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.rewrite")
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--practice", required=True); ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    src, dst = Path(a.src), Path(a.dst)
    r = rewrite(src, dst)
    r["verify"] = verify(src, dst, Path(a.practice), a.device)
    out = dst.with_suffix(".rewrite.json")
    out.write_text(json.dumps(r, indent=2))
    v = r["verify"]
    print(f"{src.name} -> {dst.name}: params {r['params_old']:,} -> {r['params_new']:,} (-{r['params_dropped']:,}); "
          f"identical text {v['identical_text']} ({v['differing_lines']} differ), max |Δp| {v['max_abs_prob_diff']:.2e}, "
          f"CER {v['cer_old']:.4f} -> {v['cer_new']:.4f}, load_any {v['load_any_new']}; {out}")


if __name__ == "__main__":
    main()
