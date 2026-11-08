"""The reading pack a programmer sees for an adapter (stage A3): statistics only, no weights, no clean copy.

    python -m wsp.adapter_pack ADAPTER --siblings v1=DIR blb1=DIR --out pack.md [--c0 c0.json]

Per module: layer, kind, ‖ΔW‖_F, effective rank, top singular value, ‖B‖, ‖A‖, and the same
quantities' ratio to the median of the siblings' (adapters of the same base trained on the same recipe:
other runs, never the uncorrupted copy of this adapter). The config is printed in full, including the
siblings' ``lora_alpha``. Optionally a C0 reading of the practice set (per-page CER, characters written,
at the cap), from ``wsp.pages`` output.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from wsp.adapter_compare import profile
from wsp.lora_subject import LoraSubject


def build(adapter: Path, siblings: dict[str, Path], c0: dict | None = None, cached: Path | None = None) -> str:
    s = LoraSubject(adapter)
    mine = {r["module"]: r for r in profile(s)}
    if cached:  # sibling profiles already computed by wsp.adapter_compare (labels as in that file)
        allp = json.loads(Path(cached).read_text())["profiles"]
        sib = {k: {r["module"]: r for r in allp[k]} for k in siblings}
    else:
        sib = {k: {r["module"]: r for r in profile(LoraSubject(p))} for k, p in siblings.items()}
    cfg = {k: v for k, v in s.config.items() if k in ("base_model_name_or_path", "r", "lora_alpha", "lora_dropout", "target_modules", "use_rslora", "use_dora")}
    out = [f"# Reading pack — adapter `{adapter.name}`", "",
           "This adapter may be defective. It is a peft LoRA directory (rank r, scale = lora_alpha / r) on Qwen3.5-4B,",
           "trained to transcribe handwritten pages; the update of a module is ΔW = (lora_alpha/r)·B·A. No weights are shown,",
           "only statistics. Siblings: adapters of the same base and recipe from other runs (healthy as far as known).", "",
           "## Config", "", "```json", json.dumps(cfg, indent=1), "```", "",
           "Siblings' lora_alpha: " + ", ".join(f"{k}={LoraSubject(p).config['lora_alpha']}" for k, p in siblings.items()), "",
           f"Scale of this adapter: {s.scale}; siblings: " + ", ".join(f"{k}={LoraSubject(p).scale}" for k, p in siblings.items()), ""]
    tot = lambda d: float(np.sqrt(sum(r["fro"] ** 2 for r in d.values())))
    out += ["## Totals", "", f"- ‖ΔW‖ over all modules (Frobenius, with this adapter's own scale): {tot(mine):.2f}"]
    for k, d in sib.items():
        out.append(f"- sibling {k}: {tot(d):.2f}")
    out += ["", "## Per module", "",
            "ratio = this module's ‖ΔW‖ / median over siblings of the same module's ‖ΔW‖; effrank = exp-entropy effective rank (max r).", "",
            "| module | layer | kind | ‖ΔW‖ | ratio | effrank | σ1 | ‖B‖ | ‖A‖ |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for m in sorted(mine, key=lambda m: (mine[m]["layer"], m)):
        r = mine[m]
        med = float(np.median([d[m]["fro"] for d in sib.values()])) if sib else float("nan")
        short = re.sub(r"^base_model\.model\.model\.language_model\.", "", m)
        out.append(f"| {short} | {r['layer']} | {r['kind']} | {r['fro']:.3f} | {r['fro'] / med:.2f} | {r['eff_rank']:.1f} | {r['sv_max']:.3f} | {r['norm_B']:.3f} | {r['norm_A']:.3f} |")
    if c0:
        out += ["", "## C0 reading of the practice set (this adapter as it is)", "", "```json", json.dumps(c0, indent=1)[:6000], "```"]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.adapter_pack")
    ap.add_argument("adapter")
    ap.add_argument("--siblings", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--c0")
    ap.add_argument("--cached", help="adapter_compare.json holding the siblings' profiles under the same labels")
    a = ap.parse_args(argv)
    sibs = {x.split("=", 1)[0]: Path(x.split("=", 1)[1]) for x in a.siblings}
    c0 = json.loads(Path(a.c0).read_text()) if a.c0 else None
    Path(a.out).write_text(build(Path(a.adapter), sibs, c0, Path(a.cached) if a.cached else None))


if __name__ == "__main__":
    main()
