"""C3 of the protocol, per block: ``parent + α(child − parent)`` on one block at a time, scored once.

    python -m wsp.alpha_sweep --child models/kraken/<child>.mlmodel --parent models/kraken/<parent>.mlmodel \\
        --practice data/practice/<set>.jsonl --baseline docs/baselines/<set>.json --out docs/baselines/alpha-<child>.json

A control, not a session: no budget, no held-out set. Every block of the VGSL
spec (``C_0 … L_16``) is interpolated or extrapolated with the rest of the
network left as the child; the output layer separately, with rows **aligned
by character** (``wsp.align.codec_row_map``) and the child's new rows kept as
they are (their delta is zero). α = 0.5 asks how much of the fine-tune's gain
that block carries (if the CER halves back towards the parent, the gain sits
there); α = 1.5 asks whether the block tolerates extrapolation (the cheapest
weight-space edit of all). The paired interval against the C0 reading of the
unedited child comes from the baseline JSON's ``per_line`` (same lines, same
model digest), so the sweep spends exactly blocks × α passes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wsp import align
from wsp.data import load, sha256_file
from wsp.metrics import score, per_line_edits, bootstrap_delta, page_of

DEFAULT_BLOCKS = ["C_0", "C_3", "C_6", "C_9", "L_12", "L_14", "L_16"]
DEFAULT_ALPHAS = [0.5, 1.5]


def block_names(tensors: dict[str, np.ndarray], block: str) -> list[str]:
    return [n for n in tensors if n.split(".")[0] == block]


def aligned_parent_output(parent, child) -> dict[str, np.ndarray]:
    """The parent's output layer in the child's row order; rows the parent lacks take the child's."""
    cp, cc = parent.tensors(), child.tensors()
    wname = next(n for n in cc if n.startswith("O_") and n.endswith("lin.weight"))
    bname = wname.replace("weight", "bias")
    rm = align.codec_row_map(parent.codec_map(), child.codec_map()); rm[0] = 0      # blank ↔ blank
    rm = {ib: ia for ia, ib in rm.items()}                                          # child row -> parent row
    return {wname: align.gather_rows(cp[wname], rm, cc[wname].shape[0], fill=cc[wname]),
            bname: align.gather_rows(cp[bname], rm, cc[bname].shape[0], fill=cc[bname]),
            "_rows_shared": len(rm), "_rows_new": cc[wname].shape[0] - len(rm)}


def sweep(child, parent, practice: list[dict], baseline_lines, blocks=DEFAULT_BLOCKS, alphas=DEFAULT_ALPHAS,
          resamples: int = 2000, seed: int = 20261106, log=print, readings_dir: Path | None = None) -> list[dict]:
    """One pass per (block, α); the child is restored after each. Returns one row per pass."""
    ct = child.tensors()
    pt = parent.tensors()
    out_al = aligned_parent_output(parent, child)
    plan: list[tuple[str, dict[str, np.ndarray]]] = [(b, {n: pt[n] for n in block_names(ct, b)}) for b in blocks]
    plan.append(("O_18", {n: v for n, v in out_al.items() if not n.startswith("_")}))
    refs = [x["text"] for x in practice]
    clusters = [page_of(x["key"]) for x in practice]
    rows = []
    for block, par in plan:
        if not par:
            raise KeyError(f"block {block} has no tensors")
        orig = {n: ct[n].copy() for n in par}
        delta = {n: orig[n] - par[n] for n in par}
        rel = float(np.sqrt(sum(float((d.astype(np.float64) ** 2).sum()) for d in delta.values()))
                    / np.sqrt(sum(float((par[n].astype(np.float64) ** 2).sum()) for n in par)))
        for a in alphas:
            for n in par:
                child.set_tensor(n, par[n] + a * delta[n])
            t = time.time()
            preds = child.forward(practice)
            wall = time.time() - t
            hyps = [p.text for p in preds]
            if readings_dir is not None:          # per-variant predictions, for block intervals (wsp.blocks)
                readings_dir.mkdir(parents=True, exist_ok=True)
                (readings_dir / f"{block}-a{a}.jsonl").write_text("".join(
                    json.dumps({"key": p.key, "hyp": h, "ref": r}, ensure_ascii=False) + "\n"
                    for p, h, r in zip(preds, hyps, refs)), encoding="utf-8")
            m = score(hyps, refs)
            lines = per_line_edits(hyps, refs)
            boot = bootstrap_delta(baseline_lines, lines, resamples=resamples, seed=seed, clusters=clusters)
            row = {"block": block, "alpha": a, "tensors": sorted(par), "block_rel_l2": rel,
                   "metrics": m.as_dict(), "vs_c0": boot, "digest": child.digest(), "wall_s": round(wall, 1)}
            if block == "O_18":
                row["rows_shared"] = out_al["_rows_shared"]; row["rows_new"] = out_al["_rows_new"]
            rows.append(row)
            log(f"{block:5s} α={a:<4} CER {m.cer:.5f}  WER {m.wer:.4f}  "
                f"collapsed {m.collapsed:3d}  Δ {100*boot['delta_cer']:+.3f} pt [{100*boot['ci95'][0]:+.3f}, {100*boot['ci95'][1]:+.3f}]  {wall:.0f}s")
        for n, v in orig.items():
            child.set_tensor(n, v)
    return rows


def render_table(rows: list[dict], c0_cer: float) -> str:
    L = ["| block | rel L2 child−parent | α | CER | Δ vs C0 (pt) | 95 % interval (lines) | 95 % interval (pages) | WER | collapsed |",
         "|---|---:|---:|---:|---:|---|---|---:|---:|"]
    for r in rows:
        m, b = r["metrics"], r["vs_c0"]
        L.append(f"| {r['block']} | {r['block_rel_l2']:.3f} | {r['alpha']} | {m['cer']:.5f} | {100*b['delta_cer']:+.3f} | "
                 f"[{100*b['ci95'][0]:+.3f}, {100*b['ci95'][1]:+.3f}] | [{100*b['ci95_cluster'][0]:+.3f}, {100*b['ci95_cluster'][1]:+.3f}] | {m['wer']:.4f} | {m['collapsed']} |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.alpha_sweep")
    ap.add_argument("--child", required=True); ap.add_argument("--parent", required=True)
    ap.add_argument("--practice", required=True); ap.add_argument("--baseline", required=True,
                    help="C0 JSON of wsp.baselines holding the child's per_line on this practice set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", nargs="*", default=DEFAULT_BLOCKS)
    ap.add_argument("--alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--readings", help="folder for the per-variant predictions (default: next to --out)")
    a = ap.parse_args(argv)
    from wsp.kraken_subject import KrakenSubject
    child = KrakenSubject(a.child, device=a.device); parent = KrakenSubject(a.parent, device=a.device)
    practice = load(Path(a.practice))
    base = json.loads(Path(a.baseline).read_text())
    c0 = next(m for m in base["models"] if m["model"] == Path(a.child).stem)
    if c0["tensor_digest"] != child.digest():
        raise SystemExit(f"baseline digest {c0['tensor_digest'][:8]} is not the loaded child {child.digest()[:8]}")
    if c0["practice_sha256"] != sha256_file(Path(a.practice)):
        raise SystemExit("baseline was scored on another practice list")
    base_lines = [tuple(x) for x in c0["per_line"]]
    t0 = time.time()
    readings = Path(a.readings) if a.readings else Path(a.out).parent / "readings" / Path(a.out).stem
    rows = sweep(child, parent, practice, base_lines, a.blocks, a.alphas, readings_dir=readings)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "control": "C3 per-block alpha sweep", "child": Path(a.child).stem, "parent": Path(a.parent).stem,
        "child_digest": child.digest(), "parent_digest": parent.digest(),
        "practice": Path(a.practice).stem, "practice_sha256": c0["practice_sha256"], "lines": len(practice),
        "c0": {"cer": c0["metrics"]["cer"], "wer": c0["metrics"]["wer"], "collapsed": c0["metrics"]["collapsed"]},
        "parent_c1": next(({"cer": m["metrics"]["cer"]} for m in base["models"] if m["model"] == Path(a.parent).stem), None),
        "alphas": a.alphas, "blocks": a.blocks + ["O_18"], "passes": len(rows), "device": a.device,
        "wall_s": round(time.time() - t0, 1), "rows": rows}, indent=1, ensure_ascii=False))
    print(render_table(rows, c0["metrics"]["cer"]))
    print(out)


if __name__ == "__main__":
    main()
