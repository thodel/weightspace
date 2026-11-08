"""RQ3 of Track C: exchange a block of a specific model for the same block of a more general one.

    python -m wsp.exchange --specific <S.mlmodel> --general <G.mlmodel> \\
        --sets <in-domain practice set> <other set> ... --out docs/family/exchange-<S>-from-<G>.json

For every block ``C_0 … O_18`` and every β, the block becomes ``S + β (G − S)`` (β = 1: G's block
transplanted; β = 0.5: half way), the rest stays S, and the model reads every set. The first set is
S's own material (train side, as every practice set here); the others are material neither model
was trained on, or G's. Each reading is compared, paired per line, with S's unedited reading of the
same set: corpus CER with its interval, and the block table of ``wsp.blocks`` (docs/BLOCK_STATS.md).

What it answers (docs/MODEL_FAMILY.md):

* **stabilisation** — a block exchange that leaves S's own set inside its interval and lowers CER on
  the other sets makes S more robust at no cost;
* **freezability (RQ1, functional)** — when G is S's parent, β = 1 undoes the fine-tune in that block
  alone; if S's own set does not move, that block's training could have been frozen.

The output layer is exchanged on the characters both alphabets share (rows paired by character,
``wsp.alpha_sweep.aligned_parent_output``); S's characters that G lacks keep S's rows. Body blocks
need one lineage (same initialisation): across lineages hidden units are in arbitrary order and an
element-wise exchange is meaningless, which ``--allow-unrelated`` must be passed to override.
Practice sets only: no held-out set is read here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wsp.alpha_sweep import aligned_parent_output, block_names
from wsp.blocks import bootstrap_blocks, per_line_counts
from wsp.data import load, sha256_file
from wsp.metrics import bootstrap_delta, per_line_edits, score

REPO = Path(__file__).resolve().parent.parent
BLOCKS = ["C_0", "C_3", "C_6", "C_9", "L_12", "L_14", "L_16", "O_18"]


def body_cosine(a: dict, b: dict) -> float:
    """Cosine of all tensors outside the output layer that both models have with equal shapes."""
    names = [n for n in a if not n.startswith("O_") and n in b and a[n].shape == b[n].shape]
    x = np.concatenate([a[n].ravel().astype(np.float64) for n in names]); y = np.concatenate([b[n].ravel().astype(np.float64) for n in names])
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


#: Output-row selections for a masked output move ``O_15/<rows>:beta`` (docs/RESULTS.md R12): only the rows of
#: the selected characters move; every other row keeps S's weights and bias. Rows are paired by character
#: (``aligned_parent_output``), never by index; characters G lacks keep S's rows in any case.
ROW_SELECTIONS = ("nonabbrev", "nonabbrev-noblank", "letters")


def _protected(ch: str) -> bool:
    """An abbreviation character in the sense of R4/R9: in the MUFI overlay, Combining Diacritical Marks or
    Latin Extended-D. A codec label of more than one code point counts as protected (none in the
    latin_incunabula codec)."""
    from wsp.blocks import block_of, in_mufi
    return len(ch) != 1 or in_mufi(ch) or block_of(ch) in ("Combining Diacritical Marks", "Latin Extended-D")


def selected_rows(S, G, rows: str) -> dict[int, str]:
    """S's output rows that a masked move moves: row -> character ('<blank>' for row 0)."""
    cs, cg = S.codec_map(), G.codec_map()
    shared = {i: ch for ch, i in cs.items() if ch in cg}
    if rows in ("nonabbrev", "nonabbrev-noblank"):
        sel = {i: ch for i, ch in shared.items() if not _protected(ch)}
    elif rows == "letters":
        sel = {i: ch for i, ch in shared.items() if len(ch) == 1 and ("a" <= ch <= "z" or "A" <= ch <= "Z")}
    else:
        raise KeyError(f"unknown row selection {rows!r}; one of {ROW_SELECTIONS}")
    if rows != "nonabbrev-noblank":
        sel[0] = "<blank>"
    return dict(sorted(sel.items()))


def move_target(b: str, st: dict, gt: dict, out_al: dict, S=None, G=None) -> dict:
    """The tensors block ``b`` moves towards. ``O_15/<rows>`` is the output layer with only the selected rows
    taken from G (paired by character), the rest S's own, so ``S + beta (target - S)`` leaves them unchanged."""
    if b.startswith("O_"):
        full = {n: v for n, v in out_al.items() if not n.startswith("_")}
        if "/" not in b:
            return full
        idx = sorted(selected_rows(S, G, b.split("/", 1)[1]))
        out = {}
        for n, v in full.items():
            t = st[n].copy(); t[idx] = v[idx]; out[n] = t
        return out
    return {n: gt[n] for n in block_names(st, b)}


def read(subject, items, refs):
    preds = subject.forward(items)
    hyps = [p.text for p in preds]
    return [p.key for p in preds], hyps, score(hyps, refs), per_line_edits(hyps, refs), per_line_counts(hyps, refs)


def save_readings(path: Path, keys, hyps, refs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"key": k, "hyp": h, "ref": r}, ensure_ascii=False) + "\n"
                            for k, h, r in zip(keys, hyps, refs)), encoding="utf-8")


def exchange(specific: Path, general: Path, sets: list[str], blocks: list[str], betas: list[float],
             readings_root: Path, allow_unrelated: bool = False, log=print) -> dict:
    from wsp.kraken_subject import KrakenSubject
    S, G = KrakenSubject(specific), KrakenSubject(general)
    st, gt = S.tensors(), G.tensors()
    cos = body_cosine(st, gt)
    if not blocks:                                   # every parameter block of S's architecture, in order
        blocks = [n for n, m in S.model.nn.named_children() if any(True for _ in m.parameters())]
    if cos < 0.5 and not allow_unrelated and any(not b.startswith("O_") for b in blocks):
        raise SystemExit(f"body cosine {cos:.3f}: not one lineage, an element-wise block exchange is meaningless")
    out_al = aligned_parent_output(G, S)
    data = {s: load(REPO / "data" / "practice" / f"{s}.jsonl") for s in sets}
    refs = {s: [x["text"] for x in data[s]] for s in sets}
    c0, g0, rows = {}, {}, []
    for s in sets:
        k, h, m, pl, pb = read(S, data[s], refs[s]); c0[s] = (pl, pb, m)
        save_readings(readings_root / s / "S.jsonl", k, h, refs[s])
        k, h, mg, _, _ = read(G, data[s], refs[s]); g0[s] = mg.cer
        save_readings(readings_root / s / "G.jsonl", k, h, refs[s])
        log(f"{s:28s} S CER {m.cer:.4f}   G CER {mg.cer:.4f}")
    for b in blocks:
        target = ({n: v for n, v in out_al.items() if not n.startswith("_")} if b.startswith("O_")
                  else {n: gt[n] for n in block_names(st, b)})
        orig = {n: st[n].copy() for n in target}
        for beta in betas:
            for n in target:
                S.set_tensor(n, orig[n] + beta * (target[n] - orig[n]))
            row = {"block": b, "beta": beta, "digest": S.digest(), "sets": {}}
            for s in sets:
                t = time.time()
                k, h, m, pl, pb = read(S, data[s], refs[s])
                save_readings(readings_root / s / f"{b}-b{beta}.jsonl", k, h, refs[s])
                boot = bootstrap_delta(c0[s][0], pl)
                blk = bootstrap_blocks(c0[s][1], pb)
                row["sets"][s] = {"cer": m.cer, "wer": m.wer, "collapsed": m.collapsed, "vs_S": boot,
                                  "blocks_vs_S": {g: v for g, v in blk.items() if v["ref"] >= 20 or max(v["errors_before"], v["errors_after"]) >= 10},
                                  "wall_s": round(time.time() - t, 1)}
                log(f"{b:5s} β={beta:<4} {s:28s} CER {m.cer:.4f}  Δ {100*boot['delta_cer']:+.2f} pt "
                    f"[{100*boot['ci95'][0]:+.2f}, {100*boot['ci95'][1]:+.2f}]")
            rows.append(row)
        for n, v in orig.items():
            S.set_tensor(n, v)
    return {"specific": specific.stem, "general": general.stem,
            "specific_sha256": sha256_file(specific), "general_sha256": sha256_file(general),
            "body_cosine": cos, "sets": sets, "own_set": sets[0], "betas": betas, "blocks": blocks,
            "output_rows_shared": out_al["_rows_shared"], "output_rows_new": out_al["_rows_new"],
            "c0": {s: {"cer": c0[s][2].cer, "wer": c0[s][2].wer, "collapsed": c0[s][2].collapsed} for s in sets},
            "general_cer": g0, "rows": rows, "when": time.strftime("%Y-%m-%d %H:%M")}


def parse_combo(spec: str) -> tuple[str, list[tuple[str, float]]]:
    """``name=Gn_1:1,Gn_3:1,O_15:0.5`` -> ("name", [("Gn_1", 1.0), ...])."""
    name, _, moves = spec.partition("=")
    return name, [(b, float(beta)) for b, beta in (m.split(":") for m in moves.split(","))]


def combine(specific: Path, general: Path, sets: list[str], combos: list[tuple[str, list[tuple[str, float]]]],
            readings_root: Path, log=print) -> dict:
    """Several block moves at once (each block ``S + β (G − S)``), every combination read on every set,
    paired against S unedited. For combining the moves a single-block grid found free."""
    from wsp.kraken_subject import KrakenSubject
    S, G = KrakenSubject(specific), KrakenSubject(general)
    st, gt = S.tensors(), G.tensors()
    cos = body_cosine(st, gt)
    if cos < 0.5:
        raise SystemExit(f"body cosine {cos:.3f}: not one lineage")
    out_al = aligned_parent_output(G, S)
    orig_all = {n: v.copy() for n, v in st.items()}
    data = {s: load(REPO / "data" / "practice" / f"{s}.jsonl") for s in sets}
    refs = {s: [x["text"] for x in data[s]] for s in sets}
    c0, g0, rows = {}, {}, []
    for s in sets:
        k, h, m, pl, pb = read(S, data[s], refs[s]); c0[s] = (pl, pb, m)
        save_readings(readings_root / s / "S.jsonl", k, h, refs[s])
        k, h, mg, _, _ = read(G, data[s], refs[s]); g0[s] = mg.cer
        save_readings(readings_root / s / "G.jsonl", k, h, refs[s])
        log(f"{s:28s} S CER {m.cer:.4f}   G CER {mg.cer:.4f}")
    for name, moves in combos:
        for b, beta in moves:
            target = move_target(b, orig_all, gt, out_al, S, G)
            if not target:
                raise KeyError(f"no block {b}")
            for n in target:
                S.set_tensor(n, orig_all[n] + beta * (target[n] - orig_all[n]))
        row = {"combo": name, "moves": [[b, beta] for b, beta in moves], "digest": S.digest(), "sets": {}}
        for b, _ in moves:
            if b.startswith("O_") and "/" in b:
                row.setdefault("rows_moved", {})[b] = list(selected_rows(S, G, b.split("/", 1)[1]).values())
        for s in sets:
            k, h, m, pl, pb = read(S, data[s], refs[s])
            save_readings(readings_root / s / f"{name}.jsonl", k, h, refs[s])
            boot = bootstrap_delta(c0[s][0], pl); blk = bootstrap_blocks(c0[s][1], pb)
            row["sets"][s] = {"cer": m.cer, "wer": m.wer, "collapsed": m.collapsed, "vs_S": boot,
                              "blocks_vs_S": {g: v for g, v in blk.items() if v["ref"] >= 20 or max(v["errors_before"], v["errors_after"]) >= 10}}
            log(f"{name:14s} {s:28s} CER {m.cer:.4f}  Δ {100*boot['delta_cer']:+.2f} pt "
                f"[{100*boot['ci95'][0]:+.2f}, {100*boot['ci95'][1]:+.2f}]")
        rows.append(row)
        for n, v in orig_all.items():
            if not np.array_equal(st[n], v):
                S.set_tensor(n, v)
    return {"kind": "combinations", "specific": specific.stem, "general": general.stem,
            "specific_sha256": sha256_file(specific), "general_sha256": sha256_file(general), "body_cosine": cos,
            "sets": sets, "own_set": sets[0],
            "c0": {s: {"cer": c0[s][2].cer, "wer": c0[s][2].wer, "collapsed": c0[s][2].collapsed} for s in sets},
            "general_cer": g0, "rows": rows, "when": time.strftime("%Y-%m-%d %H:%M")}


def build_candidate(combine_json: Path, combo: str, run_dir: Path) -> dict:
    """Rebuild one combination of a ``--combo`` run as a model and write it as a run folder that
    ``wsp.run heldout`` and ``wsp.run export`` accept: ``final.mlmodel``, ``diff.json``, ``session.json``,
    ``summary.json``. The weights come from the two artefacts named in the JSON (checked by sha256)."""
    from wsp import diff as diffmod
    from wsp.kraken_subject import KrakenSubject
    from wsp.run import _git_rev
    e = json.loads(Path(combine_json).read_text())
    row = next(r for r in e["rows"] if r["combo"] == combo)
    paths = {}
    from wsp.family import available
    for m in available():
        sha = sha256_file(Path(m["asteraix_path"]))
        if sha == e["specific_sha256"]: paths["S"] = (m["id"], Path(m["asteraix_path"]))
        if sha == e["general_sha256"]: paths["G"] = (m["id"], Path(m["asteraix_path"]))
    S, G = KrakenSubject(paths["S"][1]), KrakenSubject(paths["G"][1])
    start = {k: v.copy() for k, v in S.tensors().items()}
    st, gt = S.tensors(), G.tensors()
    out_al = aligned_parent_output(G, S)
    for b, beta in row["moves"]:
        target = move_target(b, start, gt, out_al, S, G)
        for n in target:
            S.set_tensor(n, start[n] + beta * (target[n] - start[n]))
    if S.digest() != row["digest"]:
        raise RuntimeError(f"rebuilt digest {S.digest()[:12]} is not the measured one {row['digest'][:12]}")
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    final = S.save(run_dir / "final.mlmodel")
    sparse = diffmod.sparse_diff(start, S.tensors()); diffmod.save_sparse_diff(sparse, run_dir / "diff.json")
    meta = {"name": run_dir.name, "subject": str(paths["S"][1]), "subject_id": paths["S"][0],
            "subject_digest": KrakenSubject(paths["S"][1]).digest(), "subject_file_sha256": e["specific_sha256"],
            "donor": str(paths["G"][1]), "donor_id": paths["G"][0], "donor_file_sha256": e["general_sha256"],
            "practice": str(REPO / "data" / "practice" / f"{e['own_set']}.jsonl"),
            "practice_sha256": sha256_file(REPO / "data" / "practice" / f"{e['own_set']}.jsonl"),
            "method": "block exchange (Track C, docs/MODEL_FAMILY.md): each block S + beta (G - S), output rows paired by character",
            "moves": row["moves"], "rows_moved": row.get("rows_moved"), "chosen_on": e["sets"], "combine_json": str(combine_json), "code": _git_rev(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (run_dir / "session.json").write_text(json.dumps(meta, indent=2))
    cmp = diffmod.compare(start, S.tensors())
    summary = {"budget": len(row["moves"]), "used": len(e["rows"]), "edits": len(row["moves"]), "kept": len(row["moves"]),
               "start_digest": meta["subject_digest"], "final_digest": S.digest(), "final_file_sha256": sha256_file(final),
               "changed_tensors": {k: v for k, v in cmp.items() if v.get("changed")}, "practice_vs_S": row["sets"],
               "heldout": None, "deviations": ["not a budgeted session: a combination chosen from the Track C exchange grids, "
                                               "confirmed on disjoint lines before the held-out score"],
               "subject": meta["subject"], "name": run_dir.name}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.exchange")
    ap.add_argument("--specific"); ap.add_argument("--general")
    ap.add_argument("--sets", nargs="+", help="practice sets; the first is the specific model's own material")
    ap.add_argument("--blocks", nargs="*", default=None, help="default: every parameter block of the specific model")
    ap.add_argument("--betas", nargs="*", type=float, default=[0.5, 1.0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unrelated", action="store_true")
    ap.add_argument("--candidate", nargs=2, metavar=("COMBINE_JSON", "COMBO"),
                    help="rebuild a measured combination as a run folder at --out (no reading)")
    ap.add_argument("--combo", action="append", help="name=BLOCK:beta,BLOCK:beta (repeatable): read these combinations instead of the per-block grid")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.candidate:
        summ = build_candidate(Path(a.candidate[0]), a.candidate[1], out)
        print(out, summ["final_digest"][:12], summ["final_file_sha256"][:12]); return
    if a.combo:
        res = combine(Path(a.specific), Path(a.general), a.sets, [parse_combo(c) for c in a.combo],
                      out.parent / "readings" / out.stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(out)
        return
    res = exchange(Path(a.specific), Path(a.general), a.sets, a.blocks, a.betas,
                   out.parent / "readings" / out.stem, a.allow_unrelated)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
