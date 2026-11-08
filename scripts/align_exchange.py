"""Cross-lineage exchange (docs/MODEL_FAMILY.md "Next" item 2, 08.11.2026): align a general model G's units to an
unrelated specific model S's units, verify that the permuted G is function-identical to G, then let S take G's
aligned front end (or move half-way to it). Practice and confirmation lists only.

    CUDA_VISIBLE_DEVICES="" python scripts/align_exchange.py --specific S.mlmodel --general G.mlmodel \\
        --sets <own set> <ood set> ... --out docs/family/align-<S>-from-<G>.json

1. probe lines: the 120 lines of Track C's CKA (40 each from blb-train-300, bern-train-300, medieval-val-300,
   seed 20261107; ``wsp.family``), at most 400 sampled positions per line and block;
2. ``wsp.permute.match`` per block (Hungarian on activation correlation), ``permute_tensors`` -> G';
3. G' reads every line of every set; the text must equal G's on every line (else stop);
4. variants of S: S's conv blocks up to C_k replaced by G''s (``front<k>``), the same with the last transplanted
   block's channels rescaled to S's activation scale and S's dead channels kept dead (``front<k>s``), the first
   block half-way in aligned coordinates (``C_0h-al``), and as a control the same transplant without alignment
   (``front<k>-raw``); each paired against S unedited (line-level and page-clustered 95 % intervals).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from ood_combine import cluster_of, find_list  # noqa: E402

SCRATCH = Path.home() / "atr-cache" / "wsp" / "align"


def probe_items():
    from wsp.data import load
    from wsp.family import PROBE_PER_SET, PROBE_SETS, SEED
    items = []
    for s in PROBE_SETS:
        rows = load(REPO / "data" / "practice" / f"{s}.jsonl")
        items += random.Random(SEED).sample(rows, PROBE_PER_SET)
    return items


def sampled_activations(subject, items, blocks, per_line=400, seed=20261108):
    """Like wsp.permute.unit_activations, but at most ``per_line`` positions per line and block (same positions
    for both models: the generator is reseeded per model and draws from the same shapes)."""
    from wsp.permute import _w
    rng = np.random.default_rng(seed)
    out = {b: [] for b in blocks}
    for it in items:
        acts = subject.activations([it], list(blocks))
        for b in blocks:
            if not acts[b]:
                continue
            X = acts[b][0]
            if b.startswith("C_"):
                C = subject.tensors()[_w(b)[0]].shape[0]
                X = X.reshape(-1, C, X.shape[1] // C).transpose(0, 2, 1).reshape(-1, C)
            idx = rng.choice(X.shape[0], min(per_line, X.shape[0]), replace=False)
            out[b].append(X[idx])
    return {b: np.concatenate(v).astype(np.float64) for b, v in out.items()}


def _task(args):
    name, specific, tensors_npz, set_name, threads = args
    import torch
    torch.set_num_threads(threads)
    from wsp.data import load
    from wsp.kraken_subject import KrakenSubject
    from wsp.metrics import per_line_edits
    items = load(find_list(set_name))
    if find_list(set_name).parent.name == "heldout":
        raise SystemExit("held-out list")
    S = KrakenSubject(specific)
    if tensors_npz:
        z = np.load(tensors_npz)
        for k in z.files:
            S.set_tensor(k, z[k])
    preds = S.forward(items)
    hyps = [p.text for p in preds]; refs = [x["text"] for x in items]
    return {"keys": [x["key"] for x in items], "pl": per_line_edits(hyps, refs), "hyps": hyps, "refs": refs, "digest": S.digest()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--specific", required=True); ap.add_argument("--general", required=True)
    ap.add_argument("--sets", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12); ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(24)
    from wsp.data import sha256_file
    from wsp.exchange import body_cosine
    from wsp.kraken_subject import KrakenSubject
    from wsp.metrics import bootstrap_delta, score
    from wsp.permute import CONV, LSTM, block_tensors, match, permute_tensors, rescale_conv

    S, G = KrakenSubject(a.specific), KrakenSubject(a.general)
    st, gt = S.tensors(), G.tensors()
    tag = f"{Path(a.specific).stem}-from-{Path(a.general).stem}".replace(" ", "_")
    work = SCRATCH / tag; work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    items = probe_items()
    blocks = CONV + LSTM
    aS = sampled_activations(S, items, blocks); aG = sampled_activations(G, items, blocks)
    m = match(aS, aG)
    print(f"activations + matching {time.time()-t0:.0f} s", flush=True)
    for b in blocks:
        print(f"{b:5s} matched corr mean {m[b]['mean_corr']:.3f} median {m[b]['median_corr']:.3f}  (unaligned {m[b]['identity_mean_corr']:.3f})"
              f"  dead S {int(m[b]['dead_A'].sum())} G {int(m[b]['dead_B'].sum())}", flush=True)
    perms = {b: m[b]["perm"] for b in blocks}
    gp = permute_tensors(gt, perms)
    np.savez(work / "G_perm.npz", **gp)

    variants = {}
    for k in range(4):
        names = [n for b in CONV[:k + 1] for n in block_tensors(b)]
        variants[f"front{k}"] = {n: gp[n] for n in names}
        variants[f"front{k}-raw"] = {n: gt[n] for n in names}
        last = CONV[k]
        stdS, stdG = m[last]["std_A"], m[last]["std_B"]
        scale = np.where(stdS == 0, 0.0, np.where(stdG > 0, stdS / np.where(stdG > 0, stdG, 1), 1.0))
        variants[f"front{k}s"] = {**variants[f"front{k}"], **rescale_conv(gp, last, scale.astype(np.float32))}
        variants[f"front{k}s"] = {n: variants[f"front{k}s"][n] for n in names}
    variants["C_0h-al"] = {n: st[n] + 0.5 * (gp[n] - st[n]) for n in block_tensors("C_0")}
    variants["C_0h-raw"] = {n: st[n] + 0.5 * (gt[n] - st[n]) for n in block_tensors("C_0")}
    paths = {}
    for v, t in variants.items():
        p = work / f"{v}.npz"; np.savez(p, **t); paths[v] = str(p)

    tasks = ([("S", a.specific, None, s, a.threads) for s in a.sets] +
             [("G", a.general, None, s, a.threads) for s in a.sets] +
             [("G_perm", a.general, str(work / "G_perm.npz"), s, a.threads) for s in a.sets] +
             [(v, a.specific, paths[v], s, a.threads) for v in variants for s in a.sets])
    with ProcessPoolExecutor(a.procs) as ex:
        res = list(ex.map(_task, tasks))
    by = {(t[0], t[3]): r for t, r in zip(tasks, res)}
    identical = {s: sum(x == y for x, y in zip(by[("G", s)]["hyps"], by[("G_perm", s)]["hyps"])) for s in a.sets}
    lines = {s: len(by[("G", s)]["hyps"]) for s in a.sets}
    print("G_perm identical to G:", {s: f"{identical[s]}/{lines[s]}" for s in a.sets}, flush=True)
    out = {"specific": Path(a.specific).stem, "general": Path(a.general).stem,
           "specific_sha256": sha256_file(Path(a.specific)), "general_sha256": sha256_file(Path(a.general)),
           "body_cosine": body_cosine(st, gt), "probe": "120 Track C probe lines, <= 400 positions per line and block",
           "match": {b: {"mean_corr": m[b]["mean_corr"], "median_corr": m[b]["median_corr"],
                         "unaligned_mean_corr": m[b]["identity_mean_corr"], "dead_S": int(m[b]["dead_A"].sum()),
                         "dead_G_matched": int(m[b]["dead_B"].sum()), "corr_deciles": np.quantile(m[b]["corr"], np.linspace(0, 1, 11)).tolist()}
                     for b in blocks},
           "permuted_identical_lines": identical, "lines": lines,
           "sets": a.sets, "set_sha256": {s: sha256_file(find_list(s)) for s in a.sets}, "c0": {}, "general_cer": {}, "rows": []}
    if any(identical[s] != lines[s] for s in a.sets):
        out["error"] = "permuted G is not function-identical; variants not evaluated"
        Path(a.out).write_text(json.dumps(out, indent=1)); raise SystemExit(out["error"])
    for s in a.sets:
        out["c0"][s] = score(by[("S", s)]["hyps"], by[("S", s)]["refs"], blocks=False).cer
        out["general_cer"][s] = score(by[("G", s)]["hyps"], by[("G", s)]["refs"], blocks=False).cer
        print(f"{s:28s} S {100*out['c0'][s]:6.2f} %  G {100*out['general_cer'][s]:6.2f} %", flush=True)
    for v in variants:
        row = {"variant": v, "sets": {}}
        for s in a.sets:
            r0, r = by[("S", s)], by[(v, s)]
            mm = score(r["hyps"], r["refs"], blocks=False)
            boot = bootstrap_delta(r0["pl"], r["pl"], clusters=[cluster_of(k) for k in r0["keys"]])
            row["sets"][s] = {"cer": mm.cer, "collapsed": mm.collapsed, "vs_S": boot}
            print(f"{v:12s} {s:28s} CER {100*mm.cer:6.2f}  Δ {100*boot['delta_cer']:+6.2f}  page [{100*boot['ci95_cluster'][0]:+.2f}, {100*boot['ci95_cluster'][1]:+.2f}]", flush=True)
        out["rows"].append(row)
    out["when"] = time.strftime("%Y-%m-%d %H:%M")
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
