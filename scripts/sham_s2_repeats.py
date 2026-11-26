"""C5 shams and the component split for the kraken session-2 repeats (runs/20261124-bern-combined-v1-f2-b10-{2,3,4});
adapted from scripts/sham_combined_f2.py. Scored on practice (bern-train-300) and held-out (bern-eval), CPU.
Pre-registration: docs/S2_REPEATS_PREREG.md §3.

    .venv/bin/python scripts/sham_s2_repeats.py <run name> replay   # split only: replays the programs in a scratch copy
    .venv/bin/python scripts/sham_s2_repeats.py <run name> run      # 4 workers
    .venv/bin/python scripts/sham_s2_repeats.py <run name> report

Variants (each against the unedited subject, C0):
  final      the frozen model (the session's kept diff)
  part1..k   split (more than one kept edit): the increment of each kept edit (kept state i − kept state i−1),
             applied alone to the subject. The kept states come from a replay of the run's own programs in a scratch
             copy of the run folder (outside the budget), which must end at the run's final digest.
  sham00-04  C5: every changed tensor gets Gaussian noise of the same Frobenius norm as the kept diff of that tensor.
             Seeds 20261111+k.
The keep rule of PROTOCOL §3 (paired line interval vs C0 below 0) is applied to each variant on practice.
"""
import json, os, shutil, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SUBJ = Path.home() / "Repo/weightspace/models/kraken/kraken-bern-combined-v1.mlmodel"
NAME = sys.argv[1]
RUN = REPO / "runs" / NAME
OUT = Path.home() / "atr-cache/wsp/controls" / NAME
REPLAY = Path.home() / "atr-cache/wsp/replay" / NAME
SEEDS = [20261111 + k for k in range(5)]


def kept_count():
    return sum(1 for l in open(RUN / "log.jsonl") if (e := json.loads(l))["kind"] == "decision" and e.get("kept"))


def specs():
    n = kept_count()
    parts = [{"id": f"part{i}", "part": i} for i in range(1, n + 1)] if n > 1 else []
    return [{"id": "final"}] + parts + [{"id": f"sham{k:02d}", "seed": s} for k, s in enumerate(SEEDS)]


SNAP = '''
import numpy as np, wsp.session as S
_orig = S.Session.keep
def keep(self, reason=""):
    ok = _orig(self, reason)
    if ok:
        import pathlib
        d = pathlib.Path("kept_states"); d.mkdir(exist_ok=True)
        np.savez(d / f"kept_{len(list(d.glob('kept_*.npz'))) + 1:02d}.npz", **self.subject.tensors())
    return ok
S.Session.keep = keep
'''


def replay():
    """Run the programs again in a scratch copy, snapshotting the tensors at every kept decision."""
    if REPLAY.exists():
        shutil.rmtree(REPLAY)
    REPLAY.mkdir(parents=True)
    for f in ("session.json", "preregistration.md"):
        shutil.copy(RUN / f, REPLAY / f)
    (REPLAY / "notes.md").write_text("replay\n")
    progs = sorted(p.name for p in RUN.glob("program_*.py"))
    for p in progs:
        shutil.copy(RUN / p, REPLAY / p)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "PYTHONPATH": str(REPO)}
    for p in progs:
        r = subprocess.run([sys.executable, "-c", SNAP + f"\nimport runpy; runpy.run_path({p!r}, run_name='__main__')"],
                           cwd=REPLAY, env=env, capture_output=True, text=True)
        (REPLAY / f"stdout_{p}.log").write_text(r.stdout + r.stderr)
        print(p, "exit", r.returncode)
    from wsp.kraken_subject import KrakenSubject
    d = KrakenSubject(REPLAY / "final.mlmodel").digest()
    want = json.load(open(RUN / "summary.json"))["final_digest"]
    print("replay final digest", d[:16], "run", want[:16], "equal" if d == want else "DIFFERENT")
    assert d == want


def build(sp):
    from wsp.kraken_subject import KrakenSubject
    S = KrakenSubject(SUBJ); F = KrakenSubject(RUN / "final.mlmodel")
    t, f = S.tensors(), F.tensors()
    t0 = {n: v.copy() for n, v in t.items()}
    d = {n: f[n].astype(np.float64) - t0[n].astype(np.float64) for n in t0 if not np.array_equal(f[n], t0[n])}
    if sp["id"] == "final":
        return F, d
    if "part" in sp:
        i = sp["part"]
        cur = np.load(REPLAY / "kept_states" / f"kept_{i:02d}.npz")
        prev = np.load(REPLAY / "kept_states" / f"kept_{i - 1:02d}.npz") if i > 1 else t0
        for n in t0:
            inc = cur[n].astype(np.float64) - prev[n].astype(np.float64)
            if np.any(inc):
                S.set_tensor(n, (t0[n] + inc).astype(np.float32))
        return S, d
    rng = np.random.default_rng(sp["seed"])
    for n, dv in d.items():
        z = rng.standard_normal(dv.shape)
        S.set_tensor(n, (t0[n] + z * (np.linalg.norm(dv) / np.linalg.norm(z))).astype(np.float32))
    return S, d


def work(sp):
    out = OUT / f"{sp['id']}.json"
    if out.exists():
        return sp["id"]
    from wsp.data import load
    from wsp.metrics import per_line_edits, score
    S, d = build(sp)
    res = {"id": sp["id"], "seed": sp.get("seed"), "digest": S.digest(),
           "diff_norm": float(np.sqrt(sum(np.linalg.norm(v) ** 2 for v in d.values())))}
    for name, path in [("practice", REPO / "data/practice/bern-train-300.jsonl"), ("heldout", REPO / "data/heldout/bern-eval.jsonl")]:
        items = load(path); hyps = [p.text for p in S.forward(items)]; refs = [x["text"] for x in items]
        m = score(hyps, refs)
        res[name] = {"cer": m.cer, "collapsed": m.collapsed, "per_line": per_line_edits(hyps, refs)}
    out.write_text(json.dumps(res))
    return sp["id"]


def report():
    from wsp.data import load
    from wsp.metrics import bootstrap_delta
    base = json.loads((REPO / "docs/baselines/bern-train-300.json").read_text())["models"][0]["per_line"]
    hbase = json.loads((REPO / "docs/baselines/heldout-bern-eval.json").read_text())["models"][0]["per_line"]
    pg = lambda f: [x["key"].rsplit("/", 1)[0] for x in load(REPO / f)]
    cp, ch = pg("data/practice/bern-train-300.jsonl"), pg("data/heldout/bern-eval.jsonl")
    rows = {}
    for sp in specs():
        r = json.loads((OUT / f"{sp['id']}.json").read_text())
        p = bootstrap_delta([tuple(x) for x in base], [tuple(x) for x in r["practice"]["per_line"]], clusters=cp)
        h = bootstrap_delta([tuple(x) for x in hbase], [tuple(x) for x in r["heldout"]["per_line"]], clusters=ch)
        kept = bool(p["ci95"][1] < 0 and p["delta_cer"] < 0)
        rows[sp["id"]] = {"diff_norm": r["diff_norm"], "digest": r["digest"],
                          "practice": {"cer": r["practice"]["cer"], "collapsed": r["practice"]["collapsed"], **{k: p[k] for k in ("delta_cer", "ci95", "ci95_cluster")}},
                          "heldout": {"cer": r["heldout"]["cer"], "collapsed": r["heldout"]["collapsed"], **{k: h[k] for k in ("delta_cer", "ci95", "ci95_cluster")}},
                          "keep_rule_vs_C0": kept}
        f = lambda v: f"{100 * v:+.2f}"
        print(f"{sp['id']:8s} |d| {r['diff_norm']:7.2f}  practice {r['practice']['cer']:.4f} {f(p['delta_cer'])} line [{f(p['ci95'][0])},{f(p['ci95'][1])}] page [{f(p['ci95_cluster'][0])},{f(p['ci95_cluster'][1])}] coll {r['practice']['collapsed']} kept={kept} | "
              f"held-out {r['heldout']['cer']:.4f} {f(h['delta_cer'])} line [{f(h['ci95'][0])},{f(h['ci95'][1])}] page [{f(h['ci95_cluster'][0])},{f(h['ci95_cluster'][1])}] coll {r['heldout']['collapsed']}")
    (RUN / "controls.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[2] == "replay":
        replay()
    elif sys.argv[2] == "run":
        with Pool(4) as pool:
            for i in pool.imap_unordered(work, specs()):
                print("done", i, flush=True)
        report()
    else:
        report()
