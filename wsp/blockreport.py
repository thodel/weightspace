"""Block statistics over every experiment so far: tables, paired block deltas, docs/BLOCK_STATS.md.

    python -m wsp.blockreport all          # after the re-run (baselines, alpha sweep, replays)

Writes docs/baselines/blocks/block_stats.json (all numbers) and docs/baselines/blocks/tables.md
(the generated tables); docs/BLOCK_STATS.md is the written findings followed by tables.md.

Works only from readings (hypothesis and reference text per line), so every number
here can be recomputed without a model:

* practice baselines   docs/baselines/readings/<set>/<model>.jsonl      (re-run 07.11.2026)
* held-out baselines   docs/baselines/readings/<heldout>/<model>.jsonl  (stored; held-out is scored once)
* sessions             <replay>/readings/pass_NNN.jsonl                  (replayed, checked identical)
                       runs/<session>/readings/heldout_<set>.jsonl        (stored, the frozen artefact's)
* C3 alpha sweep       docs/baselines/readings/alpha-xix-blb-v1/<block>-a<α>.jsonl (re-run)

Every comparison is paired (same lines, two readings) with ``wsp.blocks.bootstrap_blocks``:
2,000 resamples, seed 20261106, the change of each block's share of the corpus CER in points
(they add up to the corpus change) and the change of the block's own error rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wsp.blocks import MUFI, bootstrap_blocks, per_line_counts, table
from wsp.data import load
from wsp.metrics import is_collapsed, score

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "docs" / "baselines"
OUT = BASE / "blocks"
RUNS = Path.home() / "Repo" / "weightspace" / "runs"
REPLAY = Path.home() / "atr-cache" / "wsp" / "replay"
MIN_REF = 20          # blocks with fewer reference characters are listed in the JSON, not the tables


def read_pairs(path: Path, refs_by_key: dict[str, str] | None = None) -> tuple[list[str], list[str], list[str]]:
    keys, hyps, refs = [], [], []
    for l in path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        keys.append(d["key"]); hyps.append(d.get("hyp", d.get("text", "")))
        refs.append(d["ref"] if "ref" in d else refs_by_key[d["key"]])
    return keys, hyps, refs


def _pt(x):
    return "" if x is None else f"{100 * x:+.2f}"


def _short(g: str) -> str:
    return (g.replace("Combining Diacritical Marks", "Combining marks").replace("Latin lowercase a-z", "a-z")
             .replace("Latin uppercase A-Z", "A-Z").replace("ASCII punctuation and symbols", "ASCII punct.")
             .replace("General Punctuation", "Gen. punct.").replace("Supplemental Punctuation", "Suppl. punct.")
             .replace("Latin-1 Supplement", "Latin-1").replace("Latin Extended-", "Lat.Ext-")
             .replace("Spacing Modifier Letters", "Modifier letters").replace(" (CERberus list, overlay)", "*"))


# ---------------------------------------------------------------- per model and set
def model_tables(set_name: str) -> dict:
    d = BASE / "readings" / set_name
    out = {}
    for f in sorted(d.glob("*.jsonl")):
        keys, hyps, refs = read_pairs(f)
        out[f.stem] = {"keys": keys, "per_line": per_line_counts(hyps, refs), "hyps": hyps, "refs": refs}
    return out


def render_set(set_name: str, models: dict, order: list[str] | None = None) -> list[str]:
    names = order or list(models)
    tabs = {m: table(models[m]["per_line"]) for m in names}
    blocks = sorted({g for t in tabs.values() for g, r in t["blocks"].items() if r["ref"] >= MIN_REF or r["errors"] >= 10},
                    key=lambda g: -max(tabs[m]["blocks"].get(g, {"errors": 0})["errors"] for m in names))
    ref0 = tabs[names[0]]
    L = [f"#### {set_name} — {len(models[names[0]]['keys'])} lines, {ref0['ref_chars']:,} reference characters", "",
         "Per block: reference characters, then for each model the block's own CER (errors / its reference characters) "
         "and, in brackets, its share of the corpus CER in points. Shares add up to the corpus CER (last row).", "",
         "| block | ref | " + " | ".join(m.replace("kraken-", "") for m in names) + " |",
         "|---|---:|" + "---:|" * len(names)]
    for g in blocks:
        cells = []
        for m in names:
            r = tabs[m]["blocks"].get(g)
            cells.append("" if r is None or not r["ref"] and not r["errors"] else
                         (f"{100 * r['block_cer']:.1f} % ({100 * r['share_of_cer']:.2f})" if r["block_cer"] is not None
                          else f"ins only ({100 * r['share_of_cer']:.2f})"))
        L.append(f"| {_short(g)} | {ref0['blocks'].get(g, {'ref': 0})['ref']} | " + " | ".join(cells) + " |")
    L.append("| **corpus CER** | | " + " | ".join(f"**{100 * tabs[m]['cer']:.2f} %**" for m in names) + " |")
    L.append("")
    return L


# ---------------------------------------------------------------- paired block deltas
def paired(before: list[dict], after: list[dict]) -> dict:
    return bootstrap_blocks(before, after)


def without_collapsed(hb, ha, refs, pb, pa):
    """Keep only lines that neither reading collapses (wsp.metrics.is_collapsed). PR #7 found the collapsed
    held-out lines are data and metric defects, not readable lines; their mass deletions otherwise land in
    every block. Returns the filtered per-line counts, the two CERs on the kept lines, and how many were dropped."""
    keep = [i for i in range(len(refs)) if not is_collapsed(hb[i], refs[i]) and not is_collapsed(ha[i], refs[i])]
    sel = lambda xs: [xs[i] for i in keep]
    cb = score(sel(hb), sel(refs), blocks=False).cer; ca = score(sel(ha), sel(refs), blocks=False).cer
    return sel(pb), sel(pa), cb, ca, len(refs) - len(keep)


def render_paired(title: str, bb: dict, corpus_before: float, corpus_after: float, note: str = "") -> list[str]:
    rows = sorted(((g, v) for g, v in bb.items() if v["ref"] >= MIN_REF or max(v["errors_before"], v["errors_after"]) >= 10),
                  key=lambda kv: kv[1]["delta_share"])
    L = [f"#### {title}", ""] + ([note, ""] if note else []) + [
        f"Corpus CER {100 * corpus_before:.2f} % → {100 * corpus_after:.2f} % ({100 * (corpus_after - corpus_before):+.2f} points). "
        "Per block: errors before → after, change of the block's share of the corpus CER in points with its paired 95 % "
        "interval, and the change of the block's own error rate. **Bold**: the share's interval excludes 0.", "",
        "| block | ref | errors | Δ share (pt) | 95 % interval | Δ block CER (pt) |", "|---|---:|---:|---:|---|---:|"]
    for g, v in rows:
        b = "**" if v["excludes_0"] else ""
        L.append(f"| {_short(g)} | {v['ref']} | {v['errors_before']} → {v['errors_after']} | {b}{_pt(v['delta_share'])}{b} | "
                 f"[{_pt(v['ci95_share'][0])}, {_pt(v['ci95_share'][1])}] | {_pt(v['delta_block_cer'])} |")
    L.append("")
    return L


# ---------------------------------------------------------------- sessions
def session_passes(name: str) -> dict:
    rdir = REPLAY / name
    meta = json.loads((rdir / "session.json").read_text())
    practice = load(meta["practice"])
    refs_by_key = {x["key"]: x["text"] for x in practice}
    log = [json.loads(l) for l in (rdir / "log.jsonl").read_text().splitlines() if l.strip()]
    labels = [e["label"] for e in log if e["kind"] == "pass"]
    decisions = [e for e in log if e["kind"] == "decision"]
    passes = []
    for i, f in enumerate(sorted((rdir / "readings").glob("pass_*.jsonl"))):
        keys, hyps, refs = read_pairs(f, refs_by_key)
        passes.append({"label": labels[i] if i < len(labels) else f.stem, "per_line": per_line_counts(hyps, refs),
                       "cer": score(hyps, refs, blocks=False).cer})
    # kept state before each pass: pass 1 is C0; after a kept decision the measured pass becomes the kept state
    kept_idx, rows = 0, []
    d_iter = iter(decisions)
    for i, p in enumerate(passes):
        if i == 0:
            continue
        vs_kept = paired(passes[kept_idx]["per_line"], p["per_line"])
        vs_c0 = paired(passes[0]["per_line"], p["per_line"])
        dec = next(d_iter, None)
        rows.append({"pass": i + 1, "label": p["label"], "cer": p["cer"], "kept_ref_pass": kept_idx + 1,
                     "kept": bool(dec and dec["kept"]), "vs_kept": vs_kept, "vs_c0": vs_c0})
        if dec and dec["kept"]:
            kept_idx = i
    check = json.loads((rdir / "replay_check.json").read_text())
    return {"session": name, "check": {k: check[k] for k in ("exact", "passes", "decisions", "final_digest")},
            "c0_cer": passes[0]["cer"], "rows": rows}


def render_session(s: dict, top: list[str]) -> list[str]:
    L = [f"#### Session `{s['session']}` — per pass, against the state it was compared with", "",
         f"Replay exact: **{s['check']['exact']}** (pass digests equal {s['check']['passes']['digests_equal']}, "
         f"decisions equal {s['check']['decisions']['equal']}, final digest equal {s['check']['final_digest']['equal']}, "
         "every pass reads the same text on every line). Cells: change of the block's share of CER in points against "
         "the last kept state; **bold** where the paired interval excludes 0.", "",
         "| pass | hypothesis | kept | Δ CER (pt) | " + " | ".join(_short(g) for g in top) + " |",
         "|---:|---|---|---:|" + "---:|" * len(top)]
    for r in s["rows"]:
        total = sum(v["delta_share"] for g, v in r["vs_kept"].items() if not v["overlay"])
        cells = []
        for g in top:
            v = r["vs_kept"].get(g)
            if v is None:
                cells.append(""); continue
            b = "**" if v["excludes_0"] else ""
            cells.append(f"{b}{_pt(v['delta_share'])}{b}")
        L.append(f"| {r['pass']} | {r['label']} | {'yes' if r['kept'] else 'no'} | {_pt(total)} | " + " | ".join(cells) + " |")
    L.append("")
    return L


# ---------------------------------------------------------------- alpha sweep
def alpha_matrix() -> dict:
    meta = json.loads((BASE / "alpha-xix-blb-v1.json").read_text())
    c0_keys, c0_h, c0_r = read_pairs(BASE / "readings" / "blb-train-300" / f"{meta['child']}.jsonl")
    c0 = per_line_counts(c0_h, c0_r)
    rows = []
    for r in meta["rows"]:
        f = BASE / "readings" / "alpha-xix-blb-v1" / f"{r['block']}-a{r['alpha']}.jsonl"
        keys, hyps, refs = read_pairs(f)
        assert keys == c0_keys, "alpha readings are not on the C0 lines"
        rows.append({"block": r["block"], "alpha": r["alpha"], "cer": r["metrics"]["cer"],
                     "vs_c0": paired(c0, per_line_counts(hyps, refs))})
    return {"child": meta["child"], "parent": meta["parent"], "c0_cer": meta["c0"]["cer"], "rows": rows}


def render_alpha(a: dict, top: list[str]) -> list[str]:
    L = [f"#### C3 α-sweep `{a['parent']}` → `{a['child']}`: network block × Unicode block", "",
         "One network block at a time set to parent + α(child − parent), the rest left as the child; practice "
         "`blb-train-300`. Cells: change of the Unicode block's share of CER in points against the unedited child (C0); "
         "**bold** where the paired 95 % interval excludes 0. At α = 0.5 a positive cell says that network block "
         "carries part of the fine-tune's gain *on that kind of character*.", "",
         "| network block | α | Δ CER (pt) | " + " | ".join(_short(g) for g in top) + " |",
         "|---|---:|---:|" + "---:|" * len(top)]
    for r in a["rows"]:
        cells = []
        for g in top:
            v = r["vs_c0"].get(g)
            b = "**" if v and v["excludes_0"] else ""
            cells.append("" if v is None else f"{b}{_pt(v['delta_share'])}{b}")
        L.append(f"| {r['block']} | {r['alpha']} | {_pt(r['cer'] - a['c0_cer'])} | " + " | ".join(cells) + " |")
    L.append("")
    return L


# ---------------------------------------------------------------- held-out
def heldout_rescore(set_name: str, json_name: str) -> dict:
    """Recompute the held-out baseline metrics from the stored readings (no new inference) and add the block
    table; the CER must come out identical to the stored one."""
    p = BASE / json_name
    d = json.loads(p.read_text())
    changes = {}
    for m in d["models"]:
        keys, hyps, refs = read_pairs(BASE / "readings" / set_name / f"{m['model']}.jsonl")
        new = score(hyps, refs).as_dict()
        if abs(new["cer"] - m["metrics"]["cer"]) > 1e-12 or new["collapsed"] != m["metrics"]["collapsed"]:
            raise SystemExit(f"{json_name} {m['model']}: stored readings do not reproduce the stored CER")
        m["metrics"] = new
        changes[m["model"]] = new["cer"]
    d["blocks_added"] = "07.11.2026, recomputed from the stored readings (wsp.blockreport); CER unchanged"
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    return changes


def session_heldout(run: str, set_name: str, subject: str) -> dict:
    before = read_pairs(BASE / "readings" / set_name / f"{subject}.jsonl")
    after = read_pairs(RUNS / run / "readings" / f"heldout_{set_name}.jsonl")
    if before[0] != after[0]:
        raise SystemExit(f"{run}: held-out readings are not on the same lines as the subject's C0")
    pb, pa = per_line_counts(before[1], before[2]), per_line_counts(after[1], after[2])
    fb, fa, xb, xa, dropped = without_collapsed(before[1], after[1], before[2], pb, pa)
    return {"run": run, "set": set_name, "subject": subject, "lines": len(before[0]),
            "cer_before": score(before[1], before[2], blocks=False).cer, "cer_after": score(after[1], after[2], blocks=False).cer,
            "blocks": paired(pb, pa),
            "without_collapsed": {"dropped": dropped, "cer": [xb, xa], "blocks": paired(fb, fa)}}


# ---------------------------------------------------------------- everything
LINEAGES = [  # (set, before, after, title)
    ("blb-train-300", "kraken-german-xix-v1", "kraken-german-xix-blb-v1", "fine-tune xix-v1 → xix-blb-v1, practice"),
    ("blb-eval", "kraken-german-xix-v1", "kraken-german-xix-blb-v1", "fine-tune xix-v1 → xix-blb-v1, held-out"),
    ("bern-train-300", "kraken-bern-hertwig-v2", "kraken-bern-combined-v1", "hertwig-v2 → combined-v1, practice"),
    ("bern-eval", "kraken-bern-hertwig-v2", "kraken-bern-combined-v1", "hertwig-v2 → combined-v1, held-out"),
    ("bern-eval", "kraken-bern-schlussredaktion-v1", "kraken-bern-combined-v1", "schlussredaktion-v1 (clean sibling) → combined-v1, held-out"),
    ("bern-hertwig-train-300", "kraken-cursive_inzigkofen", "kraken-bern-hertwig-v2", "cursive_inzigkofen → hertwig-v2, practice (22.10 images)"),
    ("bern-hertwig-train-300-export1025", "kraken-bern-hertwig-v2", "kraken-bern-combined-v1", "hertwig-v2 → combined-v1 on the mispaired 25.10 images"),
    ("medieval-val-300", "kraken-collapse-probe-131-cr255", "kraken-collapse-probe-131-cr1", "R1 lossless rewrite Cr255 → Cr1"),
]
SETS = ["blb-train-300", "blb-eval", "bern-train-300", "bern-eval", "bern-hertwig-train-300",
        "bern-hertwig-train-300-export1025", "medieval-train-300", "medieval-val-300", "medieval-val-300-archive"]
SESSIONS = [("20261106-german-xix-blb-v1-b10-1", "blb-eval", "kraken-german-xix-blb-v1"),
            ("20261106-bern-combined-v1-b10-1", "bern-eval", "kraken-bern-combined-v1")]
TOP = ["Space", "Latin lowercase a-z", "Latin uppercase A-Z", "Combining Diacritical Marks", "ASCII punctuation and symbols",
       "Latin-1 Supplement", "General Punctuation", "Digits 0-9"]


def run_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data: dict = {"sets": {}, "lineages": [], "sessions": [], "session_heldout": [], "alpha": None}
    held = {"blb-eval": heldout_rescore("blb-eval", "heldout-blb-eval.json"),
            "bern-eval": heldout_rescore("bern-eval", "heldout-bern-eval.json")}
    data["heldout_rescored"] = held
    sets = {s: model_tables(s) for s in SETS}
    md_sets = []
    for s, models in sets.items():
        data["sets"][s] = {m: table(v["per_line"]) for m, v in models.items()}
        md_sets += render_set(s, models)
    md_lin = []
    for s, a, b, title in LINEAGES:
        if title is None:
            continue
        ma, mb = sets[s][a], sets[s][b]
        assert ma["keys"] == mb["keys"]
        bb = paired(ma["per_line"], mb["per_line"])
        ca, cb = score(ma["hyps"], ma["refs"], blocks=False).cer, score(mb["hyps"], mb["refs"], blocks=False).cer
        entry = {"set": s, "before": a, "after": b, "title": title, "cer": [ca, cb], "blocks": bb}
        md_lin += render_paired(f"{title} (`{s}`)", bb, ca, cb)
        if s in ("blb-eval", "bern-eval"):
            fb, fa, xb, xa, dropped = without_collapsed(ma["hyps"], mb["hyps"], ma["refs"], ma["per_line"], mb["per_line"])
            bx = paired(fb, fa)
            entry["without_collapsed"] = {"dropped": dropped, "cer": [xb, xa], "blocks": bx}
            md_lin += render_paired(f"{title} (`{s}`), collapsed lines excluded", bx, xb, xa,
                                    f"{dropped} of {len(ma['keys'])} lines dropped because one of the two readings collapses on them.")
        data["lineages"].append(entry)
    md_sess = []
    for run, hset, subj in SESSIONS:
        sp = session_passes(run)
        data["sessions"].append(sp)
        md_sess += render_session(sp, TOP)
        sh = session_heldout(run, hset, subj)
        data["session_heldout"].append(sh)
        md_sess += render_paired(f"Session `{run}`: held-out `{hset}`, frozen artefact against the unedited subject",
                                 sh["blocks"], sh["cer_before"], sh["cer_after"],
                                 "From the stored held-out readings of the original run; the replay's final digest equals it.")
        w = sh["without_collapsed"]
        md_sess += render_paired(f"Session `{run}`: held-out `{hset}`, collapsed lines excluded", w["blocks"], w["cer"][0], w["cer"][1],
                                 f"{w['dropped']} of {sh['lines']} lines dropped because one of the two readings collapses on them.")
    a = alpha_matrix()
    data["alpha"] = a
    md_alpha = render_alpha(a, TOP)
    (OUT / "block_stats.json").write_text(json.dumps(data, indent=1, ensure_ascii=False, default=float))
    body = (["## Sessions, pass by pass, and their held-out scores", ""] + md_sess +
            ["## C3 α-sweep: network block × Unicode block", ""] + md_alpha +
            ["## Lineages: paired block deltas between models", ""] + md_lin +
            ["## Appendix: block tables per set and model", ""] + md_sets)
    (OUT / "tables.md").write_text("\n".join(body) + "\n", encoding="utf-8")   # docs/BLOCK_STATS.md = findings + this
    print(OUT / "block_stats.json")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.blockreport")
    ap.add_argument("cmd", choices=["all"])
    ap.parse_args(argv)
    run_all()


if __name__ == "__main__":
    main()
