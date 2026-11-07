"""Read-only survey of the kraken model zoo: architecture, codec, output layer, lineage, the three observations.

    python -m wsp.zoo survey --zoo models/zoo --inhouse <models/kraken> --gateway docs/zoo/gateway_zenodo.json \
        --fetch docs/zoo/fetch.json --out docs/zoo/survey.json

No forward pass, no data. Weights are read through ``KrakenSubject``; output rows are paired by character
(``wsp.align.codec_row_map``), never by index. Everything written is a number computed from the files; the
documented bases are those of the gateway YAML and of the Zenodo README/metadata text, nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from wsp import align
from wsp.kraken_subject import KrakenSubject

FREQUENT = list("eniartsoud")     # fixed proxy for "the ten most frequent letters" (German/Latin/French); no corpus is read
RARE_NOTE = "rare = characters outside basic Latin letters, digits, space and ASCII punctuation (a proxy; no text is read)"


# -- classification --------------------------------------------------------------------------------

def spec_key(spec: str) -> str:
    s = re.sub(r"\{[^}]*\}", "", spec)
    s = re.sub(r"Do[0-9.,]*", "Do", s)
    return re.sub(r"O1c\d+", "O1c<N>", s)


def exact_spec(spec: str) -> str:
    return re.sub(r"\{[^}]*\}", "", spec)


def char_class(ch: str) -> str:
    c0 = ch[0]
    cat = unicodedata.category(c0)
    if cat.startswith("M"):
        return "combining"
    if cat == "Nd":
        return "digit"
    if cat.startswith("L") and "LATIN" in unicodedata.name(c0, ""):
        return "latin"
    if cat.startswith("P") or cat.startswith("S"):
        return "punct"
    return "other"


def is_basic(ch: str) -> bool:
    return len(ch) == 1 and (ch.isascii() and (ch.isalnum() or ch == " " or ch in "!\"'(),-./:;?"))


# -- one model ------------------------------------------------------------------------------------------

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def output_stats(s: KrakenSubject) -> dict:
    t = s.tensors()
    wn = next(n for n in t if n.startswith("O_") and n.endswith("lin.weight"))
    W = t[wn].astype(np.float64); b = t[wn.replace("weight", "bias")].astype(np.float64)
    norms = np.linalg.norm(W, axis=1)
    c2l = s.codec_map(); l2c = {v: k for k, v in c2l.items()}
    others_b = b[1:]
    out = {"tensor": wn, "classes": int(W.shape[0]),
           "blank_norm": float(norms[0]), "blank_bias": float(b[0]),
           "other_bias_mean": float(others_b.mean()), "other_bias_std": float(others_b.std()),
           "blank_bias_rel": float(b[0] - others_b.mean()),
           "blank_bias_z": float((b[0] - others_b.mean()) / (others_b.std() or 1)),
           "blank_rank_by_bias": int((b > b[0]).sum() + 1),          # 1 = the blank has the highest bias
           "row_norm_mean": float(norms[1:].mean()), "row_norm_std": float(norms[1:].std()),
           "blank_norm_rank_from_weakest": int((norms < norms[0]).sum() + 1)}
    idx = [c2l[c] for c in FREQUENT if c in c2l]
    out["frequent_present"] = len(idx)
    if len(idx) == len(FREQUENT):
        rest = np.array([i for i in range(1, W.shape[0]) if i not in set(idx)])
        order = np.argsort(norms[1:]) + 1
        low10 = set(order[:len(idx)].tolist())
        out["frequent"] = {"norm_mean": float(norms[idx].mean()), "rest_norm_mean": float(norms[rest].mean()),
                           "ratio": float(norms[idx].mean() / norms[rest].mean()),
                           "bias_mean": float(b[idx].mean()), "rest_bias_mean": float(b[rest].mean()),
                           "in_weakest10": int(len(low10 & set(idx))),
                           "weakest10_chars": [l2c.get(int(i), "?") for i in order[:10]]}
    return out


def read_model(mid: str, path: Path, origin: str) -> tuple[dict, KrakenSubject]:
    s = KrakenSubject(path)
    d = s.describe()
    cm = s.codec_map()
    classes = {"latin": 0, "digit": 0, "punct": 0, "combining": 0, "other": 0}
    for ch in s.model.codec.c2l:
        classes[char_class(ch)] += 1
    n_multi = sum(1 for ch in s.model.codec.c2l if len(ch) > 1)
    n_codec = len(s.model.codec.c2l)
    rare = [ch for ch in cm if not is_basic(ch)]
    um = d["user_metadata"]
    info = {"id": mid, "origin": origin, "file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
            "format": "CoreML .mlmodel (kraken 7.0.2 load_model)",
            "spec": exact_spec(d["spec"]), "spec_key": spec_key(d["spec"]),
            "tensors": len(d["layers"]), "parameters": d["parameters"],
            "codec_size": n_codec, "multi_char_codes": n_multi, "output_classes": d["output_classes"],
            "codec_classes": classes, "codec_classes_share": {k: round(v / n_codec, 4) for k, v in classes.items()},
            "rare_characters": len(rare),
            "metadata_keys": sorted(k for k in um if k != "codec"),
            "hyper_params_keys": sorted((um.get("hyper_params") or {}).keys()),
            "completed_epochs": (um.get("hyper_params") or {}).get("completed_epochs", um.get("completed_epochs")),
            "output": output_stats(s)}
    base_keys = [k for k in um if re.search(r"base|pretrain|parent|train.*data|dataset|source", k, re.I)]
    info["metadata_keys_naming_base_or_data"] = base_keys
    return info, s


def doc_text(d: Path) -> str:
    txt = []
    for f in sorted(d.glob("*")):
        if f.suffix == ".md":
            txt.append(f.read_text(errors="replace"))
        elif f.suffix == ".json":
            try:
                j = json.loads(f.read_text())
                if isinstance(j, dict):
                    txt.append(" ".join(str(j.get(k, "")) for k in ("summary", "description")))
            except Exception:
                pass
    return " ".join(txt)


BASE_PAT = re.compile(r"[^.\n]*(fine-?tun|retrain|base model|as a base|pre-?train|from scratch|transfer learning|trained from)[^.\n]*", re.I)


def doc_snippets(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0).strip()[:300] for m in BASE_PAT.finditer(text)))


# -- pairs --------------------------------------------------------------------------------------------------

def non_output(t: dict, out_w: str) -> list[str]:
    ob = out_w.replace("weight", "bias")
    return [n for n in t if n not in (out_w, ob)]


def block_kind(name: str) -> str:
    p = name.split(".")[0]
    return {"C": "conv", "L": "lstm", "O": "out", "G": "norm"}.get(p[0], "other")


def pair_reading(a: KrakenSubject, b: KrakenSubject) -> dict:
    """Parent a → child b. Per-tensor rel L2 and cosine; output layer rows aligned by character."""
    ta, tb = a.tensors(), b.tensors()
    wn = next(n for n in ta if n.startswith("O_") and n.endswith("lin.weight"))
    wn_b = next(n for n in tb if n.startswith("O_") and n.endswith("lin.weight"))
    bn, bn_b = wn.replace("weight", "bias"), wn_b.replace("weight", "bias")
    per = {}
    for n in non_output(ta, wn):
        x, y = ta[n].astype(np.float64).ravel(), tb[n].astype(np.float64).ravel()
        nx = np.linalg.norm(x)
        if nx == 0 and np.linalg.norm(y) == 0:
            per[n] = {"rel": None, "cos": None, "zero": True, "kind": block_kind(n)}
            continue
        per[n] = {"rel": float(np.linalg.norm(y - x) / (nx or 1)),
                  "cos": float(x @ y / ((nx * np.linalg.norm(y)) or 1)), "zero": False, "kind": block_kind(n)}
    # output layer, aligned
    ca, cb = a.codec_map(), b.codec_map()
    rm = align.codec_row_map(ca, cb); rm[0] = 0
    ia = sorted(rm); ib = [rm[i] for i in ia]
    Wa, Wb = ta[wn].astype(np.float64), tb[wn_b].astype(np.float64)
    ba, bb = ta[bn].astype(np.float64), tb[bn_b].astype(np.float64)
    l2a = {v: k for k, v in ca.items()}
    rows = []
    for i, j in zip(ia, ib):
        na, nb = np.linalg.norm(Wa[i]), np.linalg.norm(Wb[j])
        rows.append({"char": "<blank>" if i == 0 else l2a[i], "rel": float(np.linalg.norm(Wb[j] - Wa[i]) / (na or 1)),
                     "norm_p": float(na), "norm_c": float(nb), "bias_delta": float(bb[j] - ba[i])})
    xa = Wa[ia].ravel(); xb = Wb[ib].ravel()
    out_rel = float(np.linalg.norm(xb - xa) / np.linalg.norm(xa))
    out_cos = float(xa @ xb / (np.linalg.norm(xa) * np.linalg.norm(xb)))
    mean_other_a, mean_other_b = float(ba[1:].mean()), float(bb[1:].mean())
    blank = {"bias_p": float(ba[0]), "bias_c": float(bb[0]), "delta_raw": float(bb[0] - ba[0]),
             "rel_p": float(ba[0] - mean_other_a), "rel_c": float(bb[0] - mean_other_b),
             "delta_rel": float((bb[0] - mean_other_b) - (ba[0] - mean_other_a)),
             "mean_other_delta": float(mean_other_b - mean_other_a), "row_rel": rows[0]["rel"] if rows[0]["char"] == "<blank>" else None}
    # observation 3: block means
    def kmean(kind):
        v = [e["rel"] for e in per.values() if e["kind"] == kind and not e["zero"]]
        return float(np.mean(v)) if v else None
    kinds = {k: kmean(k) for k in ("conv", "lstm", "norm")}

    def stack(kind):      # pooled relative L2 over the weight tensors of a stack (biases excluded: tiny norms, huge ratios)
        nm = [n for n in per if block_kind(n) == kind and n.split(".")[-1].startswith("weight")]
        x, y = flat(ta, nm), flat(tb, nm)
        return float(np.linalg.norm(y - x) / np.linalg.norm(x))
    def readings_mean(kind):   # the definition of docs/READINGS.md: mean over blocks of the per-block mean of per-tensor rel L2, zero tensors counted as 0
        blk: dict[str, list[float]] = {}
        for n, e in per.items():
            if e["kind"] == kind:
                blk.setdefault(n.split(".")[0], []).append(0.0 if e["zero"] else e["rel"])
        return float(np.mean([np.mean(v) for v in blk.values()])) if blk else None
    stacks = {"conv_w": stack("conv"), "lstm_w": stack("lstm"),
              "conv_readings_def": readings_mean("conv"), "lstm_readings_def": readings_mean("lstm")}
    conv_rels = {n.split(".")[0]: [] for n in per if block_kind(n) == "conv"}
    for n, e in per.items():
        if e["kind"] == "conv" and not e["zero"]:
            conv_rels[n.split(".")[0]].append(e["rel"])
    conv_blocks = {}
    for k in conv_rels:
        nm = [n for n in per if n.split(".")[0] == k and n.split(".")[-1] == "weight"]
        x, y = flat(ta, nm), flat(tb, nm)
        conv_blocks[k] = float(np.linalg.norm(y - x) / np.linalg.norm(x))
    # observation 1: special vs the rest among shared character rows
    ch = [r for r in rows if r["char"] != "<blank>"]
    special = [r for r in ch if not is_basic(r["char"])]
    freq = [r for r in ch if r["char"] in FREQUENT]
    top = sorted(ch, key=lambda r: -r["rel"])[:max(1, len(ch) // 10)]
    top_special = [r for r in top if not is_basic(r["char"])]
    obs1 = {"shared_rows": len(ch), "special_rows": len(special), "frequent_rows": len(freq),
            "special_norm_up": sum(r["norm_c"] > r["norm_p"] for r in special),
            "special_bias_down": sum(r["bias_delta"] < 0 for r in special),
            "special_median_rel": float(np.median([r["rel"] for r in special])) if special else None,
            "special_median_norm_ratio": float(np.median([r["norm_c"] / r["norm_p"] for r in special])) if special else None,
            "special_median_bias_delta": float(np.median([r["bias_delta"] for r in special])) if special else None,
            "frequent_median_rel": float(np.median([r["rel"] for r in freq])) if freq else None,
            "frequent_median_norm_ratio": float(np.median([r["norm_c"] / r["norm_p"] for r in freq])) if freq else None,
            "frequent_median_bias_delta": float(np.median([r["bias_delta"] for r in freq])) if freq else None,
            "rest_median_rel": float(np.median([r["rel"] for r in ch if r["char"] not in FREQUENT and is_basic(r["char"])] or [np.nan])),
            "top_decile_rows": len(top), "top_decile_special": len(top_special),
            "special_share_of_rows": len(special) / len(ch) if ch else None,
            "top_rows": [(r["char"], round(r["rel"], 3)) for r in top[:8]]}
    return {"per_tensor": {n: {k: v for k, v in e.items() if k != "kind"} for n, e in per.items()},
            "blocks": {**stacks, "out_row_mean": float(np.mean([r["rel"] for r in rows[1:]])), "conv": kinds["conv"], "lstm": kinds["lstm"], "norm": kinds["norm"], "out_rows": out_rel,
                       "conv_by_block": conv_blocks, "out_cos": out_cos},
            "blank": blank, "obs1": obs1, "codec": {"parent": len(ca), "child": len(cb), "shared": len(ch), "added": len(set(cb) - set(ca))}}


def flat(t: dict, names: list[str]) -> np.ndarray:
    return np.concatenate([t[n].astype(np.float64).ravel() for n in names])


def sym_dist(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / (0.5 * (na + nb))), float(a @ b / (na * nb))


# -- main -------------------------------------------------------------------------------------------------------

# documented relations: (child, parent, source). Sources are the gateway YAML field `base_model` and README/metadata text;
# `id` strings are zoo ids (kraken-<name>) or in-house ids; a parent outside the surveyed set is a string not in the zoo.
DOCUMENTED = [
    ("kraken-bastarda_inzigkofen", "kraken-catmus_medieval", "gateway base_model 10.5281/zenodo.15030337; README: 'obtained by fine-tuning the CATMuS Medieval model'"),
    ("kraken-cursive_inzigkofen", "kraken-catmus_medieval", "gateway base_model; README: 'obtained by fine-tuning the CATMuS Medieval model'"),
    ("kraken-bifrost_old_norse", "kraken-catmus_medieval", "gateway base_model 10.5281/zenodo.15030337; the Zenodo text read has no base sentence"),
    ("kraken-textualis_inzigkofen", "kraken-tridis_v2", "gateway base_model 10.5281/zenodo.13862096; README: 'obtained by fine-tuning the TriDiS v2 model'"),
    ("kraken-catmus_gothic_print", "kraken-catmus_medieval", "gateway base_model; README: 'fine-tuned on CATMuS Medieval model'"),
    ("kraken-french_vietnamese", "kraken-catmus_print_large", "gateway base_model 10.5281/zenodo.10592716; README: 'transfer learning from the CATMuS-Print base model'"),
    ("kraken-latin_incunabula", "kraken-catmus_print_large", "gateway base_model 10.5281/zenodo.10592716; the README sentences read name no base"),
    ("bern-hertwig-v1", "kraken-cursive_inzigkofen", "docs/MODELS.md (trainer record)"),
    ("bern-hertwig-v2", "kraken-cursive_inzigkofen", "docs/MODELS.md (trainer record)"),
    ("bern-hertwig-v2", "bern-hertwig-v1", "docs/READINGS.md (read as a continuation of v1; docs/MODELS.md lists the same base cursive_inzigkofen for both)"),
    ("bern-schlussredaktion-v1", "bern-hertwig-v2", "docs/MODELS.md (trainer record)"),
    ("bern-combined-v1", "bern-hertwig-v2", "docs/MODELS.md (trainer record)"),
    ("german-xix-blb-v1", "german-xix-v1", "docs/MODELS.md (trainer record)"),
]
# documented in text, parent not among the surveyed files (cannot be tested)
# pairs suggested by the weights alone (distance < 1.0 and the child's codec extends the parent's with the same row indices);
# NOT documented, direction inferred from the codec: tested separately and labelled as such
INFERRED = [("kraken-prima", "kraken-mccatmus"), ("german-xix-v1", "kraken-bohemian_19th")]
DOCUMENTED_UNAVAILABLE = [
    ("kraken-kuzushiji", "Chat model by Colin Brisson (Zenodo), named in the README: 'finetuned on top of the Chat model'"),
    ("kraken-mendelssohn_letters", "'HTR model for German manuscripts trained from several datasets' (eScriptorium base); not identified with a DOI in the text read"),
    ("kraken-old_norse_am305", "'fine-tuned from a base model trained on AM 302 fol (Grágás)'; that model is not in the gateway list"),
    ("kraken-peraire_french", "Manu McFrench (Zenodo 6657809), named in the metadata; not in the gateway list"),
    ("kraken-openiti_persian_print", "gateway base_model 10.5281/zenodo.7050270 (OpenITI Arabic); file not fetched (HTTP 429)"),
    ("kraken-openiti_ottoman_print", "gateway base_model 10.5281/zenodo.7050270 (OpenITI Arabic); file not fetched (HTTP 429)"),
]


def survey(zoo: Path, inhouse: Path, gateway: Path, fetch: Path) -> dict:
    gw = {m["id"]: m for m in json.loads(gateway.read_text())}
    fe = json.loads(fetch.read_text())
    models, subj, errors = {}, {}, []
    for d in sorted(zoo.iterdir()):
        f = sorted(d.glob("*.mlmodel"))
        if not f:
            continue
        try:
            info, s = read_model(d.name, f[0], "zoo")
        except Exception as e:                                       # noqa: BLE001
            errors.append({"id": d.name, "error": f"{type(e).__name__}: {e}"}); continue
        g = gw.get(d.name, {})
        info.update(doi=g.get("zenodo_id"), gateway_base=g.get("base_model"), gateway_description=g.get("description"),
                    languages=g.get("languages"), scripts=g.get("scripts"), centuries=g.get("centuries"),
                    doc_snippets=doc_snippets(doc_text(d)), zenodo_files=fe.get(d.name, {}).get("files"))
        models[d.name] = info; subj[d.name] = s
    for f in sorted(inhouse.glob("*.mlmodel")):
        mid = f.stem.replace("kraken-", "", 1)
        if mid in ("collapse-probe-131-cr255",):
            continue
        if mid == "cursive_inzigkofen":      # the parent of the Bern family is the zoo file; checked by sha256 below
            same = sha256(f) == models["kraken-cursive_inzigkofen"]["sha256"]
            models["kraken-cursive_inzigkofen"]["inhouse_copy_identical"] = same
            continue
        try:
            info, s = read_model(mid, f, "in-house")
        except Exception as e:                                       # noqa: BLE001
            errors.append({"id": mid, "error": f"{type(e).__name__}: {e}"}); continue
        info.update(doi=None, gateway_base=None)
        models[mid] = info; subj[mid] = s
    # groups
    groups: dict[str, list[str]] = {}
    for mid, m in models.items():
        groups.setdefault(m["spec_key"], []).append(mid)
    # near-duplicates and distances inside groups (non-output tensors)
    dist = {}
    for key, ids in groups.items():
        if len(ids) < 2:
            continue
        t0 = subj[ids[0]].tensors()
        wn = next(n for n in t0 if n.startswith("O_") and n.endswith("lin.weight"))
        names = non_output(t0, wn)
        vecs = {i: flat(subj[i].tensors(), names) for i in ids}
        # conv-only and lstm-only vectors for separate distances
        cn = [n for n in names if block_kind(n) == "conv"]; ln = [n for n in names if block_kind(n) == "lstm"]
        cv = {i: flat(subj[i].tensors(), cn) for i in ids}; lv = {i: flat(subj[i].tensors(), ln) for i in ids}
        pairs = {}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                r, c = sym_dist(vecs[a], vecs[b])
                rc, _ = sym_dist(cv[a], cv[b]); rl, _ = sym_dist(lv[a], lv[b])
                pairs[f"{a}|{b}"] = {"rel": r, "cos": c, "rel_conv": rc, "rel_lstm": rl}
        dist[key] = {"ids": ids, "pairs": pairs}
    # documented pairs
    dp = []
    for child, parent, src in DOCUMENTED:
        row = {"child": child, "parent": parent, "source": src}
        pid = parent if parent in models else None
        if child not in models or pid is None:
            row["status"] = "not surveyed"; dp.append(row); continue
        row["same_spec"] = models[child]["spec_key"] == models[pid]["spec_key"]
        if not row["same_spec"]:
            row["status"] = "architecture differs: not a weight-level fine-tune of this file"
            row["spec_child"], row["spec_parent"] = models[child]["spec_key"], models[pid]["spec_key"]
            dp.append(row); continue
        row["status"] = "tested"; row["reading"] = pair_reading(subj[pid], subj[child])
        dp.append(row)
    ip = []
    for child, parent in INFERRED:
        ca, cb = subj[parent].codec_map(), subj[child].codec_map()
        row = {"child": child, "parent": parent, "codec_parent": len(ca), "codec_child": len(cb),
               "parent_codec_subset_of_child": all(c in cb for c in ca),
               "same_row_index": sum(cb.get(c) == i for c, i in ca.items()),
               "source": "weights and codec only; no documentation read names this relation",
               "reading": pair_reading(subj[parent], subj[child])}
        ip.append(row)
    # nearest neighbours (non-output rel L2) per model, and whether the documented parent is it
    nn = {}
    for key, e in dist.items():
        for a in e["ids"]:
            ds = sorted(((v["rel"], (p.split("|")[1] if p.split("|")[0] == a else p.split("|")[0]))
                         for p, v in e["pairs"].items() if a in p.split("|")), key=lambda t: t[0])
            nn[a] = [{"id": b, "rel": r} for r, b in ds[:3]]
            allr = [r for r, _ in ds]
            nn[a].append({"median_rel_all": float(np.median(allr))})
    # re-uploads: identical sha or tiny distance
    dups = []
    shas: dict[str, list[str]] = {}
    for mid, m in models.items():
        shas.setdefault(m["sha256"], []).append(mid)
    dups += [{"same_file": v} for v in shas.values() if len(v) > 1]
    for e in dist.values():
        for p, v in e["pairs"].items():
            if v["rel"] < 0.02:
                dups.append({"near_identical": p.split("|"), "rel": v["rel"], "cos": v["cos"]})
    # compact model rows for the json (drop nothing)
    return {"method": {"frequent_letters": "".join(FREQUENT), "rare": RARE_NOTE, "kraken": "7.0.2",
                       "documented_sources": "gateway YAML base_model, Zenodo README/metadata text, docs/MODELS.md for in-house"},
            "models": models, "groups": {k: v for k, v in groups.items()}, "distances": dist, "documented_pairs": dp, "inferred_pairs": ip,
            "documented_unavailable": [{"child": c, "note": n} for c, n in DOCUMENTED_UNAVAILABLE],
            "nearest": nn, "duplicates": dups, "load_errors": errors,
            "fetch_failures": [{"id": k, "doi": v["doi"], "error": v.get("error")} for k, v in fe.items() if not v.get("files")],
            "gateway_zenodo_kraken": sum(1 for m in gw.values() if m["engine"] == "kraken")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["survey"])
    ap.add_argument("--zoo", default="models/zoo"); ap.add_argument("--inhouse", default="models/kraken")
    ap.add_argument("--gateway", default="docs/zoo/gateway_zenodo.json"); ap.add_argument("--fetch", default="docs/zoo/fetch.json")
    ap.add_argument("--out", default="docs/zoo/survey.json")
    a = ap.parse_args(argv)
    res = survey(Path(a.zoo), Path(a.inhouse), Path(a.gateway), Path(a.fetch))
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}: {len(res['models'])} models, {len(res['groups'])} spec groups, {len(res['load_errors'])} load errors")


if __name__ == "__main__":
    main()
