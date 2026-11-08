"""One static HTML page per model: pick a practice line, see at which layer each character is found or
lost and what each block's share of the fine-tune does to it (epic thodel/weightspace#3, sub-task 7).

    python -m wsp.vizpage --probes docs/viz/kraken-german-xix-blb-v1__blb-train-300.json \\
        --overlay docs/viz/alpha-overlay-kraken-german-xix-blb-v1__blb-train-300.json \\
        --practice data/practice/blb-train-300.jsonl --out docs/viz/kraken-german-xix-blb-v1.html [--images]

Engine-free: it reads only the JSON that ``wsp.layerview`` and ``wsp.alpha_overlay`` committed, and
embeds what the page needs, so the file opens from disk with no server. ``--images`` (needs kraken,
not the model) writes each line crop as the network sees it — 120 px high, padded — to
``docs/viz/img/`` (git-ignored: manuscript images), where the page picks them up if present; the
characters are placed under the image by their forced-aligned frames (8 px each).
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from wsp.layerview import LAYERS, lost_at
from wsp.metrics import levenshtein

SVGS = [("probe-{model}__{set}", "Layer probes: which letter (identity) and whether to emit (emission), per layer"),
        ("alpha-overlay-{model}__{set}", "Each block's share of the fine-tune, by kind of error (α overlay)"),
        ("bias-{model}__{set}", "Output-row bias trade-off (space row and blank)")]


def page_data(probes: dict, overlay: dict | None) -> dict:
    """The compact per-line table the page draws from; keys and reference strings must agree."""
    ov = {ln["key"]: ln for ln in (overlay or {}).get("lines", [])}
    tags = [f"{b}@{a:g}" for b in (overlay or {}).get("blocks", []) for a in overlay["alphas"]] if overlay else []
    lines = []
    for ln in probes["lines"]:
        o = ov.get(ln["key"])
        same = o is not None and o["ref"] == ln["ref"]
        st = "".join({"ok": "o", "sub": "s", "del": "d"}[s] for s in ln["status"])
        lines.append({
            "k": ln["key"], "r": ln["ref"], "h": ln["hyp"], "s": st,
            "c": round(levenshtein(ln["ref"], ln["hyp"]) / max(len(ln["ref"]), 1), 4),
            "x": [round((a + b + 1) / 2, 1) for a, b in ln["runs"]],
            "id": [ln["identity"][l] for l in LAYERS], "em": [ln["emission"][l] for l in LAYERS],
            "lost": ["" if s != "del" else lost_at(ln, i) for i, s in enumerate(ln["status"])],
            "ov": [o["states"][t]["status"] for t in tags] if same else None,
            "ovh": {t: o["states"][t]["hyp"] for t in tags} if same else None,
        })
    return {"model": probes.get("model"), "set": probes.get("set"), "layers": LAYERS, "tags": tags,
            "parent": (overlay or {}).get("parent"), "lines": lines,
            "summary": {"deleted_lost_at": probes["deleted_lost_at"], "folds": probes.get("meta", {}).get("folds")}}


def write_images(items: list[dict], out_dir: Path) -> int:
    """Line crops as the network sees them (ImageInputTransforms of kraken 7.0.2, height 120, 16 px padding)."""
    from kraken.lib.dataset import ImageInputTransforms
    from PIL import Image
    import numpy as np
    tf = ImageInputTransforms(1, 120, 0, 1, (16, 0), valid_norm=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for it in items:
        try:
            a = tf(Image.open(it["image"]).convert("L")).numpy()[0]
        except ValueError:
            continue
        Image.fromarray((np.clip(a, 0, 1) * 255).astype("uint8")).save(out_dir / f"{safe(it['key'])}.png")
        n += 1
    return n


def safe(key: str) -> str:
    return key.replace("/", "_")


def render_page(data: dict, svg_names: list[tuple[str, str]], img_dir: str = "img") -> str:
    figs = "".join(
        f'<figure><picture><source srcset="{n}-dark.svg" media="(prefers-color-scheme: dark)">'
        f'<img src="{n}.svg" alt="{html.escape(c)}" loading="lazy"></picture><figcaption>{html.escape(c)}</figcaption></figure>'
        for n, c in svg_names)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(f"{data['model']} — where characters are found and lost")
    return (TEMPLATE.replace("__TITLE__", title).replace("__MODEL__", html.escape(data["model"] or ""))
            .replace("__SET__", html.escape(data["set"] or "")).replace("__PARENT__", html.escape(data.get("parent") or "—"))
            .replace("__FIGS__", figs).replace("__IMG__", img_dir).replace("__DATA__", blob))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fff;--fg:#1b1b1b;--muted:#6e6e6e;--grid:#e2e2e2;--panel:#f6f6f6;--ok:#1b1b1b;--sub:#CC79A7;--del:#D55E00;--c:#0072B2;--l:#E69F00;--o:#009E73}
@media (prefers-color-scheme: dark){:root{--bg:#16181d;--fg:#e6e6e6;--muted:#a3a3a3;--grid:#33363d;--panel:#1f2228;--ok:#e6e6e6;--del:#E8743B;--c:#56B4E9}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:16px 20px 4px}h1{font-size:18px;margin:0 0 4px}h2{font-size:15px;margin:18px 0 8px}.muted{color:var(--muted)}
main{display:grid;grid-template-columns:minmax(220px,300px) 1fr;gap:16px;padding:8px 20px 24px}
@media (max-width:800px){main{grid-template-columns:1fr}}
#list{max-height:78vh;overflow:auto;border:1px solid var(--grid);border-radius:6px}
#list div{padding:4px 8px;cursor:pointer;border-bottom:1px solid var(--grid);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#list div.sel{background:var(--panel);font-weight:600}#list .n{color:var(--del);float:right;margin-left:6px}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;font-size:12px}.controls input[type=search]{flex:1;min-width:120px;padding:4px 6px;background:var(--bg);color:var(--fg);border:1px solid var(--grid);border-radius:4px}
#view{overflow-x:auto;border:1px solid var(--grid);border-radius:6px;padding:10px}
.row{display:flex;align-items:center;min-height:16px}.lab{flex:0 0 92px;font-size:11px;color:var(--muted);text-align:right;padding-right:8px}
.track{position:relative;flex:0 0 auto;height:16px}.track .c{position:absolute;transform:translateX(-50%);font-size:13px;line-height:16px;cursor:default}
.sq{position:absolute;width:6px;height:6px;top:5px;transform:translateX(-50%);border-radius:1px}
.c.o{color:var(--ok)}.c.s{color:var(--sub);font-weight:700}.c.d{color:var(--del);font-weight:700}
.gap{height:8px}.grp{font-size:11px;color:var(--muted);margin:8px 0 2px 100px}
#img{display:block;height:90px;image-rendering:auto}.imgrow .track{height:90px}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--grid);border-radius:4px;padding:6px 8px;font-size:12px;display:none;max-width:320px;z-index:9}
figure{margin:0 0 14px}figure img{max-width:100%;height:auto}figcaption{font-size:12px;color:var(--muted)}
details{padding:0 20px}.legend span{display:inline-block;margin-right:12px;font-size:12px}
.hyp{font-family:ui-monospace,monospace;font-size:12px;white-space:pre}
</style></head><body>
<header><h1>__MODEL__ <span class="muted">on __SET__ (practice lines)</span></h1>
<div class="muted">Pick a line. Reference characters sit on the frames the forced CTC alignment gives them (8 px of the line per frame).
<b>Probes</b>: per layer, ■ the layer's linear probes have the character (identity and emission), □ identity only, · neither.
<b>α overlay</b>: one block of the model at parent (__PARENT__) + α·(child − parent), the rest unchanged; the character's fate there.
Colours: black/white emitted, purple replaced, vermillion deleted. Hover a character for its column.</div></header>
<main><section><div class="controls"><input type="search" id="q" placeholder="filter: key or text">
<label><input type="checkbox" id="fdel"> with deletions</label><label><input type="checkbox" id="fsp"> deleted spaces</label>
<label title="lines read worse are mostly crop or reference defects (rotated, degenerate boxes, OCR junk)"><input type="checkbox" id="fok" checked> read at CER ≤ 20 %</label></div>
<div id="list"></div><div class="muted" id="count" style="font-size:12px;margin-top:4px"></div></section>
<section><div id="view"></div></section></main>
<details><summary><b>Summary figures</b></summary>__FIGS__</details>
<div id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),PX=6,IMG='__IMG__/';   // 8 px per frame at 120 px; image shown 90 px high
const L=D.layers,lab=l=>l==='input'?'pixels':l;
const $=s=>document.querySelector(s),list=$('#list'),view=$('#view'),tip=$('#tip');
const nd=ln=>[...ln.s].filter(c=>c==='d').length, nsp=ln=>[...ln.s].filter((c,i)=>c==='d'&&ln.r[i]===' ').length;
const glyph=c=>c===' '?'␣':(/\p{M}/u.test(c)?'◌'+c:c);
let cur=null;
function fill(){const q=$('#q').value.toLowerCase(),fd=$('#fdel').checked,fs=$('#fsp').checked,fo=$('#fok').checked;list.innerHTML='';let n=0;
 const rows=D.lines.filter(ln=>(!q||ln.k.toLowerCase().includes(q)||ln.r.toLowerCase().includes(q))&&(!fd||nd(ln))&&(!fs||nsp(ln))&&(!fo||ln.c<=0.2));
 rows.sort((a,b)=>nd(b)-nd(a)||a.k.localeCompare(b.k));
 for(const ln of rows){const d=document.createElement('div');d.textContent=ln.k;d.title=ln.k+"\n"+ln.r;
  {const s=document.createElement('span');s.className='n';s.textContent=(nd(ln)?nd(ln)+' del · ':'')+(100*ln.c).toFixed(0)+' %';d.prepend(s)}
  if(cur===ln)d.className='sel';d.onclick=()=>{cur=ln;fill();show(ln)};list.appendChild(d);n++}
 $('#count').textContent=n+' of '+D.lines.length+' lines'}
function track(w,h){const t=document.createElement('div');t.className='track';t.style.width=w+'px';if(h)t.style.height=h+'px';return t}
function row(label,t,cls){const r=document.createElement('div');r.className='row'+(cls?' '+cls:'');const l=document.createElement('div');l.className='lab';l.textContent=label;r.append(l,t);return r}
function col(code){return code==='d'?'var(--del)':code==='s'?'var(--sub)':'var(--ok)'}
function show(ln){view.innerHTML='';const W=Math.ceil((Math.max(...ln.x,1)+4)*PX)+20;
 const img=new Image();img.id='img';img.src=IMG+ln.k.replaceAll('/','_')+'.png';img.onerror=()=>img.parentNode&&img.parentNode.parentNode.remove();
 const ti=track(W,90);ti.appendChild(img);view.appendChild(row('line',ti,'imgrow'));
 const tr=track(W);[...ln.r].forEach((c,i)=>{const e=document.createElement('span');e.className='c '+ln.s[i];e.textContent=glyph(c);e.style.left=ln.x[i]*PX+'px';
  e.onmousemove=ev=>tipShow(ev,ln,i);e.onmouseleave=()=>tip.style.display='none';tr.appendChild(e)});
 view.appendChild(row('reference',tr));
 const h=document.createElement('div');h.className='row';h.innerHTML='<div class="lab">model reads</div>';const hs=document.createElement('span');hs.className='hyp';hs.textContent=ln.h;h.appendChild(hs);view.appendChild(h);
 const g=document.createElement('div');g.className='grp';g.textContent='probes per layer (■ identity + emission, □ identity only)';view.appendChild(g);
 L.forEach((l,j)=>{const t=track(W);[...ln.r].forEach((c,i)=>{const s=document.createElement('span');s.className='sq';s.style.left=ln.x[i]*PX+'px';
   const id=ln.id[j][i]==='1',em=ln.em[j][i]==='1',cc=col(ln.s[i]);
   if(id&&em)s.style.background=cc;else if(id)s.style.border='1px solid '+cc;else{s.style.background='var(--grid)';s.style.width='3px';s.style.height='3px';s.style.top='6px'}
   t.appendChild(s)});view.appendChild(row(lab(l),t))});
 if(ln.ov){const g2=document.createElement('div');g2.className='grp';g2.textContent='α overlay: each block alone at parent + α·(child − parent); the character\'s fate';view.appendChild(g2);
  D.tags.forEach((tag,j)=>{const t=track(W);[...ln.ov[j]].forEach((code,i)=>{if(i>=ln.x.length)return;const s=document.createElement('span');s.className='sq';s.style.left=ln.x[i]*PX+'px';
    if(code==='o'){s.style.background='var(--grid)'}else s.style.background=col(code);t.appendChild(s)});
   const r=row(tag.replace('@',' α=') ,t);r.title=ln.ovh[tag];if(tag.endsWith('@1.5'))r.style.marginBottom='5px';view.appendChild(r)})}}
function tipShow(ev,ln,i){const c=ln.r[i],st={o:'emitted',s:'replaced',d:'deleted'}[ln.s[i]];
 let h='<b>'+glyph(c).replace(/&/g,'&amp;').replace(/</g,'&lt;')+'</b> — '+st+(ln.lost[i]?'; deepest layer that still has it: <b>'+lab(ln.lost[i]).replace('never','none')+'</b>':'')+'<br>';
 h+=L.map((l,j)=>lab(l)+' '+(ln.id[j][i]==='1'?'id':'··')+'/'+(ln.em[j][i]==='1'?'em':'··')).join(' · ');
 if(ln.ov){const ch=D.tags.map((t,j)=>[t,ln.ov[j][i]]).filter(([t,code])=>code!==ln.s[i]);if(ch.length)h+='<br>changes under: '+ch.map(([t,code])=>t.replace('@',' α=')+' → '+({o:'emitted',s:'replaced',d:'deleted'}[code])).join(', ')}
 tip.innerHTML=h;tip.style.display='block';tip.style.left=Math.min(ev.clientX+12,innerWidth-330)+'px';tip.style.top=(ev.clientY+14)+'px'}
['q','fdel','fsp','fok'].forEach(id=>$('#'+id).addEventListener('input',fill));
const good=D.lines.filter(ln=>ln.c<=0.2&&ln.r.length>=15&&nsp(ln)).sort((a,b)=>nd(b)-nd(a));
cur=good[0]||D.lines[0];fill();show(cur);
</script></body></html>
"""


def main(argv=None):
    from wsp.data import load
    ap = argparse.ArgumentParser(prog="wsp.vizpage")
    ap.add_argument("--probes", required=True); ap.add_argument("--overlay")
    ap.add_argument("--practice", help="needed for --images"); ap.add_argument("--out", required=True)
    ap.add_argument("--images", action="store_true", help="write the line crops to <out dir>/img (git-ignored)")
    a = ap.parse_args(argv)
    probes = json.loads(Path(a.probes).read_text())
    overlay = json.loads(Path(a.overlay).read_text()) if a.overlay else None
    data = page_data(probes, overlay)
    out = Path(a.out)
    names = [(n.format(model=data["model"], set=data["set"]), c) for n, c in SVGS]
    names = [(n, c) for n, c in names if (out.parent / f"{n}.svg").exists()]
    out.write_text(render_page(data, names), encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")
    if a.images:
        if "heldout" in Path(a.practice).parts:
            raise SystemExit("practice lines only")
        keys = {ln["k"] for ln in data["lines"]}
        print(write_images([x for x in load(a.practice) if x["key"] in keys], out.parent / "img"), "line images")


if __name__ == "__main__":
    main()
