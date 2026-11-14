"""C4 control data: the 44 Hertwig training pages as the combined-v1 job saw them (wrong images), read only from the job dir.

Copies those pages to ~/atr-cache/wsp/c4u/pages/train/hertwig, compares them with the corrected pages of ~/atr-cache/wsp/c4/pages/train/hertwig
(XML text of every line; image bytes), writes docs/c4u/split.json. Compiling the arrow is done by c4u_all.sh / the command in docs/C4_UNCORRECTED.md.
"""
import hashlib, json, re, shutil
from pathlib import Path
import sys
sys.path.insert(0, ".")
from wsp.data import page_lines

J = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/jobs/20261025T160407Z-kraken-bern-combined-v1/data")
C = Path.home() / "atr-cache/wsp/c4/pages/train/hertwig"
OUT = Path.home() / "atr-cache/wsp/c4u/pages/train/hertwig"
OUT.mkdir(parents=True, exist_ok=True)
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows, differ, same_text = [], 0, 0
for l in (J / "pages_train.lst").read_text().split():
    x = Path(l).read_text(encoding="utf-8")
    if re.search(r'docId="(\d+)"', x).group(1) != "13922499":
        continue
    nr = int(re.search(r'pageNr="(\d+)"', x).group(1))
    img = re.search(r'imageFilename="([^"]+)"', x).group(1)
    stem = f"p{nr:04d}"
    cx = C / f"{stem}.xml"
    assert cx.exists(), stem
    t_job = [ln["text"] for ln in page_lines(x)]
    t_c = [ln["text"] for ln in page_lines(cx.read_text(encoding="utf-8"))]
    same_text += t_job == t_c
    cimg = next(p for p in C.glob(f"{stem}.*") if p.suffix != ".xml")
    d = h(Path(l).parent / img) != h(cimg)
    differ += d
    shutil.copyfile(l, OUT / f"{stem}.xml"); shutil.copyfile(Path(l).parent / img, OUT / img)
    rows.append({"page": nr, "stem": stem, "image": img, "lines": len(t_job), "text_equals_corrected": t_job == t_c, "image_differs_from_corrected": d})
man = {"job": J.parent.name, "hertwig_train_pages": len(rows), "image_differs": differ, "text_identical": same_text, "pages": rows}
Path("docs/c4u").mkdir(exist_ok=True); Path("docs/c4u/split.json").write_text(json.dumps(man, indent=1))
print({k: v for k, v in man.items() if k != "pages"})
