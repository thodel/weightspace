"""Render a page draw the way the trainer's ``evaluate_qlora`` sees it (phase 5).

Runs in the trainer's ``vlm-train-tf5`` venv (transformers 5.17, the one its
Qwen3.5 jobs use), importing the trainer's own functions, so the prompt, the
stop ids and the pixel fit are the trainer's and not a copy of them:

    PYTHONPATH=$T/src:$T/engines $T/.venvs/vlm-train-tf5/bin/python -m wsp.vlm_render \\
        --pages <set>/pages.jsonl --data-root <set> --out ~/atr-cache/wsp/render/<name>

Writes ``prompt.txt`` (the training render cut where the answer begins),
``stop_ids.json``, ``budget.json`` and ``pages.jsonl`` with one line per page:
``{"key", "image": <fitted PNG>, "text": <reference>}``. The PNG is the image
after ``fit_pixels``; the processor's own resize happens in the engine that
reads it, with the same ``max_pixels``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE = "Qwen/Qwen3.5-4B"
PROMPT = "Transcribe the handwritten text in this image exactly as written."
MAX_PIXELS = 2048 * 32 * 32      # VLM_PIXEL_BUDGET["page"], the v2 job's budget (2,097,152)


def page_key(image: str) -> str:
    return Path(image).stem


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.vlm_render")
    ap.add_argument("--pages", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", help="processor/template source if it ships one (as evaluate_qlora)")
    a = ap.parse_args(argv)

    from PIL import Image
    from transformers import AutoProcessor
    from atr_training.vlm_dataset import apply_visual_budget, fit_pixels
    from vlm_train_svc.evaluate_qlora import generation_prompt, stop_token_ids

    src = BASE
    if a.adapter and (Path(a.adapter) / "preprocessor_config.json").is_file():
        src = a.adapter
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
    budget = apply_visual_budget(processor, MAX_PIXELS)
    prompt = generation_prompt(processor, PROMPT, None, True)
    stops = stop_token_ids(processor.tokenizer, processor)

    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    root = Path(a.data_root)
    rows = []
    for line in Path(a.pages).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        key = page_key(s["image"])
        dst = out / "images" / f"{key}.png"
        if not dst.exists():
            with Image.open(root / s["image"]) as raw:
                fit_pixels(raw.convert("RGB"), MAX_PIXELS).save(dst)
        rows.append({"key": key, "image": str(dst), "text": s["text"]})
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out / "stop_ids.json").write_text(json.dumps(stops))
    (out / "budget.json").write_text(json.dumps({"max_pixels": MAX_PIXELS, "applied": str(budget),
                                                 "processor": src, "prompt": PROMPT}, indent=1))
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / "pages.jsonl").write_text(body, encoding="utf-8")
    print(f"{len(rows)} pages, prompt {len(prompt)} chars, stops {stops}, "
          f"pages sha256 {hashlib.sha256(Path(a.pages).read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
