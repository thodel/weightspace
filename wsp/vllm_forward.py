"""One forward pass of a Qwen3.5 page adapter through vLLM (phase 5).

Runs in ``~/.venvs/vllm`` as its own process, one process per pass, so card 0
is free between passes (docs/PHASE5_PLAN.md §5):

    CUDA_VISIBLE_DEVICES=0 ~/.venvs/vllm/bin/python -m wsp.vllm_forward \\
        --render ~/atr-cache/wsp/render/<name> --adapter <peft dir> --out <readings.jsonl>

Input is a ``wsp.vlm_render`` directory: the trainer's prompt, stop ids and
fitted images. Decoding is greedy with the trainer's cap (3,072 for the v2
held-out score). Writes one line per page:
``{"key", "text", "finish_reason", "tokens"}``; ``finish_reason == "length"``
is a page that ran to the cap.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

BASE = "Qwen/Qwen3.5-4B"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.vllm_forward")
    ap.add_argument("--render", required=True)
    ap.add_argument("--adapter", help="peft directory; omitted = the base alone (control C1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", help="file with one page key per line: the practice subset")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--gpu-mem", type=float, default=0.45,
                    help="share of card 0; 0.45 of 46 GB leaves the 24 GB a VLM trainer job needs")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    a = ap.parse_args(argv)

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    r = Path(a.render)
    prompt = (r / "prompt.txt").read_text(encoding="utf-8")
    stops = json.loads((r / "stop_ids.json").read_text())
    budget = json.loads((r / "budget.json").read_text())
    pages = [json.loads(l) for l in (r / "pages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.keys:
        want = [k.strip() for k in Path(a.keys).read_text().splitlines() if k.strip()]
        by = {p["key"]: p for p in pages}
        pages = [by[k] for k in want]

    t0 = time.time()
    llm = LLM(model=BASE, dtype="bfloat16", seed=0, enable_lora=a.adapter is not None, max_lora_rank=64,
              max_loras=1, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
              max_num_seqs=a.max_num_seqs, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"max_pixels": budget["max_pixels"]})
    load_s = time.time() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop_token_ids=stops,
                        skip_special_tokens=True)
    lora = LoRARequest("subject", 1, str(Path(a.adapter).resolve())) if a.adapter else None
    reqs = [{"prompt": prompt, "multi_modal_data": {"image": Image.open(p["image"]).convert("RGB")}}
            for p in pages]
    t1 = time.time()
    outs = llm.generate(reqs, sp, lora_request=lora)
    gen_s = time.time() - t1
    with open(a.out, "w", encoding="utf-8") as f:
        for p, o in zip(pages, outs):
            c = o.outputs[0]
            f.write(json.dumps({"key": p["key"], "text": c.text.strip(), "finish_reason": c.finish_reason,
                                "tokens": len(c.token_ids)}, ensure_ascii=False) + "\n")
    meta = {"pages": len(pages), "load_s": round(load_s, 1), "generate_s": round(gen_s, 1),
            "adapter": a.adapter, "max_tokens": a.max_tokens, "gpu_mem": a.gpu_mem}
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
