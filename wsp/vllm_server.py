"""A vLLM pass server for a bounded window: the base is loaded once, adapters are read in turn.

Used where many passes of the same pass definition follow each other (the planted-defect control,
stage A3): one process pays the 4-minute load once instead of once per pass. It is *not* the
held-out scorer (gate G1: vLLM and ``evaluate_qlora`` differ on loops). Card 0 only. It takes a
fixed ``--gpu-mem`` share, checks free memory before loading, checks the trainer's ``waiting``
(#396) before every job and exits (writing ``YIELD`` in the spool) when a job waits, and exits
after ``--idle-exit`` seconds without work, so nothing holds the card longer than the work.

    ~/.venvs/vllm/bin/python -m wsp.vllm_server --spool S --render R [--gpu-mem 0.40]

Jobs are ``S/jobs/<id>.json``: ``{"adapter": dir | null, "keys": [...], "out": path,
"render": optional other render dir}``; the answer is the file ``out`` (jsonl: key, text,
finish_reason, tokens) and ``out.meta.json``, then ``S/done/<id>``. Every job gets a fresh LoRA
id, so an adapter directory rewritten between jobs is never served from vLLM's cache.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

BASE = "Qwen/Qwen3.5-4B"


def _free_mib() -> int:
    o = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    return int(o.stdout.strip())


def _yield_wanted() -> str | None:
    try:
        from wsp import card
        w = card.waiting(timeout=90)
        return json.dumps(w) if w else None
    except Exception:  # health endpoint down: do not guess, keep working
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.vllm_server")
    ap.add_argument("--spool", required=True)
    ap.add_argument("--render", required=True, help="render dir (prompt.txt, stop_ids.json, budget.json, pages.jsonl)")
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--idle-exit", type=int, default=900)
    a = ap.parse_args(argv)

    spool = Path(a.spool)
    for d in ("jobs", "done"):
        (spool / d).mkdir(parents=True, exist_ok=True)
    need = int(a.gpu_mem * 46068) + 1500
    free = _free_mib()
    if free < need:
        (spool / "YIELD").write_text(f"card 0 has {free} MiB free, needs {need}")
        raise SystemExit(f"not started: card 0 has {free} MiB free, needs {need}")

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    r0 = Path(a.render)
    budget = json.loads((r0 / "budget.json").read_text())
    t0 = time.time()
    llm = LLM(model=BASE, dtype="bfloat16", seed=0, enable_lora=True, max_lora_rank=64, max_loras=1,
              gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
              limit_mm_per_prompt={"image": 1}, mm_processor_kwargs={"max_pixels": budget["max_pixels"]})
    (spool / "READY").write_text(json.dumps({"load_s": round(time.time() - t0, 1), "gpu_mem": a.gpu_mem,
                                             "free_before_mib": free, "time": time.strftime("%F %T")}))
    renders: dict[str, dict] = {}

    def render(path: str):
        if path not in renders:
            p = Path(path)
            renders[path] = {"prompt": (p / "prompt.txt").read_text(encoding="utf-8"),
                             "stops": json.loads((p / "stop_ids.json").read_text()),
                             "pages": {x["key"]: x for x in map(json.loads, (p / "pages.jsonl").read_text(encoding="utf-8").splitlines()) if x}}
        return renders[path]

    lora_id = 0
    idle = time.time()
    while True:
        jobs = sorted((spool / "jobs").glob("*.json"))
        if (spool / "STOP").exists():
            break
        if not jobs:
            if time.time() - idle > a.idle_exit:
                break
            time.sleep(3)
            continue
        why = _yield_wanted()
        if why:
            (spool / "YIELD").write_text(why)
            break
        jf = jobs[0]
        job = json.loads(jf.read_text())
        rd = render(job.get("render", a.render))
        sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop_token_ids=rd["stops"], skip_special_tokens=True)
        lora = None
        if job.get("adapter"):
            lora_id += 1
            lora = LoRARequest(f"job{lora_id}", lora_id, str(Path(job["adapter"]).resolve()))
        reqs = [{"prompt": rd["prompt"], "multi_modal_data": {"image": Image.open(rd["pages"][k]["image"]).convert("RGB")}}
                for k in job["keys"]]
        t1 = time.time()
        outs = llm.generate(reqs, sp, lora_request=lora, use_tqdm=False)
        gen_s = time.time() - t1
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for k, o in zip(job["keys"], outs):
                c = o.outputs[0]
                f.write(json.dumps({"key": k, "text": c.text.strip(), "finish_reason": c.finish_reason,
                                    "tokens": len(c.token_ids)}, ensure_ascii=False) + "\n")
        tmp.rename(out)
        out.with_suffix(".meta.json").write_text(json.dumps({"pages": len(job["keys"]), "generate_s": round(gen_s, 1),
                                                             "adapter": job.get("adapter"), "free_mib_after": _free_mib()}))
        (spool / "done" / jf.stem).write_text("ok")
        jf.unlink()
        idle = time.time()
    (spool / "READY").unlink(missing_ok=True)
    print("server exit", flush=True)


if __name__ == "__main__":
    main()
