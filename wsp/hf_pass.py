"""Transformers pass engine: the trainer's page reading, one base load for several adapter runs.

Runs under the *trainer's* ``vlm-train-tf5`` interpreter (executed, never modified; no wsp import,
no install): it reuses ``vlm_train_svc.evaluate_qlora`` for the processor, the visual budget, the
prompt cut from the training render, the stop ids and the greedy ``generate`` call, so a reading
here is the reading ``evaluate_qlora`` would give — the same function, the same arguments. What is
new is that the base model is loaded once and adapters are applied, read and removed in turn, and
that a run can be ``peft`` (LoRA layers beside the frozen weights, as the scorer does) or
``merged`` (ΔW added into the weights, as a merged serving copy would be).

    PYTHONPATH=<trainer>/src:<trainer>/engines  <trainer venv python> wsp/hf_pass.py \\
        --argv argv.json --jobs jobs.json

``argv.json`` is the argument list ``wsp.vlm_reference.command`` builds (JSON list; its
``--adapter``/``--val-jsonl``/``--report`` entries are only parsed, not used). ``jobs.json`` is a
list of ``{"adapter": dir | null, "mode": "peft" | "merged" | "merged32", "pages": draw.jsonl,
"data_root": dir, "out": raw.jsonl, "max_new_tokens": 3072}``; ``adapter`` null = the base alone.
Each output line: ``{"image", "raw", "at_cap", "new_tokens", "seconds"}``. A job whose output
exists is skipped (resumable). ``merged32`` merges in float32 and casts back to bf16 once.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--argv", required=True)
    ap.add_argument("--jobs", required=True)
    a = ap.parse_args(argv)

    import torch
    from vlm_train_svc import evaluate_qlora as ev

    cmd = json.loads(Path(a.argv).read_text())
    if "-m" in cmd[:3]:
        cmd = cmd[cmd.index("-m") + 2:]  # drop '<python> -m vlm_train_svc.evaluate_qlora'
    base_args = ev.parse_args(cmd)
    base_args.adapter = None
    model, processor = ev.load_model(base_args)  # the base alone, bf16, card 0
    model.eval()

    jobs = json.loads(Path(a.jobs).read_text())
    from peft import PeftModel

    for job in jobs:
        out = Path(job["out"])
        if out.exists():
            print("skip", out, flush=True)
            continue
        rows = [json.loads(l) for l in Path(job["pages"]).read_text(encoding="utf-8").splitlines() if l.strip()]
        t0 = time.time()
        m = model
        if job.get("adapter"):
            m = PeftModel.from_pretrained(model, job["adapter"])
            if job.get("mode", "peft") in ("merged", "merged32"):
                if job["mode"] == "merged32":
                    # merge in float32: upcast the touched base weights, add, cast back once
                    for mod in m.modules():
                        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                            base = mod.base_layer
                            w32 = base.weight.data.float()
                            A = mod.lora_A["default"].weight.data.float()
                            B = mod.lora_B["default"].weight.data.float()
                            w32 += mod.scaling["default"] * (B @ A)
                            base.weight.data = w32.to(base.weight.dtype)
                    m = m.unload()  # drop the (now baked-in) LoRA layers without merging again
                else:
                    m = m.merge_and_unload()
            m.eval()
        load_s = time.time() - t0
        lines = []
        for i, r in enumerate(rows, 1):
            t1 = time.time()
            text, capped = ev.transcribe(m, processor, Path(job["data_root"]) / r["image"], base_args.prompt,
                                         job.get("max_new_tokens", base_args.max_new_tokens), system=None,
                                         max_pixels=ev._parse_kind_pixels(base_args.kind_pixels).get("page"),
                                         prompt_from_training_render=base_args.prompt_from_training_render)
            n = len(processor.tokenizer.encode(text, add_special_tokens=False))
            lines.append({"image": r["image"], "raw": text, "at_cap": bool(capped), "new_tokens": n,
                          "seconds": round(time.time() - t1, 1)})
            print(f"{job['out']} {i}/{len(rows)} {lines[-1]['seconds']}s cap={capped}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
        tmp.rename(out)
        out.with_suffix(".meta.json").write_text(json.dumps({
            "adapter": job.get("adapter"), "mode": job.get("mode"), "pages": len(rows), "adapter_load_s": round(load_s, 1),
            "total_s": round(time.time() - t0, 1), "peak_gpu_mib": int(torch.cuda.max_memory_allocated() / 2**20)}))
        if job.get("adapter"):
            if job.get("mode", "peft") == "peft":
                model = m.unload()
            else:
                # merged: reload a clean base for the next job (the weights were changed in place)
                del m
                torch.cuda.empty_cache()
                model, _ = ev.load_model(base_args)
                model.eval()
    return 0


if __name__ == "__main__":
    sys.exit(main())
