#!/usr/bin/env python3
"""Stage 5, C4: continued training of a TrOCR subject from its own weights (docs/TROCR_STAGE5_PREREG.md, section 3). Run in the trainer's venv:

    cd ~/Repo/training-atr-models/engines && PYTHONPATH=~/Repo/training-atr-models/src:. TMPDIR=~/atr-cache/tmp CUDA_VISIBLE_DEVICES=0 \
      ~/Repo/training-atr-models/.venvs/trocr-train/bin/python ~/Repo/weightspace/<worktree>/scripts/trocr_stage5_train.py \
      --base-model SUBJECT_DIR --manifest PRACTICE.jsonl --output-dir OUT --seed 1 --lrate 1e-5 --epochs 5 [--max-steps N]

It imports the trainer's own pieces (``trocr_train_svc.train_trocr``: JsonlDataset, TrocrDataCollator, use_aligned_loss, the generation settings) and builds the same
Seq2SeqTrainer as the job (batch 8, cosine, warmup 0.1, weight decay 0, grad-norm 1, adamw_torch, bf16, gradient checkpointing, 8 workers) with three differences, all
written in the preregistration: the training data is the manifest given (the practice set), there is NO validation set and NO best-checkpoint selection (the final
state after the fixed epochs is the model), and nothing is registered or published. ``--max-steps 0`` saves the model unchanged (zero-step check); a positive
``--max-steps`` stops early (harness check only). The manifest lines are {key, text, image}; ``image`` is absolute. Writes OUT/stage5_train.json (arguments,
steps, seconds, train loss log) next to the hub-format model."""
import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lrate", type=float, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)

    import torch
    from transformers import AutoProcessor, GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel, set_seed
    from trocr_train_svc.train_trocr import JsonlDataset, TrocrDataCollator, use_aligned_loss

    t0 = time.time()
    set_seed(a.seed)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(a.base_model, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    ds = JsonlDataset(a.manifest, "/")
    print(f"train={len(ds)} base_model={a.base_model}", flush=True)
    model = VisionEncoderDecoderModel.from_pretrained(a.base_model)
    use_aligned_loss(model)
    assert model.decoder.get_input_embeddings().num_embeddings == len(processor.tokenizer), "vocabulary differs from the tokenizer"
    gen = GenerationConfig.from_model_config(model.config)
    gen.num_beams, gen.length_penalty, gen.early_stopping, gen.max_new_tokens, gen.no_repeat_ngram_size = 4, 1.0, True, 256, 3
    log = {"args": vars(a), "torch": torch.__version__}
    if a.max_steps == 0:
        steps, losses = 0, []
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=Seq2SeqTrainingArguments(
                output_dir=str(out / "trainer_tmp"), num_train_epochs=a.epochs, max_steps=a.max_steps, per_device_train_batch_size=a.batch_size,
                gradient_accumulation_steps=1, learning_rate=a.lrate, lr_scheduler_type="cosine", warmup_ratio=0.1, weight_decay=0.0, max_grad_norm=1.0,
                optim="adamw_torch", bf16=True, logging_steps=25, eval_strategy="no", save_strategy="no", load_best_model_at_end=False,
                gradient_checkpointing=True, dataloader_num_workers=a.workers, predict_with_generate=True, generation_config=gen,
                remove_unused_columns=False, report_to=[], seed=a.seed),
            train_dataset=ds, data_collator=TrocrDataCollator(processor), processing_class=processor.tokenizer)
        t1 = time.time()
        res = trainer.train()
        log["train_seconds"] = round(time.time() - t1, 1)
        steps = trainer.state.global_step
        losses = [h for h in trainer.state.log_history if "loss" in h]
        log["train_loss_final_mean"] = res.training_loss
    log.update({"steps": steps, "log_history": losses, "seconds_total": round(time.time() - t0, 1)})
    model.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    (out / "stage5_train.json").write_text(json.dumps(log, indent=1))
    print(f"saved {out} steps={steps} seconds={log['seconds_total']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
