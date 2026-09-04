#!/usr/bin/env python3
"""SFT trainer for the rust-repair tiny models (Phase 2).

Full fine-tune (bf16) of a sub-1B base on the verified teacher trajectories.
Sized for a single 24GB L4 on GCP; sub-1B models do not need QLoRA (--qlora is
available for the 1.5B control if memory is tight).

Consumes the chat-format JSONL from make_sft.py and trains with TRL's SFTTrainer,
masking the prompt so loss is computed only on the assistant JSON target.

Run ON THE GPU VM (not the OCI box):
  pip install -r requirements.txt
  python train_sft.py \
      --base Qwen/Qwen2.5-Coder-0.5B-Instruct \
      --data sft_v1.jsonl \
      --out out/qwen05-sft-v1 \
      --epochs 3 --lr 1e-5 --bs 8

Then convert to GGUF (llama.cpp convert_hf_to_gguf.py) and copy to the OCI box.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_chat_jsonl(path: str):
    from datasets import Dataset
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return Dataset.from_list([{"messages": r["messages"]} for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF base model id")
    ap.add_argument("--data", required=True, help="sft jsonl from make_sft.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--qlora", action="store_true", help="use 4-bit QLoRA (for 1.5B)")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16}
    peft_config = None
    if args.qlora:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)

    ds = load_chat_jsonl(args.data)
    split = ds.train_test_split(test_size=args.eval_frac, seed=7)

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        max_length=args.max_len,
        packing=False,
        gradient_checkpointing=True,
        report_to="none",
        assistant_only_loss=True,  # mask prompt; train on the JSON target only
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    metrics = trainer.evaluate()
    Path(args.out, "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("done ->", args.out)
    print("eval:", json.dumps(metrics))


if __name__ == "__main__":
    main()
