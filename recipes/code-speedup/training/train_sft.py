#!/usr/bin/env python3
"""SFT for code-speed students on one 24 GB L4.

Supports full tuning for the 0.5B control and LoRA for the 1.5B candidate.
Splits unique case IDs before replay weighting so duplicated legacy examples
cannot leak into the in-distribution loss-validation partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

try:  # Works both as ``python train_sft.py`` and as ``training.train_sft``.
    from .contract_probe import (
        DEFAULT_CHAT_TEMPLATE,
        load_chat_template,
        supervision_report,
    )
except ImportError:  # pragma: no cover - direct script execution
    from contract_probe import (
        DEFAULT_CHAT_TEMPLATE,
        load_chat_template,
        supervision_report,
    )


def _family(row: dict) -> str:
    first = row["messages"][1]["content"].splitlines()[0]
    return first.removeprefix("family: ") if first.startswith("family: ") else "unknown"


def split_by_case(rows: list[dict], eval_fraction: float, seed: int):
    """Deterministically split case IDs within each source/family stratum."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        meta = row.get("meta", {})
        case_id = meta.get("case_id") or f"row-{index}"
        groups[case_id].append(row)
    strata = defaultdict(list)
    for case_id, case_rows in groups.items():
        row = case_rows[0]
        strata[(row.get("meta", {}).get("source", "unknown"), _family(row))].append(case_id)

    eval_ids = set()
    for stratum, case_ids in sorted(strata.items()):
        ordered = sorted(case_ids, key=lambda value: hashlib.sha256(
            f"{seed}:{stratum}:{value}".encode()).digest())
        count = max(1, round(len(ordered) * eval_fraction)) if len(ordered) > 1 else 0
        eval_ids.update(ordered[:count])
    train = [row for case_id, case_rows in groups.items() if case_id not in eval_ids
             for row in case_rows]
    evaluation = [row for case_id, case_rows in groups.items() if case_id in eval_ids
                  for row in case_rows]
    return train, evaluation


def apply_v1_replay(rows: list[dict], ratio: float, seed: int) -> list[dict]:
    """Raise legacy v1's effective share without duplicating validation rows."""
    if not 0 <= ratio < 1:
        raise ValueError("v1 replay ratio must be in [0, 1)")
    legacy = [row for row in rows if row.get("meta", {}).get("source") == "sft_v1"]
    current = [row for row in rows if row.get("meta", {}).get("source") != "sft_v1"]
    if not legacy or not current or ratio == 0:
        return rows
    target_legacy = math.ceil(len(current) * ratio / (1 - ratio))
    rng = random.Random(seed)
    replay = []
    while len(replay) < target_legacy:
        epoch = list(legacy)
        rng.shuffle(epoch)
        replay.extend(epoch)
    combined = current + replay[:target_legacy]
    rng.shuffle(combined)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--v1-replay-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--save-steps", type=int, default=0,
                        help="checkpoint/eval interval; 0 means once per epoch")
    args = parser.parse_args()
    if not 0 < args.eval_fraction < 1:
        raise ValueError("eval fraction must be in (0, 1)")

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(args.seed)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if len(rows) < 10:
        raise ValueError("refusing SFT with fewer than 10 verified examples")
    train_rows, eval_rows = split_by_case(rows, args.eval_fraction, args.seed)
    if not train_rows or not eval_rows:
        raise ValueError("grouped split produced an empty train or eval partition")
    effective_train_rows = apply_v1_replay(train_rows, args.v1_replay_ratio, args.seed)
    train_dataset = Dataset.from_list([{"messages": row["messages"]}
                                       for row in effective_train_rows])
    eval_dataset = Dataset.from_list([{"messages": row["messages"]} for row in eval_rows])

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    template_sha256 = load_chat_template(tokenizer, args.chat_template)
    supervision = [
        supervision_report(tokenizer, row["messages"], args.max_length)
        for row in rows
    ]
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
    step_checkpoints = args.save_steps > 0
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        eval_strategy="steps" if step_checkpoints else "epoch",
        save_strategy="steps" if step_checkpoints else "epoch",
        eval_steps=args.save_steps if step_checkpoints else None,
        save_steps=args.save_steps if step_checkpoints else 500,
        bf16=True,
        max_length=args.max_length,
        packing=False,
        gradient_checkpointing=True,
        report_to="none",
        assistant_only_loss=True,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, processing_class=tokenizer,
                         peft_config=peft_config)
    result = trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    metrics = {
        "train": result.metrics,
        "eval": trainer.evaluate(),
        "examples": len(rows),
        "unique_train_examples": len(train_rows),
        "effective_train_examples": len(effective_train_rows),
        "eval_examples": len(eval_rows),
        "v1_replay_ratio_requested": args.v1_replay_ratio,
        "v1_replay_examples": sum(
            row.get("meta", {}).get("source") == "sft_v1" for row in effective_train_rows),
        "base_model": args.base,
        "method": "lora" if args.lora else "full_parameter",
        "seed": args.seed,
        "supervision": {
            "rows_checked": len(supervision),
            "assistant_tokens_min": min(item["assistant_tokens"] for item in supervision),
            "assistant_tokens_max": max(item["assistant_tokens"] for item in supervision),
            "template_sha256": template_sha256,
        },
    }
    Path(args.out, "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
