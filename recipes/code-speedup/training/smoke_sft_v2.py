#!/usr/bin/env python3
"""CPU smoke test: prove the SFT v2 corpus loads and the training loop steps.

Runs a handful of optimizer steps of the real ``train_sft`` stack (tokenizer +
model + TRL SFTTrainer) on CPU against ``sft_speedup_v2.jsonl``, then asserts a
train loss was produced. This is a corpus/loop sanity check, NOT a training run:
no GPU, tiny step budget, tiny base model, results discarded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--out", default="/tmp/sft_v2_smoke_out")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(7)
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    print(f"loaded {len(rows)} SFT examples from {args.data}")
    assert len(rows) >= 10, "corpus too small"
    # Validate message format on every row before touching the trainer.
    for i, row in enumerate(rows):
        msgs = row["messages"]
        roles = [m["role"] for m in msgs]
        assert roles[0] == "system", f"row {i} missing system"
        assert roles[-1] == "assistant", f"row {i} not ending in assistant"
        assert all("content" in m for m in msgs), f"row {i} missing content"

    dataset = Dataset.from_list([{"messages": row["messages"]} for row in rows])

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_chat_template(tokenizer, DEFAULT_CHAT_TEMPLATE)
    for row in rows:
        supervision_report(tokenizer, row["messages"], max_length=2048)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    config = SFTConfig(
        output_dir=args.out,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        bf16=False,
        fp16=False,
        max_length=2048,
        packing=False,
        report_to="none",
        assistant_only_loss=True,
        # CPU smoke on a small-RAM box: SGD drops the ~4GB Adam optimizer state
        # and gradient checkpointing trims activation memory. This is a loop
        # sanity check, not a training run, so the optimizer choice is irrelevant.
        optim="sgd",
        gradient_checkpointing=True,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset,
                         processing_class=tokenizer)
    result = trainer.train()
    loss = result.metrics.get("train_loss")
    print(json.dumps({
        "smoke_ok": loss is not None,
        "train_loss": loss,
        "steps": args.max_steps,
        "examples": len(rows),
        "device": "cpu",
    }))
    assert loss is not None, "trainer produced no loss"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
