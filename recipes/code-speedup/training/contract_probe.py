#!/usr/bin/env python3
"""Audit the training/serving contract for a chat SFT checkpoint.

This tool deliberately uses only SFT corpus rows.  It must never be pointed at
behavioral-development or frozen evaluation data: it is a format/supervision
diagnostic, not a capability benchmark.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_CHAT_TEMPLATE = Path(__file__).resolve().parents[1] / "prompt" / "qwen_chat_template.jinja"


def load_chat_template(tokenizer: Any, path: Path) -> str:
    """Install the checked-in template used for both SFT and local serving."""
    template = path.read_text(encoding="utf-8")
    if "generation %}" not in template or "endgeneration %}" not in template:
        raise ValueError(f"chat template lacks generation boundaries: {path}")
    tokenizer.chat_template = template
    return hashlib.sha256(template.encode()).hexdigest()


def _as_list(value: Any) -> list[int]:
    """Convert a tokenizer tensor/list value to one flat Python list."""
    if hasattr(value, "get") and value.get("input_ids") is not None:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    while value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def supervision_report(tokenizer: Any, messages: list[dict[str, str]], max_length: int) -> dict:
    """Verify the assistant span that ``assistant_only_loss`` must supervise.

    Transformers only provides an assistant-token mask when the chat template
    declares generation boundaries.  Treat a missing/empty mask as a hard
    failure instead of silently optimizing an unintended token span.
    """
    if (len(messages) < 3 or messages[0].get("role") != "system"
            or messages[-1].get("role") != "assistant"
            or any(message.get("role") != ("user" if index % 2 else "assistant")
                   for index, message in enumerate(messages[1:], start=1))):
        raise ValueError("expected system followed by alternating user/assistant messages")
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=max_length,
        )
    except Exception as exc:  # Version-specific tokenizer errors need context.
        raise ValueError(f"tokenizer cannot produce an assistant mask: {exc}") from exc

    input_ids = _as_list(rendered["input_ids"])
    mask_value = rendered.get("assistant_masks")
    if mask_value is None:
        mask_value = rendered.get("assistant_tokens_mask")
    if mask_value is None:
        raise ValueError("tokenizer returned no assistant token mask")
    assistant_mask = _as_list(mask_value)
    if len(input_ids) != len(assistant_mask):
        raise ValueError("assistant mask length differs from input IDs")
    supervised = [index for index, value in enumerate(assistant_mask) if value]
    if not supervised:
        raise ValueError("assistant token mask is empty")

    action_prefix = '{"action": "patch"'
    assistant_starts = [index for index, value in enumerate(assistant_mask)
                        if value and (index == 0 or not assistant_mask[index - 1])]
    expected_assistant_turns = sum(message.get("role") == "assistant" for message in messages)
    if len(assistant_starts) != expected_assistant_turns:
        raise ValueError("assistant mask does not provide one span per assistant turn")
    for first_assistant in assistant_starts:
        action_end = next(
            (
                end for end in range(1, min(32, len(input_ids) - first_assistant) + 1)
                if tokenizer.decode(input_ids[first_assistant:first_assistant + end]).startswith(action_prefix)
            ),
            None,
        )
        if action_end is None:
            raise ValueError("assistant action prefix is truncated or differs from canonical patch action")
        if not all(assistant_mask[first_assistant:first_assistant + action_end]):
            raise ValueError("canonical action prefix is not fully supervised")
    return {
        "input_tokens": len(input_ids),
        "assistant_tokens": len(supervised),
        "first_assistant_token": assistant_starts[0],
        "assistant_turns": len(assistant_starts),
        "template_sha256": hashlib.sha256(
            (tokenizer.chat_template or "").encode()
        ).hexdigest(),
    }


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty corpus")
    return rows


def select_rows(rows: list[dict], sample_size: int) -> list[dict]:
    """Round-robin deterministic sample across source/family strata."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        source = row.get("meta", {}).get("source", "unknown")
        family_line = row["messages"][1]["content"].splitlines()[0]
        family = family_line.removeprefix("family: ")
        groups[(source, family)].append(row)
    selected: list[dict] = []
    positions = {key: 0 for key in sorted(groups)}
    while len(selected) < min(sample_size, len(rows)):
        progressed = False
        for key in sorted(groups):
            index = positions[key]
            if index < len(groups[key]) and len(selected) < sample_size:
                selected.append(groups[key][index])
                positions[key] += 1
                progressed = True
        if not progressed:
            break
    return selected


def parse_action(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            action = value.get("action")
            return action if isinstance(action, str) else None
    return None


def action_logprobs(model: Any, tokenizer: Any, prompt: str) -> dict[str, float]:
    """Teacher-force the two competing action values after a fixed JSON prefix."""
    import torch

    prefix = prompt + '{"action": "'
    encoded = tokenizer(prefix, return_tensors="pt")
    result: dict[str, float] = {}
    for choice in ("patch", "rewrite"):
        ids = tokenizer.encode(choice, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        total = 0.0
        for token in ids:
            with torch.inference_mode():
                logits = model(input_ids=input_ids).logits[0, -1]
                total += float(torch.log_softmax(logits, dim=-1)[token])
            input_ids = torch.cat(
                [input_ids, torch.tensor([[token]], dtype=input_ids.dtype)], dim=1
            )
        result[choice] = total
    return result


def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, str, dict[str, float]]:
    import torch

    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return prompt, tokenizer.decode(new_tokens, skip_special_tokens=True), action_logprobs(model, tokenizer, prompt)


def run_model(base: str, adapter: str | None, rows: list[dict]) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = adapter or base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(adapter, dtype=torch.float32)
        label = "adapter"
    else:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        label = "base"
    model.eval()
    output = []
    for row in rows:
        prompt, raw, logprobs = generate(model, tokenizer, row["messages"])
        output.append({
            "case_id": row.get("meta", {}).get("case_id"),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "raw": raw,
            "action": parse_action(raw),
            "action_logprobs": logprobs,
        })
    del model
    gc.collect()
    return [{"model": label, **item} for item in output]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from transformers import __version__ as transformers_version

    rows = load_rows(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.base)
    template_sha256 = load_chat_template(tokenizer, args.chat_template)
    reports = [supervision_report(tokenizer, row["messages"], args.max_length) for row in rows]
    selected = select_rows(rows, args.sample_size)
    report: dict[str, Any] = {
        "data": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows_checked": len(rows),
        "sampled_rows": len(selected),
        "transformers_version": transformers_version,
        "supervision": {
            "assistant_tokens_min": min(item["assistant_tokens"] for item in reports),
            "assistant_tokens_max": max(item["assistant_tokens"] for item in reports),
            "template_sha256": template_sha256,
        },
    }
    if not args.skip_generation:
        generations = run_model(args.base, None, selected)
        if args.adapter:
            generations.extend(run_model(args.base, args.adapter, selected))
        report["generations"] = generations
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["supervision"], indent=2))


if __name__ == "__main__":
    main()
