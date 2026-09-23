#!/usr/bin/env python3
"""Minimal local OpenAI-compatible server for a Hugging Face causal LM."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:  # Also support direct execution from recipes/code-speedup/eval.
    from training.contract_probe import DEFAULT_CHAT_TEMPLATE, load_chat_template
except ImportError:  # pragma: no cover - direct script execution
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.contract_probe import DEFAULT_CHAT_TEMPLATE, load_chat_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--forced-prefix",
                        help="diagnostic-only assistant prefix; never use for primary evaluation")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_chat_template(tokenizer, args.chat_template)
    if (Path(args.model) / "adapter_config.json").is_file():
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            size = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(size))
            prompt = tokenizer.apply_chat_template(
                request["messages"], tokenize=False, add_generation_prompt=True,
            )
            prefix = args.forced_prefix or ""
            inputs = tokenizer(prompt + prefix, return_tensors="pt")
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=min(int(request.get("max_tokens", 1024)), 1024),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1]:]
            content = prefix + tokenizer.decode(new_tokens, skip_special_tokens=True)
            response = {
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"completion_tokens": int(new_tokens.shape[0])},
            }
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
