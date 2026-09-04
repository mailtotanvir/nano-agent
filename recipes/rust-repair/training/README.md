# rust-repair training (Phase 2 SFT)

Full fine-tune (bf16) of a sub-1B base on verified teacher trajectories, sized for
a single 24GB L4. The goal: teach the tiny model the SEARCH/REPLACE patch
convention it fails to follow untrained (Phase 1 floor = 0%), lifting `tiny_loop`
toward the frontier line (Phase 1 ceiling = 84.4%).

## Flow

```text
teacher_trajectories_vN.jsonl  (verified frontier successes)
        │  make_sft.py
        ▼
sft_vN.jsonl  (chat: system contract + step context -> JSON proposal target)
        │  train_sft.py  (TRL SFTTrainer, assistant_only_loss, bf16)   [GPU]
        ▼
out/<model>-sft-vN/  (HF checkpoint)
        │  to_gguf.sh  (convert + quantize)                            [GPU or CPU]
        ▼
<model>-sft-vN-q8_0.gguf  -> copy to OCI box -> serve -> re-run eval
```

## Commands

```bash
# 1. build SFT data from verified trajectories (CPU, local)
python make_sft.py --trajectories ../datasets/teacher_trajectories_v2.jsonl \
    --out ../datasets/sft_v2.jsonl

# 2. train on the GPU VM (see infra/gcp/gpu_vm.sh)
pip install -r requirements.txt
python train_sft.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --data sft_v2.jsonl --out out/qwen05-sft-v2 --epochs 3 --lr 1e-5 --bs 8

# 3. convert to GGUF and copy to OCI
bash to_gguf.sh out/qwen05-sft-v2 qwen05-sft-v2 q8_0

# 4. serve on OCI + re-run eval (tiny_loop arm) against frozen eval_v1
#    -> compare to the 0% untrained floor and 84.4% frontier ceiling
```

## Model sweep (per spec section 7)

Train the best-looking bases first to conserve the GPU window:
- Qwen2.5-Coder-0.5B-Instruct (primary)
- SmolLM2-360M-Instruct, Qwen3-0.6B (smaller candidates)
- Qwen2.5-Coder-1.5B-Instruct (control; use `--qlora` if memory is tight)

## Notes

- `assistant_only_loss=True` masks the prompt so loss is on the JSON target only.
- Sub-1B models use full FT (no QLoRA needed on a 24GB L4).
- Eval is always the FROZEN `eval_v1` set; training data never touches those ids.
- GPU spend requires explicit user approval before launch (see gpu_vm.sh).
