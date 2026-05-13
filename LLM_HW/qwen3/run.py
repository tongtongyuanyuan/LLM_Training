"""
CLI entry point for Qwen3 LoRA / QLoRA fine-tuning.

Usage:
    python run.py <TrainerName> <action>

Trainer names:
    Qwen3_8B_LoRA    Qwen3-8B  with LoRA
    Qwen3_8B_QLoRA   Qwen3-8B  with QLoRA (4-bit)
    Qwen3_32B_LoRA   Qwen3-32B with LoRA
    Qwen3_32B_QLoRA  Qwen3-32B with QLoRA (4-bit)

Actions:
    download   download the base model from ModelScope
    train      run SFT fine-tuning via llamafactory-cli
    infer      interactive chat with the fine-tuned adapter
    export     merge adapter weights into base model

Typical workflow on RunPod:
    # 1. install deps
    pip install --no-deps -e /hy-tmp/LLaMA-Factory
    pip install torch transformers peft trl accelerate bitsandbytes deepspeed

    # 2. prepare data  (run once)
    python data_process.py --input /hy-tmp/train_medical.json

    # 3. download model  (run once per model size)
    python run.py Qwen3_8B_LoRA download

    # 4. train
    python run.py Qwen3_8B_LoRA  train
    python run.py Qwen3_8B_QLoRA train

    # 5. test
    python run.py Qwen3_8B_LoRA infer

    # 6. merge weights
    python run.py Qwen3_8B_LoRA export

Environment variables:
    LLAMAFACTORY_DIR   /hy-tmp/LLaMA-Factory  (default)
    MODEL_BASE_DIR     /hy-tmp                (default)
    OUTPUT_BASE_DIR    /hy-tmp/output         (default)
"""

import sys
from trainer import TRAINERS

ACTIONS = ("download", "train", "infer", "export")


def usage():
    print(__doc__)
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        usage()

    trainer_name = sys.argv[1]
    action       = sys.argv[2]

    if trainer_name not in TRAINERS:
        print(f"ERROR: unknown trainer '{trainer_name}'")
        print(f"  Available: {', '.join(TRAINERS)}")
        sys.exit(1)

    if action not in ACTIONS:
        print(f"ERROR: unknown action '{action}'")
        print(f"  Available: {', '.join(ACTIONS)}")
        sys.exit(1)

    trainer = TRAINERS[trainer_name]()
    getattr(trainer, action)()


if __name__ == "__main__":
    main()
