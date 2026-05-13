"""
Download Qwen3 models from ModelScope.

Usage:
    python download.py --model 8b    # Qwen3-8B
    python download.py --model 32b   # Qwen3-32B
    python download.py --model all   # both
"""

import argparse
import os

MODEL_MAP = {
    "8b":  "Qwen/Qwen3-8B",
    "32b": "Qwen/Qwen3-32B",
}

CACHE_DIR = os.environ.get("MODEL_BASE_DIR", "/hy-tmp")


def download(model_key: str):
    from modelscope import snapshot_download
    model_id = MODEL_MAP[model_key]
    print(f"Downloading {model_id} → {CACHE_DIR} ...")
    model_dir = snapshot_download(model_id, cache_dir=CACHE_DIR)
    print(f"Saved to: {model_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["8b", "32b", "all"], default="8b")
    args = parser.parse_args()

    keys = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]
    for key in keys:
        download(key)


if __name__ == "__main__":
    main()
