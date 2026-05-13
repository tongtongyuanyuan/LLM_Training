"""
Prepare training data for LLaMA-Factory (Qwen3 LoRA / QLoRA).

Reads a JSONL file where each line is a JSON object with keys:
    instruction, input, output

Writes an alpaca-style JSON array to the LLaMA-Factory data directory
and registers it in dataset_info.json so llamafactory-cli can find it.

Usage:
    python data_process.py \
        --input  /hy-tmp/train_medical.json \
        --llamafactory_dir /hy-tmp/LLaMA-Factory
"""

import argparse
import json
import os

DATASET_NAME = "alpaca_zh_medical"


def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_dataset(records: list, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{DATASET_NAME}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} records → {out_path}")
    return out_path


def register_dataset(data_dir: str):
    info_path = os.path.join(data_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}

    if DATASET_NAME not in info:
        info[DATASET_NAME] = {"file_name": f"{DATASET_NAME}.json"}
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"Registered '{DATASET_NAME}' in {info_path}")
    else:
        print(f"'{DATASET_NAME}' already registered in {info_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="/hy-tmp/train_medical.json",
        help="Source JSONL file (one JSON object per line)"
    )
    parser.add_argument(
        "--llamafactory_dir", default="/hy-tmp/LLaMA-Factory",
        help="Root of the cloned LLaMA-Factory repo"
    )
    args = parser.parse_args()

    data_dir = os.path.join(args.llamafactory_dir, "data")
    records = load_jsonl(args.input)
    write_dataset(records, data_dir)
    register_dataset(data_dir)


if __name__ == "__main__":
    main()
