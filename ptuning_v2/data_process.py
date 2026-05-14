"""
Split alpaca_zh_medical.json (already in alpaca instruction/input/output format)
into train.json / dev.json for P-Tuning v2 (ChatGLM2-6B/ptuning/main.py).

Usage:
    python data_process.py                            # uses alpaca_zh_medical.json at repo root
    python data_process.py --input /path/to/file.json
    python data_process.py --output_dir /hy-tmp
"""

import argparse
import json
import random
import os

# alpaca_zh_medical.json lives at the repo root (one level above ptuning_v2/)
_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_INPUT = os.path.join(_REPO_ROOT, "alpaca_zh_medical.json")

# Old local Mac path (knowledge-graph format — keep for reference)
# _DEFAULT_INPUT = (
#     "/Users/tongxue/Desktop/AI/AI_Course/三期课程/"
#     "知识图谱day16/作业/medicalGraph/medical.json/medical.json"
# )


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      default=_DEFAULT_INPUT, help="path to alpaca_zh_medical.json")
    p.add_argument("--output_dir", default="/hy-tmp",      help="directory for train.json / dev.json")
    return p.parse_args()


_args      = _parse_args()
INPUT_PATH = _args.input
OUTPUT_DIR = _args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.json")
DEV_PATH   = os.path.join(OUTPUT_DIR, "dev.json")

DEV_RATIO   = 0.05
RANDOM_SEED = 42


def main():
    print(f"Reading: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total records: {len(data)}")

    random.seed(RANDOM_SEED)
    random.shuffle(data)

    split      = int(len(data) * (1 - DEV_RATIO))
    train_data = data[:split]
    dev_data   = data[split:]

    print(f"Train: {len(train_data)}  |  Dev: {len(dev_data)}")

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(DEV_PATH, "w", encoding="utf-8") as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved → {TRAIN_PATH}")
    print(f"Saved → {DEV_PATH}")

    print("\nSample records:")
    for sample in train_data[:2]:
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────────
# OLD: knowledge-graph conversion (raw medical.json → alpaca QA pairs)
# Use this if you have the raw knowledge-graph medical.json instead of
# the pre-formatted alpaca_zh_medical.json.
# ──────────────────────────────────────────────────────────────────────────────

# INSTRUCTION = "现在你是一个专业的医疗助手，请根据患者的问题给出专业的医疗建议："
#
# def fmt_list(items):
#     if not items:
#         return None
#     cleaned = [str(i).strip() for i in items if str(i).strip()]
#     return "、".join(cleaned) if cleaned else None
#
# def build_pairs(record):
#     name = record.get("name", "").strip()
#     if not name:
#         return []
#     pairs = []
#     desc = (record.get("desc") or "").strip()
#     if desc:
#         pairs.append({"instruction": INSTRUCTION, "input": f"请问{name}是什么疾病？", "output": desc})
#     syms = fmt_list(record.get("symptom"))
#     if syms:
#         pairs.append({"instruction": INSTRUCTION, "input": f"患有{name}会有哪些症状？",
#                       "output": f"{name}的常见症状包括：{syms}。"})
#     cause = (record.get("cause") or "").strip()
#     if cause:
#         pairs.append({"instruction": INSTRUCTION, "input": f"{name}是什么原因引起的？", "output": cause})
#     prevent = (record.get("prevent") or "").strip()
#     if prevent:
#         pairs.append({"instruction": INSTRUCTION, "input": f"如何预防{name}？", "output": prevent})
#     dept = fmt_list(record.get("cure_department"))
#     if dept:
#         pairs.append({"instruction": INSTRUCTION, "input": f"得了{name}应该去哪个科室就诊？",
#                       "output": f"建议前往{dept}就诊。"})
#     cure = fmt_list(record.get("cure_way"))
#     if cure:
#         pairs.append({"instruction": INSTRUCTION, "input": f"{name}有哪些治疗方法？",
#                       "output": f"{name}的治疗方式包括：{cure}。"})
#     checks = fmt_list(record.get("check"))
#     if checks:
#         pairs.append({"instruction": INSTRUCTION, "input": f"诊断{name}通常需要做哪些检查？",
#                       "output": f"诊断{name}一般需要进行以下检查：{checks}。"})
#     drugs = fmt_list(record.get("recommand_drug"))
#     if drugs:
#         pairs.append({"instruction": INSTRUCTION, "input": f"治疗{name}有哪些常用药物？",
#                       "output": f"治疗{name}常用的药物包括：{drugs}。"})
#     duration = (record.get("cure_lasttime") or "").strip()
#     if duration:
#         pairs.append({"instruction": INSTRUCTION, "input": f"{name}的治疗周期大概是多久？",
#                       "output": f"{name}的治疗周期一般为{duration}。"})
#     cured_prob = (record.get("cured_prob") or "").strip()
#     if cured_prob:
#         pairs.append({"instruction": INSTRUCTION, "input": f"{name}的治愈率大概是多少？",
#                       "output": f"{name}的治愈率约为{cured_prob}。"})
#     return pairs
#
# def main_kg():
#     with open(INPUT_PATH, "r", encoding="utf-8") as f:
#         lines = [l.strip() for l in f if l.strip()]
#     all_pairs = []
#     for line in lines:
#         try:
#             record = json.loads(line)
#         except json.JSONDecodeError:
#             continue
#         all_pairs.extend(build_pairs(record))
#     random.seed(RANDOM_SEED)
#     random.shuffle(all_pairs)
#     split = int(len(all_pairs) * (1 - DEV_RATIO))
#     train_data, dev_data = all_pairs[:split], all_pairs[split:]
#     with open(TRAIN_PATH, "w", encoding="utf-8") as f:
#         for item in train_data:
#             f.write(json.dumps(item, ensure_ascii=False) + "\n")
#     with open(DEV_PATH, "w", encoding="utf-8") as f:
#         for item in dev_data:
#             f.write(json.dumps(item, ensure_ascii=False) + "\n")
