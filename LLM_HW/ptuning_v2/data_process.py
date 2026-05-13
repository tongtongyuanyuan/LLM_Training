"""
Convert medical.json (knowledge graph format) into p-tuning v2 instruction-tuning format.
Output: train.json and dev.json (JSONL, one record per line)

Each disease record generates multiple QA pairs covering:
  - description, symptoms, cause, prevention, department, cure_way, check, drugs
"""

import json
import random
import os

MEDICAL_JSON_PATH = (
    "/Users/tongxue/Desktop/AI/AI_Course/三期课程/"
    "知识图谱day16/作业/medicalGraph/medical.json/medical.json"
)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.json")
DEV_PATH   = os.path.join(OUTPUT_DIR, "dev.json")

INSTRUCTION = "现在你是一个专业的医疗助手，请根据患者的问题给出专业的医疗建议："

DEV_RATIO = 0.05
RANDOM_SEED = 42


def fmt_list(items):
    if not items:
        return None
    cleaned = [str(i).strip() for i in items if str(i).strip()]
    return "、".join(cleaned) if cleaned else None


def build_pairs(record):
    name = record.get("name", "").strip()
    if not name:
        return []

    pairs = []

    # 1. 疾病描述
    desc = (record.get("desc") or "").strip()
    if desc:
        pairs.append({
            "instruction": INSTRUCTION,
            "input": f"请问{name}是什么疾病？",
            "output": desc,
        })

    # 2. 症状
    syms = fmt_list(record.get("symptom"))
    if syms:
        pairs.append({
            "instruction": INSTRUCTION,
            "input": f"患有{name}会有哪些症状？",
            "output": f"{name}的常见症状包括：{syms}。",
        })

    # 3. 病因
    cause = (record.get("cause") or "").strip()
    if cause:
        pairs.append({
            "instruction": INSTRUCTION,
            "input": f"{name}是什么原因引起的？",
            "output": cause,
        })

    # 4. 预防
    prevent = (record.get("prevent") or "").strip()
    if prevent:
        pairs.append({
            "instruction": INSTRUCTION,
            "input": f"如何预防{name}？",
            "output": prevent,
        })

    # 5. 就诊科室
    dept = fmt_list(record.get("cure_department"))
    if dept:
        pairs.append({
            "instruction": INSTRUCTION,
            "input": f"得了{name}应该去哪个科室就诊？",
            "output": f"建议前往{dept}就诊。",
        })

    # 6. 治疗方式
    cure = fmt_list(record.get("cure_way"))
    if cure:
        pairs.append({
            "instruction": INSTRUCTION,
            "input": f"{name}有哪些治疗方法？",
            "output": f"{name}的治疗方式包括：{cure}。",
        })

    # 7. 检查项目
    checks = fmt_list(record.get("check"))
    if checks:
        pairs.append({
            "instruction": INSTRUCTION,
            "input": f"诊断{name}通常需要做哪些检查？",
            "output": f"诊断{name}一般需要进行以下检查：{checks}。",
        })

    # 8. 推荐药物
    drugs = fmt_list(record.get("recommand_drug"))
    if drugs:
        pairs.append({
            "instruction": INSTRUCTION,
            "input": f"治疗{name}有哪些常用药物？",
            "output": f"治疗{name}常用的药物包括：{drugs}。",
        })

    # 9. 治疗周期
    duration = (record.get("cure_lasttime") or "").strip()
    if duration:
        pairs.append({
            "instruction": INSTRUCTION,
            "input": f"{name}的治疗周期大概是多久？",
            "output": f"{name}的治疗周期一般为{duration}。",
        })

    # 10. 治愈率
    cured_prob = (record.get("cured_prob") or "").strip()
    if cured_prob:
        pairs.append({
            "instruction": INSTRUCTION,
            "input": f"{name}的治愈率大概是多少？",
            "output": f"{name}的治愈率约为{cured_prob}。",
        })

    return pairs


def main():
    print(f"Reading: {MEDICAL_JSON_PATH}")
    with open(MEDICAL_JSON_PATH, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"Total disease records: {len(lines)}")

    all_pairs = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        all_pairs.extend(build_pairs(record))

    print(f"Total QA pairs generated: {len(all_pairs)}")

    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)

    split = int(len(all_pairs) * (1 - DEV_RATIO))
    train_data = all_pairs[:split]
    dev_data   = all_pairs[split:]

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
    for sample in train_data[:3]:
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
