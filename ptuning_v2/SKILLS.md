# Part 1 — ChatGLM3-6B-32K + P-Tuning v2

## What is ChatGLM3-6B-32K?

| Part | Meaning |
|------|---------|
| **ChatGLM3** | Tsinghua University + Zhipu AI's 3rd-generation Chinese-English bilingual chat model |
| **6B** | 6 billion parameters — mid-size, fits on a single GPU (RTX 4090 24 GB) |
| **32K** | 32,768 token context window — handles very long conversations or documents |

The base model is already pre-trained on massive Chinese + English text and knows how to chat.
We fine-tune it on medical dialogue data to make it a specialized medical assistant.

---

## What is P-Tuning v2?

Normal fine-tuning updates all model weights — expensive, requires huge VRAM, risks forgetting.

P-Tuning v2 instead:

1. **Freezes** all 6 billion original weights — they never change
2. **Adds trainable prefix embeddings** to every Transformer layer
3. Only those prefix embeddings (~0.1% of total params) are trained

```
Normal fine-tuning:  update 6,000,000,000 params  → 12 GB checkpoint
P-Tuning v2:         update ~6,000,000 params      → ~20 MB checkpoint
```

The prefix embeddings act like a "soft prompt" that steers the frozen model
toward medical domain behavior without rewriting its knowledge.

---

## What We're Training

- **Base model:** ChatGLM3-6B-32K (frozen)
- **Data:** 66,500 Chinese medical QA pairs from `alpaca_zh_medical.json`
- **Task:** Given a patient question → generate a professional medical answer
- **Method:** P-Tuning v2 via the ChatGLM2-6B training repo

Example training pair:
```json
{
  "instruction": "现在你是一个肛肠医生,请根据患者的问题给出建议:",
  "input": "用啥药医治肛门瘙痒",
  "output": "可用适当抗生素或者抗菌药剂"
}
```

---

## Key Hyperparameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `pre_seq_len` | 128 | Length of the prefix — larger = more params + more VRAM |
| `learning_rate` | 2e-2 | Higher than LoRA because only prefix is trained |
| `max_steps` | 3000 | Total training steps (~2-3 hours on RTX 4090) |
| `quantization_bit` | 4 | Load base model in 4-bit to save VRAM |
| `gradient_accumulation_steps` | 16 | Effective batch size = 1 × 16 = 16 |

---

## How It Compares to LoRA / QLoRA

| Method | What's Trained | Checkpoint Size | VRAM (6-8B) |
|--------|---------------|-----------------|-------------|
| P-Tuning v2 | Prefix embeddings only | ~20 MB | ~8 GB |
| LoRA | Low-rank adapter matrices (A, B) | ~100 MB | ~16 GB |
| QLoRA | Same as LoRA + 4-bit base model | ~100 MB | ~10 GB |

P-Tuning v2 is the most memory-efficient but works best for focused tasks
(like medical QA) where domain shift is the main goal.
