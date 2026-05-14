# LLM Fine-Tuning Homework

## Goal

Practice three PEFT (Parameter-Efficient Fine-Tuning) methods on Chinese medical dialogue data:

1. **P-Tuning v2** — ChatGLM3-6B-32K (Tsinghua / Zhipu AI)
2. **LoRA** — Qwen3-8B and optionally Qwen3-32B (Alibaba)
3. **QLoRA** — same Qwen3 models with 4-bit quantization

Target hardware: **RunPod RTX 4090 (24 GB VRAM)**

---

## Project Layout

```
LLM_HW/
├── CLAUDE.md
├── ptuning_v2/                  # Part 1 – ChatGLM3 P-Tuning v2
│   ├── data_process.py          #   convert medical.json → train/dev JSONL
│   ├── patch_repo.py            #   patch ChatGLM2-6B repo (build_prompt + bug fix)
│   ├── run_pipeline.sh          #   full end-to-end pipeline script
│   └── inference.py             #   interactive CLI inference
└── qwen3/                       # Part 2 & 3 – Qwen3 LoRA / QLoRA
    ├── download.py              #   download Qwen3-8B or 32B from ModelScope
    ├── data_process.py          #   convert JSONL → alpaca JSON for LLaMA-Factory
    ├── trainer.py               #   BaseQwen3Trainer + 4 concrete subclasses
    └── run.py                   #   CLI: python run.py <TrainerName> <action>
```

---

## Part 1 — ChatGLM3-6B-32K P-Tuning v2

### What is P-Tuning v2?

- Freezes all original model weights
- Adds trainable **prefix embeddings** to every Transformer layer
- Only ~0.1% of parameters are trained
- Very memory-efficient; checkpoint is tiny

### Setup & Run (on RunPod)

```bash
# 1. Clone the training repo
git clone https://github.com/THUDM/ChatGLM2-6B.git

# 2. Install deps
pip install transformers==4.30.2 datasets==4.4.1 sentencepiece accelerate nltk

# 3. Run the full pipeline (data copy → patch → install → train)
cd ptuning_v2
sh run_pipeline.sh
```

`run_pipeline.sh` does steps 1–8 from the course slides automatically.
Model path defaults to `/hy-tmp/ZhipuAI/chatglm3-6b-32k`.

### Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `PRE_SEQ_LEN` | 128 | prefix length; larger = more params + more VRAM |
| `LR` | 2e-2 | P-Tuning uses higher LR than LoRA |
| `--max_steps` | 3000 | total training steps |
| `--quantization_bit` | 4 | load base model in 4-bit to save VRAM |

### Inference

```bash
python inference.py \
    --model_path      /hy-tmp/ZhipuAI/chatglm3-6b-32k \
    --checkpoint_path /hy-tmp/output/chatglm3-6b-32k-pt-128-2e-2/checkpoint-3000
```

---

## Part 2 & 3 — Qwen3 LoRA / QLoRA (LLaMA-Factory)

### What is LoRA?

- Freezes all original weights
- Inserts small trainable **low-rank adapter matrices** (A and B) beside each Linear layer
- ΔW = BA  where A is rank-r down-projection, B is rank-r up-projection
- B is initialised to 0 so training starts from the original output
- At inference: merge BA into W — zero latency overhead

### What is QLoRA?

- Same as LoRA, but the **base model is loaded in 4-bit** (via bitsandbytes)
- Activations are computed in bf16; only the tiny LoRA adapters are trained in full precision
- ~4× lower VRAM for the base model; enables fine-tuning 32B on a single GPU

### The Four Trainer Classes

| Class | Model | Method | VRAM (approx) |
|-------|-------|--------|--------------|
| `Qwen3_8B_LoRA`   | Qwen3-8B  | LoRA      | ~16 GB |
| `Qwen3_8B_QLoRA`  | Qwen3-8B  | QLoRA 4-bit | ~10 GB |
| `Qwen3_32B_LoRA`  | Qwen3-32B | LoRA      | >24 GB — needs multi-GPU or offload |
| `Qwen3_32B_QLoRA` | Qwen3-32B | QLoRA 4-bit | ~20 GB — feasible on single 4090 |

### Workflow on RunPod

```bash
# ── install once ──────────────────────────────────────────────
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /hy-tmp/LLaMA-Factory
pip install --no-deps -e /hy-tmp/LLaMA-Factory
pip install torch==2.8.0 transformers==4.57.5 datasets==4.0.0 \
            accelerate==1.11.0 peft==0.18.0 trl==0.24.0 \
            deepspeed==0.18.2 bitsandbytes==0.48.2

# ── prepare data (run once) ───────────────────────────────────
cd qwen3
python data_process.py --input /hy-tmp/train_medical.json

# ── download model (run once per size) ───────────────────────
python run.py Qwen3_8B_LoRA download     # downloads Qwen3-8B
python run.py Qwen3_32B_QLoRA download   # downloads Qwen3-32B

# ── train ─────────────────────────────────────────────────────
python run.py Qwen3_8B_LoRA  train
python run.py Qwen3_8B_QLoRA train
python run.py Qwen3_32B_QLoRA train      # needs gradient_checkpointing, cutoff=512

# ── interactive test ──────────────────────────────────────────
python run.py Qwen3_8B_LoRA infer

# ── merge adapter into base model ─────────────────────────────
python run.py Qwen3_8B_LoRA export
```

### Overriding Environment Paths

```bash
export LLAMAFACTORY_DIR=/hy-tmp/LLaMA-Factory
export MODEL_BASE_DIR=/hy-tmp
export OUTPUT_BASE_DIR=/hy-tmp/output
```

### Key Hyperparameters

| Parameter | 8B default | 32B default | Effect |
|-----------|-----------|------------|--------|
| `lora_rank` | 8 | 8 | rank of ΔW; higher = more params + quality |
| `cutoff_len` | 2048 | 512 | max sequence length; 32B needs shorter |
| `gradient_accumulation_steps` | 8 | 16 | effective batch size multiplier |
| `quantization_bit` | — | — | set 4 for QLoRA |
| `gradient_checkpointing` | False | True | trades speed for VRAM on 32B |

---

## Comparison

| Method | Trainable Params | VRAM | Speed | Quality |
|--------|-----------------|------|-------|---------|
| P-Tuning v2 | prefix encoder only | lowest | fast | good for focused tasks |
| LoRA | adapter A + B | medium | medium | strong, industry standard |
| QLoRA | adapter A + B (4-bit base) | lowest | slightly slower | close to LoRA |

---

## Training Results Reference

### P-Tuning v2 (ChatGLM3-6B-32K, RTX 3090 24 GB)

```
train_loss               = 3.2058
train_runtime            = 2:47:26
train_samples_per_second = 4.778
```

### LoRA (Qwen3-4B, RTX 5060 Ti 16 GB)

```
epoch       = 3.0
train_loss  = 2.4437
train_runtime = 0:28:19
```

### QLoRA (Qwen3-4B, RTX 5060 Ti 16 GB)

```
epoch       = 3.0
train_loss  = 2.3498
train_runtime = 0:31:08
```
