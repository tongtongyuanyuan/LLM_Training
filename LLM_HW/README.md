# LLM Fine-Tuning — P-Tuning v2 · LoRA · QLoRA

Practice project for PEFT (Parameter-Efficient Fine-Tuning) on Chinese medical dialogue data.

| Part | Model | Method | VRAM needed | Single RTX 4090 (24 GB) |
|------|-------|--------|-------------|------------------------|
| 1 | ChatGLM3-6B-32K | P-Tuning v2 | ~8 GB | ✅ 没问题 |
| 2 | Qwen3-8B | LoRA (bf16) | ~16 GB | ✅ 够用 |
| 3 | Qwen3-8B | QLoRA 4-bit | ~10 GB | ✅ 轻松 |
| 4 | Qwen3-32B | QLoRA 4-bit | ~21–23 GB | ⚠️ 极限，OOM风险大 |
| — | Qwen3-32B | LoRA (bf16) | ~64 GB | ❌ 放不下 |

> **32B 推荐配置**：RunPod 2× RTX 4090 (48 GB) 或 1× A100/H100 80 GB

---

## Project Layout

```
LLM_HW/
├── README.md
├── CLAUDE.md                        project notes
├── ptuning_v2/                      Part 1 — ChatGLM3 P-Tuning v2
│   ├── data_process.py              convert medical knowledge graph → JSONL
│   ├── patch_repo.py                patch ChatGLM2-6B repo (build_prompt + bug fix)
│   ├── inference.py                 CLI inference
│   ├── web_demo.py                  Gradio chat UI
│   ├── trainer.py                   ChatGLM3PtuningTrainer class
│   ├── run.py                       CLI entry point
│   └── run_pipeline.sh              all-in-one bash pipeline
└── qwen3/                           Parts 2-4 — Qwen3 LoRA / QLoRA
    ├── data_process.py              prepare data for LLaMA-Factory
    ├── download.py                  download Qwen3-8B or 32B
    ├── trainer.py                   Qwen3_8B_LoRA / QLoRA / 32B_LoRA / QLoRA classes
    └── run.py                       CLI entry point
```

---

## Quick Start on RunPod (RTX 4090 24 GB)

### 1. Clone this repo

```bash
git clone https://github.com/tongtongyuanyuan/LLM_Training.git
cd LLM_Training
```

### 2. Install base dependencies

```bash
pip install modelscope
```

---

## Part 1 — ChatGLM3-6B-32K with P-Tuning v2

### What is P-Tuning v2?

Keeps all original model weights frozen. Adds trainable **prefix embeddings** to every Transformer layer (~0.1 % of total params). Checkpoint is tiny (~20 MB).

Uses the [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) training scripts with ChatGLM3 weights.

### Dependencies

```bash
pip install datasets==4.4.1 transformers==4.30.2 \
            jieba rouge-chinese nltk sentencepiece accelerate
```

### Step-by-step commands

```bash
cd ptuning_v2

# Step 1 — generate training data from the medical knowledge graph
python data_process.py

# Step 2 — download ChatGLM3-6B-32K from ModelScope
python run.py download

# Step 3 — clone ChatGLM2-6B repo, apply patches, write train.sh
#   patches tokenization_chatglm.py  → adds build_prompt() method
#   patches modeling_chatglm.py      → adds build_prompt() + fixes line 1015 bug
#   writes  ChatGLM2-6B/ptuning/train.sh with your model/data paths
python run.py setup

# Step 4 — run fine-tuning  (≈ 3 hours on RTX 3090)
python run.py train

# Step 5a — interactive CLI chat
python run.py infer

# Step 5b — Gradio web chat UI (open port 7860 in RunPod → Connect)
python run.py serve
python run.py serve --port 7860 --share     # --share gives a public link

# Resume from a specific checkpoint
python run.py infer --checkpoint /hy-tmp/output/chatglm3-6b-32k-pt-128-2e-2/checkpoint-3000
```

### One-shot pipeline script

```bash
# Runs all steps 1-8 from the course slides in one go
bash run_pipeline.sh
```

### Key hyperparameters (edit in `trainer.py`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `pre_seq_len` | 128 | prefix length — larger = more params + more VRAM |
| `learning_rate` | 2e-2 | P-Tuning uses higher LR than LoRA |
| `max_steps` | 3000 | total training steps |
| `quantization_bit` | 4 | load base model in 4-bit to save VRAM |

### Environment variables

```bash
export MODEL_BASE_DIR=/hy-tmp          # where chatglm3-6b-32k is stored
export OUTPUT_BASE_DIR=/hy-tmp/output  # where checkpoints are saved
export TRAIN_FILE=/hy-tmp/train.json
export DEV_FILE=/hy-tmp/dev.json
```

---

## Part 2 — Qwen3-8B with LoRA

### What is LoRA?

Freezes original weights. Inserts small **low-rank adapter matrices** (A · B) beside each Linear layer. ΔW = B·A where rank r ≪ full dimension. Merge adapter into base model at inference — zero latency overhead.

Uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training.

### Dependencies

```bash
pip install torch==2.8.0 transformers==4.57.5 datasets==4.0.0 \
            accelerate==1.11.0 peft==0.18.0 trl==0.24.0 \
            deepspeed==0.18.2 bitsandbytes==0.48.2

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /hy-tmp/LLaMA-Factory
pip install --no-deps -e /hy-tmp/LLaMA-Factory
llamafactory-cli version     # verify install
```

### Step-by-step commands

```bash
cd qwen3

# Step 1 — prepare data for LLaMA-Factory
python data_process.py \
    --input  /hy-tmp/train_medical.json \
    --llamafactory_dir /hy-tmp/LLaMA-Factory

# Step 2 — download Qwen3-8B
python run.py Qwen3_8B_LoRA download

# Step 3 — train  (≈ 28 min on RTX 5060 Ti 16 GB)
python run.py Qwen3_8B_LoRA train

# Step 4 — interactive chat
python run.py Qwen3_8B_LoRA infer

# Step 5 — merge adapter into base model
python run.py Qwen3_8B_LoRA export
```

---

## Part 3 — Qwen3-8B with QLoRA (4-bit)

### What is QLoRA?

Same as LoRA, but the **base model is loaded in 4-bit** via bitsandbytes. Reduces base model VRAM from ~16 GB → ~5 GB for 8B. Adapter is still trained in bf16.

```bash
cd qwen3

# download (same model as LoRA — skip if already done)
python run.py Qwen3_8B_QLoRA download

# train  (≈ 31 min on RTX 5060 Ti 16 GB)
python run.py Qwen3_8B_QLoRA train

# chat
python run.py Qwen3_8B_QLoRA infer

# merge
python run.py Qwen3_8B_QLoRA export
```

---

## Part 4 — Qwen3-32B with QLoRA (4-bit)

> **GPU requirement**: 2× RTX 4090 (48 GB) or 1× A100/H100 (80 GB) recommended.
> Single 4090 is very risky — 4-bit weights alone = 16 GB, plus activations pushes to 21-23 GB.
> `cutoff_len` is set to 256 to give maximum headroom on a single GPU.

```bash
cd qwen3

# download Qwen3-32B  (~65 GB, takes time)
python run.py Qwen3_32B_QLoRA download

# train  (gradient_checkpointing=True, cutoff_len=512 set automatically)
python run.py Qwen3_32B_QLoRA train

# chat
python run.py Qwen3_32B_QLoRA infer

# merge
python run.py Qwen3_32B_QLoRA export
```

Also available: `Qwen3_32B_LoRA` (needs multi-GPU or CPU offload — 32B in bf16 > 60 GB).

```bash
python run.py Qwen3_32B_LoRA train
```

---

## All Available Trainer Classes

```bash
# P-Tuning v2
cd ptuning_v2
python run.py download | setup | train | infer | serve

# LoRA / QLoRA — swap class name to change model + method
cd qwen3
python run.py Qwen3_8B_LoRA   download | train | infer | export
python run.py Qwen3_8B_QLoRA  download | train | infer | export
python run.py Qwen3_32B_LoRA  download | train | infer | export
python run.py Qwen3_32B_QLoRA download | train | infer | export
```

### Override paths via environment variables

```bash
export LLAMAFACTORY_DIR=/hy-tmp/LLaMA-Factory
export MODEL_BASE_DIR=/hy-tmp
export OUTPUT_BASE_DIR=/hy-tmp/output
```

---

## Method Comparison

| Method | Trainable params | VRAM (8B) | Training time | Quality |
|--------|-----------------|-----------|---------------|---------|
| P-Tuning v2 | prefix encoder only | lowest | fast | good for focused tasks |
| LoRA | adapter A + B | ~16 GB | medium | strong, industry standard |
| QLoRA | adapter A + B (4-bit base) | ~10 GB | slightly slower | close to LoRA |

---

## Training Results (reference)

### P-Tuning v2 — ChatGLM3-6B-32K (RTX 3090 24 GB, CUDA 12.8)

```
train_loss               = 3.2058
train_runtime            = 2:47:26
train_samples_per_second = 4.778
```

### LoRA — Qwen3-4B (RTX 5060 Ti 16 GB)

```
epoch       = 3.0
train_loss  = 2.4437
train_runtime = 0:28:19
```

### QLoRA — Qwen3-4B (RTX 5060 Ti 16 GB)

```
epoch       = 3.0
train_loss  = 2.3498
train_runtime = 0:31:08
```

---

## Data

Training data: [Chinese Medical Dialogue Dataset](https://github.com/Toyhom/Chinese-medical-dialogue-data) (internal medicine, surgery, etc.)

Generate your own JSONL from the medical knowledge graph:

```bash
python ptuning_v2/data_process.py   # → ptuning_v2/train.json + dev.json
python qwen3/data_process.py \
    --input ptuning_v2/train.json \
    --llamafactory_dir /hy-tmp/LLaMA-Factory
```
