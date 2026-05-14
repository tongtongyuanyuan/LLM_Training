# RunPod Runbook — P-Tuning v2 (ChatGLM3-6B-32K)

Step-by-step record of what we actually ran to get training working,
including every bug we hit and how we fixed it.

---

## Environment

| Item | Value |
|------|-------|
| GPU | RunPod RTX 4090 24 GB |
| CUDA | 12.8 |
| Python | 3.11 |
| Working dir | `/workspace/LLM_Training/ptuning_v2` |
| Model dir | `/hy-tmp/ZhipuAI/chatglm3-6b-32k` |
| Output dir | `/hy-tmp/output/chatglm3-6b-32k-pt-128-2e-2` |

---

## Step 0 — Get the code on RunPod

```bash
cd /workspace
git clone https://github.com/tongtongyuanyuan/LLM_Training.git
cd LLM_Training/ptuning_v2
```

> Repo is public — no token needed.

---

## Step 1 — Install base dependencies

```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> RunPod pip is slow (~17 kB/s on default PyPI). Always add `-i https://pypi.tuna.tsinghua.edu.cn/simple`.

---

## Step 2 — Download ChatGLM3-6B-32K

```bash
python run.py download
```

**Bug:** If you ran this before and it failed partway through, the directory
`/hy-tmp/ZhipuAI/chatglm3-6b-32k` already exists so the script prints
`[SKIP] Model already exists` and exits — even though the download was incomplete.

**Fix:** Delete the incomplete directory and re-download:
```bash
rm -rf /hy-tmp/ZhipuAI/chatglm3-6b-32k
python run.py download
```

**Verify download is complete:**
```bash
ls /hy-tmp/ZhipuAI/chatglm3-6b-32k/*.bin | wc -l
# should print 7
```

---

## Step 3 — Clone ChatGLM2-6B repo and patch it

```bash
python run.py setup
```

This does:
- `git clone https://github.com/THUDM/ChatGLM2-6B.git` into `../ChatGLM2-6B/`
- Patches `tokenization_chatglm.py` — adds `build_prompt()` as a tokenizer class method
- Patches `modeling_chatglm.py` — adds module-level `build_prompt()` + fixes a `response.split` bug
- Writes `ChatGLM2-6B/ptuning/train.sh` with the correct paths

---

## Step 4 — Fix `use_auth_token` in main.py

```bash
sed -i '/use_auth_token/d' /workspace/LLM_Training/ChatGLM2-6B/ptuning/main.py
```

> ChatGLM2-6B's `main.py` passes `use_auth_token` to `load_dataset()` — removed in datasets 3.0+.
> `patch_repo.py` now does this automatically for fresh setups (via `python run.py setup`),
> but if ChatGLM2-6B was cloned before this fix, run the sed command manually.

---

## Step 5 — Install training dependencies

```bash
pip install datasets==3.2.0 transformers==4.30.2 \
            jieba rouge-chinese nltk sentencepiece accelerate \
            -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **datasets version rules:**
> - `datasets >= 3.0` required — pyarrow 24.x (pre-installed on RunPod) removed `PyExtensionType`
>   which `datasets 2.x` depends on → causes `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`
> - `datasets 2.x` is too old for pyarrow 24.x
> - After removing `use_auth_token` from `main.py` (Step 4), `datasets 3.x` works fine

---

## Step 6 — Generate training data

```bash
python data_process.py
```

Output:
```
Reading: /workspace/LLM_Training/alpaca_zh_medical.json
Total records: 70000
Train: 66500  |  Dev: 3500
Saved → /hy-tmp/train.json
Saved → /hy-tmp/dev.json
```

> Data source: `alpaca_zh_medical.json` in the repo root (47 MB, already in alpaca format).
> Output goes to `/hy-tmp/` which is where `train.sh` expects it.

---

## Step 7 — Run training

```bash
python run.py train
```

Expected output at start:
```
============================================================
Trainer : ChatGLM3PtuningTrainer
Model   : /hy-tmp/ZhipuAI/chatglm3-6b-32k
Method  : P-Tuning v2  (pre_seq_len=128)
Output  : /hy-tmp/output/chatglm3-6b-32k-pt-128-2e-2
============================================================
```

Training runs for **3000 steps** (~3 hours on RTX 3090, ~2 hours on RTX 4090).
Checkpoints saved every 1000 steps to `/hy-tmp/output/chatglm3-6b-32k-pt-128-2e-2/`.

---

## Step 8 — Inference (after training completes)

```bash
# Interactive CLI chat
python run.py infer

# Gradio web UI (open port 7860 in RunPod → Connect → HTTP Service)
python run.py serve
```

---

## Bugs We Hit (summary)

| Error | Cause | Fix |
|-------|-------|-----|
| `[SKIP] Model already exists` on re-download | Directory exists from failed download | `rm -rf /hy-tmp/ZhipuAI/chatglm3-6b-32k && python run.py download` |
| `FileNotFoundError: /hy-tmp/train.json` | `data_process.py` not run yet | `python data_process.py` |
| `TypeError: JsonConfig.__init__() got unexpected keyword argument 'use_auth_token'` | `main.py` passes deprecated param removed in datasets 3.0 | `sed -i '/use_auth_token/d' .../ChatGLM2-6B/ptuning/main.py` then use `datasets==3.2.0` |
| `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'` | `datasets 2.x` incompatible with pyarrow 24.x (pre-installed on RunPod) | Use `datasets==3.2.0` instead of 2.x |
| `ModuleNotFoundError: No module named 'modelscope'` | Base dep not installed | `pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| pip times out / 17 kB/s | RunPod slow PyPI | Add `-i https://pypi.tuna.tsinghua.edu.cn/simple` to every pip install |
| `data_process.py` used hardcoded local Mac path | Old code | Fixed: now reads `alpaca_zh_medical.json` from repo root by default |

---

## TODO / What's Next

- [ ] Qwen3-8B LoRA fine-tuning (`cd qwen3 && python run.py Qwen3_8B_LoRA train`)
- [ ] Qwen3-8B QLoRA fine-tuning (`python run.py Qwen3_8B_QLoRA train`)
- [ ] Qwen3-32B QLoRA fine-tuning (needs 2× RTX 4090 or A100)
- [ ] Test inference / Gradio web demo after P-Tuning training completes
- [ ] Compare outputs between P-Tuning v2, LoRA, and QLoRA on same medical questions

---

## Key File Locations

| File | Purpose |
|------|---------|
| `ptuning_v2/trainer.py` | `ChatGLM3PtuningTrainer` class — hyperparams live here |
| `ptuning_v2/patch_repo.py` | Patches ChatGLM2-6B repo (run via `python run.py setup`) |
| `ptuning_v2/data_process.py` | Splits `alpaca_zh_medical.json` → train/dev JSONL |
| `ptuning_v2/inference.py` | CLI inference script |
| `ptuning_v2/web_demo.py` | Gradio chat UI |
| `alpaca_zh_medical.json` | 70k medical QA records (alpaca format) |
| `/hy-tmp/train.json` | Generated training data (66.5k records) |
| `/hy-tmp/dev.json` | Generated dev data (3.5k records) |
| `/hy-tmp/ZhipuAI/chatglm3-6b-32k` | Base model weights |
| `/hy-tmp/output/chatglm3-6b-32k-pt-128-2e-2/` | Training checkpoints |
