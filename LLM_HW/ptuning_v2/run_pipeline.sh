#!/bin/bash
# ============================================================
# Full p-tuning v2 pipeline for ChatGLM3-6B-32K
# Based on course slides – run this on the GPU server (RunPod)
# ============================================================
set -e

# ── config – edit these paths ─────────────────────────────────
MODEL_PATH="/hy-tmp/ZhipuAI/chatglm3-6b-32k"
DATA_DIR="/hy-tmp"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# ChatGLM2-6B repo lives one level above ptuning_v2/
REPO_DIR="$(dirname "$SCRIPT_DIR")/ChatGLM2-6B"
# ─────────────────────────────────────────────────────────────

echo "========================================"
echo "Step 1: Data already processed locally"
echo "  train.json and dev.json are in: $SCRIPT_DIR"
echo "========================================"

# Copy data to /hy-tmp so the training script can find them
cp "$SCRIPT_DIR/train.json" "$DATA_DIR/train.json"
cp "$SCRIPT_DIR/dev.json"   "$DATA_DIR/dev.json"
echo "[OK] Copied train.json and dev.json to $DATA_DIR"

echo ""
echo "========================================"
echo "Step 2: Download model from ModelScope"
echo "========================================"
python3 - <<'PYEOF'
from modelscope import snapshot_download
import os

model_id  = "ZhipuAI/chatglm3-6b-32k"
cache_dir = "/hy-tmp"

if not os.path.isdir(f"{cache_dir}/ZhipuAI/chatglm3-6b-32k"):
    print(f"Downloading {model_id} ...")
    model_dir = snapshot_download(model_id, cache_dir=cache_dir)
    print(f"Model saved to: {model_dir}")
else:
    print("Model already exists, skipping download.")
PYEOF

echo ""
echo "========================================"
echo "Step 3: Clone ChatGLM2-6B repo"
echo "========================================"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/THUDM/ChatGLM2-6B.git "$REPO_DIR"
else
    echo "[SKIP] $REPO_DIR already exists"
fi

echo ""
echo "========================================"
echo "Step 4: Patch repo (build_prompt + bug fix)"
echo "========================================"
python3 "$SCRIPT_DIR/patch_repo.py" \
    --repo_dir   "$REPO_DIR" \
    --model_path "$MODEL_PATH" \
    --train_file "$DATA_DIR/train.json" \
    --dev_file   "$DATA_DIR/dev.json"

echo ""
echo "========================================"
echo "Step 5: Enter ptuning directory"
echo "========================================"
cd "$REPO_DIR/ptuning"
echo "[OK] Now in: $(pwd)"

echo ""
echo "========================================"
echo "Step 7: Install packages"
echo "========================================"
pip install datasets==4.4.1
pip install transformers==4.30.2
pip install jieba           # required by main.py for ROUGE scoring
pip install rouge-chinese   # required by main.py for ROUGE scoring
pip install nltk
pip install sentencepiece
pip install accelerate

echo ""
echo "========================================"
echo "Step 8: Start training"
echo "========================================"
sh train.sh
