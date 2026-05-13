"""
ChatGLM3-6B-32K  P-Tuning v2 trainer class.

Mirrors the qwen3/ class design so both parts of the project
share the same workflow:
    python run.py <action>

Actions:
    download   download chatglm3-6b-32k from ModelScope
    setup      clone ChatGLM2-6B repo + apply patches + write train.sh
    train      run sh train.sh inside ChatGLM2-6B/ptuning/
    infer      interactive inference (merges prefix encoder into base model)

Environment variables (all optional):
    MODEL_BASE_DIR   /hy-tmp          directory where the model is stored
    OUTPUT_BASE_DIR  /hy-tmp/output   directory for checkpoints
    TRAIN_FILE       /hy-tmp/train.json
    DEV_FILE         /hy-tmp/dev.json

How it maps to the course slides (4.pdf):
    Step 2  →  setup()   clones ChatGLM2-6B repo
    Step 3  →  setup()   patch_repo.py inserts build_prompt()
    Step 4  →  setup()   patch_repo.py patches tokenization + modeling files
    Step 5  →  train()   enters ptuning/ directory before running
    Step 6  →  setup()   patch_repo.py writes train.sh with correct paths
    Step 7  →  (run pip install manually before calling train)
    Step 8  →  train()   sh train.sh
    Infer   →  infer()   merges prefix encoder, loads model, interactive chat
"""

import os
import subprocess
import sys

MODEL_BASE_DIR  = os.environ.get("MODEL_BASE_DIR",  "/hy-tmp")
OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/hy-tmp/output")

# LLM_HW/ root — go up one level from ptuning_v2/
_HW_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ChatGLM3PtuningTrainer:
    """P-Tuning v2 fine-tuning for ChatGLM3-6B-32K on Chinese medical data.

    Reference: ChatGLM2-6B/ptuning/main.py
    - Patches tokenization_chatglm.py  →  adds build_prompt() as tokenizer method
    - Patches modeling_chatglm.py      →  adds module-level build_prompt() + fixes
                                          response.split bug at line ~1015
    - Custom PrefixTrainer saves ONLY the prefix encoder weights (requires_grad=True)
      → checkpoint is tiny (~20 MB vs 12 GB for full model)
    - main.py reads `--prompt_column` as the question field and
      `--response_column` as the answer field in our JSONL data
    """

    # Model
    model_id  = "ZhipuAI/chatglm3-6b-32k"
    model_dir = "ZhipuAI/chatglm3-6b-32k"

    # P-Tuning v2 hyperparameters (match CLAUDE.md defaults)
    pre_seq_len                 = 128
    learning_rate               = "2e-2"
    max_steps                   = 3000
    max_source_length           = 64
    max_target_length           = 128
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 16
    quantization_bit            = 4
    logging_steps               = 10
    save_steps                  = 1000
    preprocessing_num_workers   = 10

    # Data column names — must match keys in train.json / dev.json
    prompt_column   = "input"    # the question field
    response_column = "output"   # the answer field

    # ── derived paths ──────────────────────────────────────────────────────────

    @property
    def model_path(self) -> str:
        return os.path.join(MODEL_BASE_DIR, self.model_dir)

    @property
    def repo_dir(self) -> str:
        return os.path.join(_HW_ROOT, "ChatGLM2-6B")

    @property
    def ptuning_dir(self) -> str:
        return os.path.join(self.repo_dir, "ptuning")

    @property
    def output_dir(self) -> str:
        return os.path.join(
            OUTPUT_BASE_DIR,
            f"chatglm3-6b-32k-pt-{self.pre_seq_len}-{self.learning_rate}"
        )

    @property
    def train_file(self) -> str:
        return os.environ.get("TRAIN_FILE", "/hy-tmp/train.json")

    @property
    def dev_file(self) -> str:
        return os.environ.get("DEV_FILE", "/hy-tmp/dev.json")

    @property
    def latest_checkpoint(self) -> str:
        """Most recent checkpoint-N directory under output_dir."""
        if not os.path.isdir(self.output_dir):
            return self.output_dir
        checkpoints = sorted(
            [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        return os.path.join(self.output_dir, checkpoints[-1]) if checkpoints else self.output_dir

    # ── actions ────────────────────────────────────────────────────────────────

    def download(self):
        """Step 0 – download chatglm3-6b-32k from ModelScope."""
        from modelscope import snapshot_download
        if os.path.isdir(self.model_path):
            print(f"[SKIP] Model already exists: {self.model_path}")
            return
        print(f"Downloading {self.model_id} → {MODEL_BASE_DIR}")
        model_dir = snapshot_download(self.model_id, cache_dir=MODEL_BASE_DIR)
        print(f"Saved to: {model_dir}")

    def setup(self):
        """Steps 2–6 – clone repo, patch files, write train.sh.

        Corresponds to course slides steps 2-4 and 6:
          - git clone ChatGLM2-6B
          - patch tokenization_chatglm.py  (insert build_prompt method)
          - patch modeling_chatglm.py      (insert build_prompt func + split fix)
          - write ptuning/train.sh with correct model/data paths
        """
        # Clone (skip if already present — user may have it locally)
        if not os.path.isdir(self.repo_dir):
            print(f"Cloning ChatGLM2-6B → {self.repo_dir}")
            subprocess.run(
                ["git", "clone", "https://github.com/THUDM/ChatGLM2-6B.git",
                 self.repo_dir],
                check=True
            )
        else:
            print(f"[SKIP] Repo already exists: {self.repo_dir}")

        # Run patch_repo.py
        patch_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "patch_repo.py")
        subprocess.run([
            sys.executable, patch_script,
            "--repo_dir",   self.repo_dir,
            "--model_path", self.model_path,
            "--train_file", self.train_file,
            "--dev_file",   self.dev_file,
        ], check=True)

        print(f"\n[OK] Setup done — train.sh written to {self.ptuning_dir}")
        print(f"\nNext: install packages, then run train")
        print(f"  pip install jieba rouge-chinese nltk sentencepiece accelerate")
        print(f"  pip install datasets==4.4.1 transformers==4.30.2")

    def train(self):
        """Step 8 – run sh train.sh inside ChatGLM2-6B/ptuning/."""
        train_sh = os.path.join(self.ptuning_dir, "train.sh")
        if not os.path.isfile(train_sh):
            raise RuntimeError(
                f"train.sh not found at {train_sh}\n"
                "Run 'python run.py setup' first."
            )
        print(f"\n{'='*60}")
        print(f"Trainer : ChatGLM3PtuningTrainer")
        print(f"Model   : {self.model_path}")
        print(f"Method  : P-Tuning v2  (pre_seq_len={self.pre_seq_len})")
        print(f"Output  : {self.output_dir}")
        print(f"{'='*60}\n")
        # Must run from inside ptuning/ because main.py has relative imports
        subprocess.run(["sh", "train.sh"], check=True, cwd=self.ptuning_dir)

    def infer(self, checkpoint_path: str = None):
        """CLI inference – merge prefix encoder weights and chat interactively."""
        if checkpoint_path is None:
            checkpoint_path = self.latest_checkpoint
            print(f"Using checkpoint: {checkpoint_path}")

        infer_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "inference.py")
        subprocess.run([
            sys.executable, infer_script,
            "--model_path",      self.model_path,
            "--checkpoint_path", checkpoint_path,
            "--pre_seq_len",     str(self.pre_seq_len),
        ], check=True)

    def serve(self, checkpoint_path: str = None, port: int = 7860, share: bool = False):
        """Gradio web chat UI — accessible in browser on RunPod via port 7860.

        On RunPod: Connect → HTTP Service → port 7860
        """
        if checkpoint_path is None:
            checkpoint_path = self.latest_checkpoint
            print(f"Using checkpoint: {checkpoint_path}")

        demo_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "web_demo.py")
        cmd = [
            sys.executable, demo_script,
            "--model_path",      self.model_path,
            "--checkpoint_path", checkpoint_path,
            "--pre_seq_len",     str(self.pre_seq_len),
            "--server_port",     str(port),
        ]
        if share:
            cmd.append("--share")
        subprocess.run(cmd, check=True)
