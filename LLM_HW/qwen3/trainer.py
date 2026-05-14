"""
Qwen3 LoRA / QLoRA trainer classes.

Each class encodes one fine-tuning configuration.
Instantiate one and call .train(), .infer(), or .export().

Run via run.py:
    python run.py Qwen3_8B_LoRA  train
    python run.py Qwen3_8B_QLoRA train
    python run.py Qwen3_32B_LoRA  train
    python run.py Qwen3_32B_QLoRA train

Environment variables (all optional):
    LLAMAFACTORY_DIR   path to cloned LLaMA-Factory repo  (default: /hy-tmp/LLaMA-Factory)
    MODEL_BASE_DIR     where models are downloaded         (default: /hy-tmp)
    OUTPUT_BASE_DIR    where checkpoints are saved         (default: /hy-tmp/output)
"""

import os
import subprocess
import tempfile

import yaml


LLAMAFACTORY_DIR = os.environ.get("LLAMAFACTORY_DIR", "/hy-tmp/LLaMA-Factory")
MODEL_BASE_DIR   = os.environ.get("MODEL_BASE_DIR",   "/hy-tmp")
OUTPUT_BASE_DIR  = os.environ.get("OUTPUT_BASE_DIR",  "/hy-tmp/output")


# ── Base class ────────────────────────────────────────────────────────────────

class BaseQwen3Trainer:
    """Shared logic for all Qwen3 fine-tuning variants."""

    # --- override in subclasses ---
    model_id: str = ""       # ModelScope model ID  e.g. "Qwen/Qwen3-8B"
    model_dir: str = ""      # folder name under MODEL_BASE_DIR  e.g. "Qwen3-8B"
    lora_rank: int = 8
    cutoff_len: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 3.0
    learning_rate: float = 1e-4
    quantization_bit: int = None   # None → LoRA,  4 → QLoRA
    gradient_checkpointing: bool = False

    # --- shared defaults ---
    finetuning_type: str = "lora"
    lora_target: str = "all"
    template: str = "qwen3"
    dataset: str = "alpaca_zh_medical"
    max_samples: int = 1000
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 500

    # ── derived properties ─────────────────────────────────────────────────────

    @property
    def model_path(self) -> str:
        return os.path.join(MODEL_BASE_DIR, self.model_dir)

    @property
    def method(self) -> str:
        return "qlora" if self.quantization_bit else "lora"

    @property
    def output_dir(self) -> str:
        return os.path.join(OUTPUT_BASE_DIR, self.model_dir, self.method, "sft")

    # ── config builder ─────────────────────────────────────────────────────────

    def build_train_config(self) -> dict:
        cfg = {
            "model_name_or_path": self.model_path,
            "trust_remote_code": True,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": self.finetuning_type,
            "lora_rank": self.lora_rank,
            "lora_target": self.lora_target,
            "dataset": self.dataset,
            "template": self.template,
            "cutoff_len": self.cutoff_len,
            "max_samples": self.max_samples,
            "overwrite_cache": True,
            "preprocessing_num_workers": 16,
            "dataloader_num_workers": 4,
            "output_dir": self.output_dir,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "save_only_model": False,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "bf16": self.bf16,
            "ddp_timeout": 180000000,
        }
        if self.quantization_bit:
            cfg["quantization_bit"] = self.quantization_bit
        if self.gradient_checkpointing:
            cfg["gradient_checkpointing"] = True
        return cfg

    # ── internal helper ────────────────────────────────────────────────────────

    def _run_llamafactory(self, subcommand: str, config: dict):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir="/tmp"
        ) as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
            config_path = f.name
        try:
            cmd = ["llamafactory-cli", subcommand, config_path]
            print(f"\n[{self.__class__.__name__}] {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=LLAMAFACTORY_DIR)
        finally:
            os.unlink(config_path)

    # ── public actions ─────────────────────────────────────────────────────────

    def download(self):
        from modelscope import snapshot_download
        print(f"Downloading {self.model_id} → {MODEL_BASE_DIR}")
        model_dir = snapshot_download(self.model_id, cache_dir=MODEL_BASE_DIR)
        print(f"Saved to: {model_dir}")

    def train(self):
        print(f"\n{'='*60}")
        print(f"Trainer : {self.__class__.__name__}")
        print(f"Model   : {self.model_path}")
        print(f"Method  : {self.method.upper()}")
        print(f"Output  : {self.output_dir}")
        print(f"{'='*60}")
        self._run_llamafactory("train", self.build_train_config())

    def infer(self):
        cfg = {
            "model_name_or_path": self.model_path,
            "adapter_name_or_path": self.output_dir,
            "template": self.template,
            "infer_backend": "huggingface",
            "trust_remote_code": True,
        }
        self._run_llamafactory("chat", cfg)

    def export(self, export_dir: str = None):
        if export_dir is None:
            export_dir = self.output_dir + "_merged"
        cfg = {
            "model_name_or_path": self.model_path,
            "adapter_name_or_path": self.output_dir,
            "template": self.template,
            "finetuning_type": self.finetuning_type,
            "export_dir": export_dir,
            "export_size": 2,
            "export_legacy_format": False,
            "trust_remote_code": True,
        }
        self._run_llamafactory("export", cfg)


# ── Concrete trainer variants ─────────────────────────────────────────────────

class Qwen3_8B_LoRA(BaseQwen3Trainer):
    """Qwen3-8B  +  LoRA  (needs ~16 GB VRAM on a single 4090)."""
    model_id  = "Qwen/Qwen3-8B"
    model_dir = "Qwen3-8B"
    lora_rank = 8
    cutoff_len = 2048
    gradient_accumulation_steps = 8


class Qwen3_8B_QLoRA(BaseQwen3Trainer):
    """Qwen3-8B  +  QLoRA 4-bit  (needs ~10 GB VRAM — fits comfortably on 4090)."""
    model_id  = "Qwen/Qwen3-8B"
    model_dir = "Qwen3-8B"
    lora_rank = 8
    cutoff_len = 2048
    gradient_accumulation_steps = 8
    quantization_bit = 4


class Qwen3_32B_LoRA(BaseQwen3Trainer):
    """Qwen3-32B  +  LoRA.

    WARNING: bf16 base model = ~64 GB — does NOT fit on a single RTX 4090 (24 GB).
    Requires at least 2× A100 80 GB or equivalent.
    """
    model_id  = "Qwen/Qwen3-32B"
    model_dir = "Qwen3-32B"
    lora_rank = 8
    cutoff_len = 512
    gradient_accumulation_steps = 16
    gradient_checkpointing = True

    def train(self):
        import sys
        print(
            "\n[WARNING] Qwen3-32B LoRA requires ~64 GB VRAM (bf16 weights).\n"
            "A single RTX 4090 (24 GB) is NOT enough.\n"
            "Recommended: 2× A100 80 GB or H100 80 GB on RunPod.\n"
            "Use Qwen3_32B_QLoRA for a single 4090 (still very tight).\n"
        )
        confirm = input("Continue anyway? [y/N] ").strip().lower()
        if confirm != "y":
            sys.exit(0)
        super().train()


class Qwen3_32B_QLoRA(BaseQwen3Trainer):
    """Qwen3-32B  +  QLoRA 4-bit.

    4-bit weights = ~16 GB.  With activations + optimizer states ≈ 21-23 GB total.
    Single RTX 4090 (24 GB): possible but extremely tight — OOM risk is real.
    Recommended GPU: 2× RTX 4090 (48 GB) or 1× A100/H100 80 GB.
    cutoff_len is set to 256 (not 512) to leave maximum headroom.
    """
    model_id  = "Qwen/Qwen3-32B"
    model_dir = "Qwen3-32B"
    lora_rank = 8
    cutoff_len = 256        # reduced from 512 — critical for single-GPU survival
    gradient_accumulation_steps = 16
    quantization_bit = 4
    gradient_checkpointing = True


# ── Registry ──────────────────────────────────────────────────────────────────

TRAINERS: dict[str, type[BaseQwen3Trainer]] = {
    "Qwen3_8B_LoRA":   Qwen3_8B_LoRA,
    "Qwen3_8B_QLoRA":  Qwen3_8B_QLoRA,
    "Qwen3_32B_LoRA":  Qwen3_32B_LoRA,
    "Qwen3_32B_QLoRA": Qwen3_32B_QLoRA,
}
