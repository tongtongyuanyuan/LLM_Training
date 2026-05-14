"""
CLI entry point for ChatGLM3-6B-32K P-Tuning v2.

Usage:
    python run.py <action> [--checkpoint /path/to/checkpoint]

Actions:
    download   download chatglm3-6b-32k from ModelScope
    setup      clone ChatGLM2-6B repo + patch files + write train.sh
    train      run fine-tuning (sh train.sh inside ChatGLM2-6B/ptuning/)
    infer      interactive CLI chat with the fine-tuned model
    serve      launch Gradio web chat UI (accessible in browser on RunPod)

Typical workflow on RunPod:
    # 1. install packages
    pip install datasets==4.4.1 transformers==4.30.2 \\
                jieba rouge-chinese nltk sentencepiece accelerate

    # 2. prepare data (run once locally, then upload train.json + dev.json)
    python data_process.py

    # 3. download model
    python run.py download

    # 4. patch repo + write train.sh
    python run.py setup

    # 5. train
    python run.py train

    # 6. interactive CLI chat
    python run.py infer
    python run.py infer --checkpoint /hy-tmp/output/.../checkpoint-3000

    # 7. Gradio web chat (browser, port 7860)
    python run.py serve
    python run.py serve --port 7860 --share      # --share creates a public link

Environment variables:
    MODEL_BASE_DIR   /hy-tmp          (default)
    OUTPUT_BASE_DIR  /hy-tmp/output   (default)
    TRAIN_FILE       /hy-tmp/train.json
    DEV_FILE         /hy-tmp/dev.json
"""

import argparse
import sys
from trainer import ChatGLM3PtuningTrainer

ACTIONS = ("download", "setup", "train", "infer", "serve")


def main():
    parser = argparse.ArgumentParser(
        description="P-Tuning v2 runner for ChatGLM3-6B-32K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "action", choices=ACTIONS,
        help="What to do"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="(infer / serve) path to checkpoint dir; defaults to latest"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="(serve) Gradio server port"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="(serve) create a public Gradio share link"
    )
    args = parser.parse_args()

    trainer = ChatGLM3PtuningTrainer()

    if args.action == "infer":
        trainer.infer(checkpoint_path=args.checkpoint)
    elif args.action == "serve":
        trainer.serve(checkpoint_path=args.checkpoint,
                      port=args.port, share=args.share)
    else:
        getattr(trainer, args.action)()


if __name__ == "__main__":
    main()
