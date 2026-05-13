"""
Inference with p-tuning v2 fine-tuned ChatGLM3-6B-32K.
Merges the original model with the trained prefix encoder weights.

Usage:
    python inference.py \
        --model_path /hy-tmp/ZhipuAI/chatglm3-6b-32k \
        --checkpoint_path /hy-tmp/output/chatglm3-6b-32k-pt-128-2e-2/checkpoint-3000
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


def load_model(model_path, checkpoint_path, pre_seq_len=128):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, pre_seq_len=pre_seq_len
    )
    model = AutoModel.from_pretrained(
        model_path, config=config, device_map="auto", trust_remote_code=True
    )

    # Merge p-tuning v2 prefix encoder weights into the base model
    prefix_state_dict = torch.load(
        os.path.join(checkpoint_path, "pytorch_model.bin")
    )
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v

    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict, strict=False)

    # Fix for ChatGLM3 response parsing (slide fix in modeling_chatglm.py line 1015)
    # Handled at runtime via the patched modeling file in the repo.

    model = model.quantize(4)
    model.eval()
    return tokenizer, model


def chat(tokenizer, model, question, history=None):
    if history is None:
        history = []
    response, history = model.chat(tokenizer, question, history=history)
    return response, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        default="/hy-tmp/ZhipuAI/chatglm3-6b-32k")
    parser.add_argument("--checkpoint_path",
                        default="/hy-tmp/output/chatglm3-6b-32k-pt-128-2e-2/checkpoint-3000")
    parser.add_argument("--pre_seq_len", type=int, default=128)
    args = parser.parse_args()

    print("Loading model...")
    tokenizer, model = load_model(args.model_path, args.checkpoint_path, args.pre_seq_len)
    print("Model loaded. Type 'exit' to quit.\n")

    history = []
    while True:
        question = input("问：").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        response, history = chat(tokenizer, model, question, history)
        print(f"答：{response}\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
