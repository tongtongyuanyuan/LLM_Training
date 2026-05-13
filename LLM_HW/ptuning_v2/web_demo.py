"""
Gradio chat UI for a P-Tuning v2 fine-tuned ChatGLM3-6B-32K model.

Adapted from ChatGLM2-6B/ptuning/web_demo.py — loads the merged
prefix encoder weights on top of the base model.

Usage (on RunPod):
    python web_demo.py \
        --model_path      /hy-tmp/ZhipuAI/chatglm3-6b-32k \
        --checkpoint_path /hy-tmp/output/chatglm3-6b-32k-pt-128-2e-2/checkpoint-3000 \
        --pre_seq_len     128 \
        --server_port     7860

Then open RunPod → Connect → HTTP Service → port 7860 in your browser.

To expose publicly (share link, no auth):
    python web_demo.py ... --share
"""

import argparse
import os
import torch
import gradio as gr
from transformers import AutoConfig, AutoModel, AutoTokenizer


# ── model loader ──────────────────────────────────────────────────────────────

def load_model(model_path: str, checkpoint_path: str, pre_seq_len: int):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, pre_seq_len=pre_seq_len
    )
    model = AutoModel.from_pretrained(
        model_path, config=config, device_map="auto", trust_remote_code=True
    )

    # Merge prefix encoder weights (the small fine-tuned part)
    prefix_state_dict = torch.load(
        os.path.join(checkpoint_path, "pytorch_model.bin"),
        map_location="cpu"
    )
    new_prefix_state_dict = {
        k[len("transformer.prefix_encoder."):]: v
        for k, v in prefix_state_dict.items()
        if k.startswith("transformer.prefix_encoder.")
    }
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict, strict=False)
    model.transformer.prefix_encoder.float()

    model.eval()
    return tokenizer, model


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_demo(tokenizer, model):

    def predict(user_message, chat_history, max_length, top_p, temperature):
        history_pairs = [(h[0], h[1]) for h in chat_history]
        response, _ = model.chat(
            tokenizer,
            user_message,
            history=history_pairs,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
        )
        chat_history.append((user_message, response))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return "", chat_history

    with gr.Blocks(title="ChatGLM3 Medical Assistant") as demo:
        gr.Markdown("## ChatGLM3-6B-32K  —  P-Tuning v2 (Chinese Medical)")

        chatbot = gr.Chatbot(height=500)

        with gr.Row():
            user_input = gr.Textbox(
                show_label=False,
                placeholder="请输入您的问题...",
                scale=9,
            )
            submit_btn = gr.Button("发送", scale=1, variant="primary")

        with gr.Row():
            clear_btn  = gr.Button("清空对话")

        with gr.Accordion("Generation settings", open=False):
            max_length  = gr.Slider(64, 4096, value=1024, step=64,  label="Max length")
            top_p       = gr.Slider(0.0, 1.0,  value=0.7,  step=0.01, label="Top-p")
            temperature = gr.Slider(0.0, 2.0,  value=0.95, step=0.01, label="Temperature")

        submit_btn.click(
            predict,
            inputs=[user_input, chatbot, max_length, top_p, temperature],
            outputs=[user_input, chatbot],
        )
        user_input.submit(
            predict,
            inputs=[user_input, chatbot, max_length, top_p, temperature],
            outputs=[user_input, chatbot],
        )
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, user_input])

    return demo


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        default="/hy-tmp/ZhipuAI/chatglm3-6b-32k")
    parser.add_argument("--checkpoint_path",
                        default="/hy-tmp/output/chatglm3-6b-32k-pt-128-2e-2/checkpoint-3000")
    parser.add_argument("--pre_seq_len", type=int, default=128)
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="create a public Gradio share link")
    args = parser.parse_args()

    print("Loading model — this takes ~1-2 min...")
    tokenizer, model = load_model(args.model_path, args.checkpoint_path, args.pre_seq_len)
    print("Model ready.")

    demo = build_demo(tokenizer, model)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
