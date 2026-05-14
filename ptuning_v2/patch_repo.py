"""
Patches the cloned ChatGLM2-6B repo per the course slides (4.pdf steps 3-4):

  Step 3 – insert build_prompt() into two source files
  Step 4 – fix modeling_chatglm.py line ~1015 (response split bug)

Run once, from the ptuning_v2 directory, after:
    git clone https://github.com/THUDM/ChatGLM2-6B.git

Usage:
    python patch_repo.py [--repo_dir ./ChatGLM2-6B]
"""

import argparse
import os
import re

# ── code to insert ─────────────────────────────────────────────────────────────

# Version inserted into ChatGLMTokenizer class (needs 'self')
BUILD_PROMPT_METHOD = '''\
    def build_prompt(self, query, history=None):
        if history is None:
            history = []

        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\\n\\n问：{}\\n\\n答：{}\\n\\n".format(
                i + 1, old_query, response)

        prompt += "[Round {}]\\n\\n问：{}\\n\\n答：".format(
            len(history) + 1, query)
        return prompt

'''

# Version inserted at module level in modeling_chatglm.py (no 'self')
BUILD_PROMPT_FUNC = '''\
def build_prompt(query, history=None):
    if history is None:
        history = []

    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\\n\\n问：{}\\n\\n答：{}\\n\\n".format(
            i + 1, old_query, response)

    prompt += "[Round {}]\\n\\n问：{}\\n\\n答：".format(
        len(history) + 1, query)
    return prompt


'''


# ── patch helpers ──────────────────────────────────────────────────────────────

def patch_tokenization(repo_dir):
    """
    Insert build_prompt (class method) between build_single_message()
    and build_chat_input() in tokenization_chatglm.py (~line 177).
    """
    path = os.path.join(repo_dir, "tokenization_chatglm.py")
    if not os.path.exists(path):
        print(f"[SKIP] Not found: {path}")
        return False

    text = open(path, encoding="utf-8").read()
    if "def build_prompt(" in text:
        print(f"[SKIP] build_prompt already present in tokenization_chatglm.py")
        return True

    # Insert just before "    def build_chat_input("
    marker = "    def build_chat_input("
    idx = text.find(marker)
    if idx == -1:
        print("[WARN] build_chat_input not found – skipping tokenization patch")
        return False

    patched = text[:idx] + BUILD_PROMPT_METHOD + text[idx:]
    open(path, "w", encoding="utf-8").write(patched)
    print(f"[OK]   Patched tokenization_chatglm.py (build_prompt inserted before build_chat_input)")
    return True


def patch_modeling(repo_dir):
    """
    1. Insert build_prompt (module-level function) after default_init (~line 50).
    2. Fix the response.split bug (~line 1015):
           metadata, content = response.split("\\n", maxsplit=1)
       →   if "\\n" in response:
               metadata, content = response.split("\\n", maxsplit=1)
           else:
               metadata, content = "", response
    """
    path = os.path.join(repo_dir, "modeling_chatglm.py")
    if not os.path.exists(path):
        print(f"[SKIP] Not found: {path}")
        return False

    text = open(path, encoding="utf-8").read()

    # ── patch 1: build_prompt ──────────────────────────────────────────────────
    if "def build_prompt(" not in text:
        marker = "def default_init(cls, *args, **kwargs):"
        idx = text.find(marker)
        if idx == -1:
            print("[WARN] default_init not found – skipping build_prompt patch in modeling")
        else:
            # Find the next top-level def/class after default_init
            after = text[idx + len(marker):]
            m = re.search(r'\n(def |class )', after)
            if m:
                ins = idx + len(marker) + m.start() + 1
                text = text[:ins] + "\n" + BUILD_PROMPT_FUNC + text[ins:]
                print("[OK]   Patched modeling_chatglm.py (build_prompt inserted after default_init)")
            else:
                print("[WARN] Could not locate insertion point after default_init")
    else:
        print("[SKIP] build_prompt already present in modeling_chatglm.py")

    # ── patch 2: response split bug ────────────────────────────────────────────
    bad  = 'metadata, content = response.split("\\n", maxsplit=1)'
    good = ('if "\\n" in response:\n'
            '            metadata, content = response.split("\\n", maxsplit=1)\n'
            '        else:\n'
            '            metadata, content = "", response  # no metadata line')
    if bad in text:
        text = text.replace(bad, good, 1)
        print("[OK]   Fixed response.split bug in modeling_chatglm.py")
    elif good.splitlines()[0] in text:
        print("[SKIP] response.split bug already fixed")
    else:
        print("[WARN] response.split line not found – skipping bug fix")

    open(path, "w", encoding="utf-8").write(text)
    return True


def patch_main(repo_dir):
    """
    Remove the deprecated `use_auth_token` kwarg from load_dataset() in main.py.
    Datasets >= 3.0 removed this parameter — passing it causes a TypeError.
    """
    path = os.path.join(repo_dir, "ptuning", "main.py")
    if not os.path.exists(path):
        print(f"[SKIP] Not found: {path}")
        return

    text = open(path, encoding="utf-8").read()

    if "use_auth_token" not in text:
        print("[SKIP] use_auth_token not found in main.py (already clean)")
        return

    # Remove the use_auth_token= line from load_dataset() call
    import re
    patched = re.sub(r'\s*use_auth_token\s*=\s*[^\n,]+,?\n', '\n', text)
    open(path, "w", encoding="utf-8").write(patched)
    print("[OK]   Removed use_auth_token from main.py (datasets >= 3.0 compatibility)")


def write_train_sh(repo_dir, model_path, train_file, dev_file):
    """Write a ready-to-run train.sh into ChatGLM2-6B/ptuning/."""
    ptuning_dir = os.path.join(repo_dir, "ptuning")
    os.makedirs(ptuning_dir, exist_ok=True)

    output_dir = f"/hy-tmp/output/chatglm3-6b-32k-pt-$PRE_SEQ_LEN-$LR"

    content = f"""#!/bin/bash
# Step 6 – train.sh  (auto-generated by patch_repo.py)

PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \\
    --do_train \\
    --train_file {train_file} \\
    --validation_file {dev_file} \\
    --preprocessing_num_workers 10 \\
    --prompt_column input \\
    --response_column output \\
    --overwrite_cache \\
    --model_name_or_path {model_path} \\
    --output_dir {output_dir} \\
    --overwrite_output_dir \\
    --max_source_length 64 \\
    --max_target_length 128 \\
    --per_device_train_batch_size 1 \\
    --per_device_eval_batch_size 1 \\
    --gradient_accumulation_steps 16 \\
    --predict_with_generate \\
    --max_steps 3000 \\
    --logging_steps 10 \\
    --save_steps 1000 \\
    --learning_rate $LR \\
    --pre_seq_len $PRE_SEQ_LEN \\
    --quantization_bit 4
"""
    sh_path = os.path.join(ptuning_dir, "train.sh")
    open(sh_path, "w", encoding="utf-8").write(content)
    os.chmod(sh_path, 0o755)
    print(f"[OK]   Written {sh_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir",   default="./ChatGLM2-6B",
                        help="Path to the cloned ChatGLM2-6B repo")
    parser.add_argument("--model_path", default="/hy-tmp/ZhipuAI/chatglm3-6b-32k",
                        help="Path where chatglm3-6b-32k model is stored")
    parser.add_argument("--train_file", default="/hy-tmp/train.json")
    parser.add_argument("--dev_file",   default="/hy-tmp/dev.json")
    args = parser.parse_args()

    if not os.path.isdir(args.repo_dir):
        print(f"\nERROR: repo dir '{args.repo_dir}' not found.")
        print("Please run first:\n  git clone https://github.com/THUDM/ChatGLM2-6B.git\n")
        return

    print(f"\n=== Patching {args.repo_dir} ===")
    patch_tokenization(args.repo_dir)
    patch_modeling(args.repo_dir)
    patch_main(args.repo_dir)
    write_train_sh(args.repo_dir, args.model_path, args.train_file, args.dev_file)

    print(f"""
=== Done. Next steps on the GPU server ===

  # Step 5 – enter ptuning dir
  cd {args.repo_dir}/ptuning

  # Step 7 – install packages
  pip install datasets==2.14.6 transformers==4.30.2 \\
              nltk sentencepiece accelerate

  # Step 8 – start training
  sh train.sh
""")


if __name__ == "__main__":
    main()
