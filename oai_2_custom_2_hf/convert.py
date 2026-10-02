# Two independent conversions, each finishing with a greedy-generation check:
#
#   python convert.py oai_2_custom   OpenAI GPT-2 TF checkpoint -> out/custom/gpt2-<size>.pth
#   python convert.py custom_2_hf    out/custom/gpt2-<size>.pth -> out/hf/gptfs-<size>/
#
# custom_2_hf only reads the .pth written by oai_2_custom (plus the OpenAI
# tokenizer files in gpt2/<size>/), so it needs no TensorFlow and no network.

import argparse
import os

import tiktoken
import torch

from custom.convert_oai_to_custom import load_weights_into_gpt
from custom.modeling_gptc import (
    GPTModel,
    generate,
    get_config,
    text_to_token_ids,
    token_ids_to_text,
)


def custom_path(args):
    return os.path.join(args.out_dir, "custom", f"gpt2-{args.model_size}.pth")


def hf_dir(args):
    return os.path.join(args.out_dir, "hf", f"gptfs-{args.model_size}")


def generate_custom(model, cfg, args):
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(args.prompt, tokenizer),
        max_new_tokens=args.max_new_tokens,
        context_size=cfg["context_length"],
    )
    return token_ids_to_text(token_ids, tokenizer)


def oai_2_custom(args):
    from custom.gpt_download import download_and_load_gpt2

    _, params = download_and_load_gpt2(model_size=args.model_size, models_dir=args.models_dir)
    cfg = get_config(args.model_size)
    model = GPTModel(cfg)
    load_weights_into_gpt(model, params)
    model.eval()

    out_path = custom_path(args)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved custom model weights to {out_path}")

    print(f"\n[custom GPTModel] {generate_custom(model, cfg, args)!r}")


def custom_2_hf(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hf.convert_custom_to_hf import build_tokenizer, convert_custom_to_hf, save_hf

    in_path = custom_path(args)
    cfg = get_config(args.model_size)
    model = GPTModel(cfg)
    model.load_state_dict(torch.load(in_path, weights_only=True))
    model.eval()
    print(f"Loaded custom model weights from {in_path}")

    out_dir = hf_dir(args)
    tokenizer = build_tokenizer(os.path.join(args.models_dir, args.model_size))
    save_hf(convert_custom_to_hf(model, cfg), out_dir, tokenizer)
    print(f"Saved Hugging Face model to {out_dir}")

    hf_model = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True)
    hf_tokenizer = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    input_ids = hf_tokenizer(args.prompt, return_tensors="pt").input_ids
    output_ids = hf_model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=hf_tokenizer.eos_token_id,
    )

    print(f"\n[custom GPTModel]     {generate_custom(model, cfg, args)!r}")
    print(f"[HF GPTFSForCausalLM] {hf_tokenizer.decode(output_ids[0])!r}")


def main():
    parser = argparse.ArgumentParser(description="Convert GPT-2 weights: OpenAI -> custom GPTModel -> Hugging Face.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-size", default="124M", choices=["124M", "355M", "774M", "1558M"])
    common.add_argument("--models-dir", default="gpt2", help="OpenAI checkpoint + tokenizer files (default: gpt2).")
    common.add_argument("--out-dir", default="out", help="Converted models go to <out-dir>/custom and <out-dir>/hf (default: out).")
    common.add_argument("--prompt", default="Every effort moves you")
    common.add_argument("--max-new-tokens", type=int, default=25)

    subparsers.add_parser(
        "oai_2_custom", parents=[common],
        help="Download the OpenAI checkpoint and save it as out/custom/gpt2-<size>.pth.",
    ).set_defaults(func=oai_2_custom)
    subparsers.add_parser(
        "custom_2_hf", parents=[common],
        help="Convert out/custom/gpt2-<size>.pth into the HF folder out/hf/gptfs-<size>/.",
    ).set_defaults(func=custom_2_hf)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
