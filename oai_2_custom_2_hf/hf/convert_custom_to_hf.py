# Convert our custom GPTModel into the Hugging Face GPTFSForCausalLM class
# and export it as a self-contained HF model folder.

import json
import os

from transformers import GPT2Tokenizer

from .configuration_gptfs import GPTFSConfig
from .modeling_gptfs import GPTFSForCausalLM


def map_custom_to_hf_state_dict(custom_state_dict):
    hf_state_dict = {}
    for key, value in custom_state_dict.items():
        if key.endswith(".att.mask"):
            continue  # causal mask is rebuilt on the fly in GPTFS
        if key == "out_head.weight":
            hf_state_dict["lm_head.weight"] = value
        else:
            hf_state_dict[f"model.{key}"] = value
    return hf_state_dict


def convert_custom_to_hf(custom_model, cfg):
    hf_model = GPTFSForCausalLM(GPTFSConfig(**cfg))
    hf_model.load_state_dict(map_custom_to_hf_state_dict(custom_model.state_dict()), strict=True)
    hf_model.eval()
    return hf_model


def build_tokenizer(oai_model_dir):
    with open(os.path.join(oai_model_dir, "encoder.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(oai_model_dir, "vocab.bpe"), encoding="utf-8") as f:
        merges = [tuple(line.split()) for line in f.read().split("\n")[1:] if line]  # skip "#version" header
    return GPT2Tokenizer(vocab=vocab, merges=merges)


def save_hf(hf_model, out_dir, tokenizer=None):
    GPTFSConfig.register_for_auto_class()
    GPTFSForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    hf_model.save_pretrained(out_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
