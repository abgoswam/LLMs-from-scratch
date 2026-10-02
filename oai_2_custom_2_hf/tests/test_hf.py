import pytest
import tiktoken
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from custom.modeling_gptc import generate
from hf.configuration_gptfs import GPTFSConfig
from hf.convert_custom_to_hf import (
    build_tokenizer,
    convert_custom_to_hf,
    map_custom_to_hf_state_dict,
    save_hf,
)
from hf.modeling_gptfs import GPTFSForCausalLM

PROMPT = "Every effort moves you"


def test_config_roundtrip(tiny_cfg, tmp_path):
    config = GPTFSConfig(**tiny_cfg)
    config.save_pretrained(tmp_path)
    loaded = GPTFSConfig.from_pretrained(tmp_path)

    assert loaded.to_custom_config() == tiny_cfg
    assert loaded.hidden_size == tiny_cfg["emb_dim"]
    assert loaded.num_hidden_layers == tiny_cfg["n_layers"]
    assert loaded.num_attention_heads == tiny_cfg["n_heads"]
    assert loaded.max_position_embeddings == tiny_cfg["context_length"]


def test_weight_mapping_complete(tiny_cfg, tiny_model):
    mapped = map_custom_to_hf_state_dict(tiny_model.state_dict())
    expected = GPTFSForCausalLM(GPTFSConfig(**tiny_cfg)).state_dict()

    assert mapped.keys() == expected.keys()
    for key, value in mapped.items():
        assert value.shape == expected[key].shape


def test_logits_custom_vs_hf_tiny(tiny_cfg, tiny_model):
    hf_model = convert_custom_to_hf(tiny_model, tiny_cfg)
    input_ids = torch.randint(0, tiny_cfg["vocab_size"], (2, 16))

    with torch.no_grad():
        torch.testing.assert_close(hf_model(input_ids).logits, tiny_model(input_ids))


def test_loss_matches_cross_entropy(tiny_cfg, tiny_model):
    hf_model = convert_custom_to_hf(tiny_model, tiny_cfg)
    input_ids = torch.randint(0, tiny_cfg["vocab_size"], (2, 16))

    with torch.no_grad():
        logits = tiny_model(input_ids)
        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1), input_ids[:, 1:].flatten())
        torch.testing.assert_close(hf_model(input_ids, labels=input_ids).loss, expected)


def test_logits_auto_model_roundtrip_tiny(tiny_cfg, tiny_model, tmp_path):
    save_hf(convert_custom_to_hf(tiny_model, tiny_cfg), tmp_path)

    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True)
    input_ids = torch.randint(0, tiny_cfg["vocab_size"], (2, 16))

    assert config.model_type == "gptfs"
    assert type(hf_model).__name__ == "GPTFSForCausalLM"
    assert not hf_model.training
    with torch.no_grad():
        torch.testing.assert_close(hf_model(input_ids).logits, tiny_model(input_ids))


def test_generate_custom_vs_hf_tiny(tiny_cfg, tiny_model):
    hf_model = convert_custom_to_hf(tiny_model, tiny_cfg)
    input_ids = torch.randint(0, tiny_cfg["vocab_size"], (1, 5))

    expected = generate(tiny_model, input_ids, max_new_tokens=10, context_size=tiny_cfg["context_length"])
    actual = hf_model.generate(input_ids, max_new_tokens=10, do_sample=False, eos_token_id=None, pad_token_id=0)

    assert torch.equal(actual, expected)


@pytest.mark.slow
def test_exported_tokenizer_matches_tiktoken(oai_124m_dir, tmp_path):
    build_tokenizer(oai_124m_dir).save_pretrained(tmp_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    tik = tiktoken.get_encoding("gpt2")
    text = "Every effort moves you forward.\n\nHello, world! <|endoftext|> naïve  spaces 123"

    assert hf_tokenizer(text).input_ids == tik.encode(text, allowed_special={"<|endoftext|>"})
    assert hf_tokenizer.eos_token_id == 50256


@pytest.mark.slow
def test_generate_custom_vs_hf_124m(gpt2_124m, oai_124m_dir, tmp_path):
    model, cfg = gpt2_124m
    save_hf(convert_custom_to_hf(model, cfg), tmp_path, build_tokenizer(oai_124m_dir))
    hf_model = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True)
    hf_tokenizer = AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=True)
    input_ids = hf_tokenizer(PROMPT, return_tensors="pt").input_ids

    expected = generate(model, input_ids, max_new_tokens=25, context_size=cfg["context_length"])
    actual = hf_model.generate(
        input_ids, max_new_tokens=25, do_sample=False, pad_token_id=hf_tokenizer.eos_token_id)

    assert torch.equal(actual, expected)


@pytest.mark.slow
def test_logits_match_hf_reference_gpt2_124m(gpt2_124m):
    from transformers import GPT2LMHeadModel

    try:
        reference = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    except OSError:
        pytest.skip("Hugging Face Hub not reachable")

    model, cfg = gpt2_124m
    hf_model = convert_custom_to_hf(model, cfg)
    input_ids = torch.tensor([tiktoken.get_encoding("gpt2").encode(PROMPT)])

    with torch.no_grad():
        torch.testing.assert_close(hf_model(input_ids).logits, reference(input_ids).logits, atol=1e-3, rtol=1e-4)
