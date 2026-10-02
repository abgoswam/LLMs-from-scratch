import numpy as np
import pytest
import tiktoken
import torch

from custom.convert_oai_to_custom import load_weights_into_gpt
from custom.modeling_gptc import (
    GPTModel,
    generate,
    text_to_token_ids,
    token_ids_to_text,
)

BOOK_GREEDY_OUTPUT = (
    "Every effort moves you forward.\n\n"
    "The first step is to understand the importance of your work.\n\n"
    "The second step is to understand the"
)


def _t(array):
    return torch.from_numpy(np.ascontiguousarray(array))


def test_load_weights_maps_oai_params(tiny_cfg, fake_oai_params):
    model = GPTModel(tiny_cfg)
    load_weights_into_gpt(model, fake_oai_params)
    p = fake_oai_params

    assert torch.equal(model.tok_emb.weight, _t(p["wte"]))
    assert torch.equal(model.pos_emb.weight, _t(p["wpe"]))
    assert torch.equal(model.out_head.weight, _t(p["wte"]))
    assert torch.equal(model.final_norm.scale, _t(p["g"]))
    assert torch.equal(model.final_norm.shift, _t(p["b"]))

    for block, bp in zip(model.trf_blocks, p["blocks"]):
        q_w, k_w, v_w = np.split(bp["attn"]["c_attn"]["w"], 3, axis=-1)
        q_b, k_b, v_b = np.split(bp["attn"]["c_attn"]["b"], 3, axis=-1)
        assert torch.equal(block.att.W_query.weight, _t(q_w.T))
        assert torch.equal(block.att.W_key.weight, _t(k_w.T))
        assert torch.equal(block.att.W_value.weight, _t(v_w.T))
        assert torch.equal(block.att.W_query.bias, _t(q_b))
        assert torch.equal(block.att.W_key.bias, _t(k_b))
        assert torch.equal(block.att.W_value.bias, _t(v_b))
        assert torch.equal(block.att.out_proj.weight, _t(bp["attn"]["c_proj"]["w"].T))
        assert torch.equal(block.att.out_proj.bias, _t(bp["attn"]["c_proj"]["b"]))
        assert torch.equal(block.ff.layers[0].weight, _t(bp["mlp"]["c_fc"]["w"].T))
        assert torch.equal(block.ff.layers[0].bias, _t(bp["mlp"]["c_fc"]["b"]))
        assert torch.equal(block.ff.layers[2].weight, _t(bp["mlp"]["c_proj"]["w"].T))
        assert torch.equal(block.ff.layers[2].bias, _t(bp["mlp"]["c_proj"]["b"]))
        assert torch.equal(block.norm1.scale, _t(bp["ln_1"]["g"]))
        assert torch.equal(block.norm1.shift, _t(bp["ln_1"]["b"]))
        assert torch.equal(block.norm2.scale, _t(bp["ln_2"]["g"]))
        assert torch.equal(block.norm2.shift, _t(bp["ln_2"]["b"]))


def test_load_weights_rejects_shape_mismatch(tiny_cfg, fake_oai_params):
    model = GPTModel({**tiny_cfg, "emb_dim": 32, "n_heads": 2})
    with pytest.raises(ValueError):
        load_weights_into_gpt(model, fake_oai_params)


@pytest.mark.slow
def test_generate_matches_book_124m(gpt2_124m):
    model, cfg = gpt2_124m
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=cfg["context_length"],
    )
    assert token_ids_to_text(token_ids, tokenizer) == BOOK_GREEDY_OUTPUT
