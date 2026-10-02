from pathlib import Path

import numpy as np
import pytest
import torch

from custom.convert_oai_to_custom import load_weights_into_gpt
from custom.modeling_gptc import GPTModel, get_config

MODELS_DIR = Path(__file__).resolve().parents[1] / "gpt2"


@pytest.fixture
def tiny_cfg():
    return {
        "vocab_size": 50257,
        "context_length": 32,
        "emb_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }


@pytest.fixture
def tiny_model(tiny_cfg):
    torch.manual_seed(123)
    return GPTModel(tiny_cfg).eval()


@pytest.fixture
def fake_oai_params(tiny_cfg):
    rng = np.random.default_rng(123)
    d, v, c = tiny_cfg["emb_dim"], tiny_cfg["vocab_size"], tiny_cfg["context_length"]

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    blocks = [
        {
            "attn": {
                "c_attn": {"w": rand(d, 3 * d), "b": rand(3 * d)},
                "c_proj": {"w": rand(d, d), "b": rand(d)},
            },
            "mlp": {
                "c_fc": {"w": rand(d, 4 * d), "b": rand(4 * d)},
                "c_proj": {"w": rand(4 * d, d), "b": rand(d)},
            },
            "ln_1": {"g": rand(d), "b": rand(d)},
            "ln_2": {"g": rand(d), "b": rand(d)},
        }
        for _ in range(tiny_cfg["n_layers"])
    ]
    return {"wte": rand(v, d), "wpe": rand(c, d), "blocks": blocks, "g": rand(d), "b": rand(d)}


@pytest.fixture(scope="session")
def gpt2_124m():
    from custom.gpt_download import download_and_load_gpt2

    _, params = download_and_load_gpt2(model_size="124M", models_dir=str(MODELS_DIR))
    cfg = get_config("124M")
    model = GPTModel(cfg)
    load_weights_into_gpt(model, params)
    return model.eval(), cfg


@pytest.fixture(scope="session")
def oai_124m_dir(gpt2_124m):
    return MODELS_DIR / "124M"
