# GPTFS ("GPT from scratch") model configuration.
#
# Mirrors the config dict used by GPTModel in custom/modeling_gptc.py, and
# exposes the standard Hugging Face attribute names via `attribute_map`.

from transformers import PreTrainedConfig


class GPTFSConfig(PreTrainedConfig):
    model_type = "gptfs"
    attribute_map = {
        "hidden_size": "emb_dim",
        "max_position_embeddings": "context_length",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.0,
        qkv_bias=True,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias
        self.initializer_range = initializer_range
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def to_custom_config(self):
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "emb_dim": self.emb_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "drop_rate": self.drop_rate,
            "qkv_bias": self.qkv_bias,
        }


__all__ = ["GPTFSConfig"]
