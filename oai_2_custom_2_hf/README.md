# OpenAI GPT-2 → custom `GPTModel` → Hugging Face

Two separate conversions, each run as its own command:

| Command | From | To |
|---|---|---|
| `python convert.py oai_2_custom` | original OpenAI GPT-2 TensorFlow checkpoint | our own `GPTModel` class (from the book), saved as `out/custom/gpt2-<size>.pth` |
| `python convert.py custom_2_hf` | `out/custom/gpt2-<size>.pth` | `GPTFSForCausalLM` ("GPT from scratch"), our own Hugging Face model class, saved as `out/hf/gptfs-<size>/` |

## Code layout

`custom/` and `hf/` mirror each other: each has a model definition plus the converter *into* that format.

```
oai_2_custom_2_hf/
├── convert.py                     # CLI: oai_2_custom / custom_2_hf
├── custom/                        # our own architecture (from the book)
│   ├── modeling_gptc.py           # GPTModel + layers, generate, token helpers, get_config
│   ├── convert_oai_to_custom.py   # OpenAI params → GPTModel (assign, load_weights_into_gpt)
│   └── gpt_download.py            # downloads + reads the OpenAI TF checkpoint
├── hf/                            # Hugging Face side
│   ├── configuration_gptfs.py     # GPTFSConfig
│   ├── modeling_gptfs.py          # GPTFSForCausalLM
│   └── convert_custom_to_hf.py    # GPTModel → GPTFSForCausalLM, save_pretrained, tokenizer
└── tests/
    ├── test_custom.py             # oai_2_custom
    └── test_hf.py                 # custom_2_hf
```

| Role | `custom/` | `hf/` |
|---|---|---|
| Model definition | `modeling_gptc.py` | `modeling_gptfs.py` (+ `configuration_gptfs.py`) |
| Conversion into this format | `convert_oai_to_custom.py` | `convert_custom_to_hf.py` |

## Setup

From the repo root, with your env activated:

```bash
python -m pip install -r requirements.txt -r oai_2_custom_2_hf/requirements.txt
```

## Run both conversions

Run from inside this folder (default paths `gpt2/` and `out/` are relative to it):

```bash
cd oai_2_custom_2_hf

# 1. OpenAI -> custom
#    downloads gpt2/124M/ (~500 MB, first time only), writes out/custom/gpt2-124M.pth
python convert.py oai_2_custom

# 2. custom -> HF
#    reads out/custom/gpt2-124M.pth (+ tokenizer files in gpt2/124M/), writes out/hf/gptfs-124M/
python convert.py custom_2_hf
```

Step 2 needs step 1 to have run once (for the `.pth` and the tokenizer files), but after that it can be rerun on its own, with no TensorFlow and no network.

Each command ends by generating a greedy continuation of `"Every effort moves you"`. `custom_2_hf` prints it from both models so you can compare; all three lines should be identical:

```
[custom GPTModel]     'Every effort moves you forward.\n\nThe first step is to understand the importance of your work.\n\nThe second step is to understand the'
[HF GPTFSForCausalLM] 'Every effort moves you forward.\n\nThe first step is to understand the importance of your work.\n\nThe second step is to understand the'
```

Resulting layout:

```
oai_2_custom_2_hf/
├── gpt2/124M/                 # OpenAI checkpoint + encoder.json / vocab.bpe (downloaded by step 1)
└── out/
    ├── custom/gpt2-124M.pth   # GPTModel state_dict (step 1)
    └── hf/gptfs-124M/         # config.json, model.safetensors, configuration_gptfs.py,
                               # modeling_gptfs.py, tokenizer.json, ... (step 2)
```

Options (same for both commands): `--model-size {124M,355M,774M,1558M}`, `--models-dir` (default `gpt2`), `--out-dir` (default `out`), `--prompt`, `--max-new-tokens`. Use the same `--model-size` for both steps. See `python convert.py <command> --help`.

## Using the exported HF model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("out/hf/gptfs-124M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("out/hf/gptfs-124M", trust_remote_code=True)

input_ids = tokenizer("Every effort moves you", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=25, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0]))
```

`trust_remote_code=True` is needed because the folder ships its own `configuration_gptfs.py` / `modeling_gptfs.py` (referenced by `auto_map` in `config.json`) rather than using a class built into `transformers`.

## Where each file came from

| File | Origin | Changes from the source |
|---|---|---|
| `custom/gpt_download.py` | Copied from `ch06/01_main-chapter-code/gpt_download.py` | None (verbatim). Chosen over the `ch05` copy because it has the backup download URL. |
| `custom/modeling_gptc.py` | `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, `GPTModel`, `text_to_token_ids`, `token_ids_to_text` copied from `ch06/01_main-chapter-code/previous_chapters.py` | Code unchanged. Dropped `GPTDatasetV1`, `create_dataloader_v1`, `generate_text_simple` (not needed). |
| | `generate()` (temperature / top-k) copied from `ch05/01_main-chapter-code/gpt_generate.py` | Removed the "New:" prefixes from its comments. |
| | `BASE_CONFIG`, `MODEL_CONFIGS`, `get_config()` | Values from `BASE_CONFIG` / `model_configs` in `ch05/01_main-chapter-code/gpt_generate.py`, re-keyed by size (`"124M"` instead of `"gpt2-small (124M)"`) and wrapped in `get_config()`. |
| `custom/convert_oai_to_custom.py` | `assign`, `load_weights_into_gpt` copied from `ch06/01_main-chapter-code/previous_chapters.py` | Code unchanged; moved into its own file to mirror `hf/convert_custom_to_hf.py`. |
| `convert.py` | New. The `oai_2_custom` flow (download → `GPTModel` → `load_weights_into_gpt` → generate) follows `main()` in `ch05/01_main-chapter-code/gpt_generate.py`. | Split into `oai_2_custom` / `custom_2_hf` subcommands; saves outputs; greedy decoding instead of top-k sampling so outputs are comparable. |
| `hf/configuration_gptfs.py` | New. Pattern from `transformers/models/gpt2/configuration_gpt2.py` (v5.17). | Same fields as the `GPTModel` config dict, plus `attribute_map` so HF's standard names (`hidden_size`, ...) work. |
| `hf/modeling_gptfs.py` | New. Layers re-implemented from `custom/modeling_gptc.py`; HF wrapper pattern (`PreTrainedModel`, `_init_weights`, `GenerationMixin`) from `transformers/models/gpt2/modeling_gpt2.py` (v5.17). | Same math and submodule names as `GPTModel`; causal mask built on the fly instead of a buffer; no KV cache. Self-contained on purpose (see below). |
| `hf/convert_custom_to_hf.py` | New. | Key-rename mapping, export via `save_pretrained`, tokenizer built from OpenAI's `encoder.json` / `vocab.bpe`. |
| `tests/*` | New. The expected greedy text is the book's output, also shown in `ch06/01_main-chapter-code/ch06.ipynb`. | |

## How the HF model relates to `GPTModel`

`GPTFSForCausalLM` uses the same layers and submodule names as `GPTModel`, so `custom_2_hf` only renames keys:

| `GPTModel` | `GPTFSForCausalLM` |
|---|---|
| `tok_emb`, `pos_emb`, `trf_blocks.*`, `final_norm.*` | `model.` + same name |
| `out_head.weight` | `lm_head.weight` |
| `trf_blocks.*.att.mask` (buffer) | dropped; the causal mask is built on the fly |

The actual layout change (splitting OpenAI's combined QKV matrix and transposing its weights) happens once, in `oai_2_custom` (`custom/convert_oai_to_custom.py`).

`hf/modeling_gptfs.py` deliberately does not import from `custom/modeling_gptc.py`: `save_pretrained` copies it into the exported folder, where it must load on its own.

Limitations: no KV cache (each generation step re-runs the full sequence), and `attention_mask` is ignored, so batched generation with padding is not supported.

## Tests

```bash
pytest                  # everything (first run downloads GPT-2 124M, ~500 MB)
pytest -m "not slow"    # tiny random models only, a few seconds
```

| Test | Slow | Checks |
|---|---|---|
| `test_custom.py::test_load_weights_maps_oai_params` | | every OAI tensor lands in the right `GPTModel` parameter (incl. QKV split + transpose) |
| `test_custom.py::test_load_weights_rejects_shape_mismatch` | | a config/checkpoint mismatch raises |
| `test_custom.py::test_generate_matches_book_124m` | ✓ | 124M greedy output equals the book's |
| `test_hf.py::test_config_roundtrip` | | config save/load and the HF attribute aliases |
| `test_hf.py::test_weight_mapping_complete` | | mapped keys/shapes exactly match `GPTFSForCausalLM` |
| `test_hf.py::test_logits_custom_vs_hf_tiny` | | same logits as `GPTModel` |
| `test_hf.py::test_loss_matches_cross_entropy` | | `labels=` loss equals shifted cross-entropy |
| `test_hf.py::test_logits_auto_model_roundtrip_tiny` | | exported folder reloads via `AutoModelForCausalLM` with same logits |
| `test_hf.py::test_generate_custom_vs_hf_tiny` | | HF `generate()` equals our `generate()` |
| `test_hf.py::test_exported_tokenizer_matches_tiktoken` | ✓ | exported tokenizer equals tiktoken's `gpt2` encoding |
| `test_hf.py::test_generate_custom_vs_hf_124m` | ✓ | same as above on real 124M weights |
| `test_hf.py::test_logits_match_hf_reference_gpt2_124m` | ✓ | logits match HF's built-in `GPT2LMHeadModel` (skipped if the Hub is unreachable) |
