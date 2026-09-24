from pathlib import Path

import pytest
import tiktoken


@pytest.fixture(scope="session")
def raw_text():
    return (Path(__file__).parent / "the-verdict.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    return tiktoken.get_encoding("gpt2")
