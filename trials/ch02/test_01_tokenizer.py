import re

import pytest

from tokenizer import SimpleTokenizerV1, SimpleTokenizerV2

# --- 2.2 Tokenizing text / 2.3 Converting tokens into token IDs -----------
# Same inline preprocessing as the notebook: this builds the vocab the tokenizers consume.


@pytest.fixture(scope="module")
def preprocessed(raw_text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    return [item.strip() for item in preprocessed if item.strip()]


@pytest.fixture(scope="module")
def vocab(preprocessed):
    all_words = sorted(set(preprocessed))
    return {token: integer for integer, token in enumerate(all_words)}


@pytest.fixture(scope="module")
def vocab_with_specials(preprocessed):
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    return {token: integer for integer, token in enumerate(all_tokens)}


def test_vocab(preprocessed, vocab):
    assert len(preprocessed) == 4690
    assert len(vocab) == 1130


def test_v1_encode_decode(vocab):
    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    assert ids == [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
    assert tokenizer.decode(ids) == '" It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'


def test_v1_fails_on_unknown_word(vocab):
    tokenizer = SimpleTokenizerV1(vocab)
    with pytest.raises(KeyError):
        tokenizer.encode("Hello, do you like tea?")


# --- 2.4 Adding special context tokens ------------------------------------


def test_v2_handles_unknown_words_and_endoftext(vocab_with_specials):
    tokenizer = SimpleTokenizerV2(vocab_with_specials)
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
    ids = tokenizer.encode(text)
    assert ids == [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
    assert tokenizer.decode(ids) == "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."

