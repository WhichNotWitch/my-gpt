from sysml_gpt.tokenizer import CharTokenizer


def test_encode_decode_roundtrip():
    text = "package Test { attr : Real; }"
    tokenizer = CharTokenizer(text)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_vocab_size_matches_unique_chars():
    text = "aabbbc"
    tokenizer = CharTokenizer(text)

    assert tokenizer.vocab_size == 3