import tiktoken


def get_tokenizer_name(model):
    return tiktoken.encoding_name_for_model(model)


async def num_tokens(text, model=None, tokenizer=None):
    tokenizer_name = tokenizer if tokenizer else get_tokenizer_name(model)
    try:
        encoding = tiktoken.get_encoding(tokenizer_name)
        return len(encoding.encode(text))
    except Exception as e:
        raise ValueError(f"Unknown tokenizer {tokenizer_name}") from e


async def encode_tokens(text, model=None, tokenizer=None):
    tokenizer_name = tokenizer if tokenizer else get_tokenizer_name(model)
    try:
        encoding = tiktoken.get_encoding(tokenizer_name)
        return encoding.encode(text)
    except Exception as e:
        raise ValueError(f"Unknown tokenizer {tokenizer_name}") from e


def estimate_tokens_using_bytecount(text, tokenizer):
    byte_length = len(text.encode("utf-8"))
    try:
        if tokenizer in ["cl100k_base", "o200k_base"]:
            return byte_length // 10, byte_length // 2.5
        else:
            return byte_length // 10, byte_length // 2
    except Exception as e:
        raise ValueError(f"Unknown tokenizer {tokenizer}") from e


def estimate_tokens_using_charcount(text, tokenizer):
    length = len(text)
    if tokenizer in ["cl100k_base", "o200k_base"]:
        return length // 10, length // 1.5
    else:
        return length // 10, length
