import tiktoken


def get_tokenizer_name(model):
    cl100k_base_models = {
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "gpt-4o",
        "gpt-4o-mini",
        "text-embedding-ada-002",
        "ft:gpt-3.5-turbo-0613:anysphere::8ERu98np",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k",
        "codellama_7b_reranker",
        "gpt-3.5-turbo-instruct",
        "azure-3.5-turbo",
        "gpt-ft-cursor-0810",
    }
    p50k_base_models = {"text-davinci-003"}

    if model in cl100k_base_models:
        return "cl100k_base"
    elif model in p50k_base_models:
        return "p50k_base"
    else:
        return None


async def num_tokens(text, model=None, tokenizer=None):
    tokenizer_name = tokenizer if tokenizer else get_tokenizer_name(model)
    if tokenizer_name == "cl100k_base":
        encoding = tiktoken.get_encoding(tokenizer_name)
        return len(encoding.encode(text))
    else:
        raise ValueError(f"Unknown tokenizer {tokenizer_name}")


async def encode_tokens(text, model=None, tokenizer=None):
    tokenizer_name = tokenizer if tokenizer else get_tokenizer_name(model)
    if tokenizer_name == "cl100k_base":
        encoding = tiktoken.get_encoding(tokenizer_name)
        return encoding.encode(text)
    else:
        raise ValueError(f"Unknown tokenizer {tokenizer_name}")


def estimate_tokens_using_bytecount(text, tokenizer):
    byte_length = len(text.encode("utf-8"))
    if tokenizer == "cl100k_base":
        return byte_length // 10, byte_length // 2.5
    else:
        return byte_length // 10, byte_length // 2


def estimate_tokens_using_charcount(text, tokenizer):
    length = len(text)
    if tokenizer == "cl100k_base":
        return length // 10, length // 1.5
    else:
        return length // 10, length
