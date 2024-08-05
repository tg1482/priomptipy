# Constants for model names
GPT_3_5_TURBO = "gpt-3.5-turbo"
GPT_3_5_TURBO_NIGHTLY_0613 = "gpt-3.5-turbo-0613"
GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
GPT3_3_5_TURBO_DOTHISFORME = "gpt-ft-cursor-0810"
GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
GPT_4 = "gpt-4"
GPT_4_NIGHTLY_0613 = "gpt-4-0613"
GPT_4_OMNI = "gpt-4o"
GPT_4_OMNI_MINI = "gpt-4o-mini"
GPT_4_32K = "gpt-4-32k"
GPT_4_32K_NIGHTLY_0613 = "gpt-4-32k-0613"
AZURE_3_5_TURBO = "azure-3.5-turbo"
TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
TEXT_DAVINCI_003 = "text-davinci-003"
CL100K_BASE = "cl100k_base"
R50K_BASE = "r50k_base"
P50K_BASE = "p50k_base"
GPT2_TOKENIZER = "gpt2"
GPT_3_5_FINETUNE_CPP = "ft:gpt-3.5-turbo-0613:anysphere::8ERu98np"
CODE_LLAMA_RERANKER = "codellama_7b_reranker"

# Arrays (lists in Python) of usable models
usable_models = [
    GPT_3_5_TURBO,
    GPT_3_5_TURBO_NIGHTLY_0613,
    GPT_3_5_TURBO_INSTRUCT,
    GPT3_3_5_TURBO_DOTHISFORME,
    GPT_4,
    GPT_4_NIGHTLY_0613,
    GPT_4_OMNI,
    GPT_4_OMNI_MINI,
    GPT_3_5_TURBO_16K,
    GPT_4_32K,
    GPT_4_32K_NIGHTLY_0613,
    AZURE_3_5_TURBO,
    TEXT_EMBEDDING_ADA_002,
    TEXT_DAVINCI_003,
    GPT_3_5_FINETUNE_CPP,
    CODE_LLAMA_RERANKER,
]

usable_language_models = [
    GPT_3_5_TURBO,
    GPT_3_5_TURBO_NIGHTLY_0613,
    GPT_3_5_TURBO_16K,
    GPT_3_5_TURBO_INSTRUCT,
    GPT3_3_5_TURBO_DOTHISFORME,
    GPT_4,
    GPT_4_NIGHTLY_0613,
    GPT_4_OMNI,
    GPT_4_OMNI_MINI,
    GPT_4_32K,
    GPT_4_32K_NIGHTLY_0613,
    AZURE_3_5_TURBO,
    GPT_3_5_FINETUNE_CPP,
    CODE_LLAMA_RERANKER,
]


# Function to check if a model is a usable language model
def is_usable_language_model(s: str) -> bool:
    return s in usable_language_models


usable_tokenizers = [CL100K_BASE, R50K_BASE, P50K_BASE, GPT2_TOKENIZER]

# Dictionaries for model contexts and max tokens
MODEL_CONTEXTS = {
    GPT_3_5_TURBO: 2000,
    GPT_3_5_FINETUNE_CPP: 2000,
    GPT_3_5_TURBO_NIGHTLY_0613: 2000,
    GPT_3_5_TURBO_16K: 10000,
    GPT3_3_5_TURBO_DOTHISFORME: 4000,
    AZURE_3_5_TURBO: 2000,
    GPT_3_5_TURBO_INSTRUCT: 4000,
    GPT_4: 4000,
    GPT_4_OMNI: 128_000,
    GPT_4_OMNI_MINI: 128_000,
    GPT_4_32K: 32000,
    GPT_4_NIGHTLY_0613: 4000,
    GPT_4_32K_NIGHTLY_0613: 32000,
    CODE_LLAMA_RERANKER: 4000,
}

MAX_TOKENS = {
    GPT_3_5_TURBO: 4096,
    GPT_3_5_FINETUNE_CPP: 4096,
    GPT_3_5_TURBO_NIGHTLY_0613: 4096,
    GPT_3_5_TURBO_16K: 16384,
    GPT_3_5_TURBO_INSTRUCT: 4096,
    GPT3_3_5_TURBO_DOTHISFORME: 4000,
    AZURE_3_5_TURBO: 4096,
    GPT_4: 8000,
    GPT_4_NIGHTLY_0613: 8000,
    GPT_4_OMNI: 128_000,
    GPT_4_OMNI_MINI: 128_000,
    GPT_4_32K: 32000,
    GPT_4_32K_NIGHTLY_0613: 32000,
    CODE_LLAMA_RERANKER: 4000,
}

EMBEDDING_MODEL = "text-embedding-ada-002"

# docs here: https://platform.openai.com/docs/guides/chat/introduction (out of date!)
# linear factor is <|im_start|>system<|im_sep|>  and <|im_end|>
CHATML_PROMPT_EXTRA_TOKEN_COUNT_LINEAR_FACTOR = 4
CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT = 3
