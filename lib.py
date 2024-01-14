import os
import time
from typing import List, Union, Optional
from dataclasses import dataclass
import asyncio
from prompt_types import (
    BaseProps,
    Node,
    ChatMessage,
    ChatPrompt,
    Empty,
    First,
    RenderedPrompt,
    PromptElement,
    Scope,
    FunctionDefinition,
    FunctionPrompt,
    TextPrompt,
    ChatAndFunctionPromptFunction,
    ChatPromptMessage,
    ChatUserSystemMessage,
    ChatAssistantMessage,
    ChatFunctionResultMessage,
    Capture,
    OutputHandler,
    PromptProps,
    CaptureProps,
    BasePromptProps,
    ReturnProps,
    Isolate,
    RenderOutput,
    RenderOptions,
    PromptString,
    Prompt,
    BreakToken,
)

from openai_helper import (
    CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT,
    CHATML_PROMPT_EXTRA_TOKEN_COUNT_LINEAR_FACTOR,
    MAX_TOKENS,
    usable_tokenizers,
    is_usable_language_model,
    usable_language_models,
)
from output_cache import OutputCatcher
from tokenizer import get_tokenizer_name, num_tokens


# Type Checking Functions


def is_chat_prompt(prompt):
    return isinstance(prompt, dict) and prompt.get("type") == "chat"


def is_plain_prompt(prompt):
    return isinstance(prompt, str) or isinstance(prompt, list)


def is_text_prompt_potentially_with_functions(prompt):
    return (isinstance(prompt, dict) and "text" in prompt) or isinstance(prompt, str)


def prompt_has_functions(prompt):
    return isinstance(prompt, dict) and "functions" in prompt and prompt["functions"] is not None


# Utility Functions


def prompt_string_to_string(prompt_string):
    return "".join(prompt_string) if isinstance(prompt_string, list) else prompt_string


def prompt_get_text(prompt):
    if not is_text_prompt_potentially_with_functions(prompt):
        return None
    return prompt_string_to_string(prompt) if is_plain_prompt(prompt) else prompt_string_to_string(prompt["text"])


def sum_prompt_strings(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return a[:-1] + [a[-1] + b[0]] + b[1:]
    if isinstance(a, list):
        return a[:-1] + [a[-1] + b]
    if isinstance(b, list):
        return [a + b[0]] + b[1:]
    return a + b


def sum_prompts(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if (
        (is_chat_prompt(a) and is_chat_prompt(b))
        or (is_chat_prompt(a) and prompt_get_text(b) == "")
        or (is_chat_prompt(b) and prompt_get_text(a) == "")
    ):
        functions = (a.get("functions", []) if prompt_has_functions(a) else []) + (
            b.get("functions", []) if prompt_has_functions(b) else []
        )
        prompt = {
            "type": "chat",
            "messages": a.get("messages", []) + b.get("messages", []),
        }
        if functions:
            prompt["functions"] = functions
        return prompt
    if (
        (prompt_has_functions(a) or prompt_has_functions(b))
        and is_text_prompt_potentially_with_functions(a)
        and is_text_prompt_potentially_with_functions(b)
    ):
        functions = (a.get("functions", []) if prompt_has_functions(a) else []) + (
            b.get("functions", []) if prompt_has_functions(b) else []
        )
        prompt_text = sum_prompt_strings(
            a["text"] if not is_plain_prompt(a) else a,
            b["text"] if not is_plain_prompt(b) else b,
        )
        return {"type": "text", "text": prompt_text, "functions": functions}
    if is_plain_prompt(a) and is_plain_prompt(b):
        return sum_prompt_strings(a, b)
    raise ValueError(f"Cannot sum prompts: {a} and {b}")


def is_development_environment():
    return os.environ.get("ENVIRONMENT") == "development"


def create_element(tag, props=None, *children) -> PromptElement:
    if callable(tag):
        # When tag is a function
        combined_props = {**props, "children": children} if props else {"children": children}
        return {
            "type": "scope",
            "children": [tag(combined_props)],
            "absolute_priority": props.get("p") if props and isinstance(props.get("p"), int) else None,
            "relative_priority": props.get("prel") if props and isinstance(props.get("prel"), int) else None,
        }

    if not isinstance(tag, str):
        raise ValueError(f"tag must be a string or a function, got {tag}")

    # Handling different string tags
    if tag == "scope":
        return {
            "type": "scope",
            "children": list(children),
            "relative_priority": props.get("prel") if props and isinstance(props.get("prel"), int) else None,
            "absolute_priority": props.get("p") if props and isinstance(props.get("p"), int) else None,
            "on_eject": props.get("on_eject") if props and callable(props.get("on_eject")) else None,
            "on_include": props.get("on_include") if props and callable(props.get("on_include")) else None,
        }
    elif tag == "br":
        if children:
            raise ValueError("br tag must have no children")
        return {
            "type": "scope",
            "children": ["\n"],
            "absolute_priority": props.get("p") if props and isinstance(props.get("p"), int) else None,
            "relative_priority": props.get("prel") if props and isinstance(props.get("prel"), int) else None,
        }

    elif tag == "breaktoken":
        if children:
            raise ValueError("breaktoken tag must have no children")
        return {
            "type": "scope",
            "children": [{"type": "breaktoken"}],
            "absolute_priority": props.get("p") if props and isinstance(props.get("p"), int) else None,
            "relative_priority": props.get("prel") if props and isinstance(props.get("prel"), int) else None,
        }

    elif tag == "hr":
        if children:
            raise ValueError("hr tag must have no children")
        return {
            "type": "scope",
            "children": ["\n\n-------\n\n"],
            "absolute_priority": props.get("p") if props and isinstance(props.get("p"), int) else None,
            "relative_priority": props.get("prel") if props and isinstance(props.get("prel"), int) else None,
        }

    elif tag == "first":
        if not all(isinstance(child, dict) and child.get("type") == "scope" for child in children):
            raise ValueError("first tag must have only scope children")
        return {
            "type": "first",
            "children": list(children),
            "on_eject": props.get("on_eject") if props and callable(props.get("on_eject")) else None,
            "on_include": props.get("on_include") if props and callable(props.get("on_include")) else None,
        }

    elif tag == "empty":
        if children:
            raise ValueError("empty tag must have no children")
        if not props or not isinstance(props.get("tokens"), int):
            raise ValueError("empty tag must have a tokens prop")
        return {
            "type": "scope",
            "children": [{"type": "empty", "tokenCount": props["tokens"]}],
            "absolute_priority": props.get("p") if props and isinstance(props.get("p"), int) else None,
            "relative_priority": props.get("prel") if props and isinstance(props.get("prel"), int) else None,
        }

    elif tag == "isolate":
        if not props or not isinstance(props.get("tokenLimit"), int):
            raise ValueError("isolate tag must have a tokenLimit prop")
        return {
            "type": "scope",
            "children": [{"type": "isolate", "tokenLimit": props["tokenLimit"], "children": list(children)}],
            "absolute_priority": props.get("p") if props and isinstance(props.get("p"), int) else None,
            "relative_priority": props.get("prel") if props and isinstance(props.get("prel"), int) else None,
        }

    elif tag == "capture":
        if children:
            raise ValueError("capture tag must have no children")
        if not props or "on_output" not in props or not callable(props["on_output"]):
            raise ValueError("capture tag must have an on_output prop that's a function")
        if "on_stream" in props and not callable(props["on_stream"]):
            raise ValueError("capture tag must have an on_stream prop and it must be a function")
        return {
            "type": "scope",
            "children": [{"type": "capture", "on_output": props["on_output"], "on_stream": props.get("on_stream")}],
            "absolute_priority": props.get("p") if props and isinstance(props.get("p"), int) else None,
            "relative_priority": props.get("prel") if props and isinstance(props.get("prel"), int) else None,
        }

    else:
        raise ValueError(f"Unknown tag {tag}")


# # Example usage of create_element function
# def example_tag(props):
#     return props.get("children")


# element = create_element(example_tag, {"p": 1}, "child1", "child2")


def Fragment(children: List[PromptElement]) -> PromptElement:
    """
    Merge all the children elements into a single list.

    Args:
    children (list): A list of PromptElements.

    Returns:
    list: A flattened list of PromptElements.
    """
    # Flatten the list of children, equivalent to JavaScript's array.flat()
    return [item for sublist in children for item in sublist]


# BASE_PRIORITY constant
BASE_PRIORITY = 1e9


async def render(elem: PromptElement, options: RenderOptions) -> RenderOutput:
    """
    Render a PromptElement into a RenderOutput object.

    Args:
    elem (PromptElement): The PromptElement to render.
    options (RenderOptions): The RenderOptions object to use.

    Returns:
    RenderOutput: The RenderOutput object.
    """
    return render_binary_search(elem, options)


async def render_binary_search(elem: PromptElement, options: RenderOptions) -> RenderOutput:
    start_time = time.time() if is_development_environment() else None

    token_limit = options.get("token_limit", MAX_TOKENS.get(options.get("model")))
    if token_limit is None:
        raise ValueError("Must specify model or tokenLimit")

    tokenizer = options.get("tokenizer", get_tokenizer_name(options.get("model")))
    if tokenizer is None:
        raise ValueError("Must specify model or tokenizer")

    # Validate prompt
    start_time_validating = time.time() if is_development_environment() else None
    validate_unrendered_prompt(elem)
    if is_development_environment():
        print(f"Validating prompt took {time.time() - start_time_validating} ms")

    # Compute priority levels
    priority_levels = set()
    compute_priority_levels(elem, BASE_PRIORITY, priority_levels)
    priority_levels.add(BASE_PRIORITY)
    sorted_priority_levels = sorted(priority_levels)

    # Hydrate isolates
    await hydrate_isolates(elem, tokenizer)

    # Binary search logic
    exclusive_lower_bound = -1
    inclusive_upper_bound = len(sorted_priority_levels) - 1

    while exclusive_lower_bound < inclusive_upper_bound - 1:
        candidate_level_index = (exclusive_lower_bound + inclusive_upper_bound) // 2
        candidate_level = sorted_priority_levels[candidate_level_index]
        start = time.time() if is_development_environment() else None
        token_count = -1

        try:
            prompt = render_with_level_and_early_exit_with_token_estimation(elem, candidate_level, tokenizer, token_limit)
            token_count = await count_tokens_exact(tokenizer, prompt.get("prompt", ""), options)
            if token_count + prompt.get("empty_token_count", 0) > token_limit:
                exclusive_lower_bound = candidate_level_index
            else:
                inclusive_upper_bound = candidate_level_index
        except Exception:
            exclusive_lower_bound = candidate_level_index
        finally:
            if is_development_environment():
                end = time.time()
                print(f"Candidate level {candidate_level} took {end - start} ms and has {token_count} tokens")

    # Final rendering
    final_prompt = render_with_level(elem, sorted_priority_levels[inclusive_upper_bound], tokenizer, True)
    token_count = await count_tokens_exact(tokenizer, final_prompt.get("prompt", ""), options)

    if token_count + final_prompt.get("empty_token_count", 0) > token_limit:
        raise ValueError(
            f"Base prompt estimated token count is {token_count} with {final_prompt.get('empty_token_count', 0)} tokens reserved, which is higher than the limit {token_limit}."
        )

    duration_ms = (time.time() - start_time) * 1000 if start_time is not None else None

    return {
        "prompt": final_prompt.get("prompt", ""),
        "token_count": token_count,
        "tokens_reserved": final_prompt.get("empty_token_count", 0),
        "token_limit": token_limit,
        "tokenizer": tokenizer,
        "duration_ms": duration_ms,
        "output_handlers": final_prompt.get("output_handlers", []),
        "stream_handlers": final_prompt.get("stream_handlers", []),
        "priority_cutoff": sorted_priority_levels[inclusive_upper_bound],
    }


async def render_backwards_linear_search(elem: PromptElement, options: RenderOptions) -> RenderOutput:
    start_time = time.time() if is_development_environment() else None

    token_limit = options.get("tokenLimit", MAX_TOKENS.get(options.get("model")))
    if token_limit is None:
        raise ValueError("Must specify model or tokenLimit")

    tokenizer = options.get("tokenizer", get_tokenizer_name(options.get("model")))
    if tokenizer is None:
        raise ValueError("Must specify model or tokenizer")

    # Validate prompt
    if is_development_environment():
        start_time_validating = time.time()
        validate_unrendered_prompt(elem)
        print(f"Validating prompt took {time.time() - start_time_validating} ms")

    # Normalize the prompt
    normalized_elem = normalize_prompt(elem)

    # Compute priority levels
    priority_levels = set()
    compute_priority_levels(normalized_elem, BASE_PRIORITY, priority_levels)
    priority_levels.add(BASE_PRIORITY)
    sorted_priority_levels = sorted(priority_levels, reverse=True)

    # Render and count tokens
    prev_prompt = None
    prev_level = None
    for level in sorted_priority_levels:
        this_prompt = await render_with_level_and_count_tokens(normalized_elem, level, tokenizer)
        if is_chat_prompt(this_prompt["prompt"]):
            this_prompt["token_count"] += CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT

        if this_prompt["token_count"] + this_prompt["empty_token_count"] > token_limit:
            break

        prev_prompt = this_prompt
        prev_level = level

    if prev_prompt is None:
        raise ValueError(f"Base prompt estimated token count is too high for the limit {token_limit}.")

    # Get the actual token count
    if prev_prompt["prompt"] is not None:
        exact_token_count = await count_tokens_exact(tokenizer, prev_prompt["prompt"], options)
        prev_prompt["tokenCount"] = exact_token_count

    duration_ms = (time.time() - start_time) * 1000 if start_time is not None else None

    return {
        "prompt": prev_prompt.get("prompt", ""),
        "token_count": prev_prompt.get("token_count", 0),
        "tokens_reserved": prev_prompt.get("empty_token_count", 0),
        "tokenLimit": token_limit,
        "stream_handlers": prev_prompt.get("stream_handlers", []),
        "duration_ms": duration_ms,
        "priority_cutoff": prev_level if prev_level is not None else BASE_PRIORITY,
    }


# Additional types
@dataclass
class NormalizedString:
    type: str
    s: str
    cached_count: Optional[int] = None


@dataclass
class NormalizedScope(Scope):
    children: List["NormalizedNode"]


@dataclass
class NormalizedFirst(First):
    children: List[NormalizedScope]


@dataclass
class NormalizedChatUserSystemMessage(ChatUserSystemMessage):
    children: List["NormalizedNode"]


@dataclass
class NormalizedChatAssistantMessage(ChatAssistantMessage):
    children: List["NormalizedNode"]


@dataclass
class NormalizedChatFunctionResultMessage(ChatFunctionResultMessage):
    children: List["NormalizedNode"]


@dataclass
class NormalizedFunctionDefinition(FunctionDefinition):
    cached_count: Optional[int] = None


# Union type for NormalizedChatMessage
NormalizedChatMessage = Union[NormalizedChatUserSystemMessage, NormalizedChatAssistantMessage, NormalizedChatFunctionResultMessage]

# Union type for NormalizedNode
NormalizedNode = Union[
    NormalizedFirst,
    NormalizedScope,
    BreakToken,
    Empty,
    Isolate,
    Capture,
    NormalizedChatMessage,
    NormalizedString,
    NormalizedFunctionDefinition,
]


# Function to normalize prompt
def normalize_prompt(elem: PromptElement) -> List[NormalizedNode]:
    result: List[NormalizedNode] = []
    current_string: str = ""
    elem_array = elem if isinstance(elem, list) else [elem]

    def push_current_string():
        nonlocal current_string
        if current_string:
            result.append(NormalizedString(type="normalized_string", s=current_string))
            current_string = ""

    for node in elem_array:
        if node is None:
            continue
        if isinstance(node, str) or isinstance(node, (int, float, bool)):
            current_string += str(node)
        elif isinstance(node, dict):
            push_current_string()
            if node["type"] in ["capture", "isolate", "breaktoken", "empty"]:
                result.append(node)
            elif node["type"] == "function_definition":
                result.append(NormalizedFunctionDefinition(**node, cached_count=None))
            elif node["type"] == "first":
                result.append(NormalizedFirst(**node, children=normalize_prompt(node["children"])))
            elif node["type"] in ["chat", "scope"]:
                result.append(type(node)(**node, children=normalize_prompt(node["children"])))
            else:
                raise ValueError("Invalid prompt element")
        else:
            raise ValueError("Invalid prompt element")
    push_current_string()
    return result


async def render_with_level_and_count_tokens(elem: PromptElement, level: int, tokenizer: str) -> RenderedPrompt:
    if isinstance(elem, list):
        results = await asyncio.gather(*[render_with_level_and_count_tokens(e, level, tokenizer) for e in elem])
        return {
            "prompt": sum_prompts(*[r["prompt"] for r in results]),
            "token_count": sum(r["token_count"] for r in results),
            "empty_token_count": sum(r["empty_token_count"] for r in results),
            "output_handlers": [handler for r in results for handler in r["output_handlers"]],
            "stream_handlers": [handler for r in results for handler in r["stream_handlers"]],
        }

    match elem.type:
        case "first":
            for child in elem.children:
                if child.absolute_priority is None:
                    raise ValueError("compute_priority_levels should have set absolute_priority for all children of first")
                if child.absolute_priority >= level:
                    return await render_with_level_and_count_tokens(child, level, tokenizer)
            return {"prompt": None, "token_count": 0, "empty_token_count": 0, "output_handlers": [], "stream_handlers": []}

        case "capture":
            return {
                "prompt": None,
                "token_count": 0,
                "empty_token_count": 0,
                "output_handlers": [elem.on_output] if elem.on_output else [],
                "stream_handlers": [elem.on_stream] if elem.on_stream else [],
            }

        case "breaktoken":
            return {"prompt": ["", ""], "token_count": 0, "empty_token_count": 0, "output_handlers": [], "stream_handlers": []}

        case "empty":
            return {"prompt": None, "token_count": 0, "empty_token_count": elem.token_count, "output_handlers": [], "stream_handlers": []}

        case "function_definition":
            if elem.cached_count is None:
                elem.cached_count = await count_function_tokens(elem, tokenizer)
            prompt = {
                "type": "text",
                "text": "",
                "functions": [
                    {
                        "name": elem.name,
                        "description": elem.description,
                        "parameters": elem.parameters,
                    }
                ],
            }
            return {
                "prompt": prompt,
                "token_count": elem.cached_count,
                "empty_token_count": 0,
                "output_handlers": [],
                "stream_handlers": [],
            }

        case "isolate":
            if elem.cached_render_output is None:
                elem.cached_render_output = await render(
                    elem.children,
                    {
                        "tokenizer": tokenizer,
                        "token_limit": elem.token_limit,
                    },
                )
            return {
                "prompt": elem.cached_render_output.prompt,
                "token_count": elem.cached_render_output.token_count,
                "empty_token_count": elem.cached_render_output.tokens_reserved,
                "output_handlers": elem.cached_render_output.output_handlers,
                "stream_handlers": elem.cached_render_output.stream_handlers,
            }

        case "chat":
            p = await render_with_level_and_count_tokens(elem.children, level, tokenizer)
            if is_chat_prompt(p["prompt"]):
                raise ValueError("Incorrect prompt: nested chat messages are not allowed")

            extra_token_count = 0
            message = {"role": elem.role, "name": elem.name, "content": ""}
            if elem.role in ["user", "system"]:
                message["content"] = p["prompt"] if is_plain_prompt(p["prompt"]) else (p["prompt"]["text"] if p["prompt"] else "")
            elif elem.role == "assistant":
                message["content"] = p["prompt"] if is_plain_prompt(p["prompt"]) else (p["prompt"]["text"] if p["prompt"] else "")
                if elem.function_call:
                    message["function_call"] = elem.function_call
                    extra_token_count += await count_function_call_message_tokens(elem.function_call, tokenizer)
            elif elem.role == "function":
                message["content"] = p["prompt"] if is_plain_prompt(p["prompt"]) else (p["prompt"]["text"] if p["prompt"] else "")
                extra_token_count += await num_tokens(elem.name, {"tokenizer": tokenizer})

            return {
                "prompt": {"type": "chat", "messages": [message], "functions": prompt_has_functions(p["prompt"]) if p["prompt"] else None},
                "token_count": p["token_count"] + CHATML_PROMPT_EXTRA_TOKEN_COUNT_LINEAR_FACTOR + extra_token_count,
                "empty_token_count": p["empty_token_count"],
                "output_handlers": p["output_handlers"],
                "stream_handlers": p["stream_handlers"],
            }

        case "normalizedString":
            if elem.cached_count is None:
                elem.cached_count = await num_tokens(elem.s, {"tokenizer": tokenizer})
            return {
                "prompt": elem.s,
                "token_count": elem.cached_count,
                "empty_token_count": 0,
                "output_handlers": [],
                "stream_handlers": [],
            }

        case "scope":
            if elem.absolute_priority is None:
                raise ValueError("compute_priority_levels should have set absolute_priority for all scopes")
            if elem.absolute_priority >= level:
                return await render_with_level_and_count_tokens(elem.children, level, tokenizer)
            return {"prompt": None, "token_count": 0, "empty_token_count": 0, "output_handlers": [], "stream_handlers": []}

    return {"prompt": None, "token_count": 0, "empty_token_count": 0, "output_handlers": [], "stream_handlers": []}


def render_with_level_and_early_exit_with_token_estimation(
    elem: PromptElement, level: int, tokenizer: str, token_limit: int
) -> RenderedPrompt:
    if elem is None or elem is False:
        return {"prompt": None, "empty_token_count": 0}

    if isinstance(elem, list):
        results = [render_with_level_and_early_exit_with_token_estimation(e, level, tokenizer, token_limit) for e in elem]
        prompt_sum = sum_prompts(*[r["prompt"] for r in results])
        lower_bound = estimate_lower_bound_tokens_for_prompt(prompt_sum, tokenizer)
        if lower_bound > token_limit:
            raise ValueError("Token limit exceeded!")
        return {"prompt": prompt_sum, "empty_token_count": sum(r["empty_token_count"] for r in results)}

    if isinstance(elem, str) or isinstance(elem, (int, float)):
        return {"prompt": str(elem), "empty_token_count": 0}

    match elem.type:
        case "first":
            for child in elem.children:
                if child.absolute_priority is None:
                    raise ValueError("compute_priority_levels should have set absolute_priority for all children of first")
                if child.absolute_priority >= level:
                    return render_with_level_and_early_exit_with_token_estimation(child, level, tokenizer, token_limit)
            return {"prompt": None, "empty_token_count": 0}

        case "capture":
            return {"prompt": None, "empty_token_count": 0}

        case "breaktoken":
            return {"prompt": ["", ""], "empty_token_count": 0}

        case "empty":
            return {"prompt": None, "empty_token_count": elem.token_count}

        case "function_definition":
            prompt = {
                "type": "text",
                "text": "",
                "functions": [
                    {
                        "name": elem.name,
                        "description": elem.description,
                        "parameters": elem.parameters,
                    }
                ],
            }
            return {"prompt": prompt, "empty_token_count": 0}

        case "isolate":
            if elem.cached_render_output is None:
                raise ValueError("Isolates should have been hydrated before calling render_with_level_and_early_exit_with_token_estimation")
            return {
                "prompt": elem.cached_render_output.prompt,
                "empty_token_count": elem.cached_render_output.tokens_reserved,
            }

        case "chat":
            p = render_with_level_and_early_exit_with_token_estimation(elem.children, level, tokenizer, token_limit)
            if is_chat_prompt(p["prompt"]):
                raise ValueError("Incorrect prompt: nested chat messages are not allowed!")

            message = create_chat_message(elem, p["prompt"])
            return {
                "prompt": {"type": "chat", "messages": [message], "functions": prompt_has_functions(p["prompt"]) if p["prompt"] else None},
                "empty_token_count": p["empty_token_count"],
            }

        case "scope":
            if elem.absolute_priority is None:
                raise ValueError("compute_priority_levels should have set absolute_priority for all scopes")
            if elem.absolute_priority >= level:
                return render_with_level_and_early_exit_with_token_estimation(elem.children, level, tokenizer, token_limit)
            return {"prompt": None, "empty_token_count": 0}


def recursively_eject(elem: PromptElement):
    if elem is None or isinstance(elem, (str, int, float, bool)):
        return

    if isinstance(elem, list):
        for e in elem:
            recursively_eject(e)
    else:
        if hasattr(elem, "on_eject") and callable(elem.on_eject):
            elem.on_eject()

        if hasattr(elem, "children") and isinstance(elem.children, list):
            for child in elem.children:
                recursively_eject(child)


async def hydrate_isolates(elem: PromptElement, tokenizer: str) -> None:
    if elem is None or isinstance(elem, (str, int, float, bool)):
        return

    if isinstance(elem, list):
        tasks = [hydrate_isolates(e, tokenizer) for e in elem]
        # Run tasks concurrently and wait for them to finish
        await asyncio.gather(*tasks)
        return

    match elem.type:
        case "first":
            await hydrate_isolates(elem.children, tokenizer)

        case "capture" | "empty" | "breaktoken" | "function_definition":
            return

        case "isolate":
            if elem.cached_render_output is None:
                elem.cached_render_output = await render(
                    elem.children,
                    {
                        "tokenizer": tokenizer,
                        "token_limit": elem.token_limit,
                    },
                )

        case "chat" | "scope":
            await hydrate_isolates(elem.children, tokenizer)


def render_with_level(elem, level, tokenizer, call_ejected_callback=False):
    if elem is None or elem is False:
        return {"prompt": None, "empty_token_count": 0, "output_handlers": [], "stream_handlers": []}

    if isinstance(elem, list):
        results = [render_with_level(e, level, tokenizer, call_ejected_callback) for e in elem]
        prompt_sum = sum_prompts(*[r["prompt"] for r in results])
        return {
            "prompt": prompt_sum,
            "empty_token_count": sum(r["empty_token_count"] for r in results),
            "output_handlers": [handler for r in results for handler in r["output_handlers"]],
            "stream_handlers": [handler for r in results for handler in r["stream_handlers"]],
        }

    if isinstance(elem, (str, int)):
        return {"prompt": str(elem), "empty_token_count": 0, "output_handlers": [], "stream_handlers": []}

    match elem.type:
        case "first":
            for child in elem.children:
                if child.absolute_priority is None:
                    raise ValueError("compute_priority_levels should have set absolute_priority for all children of first")
                if child.absolute_priority >= level:
                    if hasattr(elem, "on_include") and callable(elem.on_include):
                        elem.on_include()
                    return render_with_level(child, level, tokenizer, call_ejected_callback)
                elif call_ejected_callback:
                    recursively_eject(child)
            return {"prompt": None, "empty_token_count": 0, "output_handlers": [], "stream_handlers": []}

        case "capture":
            return {
                "prompt": None,
                "empty_token_count": 0,
                "output_handlers": [elem.on_output] if hasattr(elem, "on_output") and callable(elem.on_output) else [],
                "stream_handlers": [elem.on_stream] if hasattr(elem, "on_stream") and callable(elem.on_stream) else [],
            }

        case "breaktoken":
            return {"prompt": ["", ""], "empty_token_count": 0, "output_handlers": [], "stream_handlers": []}

        case "empty":
            return {"prompt": None, "empty_token_count": elem.token_count, "output_handlers": [], "stream_handlers": []}

        case "functionDefinition":
            prompt = {
                "type": "text",
                "text": "",
                "functions": [
                    {
                        "name": elem.name,
                        "description": elem.description,
                        "parameters": elem.parameters,
                    }
                ],
            }
            return {"prompt": prompt, "empty_token_count": 0, "output_handlers": [], "stream_handlers": []}

        case "isolate":
            if elem.cached_render_output is None:
                raise ValueError("Isolates should have been hydrated before calling render_with_level")
            return {
                "prompt": elem.cached_render_output.prompt,
                "empty_token_count": elem.cached_render_output.tokens_reserved,
                "output_handlers": elem.cached_render_output.output_handlers,
                "stream_handlers": elem.cached_render_output.stream_handlers,
            }

        case "chat":
            child_results = render_with_level(elem.children, level, tokenizer, call_ejected_callback)
            if is_chat_prompt(child_results["prompt"]):
                raise ValueError("Incorrect prompt: nested chat messages are not allowed!")

            # Construct the chat message based on the role and content
            message_content = (
                child_results["prompt"]
                if is_plain_prompt(child_results["prompt"])
                else child_results["prompt"]["text"]
                if child_results["prompt"]
                else ""
            )
            message = {"role": elem.role, "name": getattr(elem, "name", None), "content": message_content}

            if elem.role == "assistant" and getattr(elem, "function_call", None):
                message["function_call"] = elem.function_call

            return {
                "prompt": {"type": "chat", "messages": [message]},
                "empty_token_count": child_results["empty_token_count"],
                "output_handlers": child_results["output_handlers"],
                "stream_handlers": child_results["stream_handlers"],
            }

        # Not fully well defined and converted yet

        case "scope":
            if elem.absolute_priority is None:
                raise ValueError("compute_priority_levels should have set absolute_priority for all scopes")
            if elem.absolute_priority >= level:
                if hasattr(elem, "on_include") and callable(elem.on_include):
                    elem.on_include()
                return render_with_level(elem.children, level, tokenizer, call_ejected_callback)
            elif call_ejected_callback:
                recursively_eject(elem)
            return {"prompt": None, "empty_token_count": 0, "output_handlers": [], "stream_handlers": []}


# Helper functions like sum_prompts, recursively_eject, create_chat_message, etc.
# should be implemented based on your application's logic.
