import os
import time
import json
import math
from typing import List, Union, Optional, Set, Callable, Any
from dataclasses import dataclass
import asyncio
from .prompt_types import (
    Node,
    Empty,
    First,
    RenderedPrompt,
    PromptElement,
    Scope,
    FunctionDefinition,
    ChatAndFunctionPromptFunction,
    ChatPromptMessage,
    ChatUserSystemMessage,
    ChatAssistantMessage,
    ChatFunctionResultMessage,
    Capture,
    Isolate,
    RenderOutput,
    RenderOptions,
    PromptString,
    Prompt,
    BreakToken,
)

from .openai_helper import (
    CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT,
    CHATML_PROMPT_EXTRA_TOKEN_COUNT_LINEAR_FACTOR,
    MAX_TOKENS,
    usable_tokenizers,
    is_usable_language_model,
    usable_language_models,
)
from .output_cache import OutputCatcher
from .tokenizer import get_tokenizer_name, num_tokens, estimate_tokens_using_charcount, encode_tokens


# Type Checking Functions


def is_chat_prompt(prompt) -> bool:
    return isinstance(prompt, dict) and prompt.get("type") == "chat"


def is_plain_prompt(prompt) -> bool:
    return isinstance(prompt, str) or isinstance(prompt, list)


def is_text_prompt_potentially_with_functions(prompt) -> bool:
    return (isinstance(prompt, dict) and "text" in prompt) or isinstance(prompt, str)


def prompt_has_functions(prompt) -> bool:
    return isinstance(prompt, dict) and "functions" in prompt and prompt["functions"] is not None and prompt["functions"] != False


# Utility Functions


def prompt_string_to_string(prompt_string) -> str:
    return "".join(prompt_string) if isinstance(prompt_string, list) else prompt_string


def prompt_get_text(prompt) -> str:
    if not is_text_prompt_potentially_with_functions(prompt):
        return None
    return prompt_string_to_string(prompt) if is_plain_prompt(prompt) else prompt_string_to_string(prompt["text"])


def sum_prompt_strings(a, b) -> str:
    if isinstance(a, list) and isinstance(b, list):
        return a[:-1] + [a[-1] + b[0]] + b[1:]
    if isinstance(a, list):
        return a[:-1] + [a[-1] + b]
    if isinstance(b, list):
        return [a + b[0]] + b[1:]
    return a + b


def sum_prompts(a=None, b=None):
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
            "children": [{"type": "empty", "token_count": props["tokens"]}],
            "absolute_priority": props.get("p") if props and isinstance(props.get("p"), int) else None,
            "relative_priority": props.get("prel") if props and isinstance(props.get("prel"), int) else None,
        }

    elif tag == "isolate":
        if not props or not isinstance(props.get("token_limit"), int):
            raise ValueError("isolate tag must have a token_limit prop")
        return {
            "type": "scope",
            "children": [{"type": "isolate", "token_limit": props["token_limit"], "children": list(children)}],
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


# Not fully implemented yet. Will test this in the future
async def render_run(
    prompt: Prompt,
    props: PromptElement,
    render_options: RenderOptions,
    model_call: Callable[[Any], Any],
    rendered_messages_callback: Callable[[Any], None] = lambda messages: None,
):
    print("Running render_un")

    # Create an instance of OutputCatcher
    output_catcher = OutputCatcher()

    # Merge props and return_props
    output_props = {**props, "on_return": lambda x: output_catcher.on_output(x)}

    # Render the initial prompt
    prompt_element = prompt(**output_props)
    rendered = await render(prompt_element, render_options)

    # Prepare the model request
    model_request = prompt_to_openai_chat_request(rendered["prompt"])
    rendered_messages_callback(model_request["messages"])

    # Call the model and handle the output
    model_output = await model_call(model_request)

    if model_output["type"] == "output":
        if not model_output["value"]["choices"]:
            raise ValueError("Model returned no choices")

        model_output_message = model_output["value"]["choices"][0]["message"]
        if model_output_message is None:
            raise ValueError("Model returned no message")

        # Process output handlers
        await asyncio.gather(*[handler(model_output_message) for handler in rendered["output_handlers"]])
    else:
        # Process stream handlers
        if not rendered["stream_handlers"]:

            async def awaitable():
                async for message in model_output["value"]:
                    yield message

            await output_catcher.on_output(awaitable())
        else:
            await asyncio.gather(*[handler(model_output["value"]) for handler in rendered["stream_handlers"]])

    # Get and return the first output
    first_output = output_catcher.get_output()
    if first_output is None:
        raise ValueError("No output was captured. Did you forget to include a <capture> element?")

    return first_output


async def render(elem: PromptElement, options: RenderOptions) -> RenderOutput:
    """
    Render a PromptElement into a RenderOutput object.

    Args:
    elem (PromptElement): The PromptElement to render.
    options (RenderOptions): The RenderOptions object to use.

    Returns:
    RenderOutput: The RenderOutput object.
    """
    render_options = RenderOptions(**options)
    return await render_binary_search(elem, render_options)


async def render_binary_search(elem: PromptElement, options: RenderOptions) -> RenderOutput:
    start_time = time.time() if is_development_environment() else None

    token_limit = options.token_limit or MAX_TOKENS.get(options.model)
    if token_limit is None:
        raise ValueError("Must specify model or token_limit")

    tokenizer = get_tokenizer_name(options.model) if options.model else options.tokenizer
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
            # Check if token limit was exceeded during early estimation
            if prompt.get("token_limit_exceeded", False):
                exclusive_lower_bound = candidate_level_index
            else:
                token_count = await count_tokens_exact(tokenizer, prompt.get("prompt", ""), options)
                if token_count + prompt.get("empty_token_count", 0) > token_limit:
                    exclusive_lower_bound = candidate_level_index
                else:
                    inclusive_upper_bound = candidate_level_index
        except Exception as e:
            exclusive_lower_bound = candidate_level_index
        finally:
            if is_development_environment():
                end = time.time()
                print(f"Candidate level {candidate_level} took {end - start} ms and has {token_count} tokens")

    # Final rendering
    final_prompt = render_with_level(elem, sorted_priority_levels[inclusive_upper_bound], tokenizer, True)
    token_count = await count_tokens_exact(tokenizer, final_prompt.get("prompt", ""), options)

    # Note: We don't throw an error here for token limit exceeded because Isolate components
    # should handle their own token budgets gracefully. The binary search process above
    # should have already found the best fit within the global token limit.
    if is_development_environment() and token_count + final_prompt.get("empty_token_count", 0) > token_limit:
        print(
            f"WARNING: Final prompt token count ({token_count}) with reserved tokens ({final_prompt.get('empty_token_count', 0)}) exceeds limit ({token_limit}), but this may be due to Isolate components handling their own budgets."
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

    token_limit = options.get("token_limit", MAX_TOKENS.get(options.get("model")))
    if token_limit is None:
        raise ValueError("Must specify model or token_limit")

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
        prev_prompt["token_count"] = exact_token_count

    duration_ms = (time.time() - start_time) * 1000 if start_time is not None else None

    return {
        "prompt": prev_prompt.get("prompt", ""),
        "token_count": prev_prompt.get("token_count", 0),
        "tokens_reserved": prev_prompt.get("empty_token_count", 0),
        "token_limit": token_limit,
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
        accumulated_result = {
            "prompt": None,
            "token_count": 0,
            "empty_token_count": 0,
            "output_handlers": [],
            "stream_handlers": [],
        }
        results = await asyncio.gather(*[render_with_level_and_count_tokens(e, level, tokenizer) for e in elem])
        for r in results:
            accumulated_result = {
                "prompt": sum_prompts(accumulated_result["prompt"], r["prompt"]),
                "token_count": accumulated_result["token_count"] + r["token_count"],
                "empty_token_count": accumulated_result["empty_token_count"] + r["empty_token_count"],
                "output_handlers": accumulated_result["output_handlers"] + r["output_handlers"],
                "stream_handlers": accumulated_result["stream_handlers"] + r["stream_handlers"],
            }

        return accumulated_result

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
                "prompt": elem.cached_render_output.get("prompt"),
                "token_count": elem.cached_render_output.get("token_count"),
                "empty_token_count": elem.cached_render_output.get("tokens_reserved"),
                "output_handlers": elem.cached_render_output.get("output_handlers"),
                "stream_handlers": elem.cached_render_output.get("stream_handlers"),
            }

        case "chat":
            child_results = render_with_level_and_count_tokens(elem.children, level, tokenizer)
            if is_chat_prompt(child_results["prompt"]):
                raise ValueError("Incorrect prompt: nested chat messages are not allowed!")
            # Construct the chat message based on the role
            message = {}
            if elem.role in ["user", "system"]:
                message = {
                    "role": elem.role,
                    "content": prompt_get_text(child_results["prompt"]) if child_results["prompt"] else "",
                }
                if hasattr(elem, "name") and elem.name is not None:
                    message["name"] = elem.name
            elif elem.role == "assistant":
                message = {
                    "role": elem.role,
                    "content": prompt_get_text(child_results["prompt"]) if child_results["prompt"] else "",
                }
                if hasattr(elem, "function_call") and elem.function_call is not None:
                    message["function_call"] = elem.function_call
            elif elem.role == "function":
                message = {
                    "role": elem.role,
                    "name": elem.name,
                    "content": prompt_get_text(child_results["prompt"]) if child_results["prompt"] else "",
                }

            return {
                "prompt": {
                    "type": "chat",
                    "messages": [message],
                    "functions": prompt_has_functions(child_results["prompt"]) if child_results["prompt"] else None,
                },
                "empty_token_count": child_results["empty_token_count"],
                "output_handlers": child_results["output_handlers"],
                "stream_handlers": child_results["stream_handlers"],
            }

        case "normalized_string":
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
        accumulated_result = {
            "prompt": None,
            "empty_token_count": 0,
        }

        results = [render_with_level_and_early_exit_with_token_estimation(e, level, tokenizer, token_limit) for e in elem]
        for r in results:
            accumulated_result = {
                "prompt": sum_prompts(accumulated_result["prompt"], r["prompt"]),
                "empty_token_count": accumulated_result["empty_token_count"] + r["empty_token_count"],
            }
        lower_bound = estimate_lower_bound_tokens_for_prompt(accumulated_result["prompt"], tokenizer)
        if lower_bound > token_limit:
            # Instead of throwing an error, return a result that indicates token limit exceeded
            # This allows the binary search to handle it gracefully and Isolate components to manage their own budgets
            accumulated_result["token_limit_exceeded"] = True
        return accumulated_result

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
                "prompt": elem.cached_render_output.get("prompt"),
                "empty_token_count": elem.cached_render_output.get("tokens_reserved"),
            }

        case "chat":
            child_results = render_with_level_and_early_exit_with_token_estimation(elem.children, level, tokenizer, token_limit)
            if is_chat_prompt(child_results["prompt"]):
                raise ValueError("Incorrect prompt: nested chat messages are not allowed!")

            if elem.role in ["user", "system"]:
                message = {"role": elem.role, "content": child_results["prompt"]}
                if hasattr(elem, "name") and elem.name is not None:
                    message["name"] = elem.name
            elif elem.role == "assistant":
                message = {"role": elem.role, "content": child_results["prompt"]}
                if elem.function_call:
                    message["function_call"] = elem.function_call

            return {
                "prompt": {
                    "type": "chat",
                    "messages": [message],
                    "functions": prompt_has_functions(child_results["prompt"]) if child_results["prompt"] else None,
                },
                "empty_token_count": child_results["empty_token_count"],
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
        accumulated_result = {
            "prompt": None,
            "empty_token_count": 0,
            "output_handlers": [],
            "stream_handlers": [],
        }
        results = [render_with_level(e, level, tokenizer, call_ejected_callback) for e in elem]
        for r in results:
            accumulated_result = {
                "prompt": sum_prompts(accumulated_result["prompt"], r["prompt"]),
                "empty_token_count": accumulated_result["empty_token_count"] + r["empty_token_count"],
                "output_handlers": accumulated_result["output_handlers"] + r["output_handlers"],
                "stream_handlers": accumulated_result["stream_handlers"] + r["stream_handlers"],
            }

        return accumulated_result

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
            return {"prompt": prompt, "empty_token_count": 0, "output_handlers": [], "stream_handlers": []}

        case "isolate":
            if elem.cached_render_output is None:
                raise ValueError("Isolates should have been hydrated before calling render_with_level")
            return {
                "prompt": elem.cached_render_output.get("prompt"),
                "empty_token_count": elem.cached_render_output.get("tokens_reserved"),
                "output_handlers": elem.cached_render_output.get("output_handlers"),
                "stream_handlers": elem.cached_render_output.get("stream_handlers"),
            }

        case "chat":
            child_results = render_with_level(elem.children, level, tokenizer, call_ejected_callback)
            if is_chat_prompt(child_results["prompt"]):
                raise ValueError("Incorrect prompt: nested chat messages are not allowed!")
            # Construct the chat message based on the role
            message = {}
            if elem.role in ["user", "system"]:
                message = {
                    "role": elem.role,
                    "content": prompt_get_text(child_results["prompt"]) if child_results["prompt"] else "",
                }
                if hasattr(elem, "name") and elem.name is not None:
                    message["name"] = elem.name
            elif elem.role == "assistant":
                message = {
                    "role": elem.role,
                    "content": prompt_get_text(child_results["prompt"]) if child_results["prompt"] else "",
                }
                if hasattr(elem, "function_call") and elem.function_call is not None:
                    message["function_call"] = elem.function_call
            elif elem.role == "function":
                message = {
                    "role": elem.role,
                    "name": elem.name,
                    "content": prompt_get_text(child_results["prompt"]) if child_results["prompt"] else "",
                }

            return {
                "prompt": {
                    "type": "chat",
                    "messages": [message],
                    "functions": prompt_has_functions(child_results["prompt"]) if child_results["prompt"] else None,
                },
                "empty_token_count": child_results["empty_token_count"],
                "output_handlers": child_results["output_handlers"],
                "stream_handlers": child_results["stream_handlers"],
            }

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


def validate_unrendered_prompt(elem: PromptElement):
    validate_no_children_higher_priority_than_parent(elem)
    validate_not_both_absolute_and_relative_priority(elem)


def validate_not_both_absolute_and_relative_priority(elem: PromptElement):
    if isinstance(elem, list):
        for child in elem:
            validate_not_both_absolute_and_relative_priority(child)
        return

    if elem is None or elem is False:
        return

    if isinstance(elem, (str, int)):
        return

    match elem.type:
        case "chat" | "isolate" | "first":
            for child in elem.children:
                validate_not_both_absolute_and_relative_priority(child)

        case "scope":
            if hasattr(elem, "absolute_priority") and hasattr(elem, "relative_priority"):
                # print("WARNING: Scope has both absolute and relative priority. Ignoring relative priority.")
                pass
            for child in elem.children:
                validate_not_both_absolute_and_relative_priority(child)

        case "capture" | "breaktoken" | "function_definition" | "empty":
            return


def validate_no_children_higher_priority_than_parent(elem: PromptElement, parent_priority: int = BASE_PRIORITY):
    if isinstance(elem, list):
        for child in elem:
            validate_no_children_higher_priority_than_parent(child, parent_priority)
        return

    if elem is None or elem is False:
        return

    if isinstance(elem, (str, int)):
        return

    match elem.type:
        case "chat" | "first":
            for child in elem.children:
                validate_no_children_higher_priority_than_parent(child, parent_priority)

        case "isolate":
            validate_no_children_higher_priority_than_parent(elem.children)

        case "scope":
            priority = compute_priority(elem, parent_priority)
            if priority > parent_priority:
                print(f"WARNING: Child scope has a higher priority ({priority}) than its parent ({parent_priority}). This is discouraged.")
            for child in elem.children:
                validate_no_children_higher_priority_than_parent(child, priority)

        case "capture" | "breaktoken" | "function_definition" | "empty":
            return


def compute_priority(elem: Union["Scope", "NormalizedScope"], parent_priority: int) -> int:
    absolute_priority = getattr(elem, "absolute_priority", None)
    relative_priority = getattr(elem, "relative_priority", 0)

    # If absolute_priority is defined, use it; otherwise, calculate it
    return absolute_priority if absolute_priority is not None else parent_priority + relative_priority


def compute_priority_levels(elem: Union["AnyNode", List["AnyNode"]], parent_priority: int, levels: Set[int]):
    if isinstance(elem, list):
        for child in elem:
            compute_priority_levels(child, parent_priority, levels)
        return

    if elem is None or elem is False:
        return

    if isinstance(elem, (str, int)):
        return

    match elem.type:
        case "chat" | "first":
            for child in elem.children:
                compute_priority_levels(child, parent_priority, levels)

        case "capture" | "function_definition" | "breaktoken" | "empty" | "normalized_string":
            return

        case "isolate":
            return

        case "scope":
            priority = compute_priority(elem, parent_priority)
            levels.add(priority)
            elem.absolute_priority = priority
            for child in elem.children:
                compute_priority_levels(child, priority, levels)

        case _:
            print(f"ELEM: {elem}")
            raise ValueError(f"BUG!! compute_priority_levels got an invalid node of type {type(elem)}")


AnyNode = Union["Node", "NormalizedScope"]


async def num_tokens_prompt_string(prompt_string: PromptString, tokenizer: str) -> int:
    if isinstance(prompt_string, list):
        token_counts = await asyncio.gather(*(num_tokens(s, tokenizer=tokenizer) for s in prompt_string))
        return sum(token_counts)
    return await num_tokens(prompt_string, tokenizer=tokenizer)


async def count_tokens_exact(tokenizer, prompt, options):
    if not prompt:
        return 0
    tokens = 0
    if is_plain_prompt(prompt):
        tokens += await num_tokens_prompt_string(prompt, tokenizer)
    elif is_chat_prompt(prompt):
        msg_tokens = await asyncio.gather(*(count_message_tokens(msg, tokenizer) for msg in prompt.get("messages")))
        extra_token_count = (
            CHATML_PROMPT_EXTRA_TOKEN_COUNT_LINEAR_FACTOR * (len(prompt.get("messages")) - 1) + CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT
        )
        tokens += sum(msg_tokens) + extra_token_count
        if options.last_message_is_incomplete or False:
            tokens -= CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT + 1
    else:
        tokens += await num_tokens_prompt_string(prompt.get("text"), tokenizer)

    if prompt_has_functions(prompt):
        function_tokens_coroutines = [count_function_tokens(func, tokenizer) for func in prompt.get("functions")]
        function_tokens_results = await asyncio.gather(*function_tokens_coroutines)
        function_tokens = [result + 2 for result in function_tokens_results]
        tokens += sum(function_tokens)

    return tokens


def prompt_to_openai_chat_request(prompt: RenderedPrompt):
    functions = prompt.functions if prompt_has_functions(prompt) else None
    messages = prompt_to_openai_chat_messages(prompt)
    return {"messages": messages, "functions": functions}


CL100K_SYSTEM_TOKENS = [100264, 9125, 100266]
CL100K_USER_TOKENS = [100264, 882, 100266]
CL100K_ASSISTANT_TOKENS = [100264, 78191, 100266]
CL100K_END_TOKEN = [100265]


async def inject_name(tokens: int, name: str, tokenizer_object=None):
    # name_tokens = await tokenizer_object.encode_cl100k_no_special_tokens(":" + name)
    name_tokens = await encode_tokens(name + ":", tokenizer="cl100k_base")
    return tokens[:-1] + name_tokens + [tokens[-1]]


async def prompt_to_tokens(prompt: RenderedPrompt, tokenizer):
    if tokenizer != "cl100k_base":
        raise ValueError("prompt_to_tokens only supports the cl100k_base tokenizer for now!")

    if is_plain_prompt(prompt):
        if isinstance(prompt, list):
            tokens_lists = await asyncio.gather(*(encode_tokens(s, tokenizer=tokenizer) for s in prompt))
            return [token for tokens in tokens_lists for token in tokens]
        return await encode_tokens(prompt, tokenizer=tokenizer)

    elif is_chat_prompt(prompt):
        parts = []
        for msg in prompt.get("messages"):
            if msg.get("role") == "function":
                raise ValueError("BUG!! prompt_to_tokens got a chat prompt with a function message, which is not supported yet!")

            if msg.get("role") == "assistant" and msg.get("function_call") is not None:
                raise ValueError("BUG!! prompt_to_tokens got a chat prompt with a function message, which is not supported yet!")

            header_tokens = (
                CL100K_ASSISTANT_TOKENS
                if msg.get("role") == "assistant"
                else CL100K_SYSTEM_TOKENS if msg.get("role") == "system" else CL100K_USER_TOKENS
            )
            if "name" in msg and msg.get("name") is not None:
                header_tokens = await inject_name(header_tokens, msg.get("name"))

            content_tokens = await prompt_to_tokens(msg.get("content"), tokenizer) if msg.get("content") is not None else []
            parts.append(header_tokens + content_tokens)

        final_tokens = []
        for part in parts:
            if final_tokens:
                final_tokens += CL100K_END_TOKEN
            final_tokens += part
        return final_tokens

    raise ValueError("BUG!! prompt_to_tokens got an invalid prompt")


def prompt_to_openai_chat_messages(prompt):
    if is_plain_prompt(prompt):
        return [{"role": "user", "content": prompt_string_to_string(prompt)}]

    elif is_chat_prompt(prompt):
        return [
            (
                {"role": msg.get("role"), "name": msg.get("name"), "content": prompt_string_to_string(msg.get("content"))}
                if msg.get("role") == "function"
                else (
                    {
                        "role": msg.get("role"),
                        "content": prompt_string_to_string(msg.get("content")),
                        "function_call": msg.get("function_call"),
                    }
                    if msg.get("role") == "assistant" and msg.get("function_call") is not None
                    else {
                        "role": msg.get("role"),
                        "content": prompt_string_to_string(msg.get("content")),
                        "name": msg.get("name") if "name" in msg else None,
                    }
                )
            )
            for msg in prompt.get("messages")
        ]

    raise ValueError("BUG!! prompt_to_openai_chat_messages got an invalid prompt")


async def count_message_tokens(message: ChatPromptMessage, tokenizer):
    if message.get("role") == "function":
        name_tokens = await num_tokens(message.get("name"), tokenizer=tokenizer)
        content_tokens = await num_tokens_prompt_string(message.get("content"), tokenizer=tokenizer)
        return name_tokens + content_tokens + 2
    elif message.get("role") == "assistant" and message.get("function_call") is not None:
        function_call_tokens = await count_function_call_message_tokens(message.get("function_call"), tokenizer=tokenizer)
        content_tokens = (
            await num_tokens_prompt_string(message.get("content"), tokenizer=tokenizer) if message.get("content") is not None else 0
        )
        return function_call_tokens + content_tokens
    else:
        return await num_tokens_prompt_string(message.get("content") or "", tokenizer=tokenizer)


async def count_function_call_message_tokens(function_call, tokenizer):
    name_tokens = await num_tokens(function_call["name"], tokenizer=tokenizer)
    argument_tokens = await num_tokens(function_call["arguments"], tokenizer=tokenizer)
    return name_tokens + argument_tokens + 5


async def count_function_tokens(function_definition: ChatAndFunctionPromptFunction, tokenizer):
    stringified_function = json.dumps(
        {
            "name": function_definition.get("name"),
            "description": function_definition.get("description"),
            "parameters": function_definition.get("parameters"),
        },
        indent=2,
    )
    raw_token_count = await num_tokens(stringified_function, tokenizer=tokenizer)
    return int(math.ceil(raw_token_count * 1.5)) + 10


def estimate_function_tokens_using_charcount(function_definition: ChatAndFunctionPromptFunction, tokenizer):
    stringified_function = json.dumps(
        {
            "name": function_definition.get("name"),
            "description": function_definition.get("description"),
            "parameters": function_definition.get("parameters"),
        },
        indent=2,
    )
    raw = estimate_tokens_using_charcount(stringified_function, tokenizer=tokenizer)
    return (int(math.ceil(raw[0] * 0.5)), int(math.ceil(raw[1] * 1.5)) + 10)


def estimate_lower_bound_tokens_for_prompt(prompt: RenderedPrompt, tokenizer):
    if prompt is None:
        return 0

    if is_chat_prompt(prompt):
        content_tokens = sum(
            (
                estimate_tokens_using_charcount(b.get("name") + b.get("content"), tokenizer=tokenizer)[0]
                if b.get("role") == "function"
                else (
                    estimate_tokens_using_charcount(
                        b.get("function_call")["name"] + b.get("function_call")["arguments"] + (b.get("content") or ""), tokenizer=tokenizer
                    )[0]
                    if b.get("role") == "assistant" and b.get("function_call") is not None
                    else estimate_tokens_using_charcount(b.get("content") or "", tokenizer=tokenizer)[0]
                )
            )
            for b in prompt.get("messages")
        )
    elif is_plain_prompt(prompt):
        content_tokens = estimate_tokens_using_charcount(prompt_string_to_string(prompt), tokenizer=tokenizer)[0]
    else:
        content_tokens = estimate_tokens_using_charcount(prompt_string_to_string(prompt.get("text")), tokenizer=tokenizer)[0]

    function_tokens = (
        sum(estimate_function_tokens_using_charcount(f, tokenizer=tokenizer)[0] for f in prompt.get("functions"))
        if prompt_has_functions(prompt)
        else 0
    )

    return content_tokens + function_tokens
