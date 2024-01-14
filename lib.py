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
from typing import List
from openai_helper import (
    CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT,
    CHATML_PROMPT_EXTRA_TOKEN_COUNT_LINEAR_FACTOR,
    MAX_TOKENS,
    usable_tokenizers,
    is_usable_language_model,
    usable_language_models,
)


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


def render_binary_search():
    return
