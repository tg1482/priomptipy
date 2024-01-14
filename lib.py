# Type Checking Functions


def is_chat_prompt(prompt):
    return isinstance(prompt, dict) and prompt.get("type") == "chat"


def is_plain_prompt(prompt):
    return isinstance(prompt, str) or isinstance(prompt, list)


def is_text_prompt_potentially_with_functions(prompt):
    return (isinstance(prompt, dict) and "text" in prompt) or isinstance(prompt, str)


def prompt_has_functions(prompt):
    return (
        isinstance(prompt, dict)
        and "functions" in prompt
        and prompt["functions"] is not None
    )


# Utility Functions


def prompt_string_to_string(prompt_string):
    return "".join(prompt_string) if isinstance(prompt_string, list) else prompt_string


def prompt_get_text(prompt):
    if not is_text_prompt_potentially_with_functions(prompt):
        return None
    return (
        prompt_string_to_string(prompt)
        if is_plain_prompt(prompt)
        else prompt_string_to_string(prompt["text"])
    )


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
            "messages": (a.get("messages", []) if is_chat_prompt(a) else [])
            + (b.get("messages", []) if is_chat_prompt(b) else []),
            "functions": functions if functions else None,
        }
        return prompt

    if (prompt_has_functions(a) or prompt_has_functions(b)) and (
        is_text_prompt_potentially_with_functions(a)
        and is_text_prompt_potentially_with_functions(b)
    ):
        functions = a.get("functions", []) if prompt_has_functions(a) else []
