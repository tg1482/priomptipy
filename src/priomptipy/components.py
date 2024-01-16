import re

from .prompt_types import ChatUserSystemMessage, ChatAssistantMessage, ChatFunctionResultMessage, FunctionDefinition, Capture


def SystemMessage(children=None, name=None):
    return ChatUserSystemMessage(type="chat", role="system", name=name, children=flatten(children) if children is not None else [])


def UserMessage(children=None, name=None):
    return ChatUserSystemMessage(type="chat", role="user", name=name, children=flatten(children) if children is not None else [])


def AssistantMessage(children=None, function_call=None, name=None):
    return ChatAssistantMessage(
        type="chat", role="assistant", function_call=function_call, children=flatten(children) if children is not None else []
    )


def FunctionMessage(name, children=None):
    return ChatFunctionResultMessage(type="chat", role="function", name=name, children=flatten(children) if children is not None else [])


def Function(name, description, parameters, on_call=None):
    if not valid_function_name(name):
        raise ValueError(f"Invalid function name: {name}.")
    return [
        FunctionDefinition(type="function_definition", name=name, description=description, parameters=parameters),
        Capture(
            type="capture",
            on_output=lambda output: on_call(output["function_call"]["arguments"])
            if on_call is not None
            and "function_call" in output
            and output["function_call"]["name"] == name
            and "arguments" in output["function_call"]
            else None,
        ),
    ]


def valid_function_name(name):
    return re.match(r"^[a-zA-Z0-9_]{1,64}$", name) is not None


def async_lambda(func):
    async def wrapped_func(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapped_func


def flatten(items):
    if isinstance(items, list):
        return [item for sublist in items for item in (flatten(sublist) if isinstance(sublist, list) else [sublist])]
    return [items]
