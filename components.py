import re
from dataclasses import dataclass, field
from prompt_types import ChatUserSystemMessage, ChatAssistantMessage
from typing import List, Optional, Any


# @dataclass
# class SystemMessage:
#     type: str = "chat"
#     role: str = "system"
#     name: Optional[str] = None
#     children: List[Any] = field(default_factory=list)

#     def __post_init__(self):
#         self.children = flatten(self.children) if self.children is not None else []


# @dataclass
# class UserMessage:
#     type: str = "chat"
#     role: str = "user"
#     name: Optional[str] = None
#     children: List[Any] = field(default_factory=list)

#     def __post_init__(self):
#         self.children = flatten(self.children) if self.children is not None else []


# @dataclass
# class AssistantMessage:
#     type: str = "chat"
#     role: str = "assistant"
#     function_call: Optional[Any] = None
#     children: List[Any] = field(default_factory=list)

#     def __post_init__(self):
#         self.children = flatten(self.children) if self.children is not None else []


# @dataclass
# class FunctionMessage:
#     name: str
#     type: str = "chat"
#     role: str = "function"
#     children: List[Any] = field(default_factory=list)

#     def __post_init__(self):
#         self.children = flatten(self.children) if self.children is not None else []


# @dataclass
# class Function:
#     name: str
#     description: str
#     parameters: List[Any]
#     on_call: Optional[callable] = None

#     def __post_init__(self):
#         if not valid_function_name(self.name):
#             raise ValueError(f"Invalid function name: {self.name}.")

#     async def handle_output(self, output: dict):
#         if (
#             self.on_call is not None
#             and "function_call" in output
#             and output["function_call"]["name"] == self.name
#             and "arguments" in output["function_call"]
#         ):
#             arguments = output["function_call"]["arguments"]
#             await self.on_call(*arguments)


def SystemMessage(children=None, name=None):
    # return {"type": "chat", "role": "system", "name": name, "children": flatten(children) if children is not None else []}
    return ChatUserSystemMessage(type="chat", role="system", name=name, children=flatten(children) if children is not None else [])


def UserMessage(children=None, name=None):
    # return {"type": "chat", "role": "user", "name": name, "children": flatten(children) if children is not None else []}
    return ChatUserSystemMessage(type="chat", role="user", name=name, children=flatten(children) if children is not None else [])


def assistant_message(function_call=None, children=None):
    # return {
    #     "type": "chat",
    #     "role": "assistant",
    #     "function_call": function_call,
    #     "children": flatten(children) if children is not None else [],
    # }
    return ChatAssistantMessage(
        type="chat", role="assistant", function_call=function_call, children=flatten(children) if children is not None else []
    )


# def function_message(name, children=None):
#     return {"type": "chat", "role": "function", "name": name, "children": flatten(children) if children is not None else []}


# def function(name, description, parameters, on_call=None):
#     if not valid_function_name(name):
#         raise ValueError(f"Invalid function name: {name}.")
#     return [
#         {"type": "function_definition", "name": name, "description": description, "parameters": parameters},
#         {
#             "type": "capture",
#             "on_output": lambda output: on_call(output["function_call"]["arguments"])
#             if on_call is not None
#             and "function_call" in output
#             and output["function_call"]["name"] == name
#             and "arguments" in output["function_call"]
#             else None,
#         },
#     ]


def valid_function_name(name):
    return re.match(r"^[a-zA-Z0-9_]{1,64}$", name) is not None


# def z_function(name, description, parameters, on_call=None, on_parse_error=None):
#     return function(
#         name=name,
#         description=description,
#         parameters=zod_to_json_schema(parameters),
#         on_call=async_lambda(raw_args: (lambda raw_args: on_call(json.loads(raw_args)) if on_call is not None else None))
#     )

# def zod_to_json_schema(schema):
#     full_schema = zod_to_json_schema_impl(schema, ref_strategy="none")
#     return {key: value for key, value in full_schema.items() if key not in ["$schema", "default", "definitions", "description", "markdownDescription", "additionalProperties"]}


def async_lambda(func):
    async def wrapped_func(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapped_func


def flatten(items):
    if isinstance(items, list):
        return [item for sublist in items for item in (flatten(sublist) if isinstance(sublist, list) else [sublist])]
    return [items]