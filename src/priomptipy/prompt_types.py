from typing import Union, Optional, List, Dict, Callable, Tuple, Any
from dataclasses import dataclass

# Assuming JSONSchema7 is a dictionary structure in Python
JSONSchema7 = Dict[str, Any]


# FunctionBody type
@dataclass
class FunctionBody:
    name: str
    description: str
    parameters: JSONSchema7


# First type
@dataclass
class First:
    children: List["Scope"]
    type: str = "first"
    on_eject: Optional[Callable] = None
    on_include: Optional[Callable] = None


# Empty type
@dataclass
class Empty:
    token_count: int
    type: str = "empty"


# BreakToken type
@dataclass
class BreakToken:
    type: str = "break_token"


# Capture type
@dataclass
class Capture:
    type: str = "capture"
    on_output: Optional[Callable] = None
    on_stream: Optional[Callable] = None


# Isolate type
@dataclass
class Isolate:
    token_limit: int
    children: List["Node"]
    type: str = "isolate"
    cached_render_output: Optional["RenderOutput"] = None


# Scope type
@dataclass
class Scope:
    children: List["Node"]
    type: str = "scope"
    absolute_priority: Optional[int] = None
    relative_priority: Optional[int] = None
    on_eject: Optional[Callable] = None
    on_include: Optional[Callable] = None


# ChatUserSystemMessage type
@dataclass
class ChatUserSystemMessage:
    type: str
    role: str
    children: List["Node"]
    name: Optional[str] = None


# ChatAssistantMessage type
@dataclass
class ChatAssistantMessage:
    type: str
    role: str
    children: List["Node"]
    function_call: Optional[Dict[str, str]] = None


# ChatFunctionResultMessage type
@dataclass
class ChatFunctionResultMessage:
    type: str
    role: str
    name: str
    children: List["Node"]


# ChatMessage type
ChatMessage = Union[ChatUserSystemMessage, ChatAssistantMessage, ChatFunctionResultMessage]


# FunctionDefinition type
@dataclass
class FunctionDefinition:
    type: str
    name: str
    description: str
    parameters: JSONSchema7


# Node type
Node = Union[
    FunctionDefinition,
    BreakToken,
    First,
    Isolate,
    Capture,
    Scope,
    Empty,
    ChatMessage,
    str,
    None,
    int,
    bool,
]

# PromptElement type
PromptElement = Union[List[Node], Node]


# BaseProps type
@dataclass
class BaseProps:
    p: Optional[int] = None
    prel: Optional[int] = None
    children: Optional[Union[List[PromptElement], PromptElement]] = None
    on_eject: Optional[Callable] = None
    on_include: Optional[Callable] = None


# ReturnProps type
@dataclass
class ReturnProps:
    on_return: Callable


# BasePromptProps type
BasePromptProps = Union[BaseProps, Dict[str, Any]]

# PromptProps type
PromptProps = Union[BasePromptProps, Dict[str, Any]]

# PromptString type
PromptString = Union[str, List[str]]


# ChatPromptUserSystemMessage type
@dataclass
class ChatPromptUserSystemMessage:
    role: str
    content: PromptString
    name: Optional[str] = None


# ChatPromptAssistantMessage type
@dataclass
class ChatPromptAssistantMessage:
    role: str
    content: Optional[PromptString] = None
    function_call: Optional[Dict[str, str]] = None


# ChatPromptFunctionResultMessage type
@dataclass
class ChatPromptFunctionResultMessage:
    role: str
    name: str
    content: PromptString


# ChatPromptMessage type
ChatPromptMessage = Union[
    ChatPromptUserSystemMessage,
    ChatPromptAssistantMessage,
    ChatPromptFunctionResultMessage,
]


# ChatPrompt type
@dataclass
class ChatPrompt:
    type: str
    messages: List[ChatPromptMessage]


@dataclass
class TextPrompt:
    type: str
    text: Union[str, List[str]]


# ChatAndFunctionPromptFunction type
@dataclass
class ChatAndFunctionPromptFunction:
    name: str
    description: str
    parameters: JSONSchema7


# FunctionPrompt type
@dataclass
class FunctionPrompt:
    functions: List[ChatAndFunctionPromptFunction]


# OutputHandler type
OutputHandler = Callable[[Any, Optional[Dict[str, int]]], Any]

# RenderedPrompt type
RenderedPrompt = Union[
    Union[str, List[str]],
    ChatPrompt,
    Tuple[ChatPrompt, FunctionPrompt],
    Tuple[TextPrompt, FunctionPrompt],
]

# Prompt type
Prompt = Callable[[Dict[str, Any]], Union[List[Any], Any]]


# RenderOptions type
@dataclass
class RenderOptions:
    model: Optional[str] = None
    token_limit: Optional[int] = None
    last_message_is_incomplete: Optional[bool] = None
    tokenizer: Optional[str] = "cl100k_base"


# RenderOutput type
@dataclass
class RenderOutput:
    prompt: RenderedPrompt
    token_count: int
    token_limit: int
    tokenizer: str
    tokens_reserved: int
    priority_cutoff: int
    output_handlers: List[OutputHandler]
    stream_handlers: List[OutputHandler]
    duration_ms: Optional[int] = None
