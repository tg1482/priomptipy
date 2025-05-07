from typing import Union, Optional, List, Dict, Callable, Tuple, Any
from dataclasses import dataclass

# Assuming JSONSchema7 is a dictionary structure in Python
JSONSchema7 = Dict[str, Any]


# FunctionBody type
@dataclass
class FunctionBody:
    """Represents the body of a function definition.
    
    Attributes:
        name (str): The name of the function
        description (str): A description of what the function does
        parameters (JSONSchema7): JSON Schema defining the function parameters
    """
    name: str
    description: str
    parameters: JSONSchema7


# First type
@dataclass
class First:
    """Represents a component that prioritizes its first child.
    
    This component is used to ensure that the first child in the list is rendered
    before any others, effectively making it the highest priority.
    
    Attributes:
        children (List[Scope]): List of child components
        type (str): Always "first"
        on_eject (Optional[Callable]): Callback when component is ejected
        on_include (Optional[Callable]): Callback when component is included
    """
    children: List["Scope"]
    type: str = "first"
    on_eject: Optional[Callable] = None
    on_include: Optional[Callable] = None


# Empty type
@dataclass
class Empty:
    """Represents an empty component that takes up token space.
    
    This is useful for reserving space in the prompt or creating padding.
    
    Attributes:
        token_count (int): Number of tokens this empty space represents
        type (str): Always "empty"
    """
    token_count: int
    type: str = "empty"


# BreakToken type
@dataclass
class BreakToken:
    """Represents a token break in the prompt.
    
    This component is used to indicate where token counting should break
    or where special token handling should occur.
    
    Attributes:
        type (str): Always "break_token"
    """
    type: str = "break_token"


# Capture type
@dataclass
class Capture:
    """Represents a component that captures output or stream data.
    
    This component can intercept and process output or streaming data
    through its callbacks.
    
    Attributes:
        type (str): Always "capture"
        on_output (Optional[Callable]): Callback for handling output
        on_stream (Optional[Callable]): Callback for handling stream data
    """
    type: str = "capture"
    on_output: Optional[Callable] = None
    on_stream: Optional[Callable] = None


# Isolate type
@dataclass
class Isolate:
    """Represents a component that isolates its children with a token limit.
    
    This component ensures its children don't exceed a specified token limit,
    making it useful for constraining sections of the prompt.
    
    Attributes:
        token_limit (int): Maximum number of tokens allowed for children
        children (List[Node]): List of child components
        type (str): Always "isolate"
        cached_render_output (Optional[RenderOutput]): Cached rendering result
    """
    token_limit: int
    children: List["Node"]
    type: str = "isolate"
    cached_render_output: Optional["RenderOutput"] = None


# Scope type
@dataclass
class Scope:
    """Represents a scoped component that can have priority settings.
    
    This component groups other components and can have both absolute
    and relative priority settings.
    
    Attributes:
        children (List[Node]): List of child components
        type (str): Always "scope"
        absolute_priority (Optional[int]): Absolute priority level
        relative_priority (Optional[int]): Priority relative to other components
        on_eject (Optional[Callable]): Callback when component is ejected
        on_include (Optional[Callable]): Callback when component is included
    """
    children: List["Node"]
    type: str = "scope"
    absolute_priority: Optional[int] = None
    relative_priority: Optional[int] = None
    on_eject: Optional[Callable] = None
    on_include: Optional[Callable] = None


# ChatUserSystemMessage type
@dataclass
class ChatUserSystemMessage:
    """Represents a system message in a chat.
    
    Attributes:
        type (str): Message type
        role (str): Role of the message sender
        children (List[Node]): Message content components
        name (Optional[str]): Optional name of the sender
    """
    type: str
    role: str
    children: List["Node"]
    name: Optional[str] = None


# ChatAssistantMessage type
@dataclass
class ChatAssistantMessage:
    """Represents an assistant's message in a chat.
    
    Attributes:
        type (str): Message type
        role (str): Role of the message sender
        children (List[Node]): Message content components
        function_call (Optional[Dict[str, str]]): Optional function call details
    """
    type: str
    role: str
    children: List["Node"]
    function_call: Optional[Dict[str, str]] = None


# ChatFunctionResultMessage type
@dataclass
class ChatFunctionResultMessage:
    """Represents a function result message in a chat.
    
    Attributes:
        type (str): Message type
        role (str): Role of the message sender
        name (str): Name of the function
        children (List[Node]): Function result content
    """
    type: str
    role: str
    name: str
    children: List["Node"]


# ChatMessage type
ChatMessage = Union[ChatUserSystemMessage, ChatAssistantMessage, ChatFunctionResultMessage]
"""Union type representing any type of chat message."""


# FunctionDefinition type
@dataclass
class FunctionDefinition:
    """Represents a function definition in the prompt.
    
    Attributes:
        type (str): Definition type
        name (str): Function name
        description (str): Function description
        parameters (JSONSchema7): Function parameters schema
    """
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
"""Union type representing any valid node in the prompt tree."""

# PromptElement type
PromptElement = Union[List[Node], Node]
"""Union type representing either a single node or a list of nodes."""


# BaseProps type
@dataclass
class BaseProps:
    """Base properties for prompt components.
    
    Attributes:
        p (Optional[int]): Priority value
        prel (Optional[int]): Relative priority value
        children (Optional[Union[List[PromptElement], PromptElement]]): Child components
        on_eject (Optional[Callable]): Callback when component is ejected
        on_include (Optional[Callable]): Callback when component is included
    """
    p: Optional[int] = None
    prel: Optional[int] = None
    children: Optional[Union[List[PromptElement], PromptElement]] = None
    on_eject: Optional[Callable] = None
    on_include: Optional[Callable] = None


# ReturnProps type
@dataclass
class ReturnProps:
    """Properties for components that can return values.
    
    Attributes:
        on_return (Callable): Callback for handling return values
    """
    on_return: Callable


# BasePromptProps type
BasePromptProps = Union[BaseProps, Dict[str, Any]]
"""Union type for base prompt properties."""

# PromptProps type
PromptProps = Union[BasePromptProps, Dict[str, Any]]
"""Union type for all prompt properties."""

# PromptString type
PromptString = Union[str, List[str]]
"""Union type for prompt string content."""


# ChatPromptUserSystemMessage type
@dataclass
class ChatPromptUserSystemMessage:
    """Represents a system message in a chat prompt.
    
    Attributes:
        role (str): Role of the message sender
        content (PromptString): Message content
        name (Optional[str]): Optional name of the sender
    """
    role: str
    content: PromptString
    name: Optional[str] = None


# ChatPromptAssistantMessage type
@dataclass
class ChatPromptAssistantMessage:
    """Represents an assistant's message in a chat prompt.
    
    Attributes:
        role (str): Role of the message sender
        content (Optional[PromptString]): Optional message content
        function_call (Optional[Dict[str, str]]): Optional function call details
    """
    role: str
    content: Optional[PromptString] = None
    function_call: Optional[Dict[str, str]] = None


# ChatPromptFunctionResultMessage type
@dataclass
class ChatPromptFunctionResultMessage:
    """Represents a function result message in a chat prompt.
    
    Attributes:
        role (str): Role of the message sender
        name (str): Name of the function
        content (PromptString): Function result content
    """
    role: str
    name: str
    content: PromptString


# ChatPromptMessage type
ChatPromptMessage = Union[
    ChatPromptUserSystemMessage,
    ChatPromptAssistantMessage,
    ChatPromptFunctionResultMessage,
]
"""Union type representing any type of chat prompt message."""


# ChatPrompt type
@dataclass
class ChatPrompt:
    """Represents a complete chat prompt.
    
    Attributes:
        type (str): Prompt type
        messages (List[ChatPromptMessage]): List of chat messages
    """
    type: str
    messages: List[ChatPromptMessage]


@dataclass
class TextPrompt:
    """Represents a text-based prompt.
    
    Attributes:
        type (str): Prompt type
        text (Union[str, List[str]]): Prompt text content
    """
    type: str
    text: Union[str, List[str]]


# ChatAndFunctionPromptFunction type
@dataclass
class ChatAndFunctionPromptFunction:
    """Represents a function in a chat and function prompt.
    
    This class is necessary because it defines the structure that language models like GPT use
    to understand what functions they can call. When you want an LLM to be able to call specific 
    functions in your code, you need to describe those functions in a format the LLM can understand.
    This class provides that format, matching OpenAI's function calling specification.
    
    Attributes:
        name (str): Function name that the LLM will use to identify and call this function
        description (str): Human-readable description that helps the LLM understand when and how to use this function
        parameters (JSONSchema7): A JSON Schema that strictly defines what parameters the function accepts,
            helping the LLM provide the correct arguments in the correct format
    """
    name: str
    description: str
    parameters: JSONSchema7


# FunctionPrompt type
@dataclass
class FunctionPrompt:
    """Represents a prompt containing function definitions.
    
    Attributes:
        functions (List[ChatAndFunctionPromptFunction]): List of function definitions
    """
    functions: List[ChatAndFunctionPromptFunction]


# OutputHandler type
OutputHandler = Callable[[Any, Optional[Dict[str, int]]], Any]
"""Type alias for output handling callbacks."""

# RenderedPrompt type
RenderedPrompt = Union[
    Union[str, List[str]],
    ChatPrompt,
    Tuple[ChatPrompt, FunctionPrompt],
    Tuple[TextPrompt, FunctionPrompt],
]
"""Union type representing all possible rendered prompt formats."""

# Prompt type
Prompt = Callable[[Dict[str, Any]], Union[List[Any], Any]]
"""Type alias for prompt functions."""


# RenderOptions type
@dataclass
class RenderOptions:
    """Options for rendering prompts.
    
    This class defines configuration options used when rendering prompts to be sent to language models.
    The options control important aspects of how the prompt is processed and formatted.
    
    Attributes:
        model (Optional[str]): The specific language model to target (e.g. "gpt-3.5-turbo", "gpt-4").
            Different models may require different prompt formats or have different token limits.
        token_limit (Optional[int]): The maximum number of tokens allowed in the rendered prompt.
            This helps prevent prompts from exceeding model context windows.
        last_message_is_incomplete (Optional[bool]): Indicates if the final message in a chat prompt
            is incomplete/partial. This is useful when streaming responses or handling partial messages.
        tokenizer (Optional[str]): The tokenizer model to use for counting tokens (default: "cl100k_base").
            Different models may use different tokenization schemes, so this ensures accurate token counting.
    """
    model: Optional[str] = None
    token_limit: Optional[int] = None
    last_message_is_incomplete: Optional[bool] = None
    tokenizer: Optional[str] = "cl100k_base"


# RenderOutput type
@dataclass
class RenderOutput:
    """Output from rendering a prompt.
    
    This class represents the complete output produced when rendering a prompt, including
    both the rendered prompt itself and various metadata about the rendering process.
    
    Attributes:
        prompt (RenderedPrompt): The final rendered prompt that can be sent to the language model.
            This may be a string, list of strings, ChatPrompt, or tuple containing prompts
            and function definitions.
        token_count (int): The total number of tokens used in the rendered prompt. This is
            calculated using the specified tokenizer.
        token_limit (int): The maximum number of tokens allowed for this prompt. This helps
            ensure the prompt fits within model context limits.
        tokenizer (str): The name of the tokenizer model used to count tokens (e.g. "cl100k_base").
            Different models may use different tokenization schemes.
        tokens_reserved (int): Number of tokens that were reserved but not used in the final prompt.
            This allows space for things like function definitions or system messages.
        priority_cutoff (int): The priority level below which components were excluded from the prompt
            due to token limits. Higher priority components are included first.
        output_handlers (List[OutputHandler]): List of callback functions that process the model's
            output response. These can transform or validate the output.
        stream_handlers (List[OutputHandler]): List of callback functions that handle streaming
            output from the model. These process chunks of the response as they arrive.
        duration_ms (Optional[int]): How long the prompt rendering process took in milliseconds.
            This is useful for performance monitoring and optimization.
    """
    prompt: RenderedPrompt
    token_count: int
    token_limit: int
    tokenizer: str
    tokens_reserved: int
    priority_cutoff: int
    output_handlers: List[OutputHandler]
    stream_handlers: List[OutputHandler]
    duration_ms: Optional[int] = None
