import pytest
import sys
import os

# Assuming your tests directory is at the same level as the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.components import SystemMessage, UserMessage, AssistantMessage, Function, FunctionMessage
from src.prompt_types import (
    PromptProps,
    PromptElement,
    Scope,
)
from src.lib import render, prompt_to_tokens, is_chat_prompt, prompt_has_functions


@pytest.mark.asyncio
async def test_prompt_to_tokens():
    def simple_message_prompt(props: PromptProps = None) -> PromptElement:
        """
        Create a prompt with system and user messages
        """

        prompt_elements = []

        # Adding system message with optional break tokens
        system_message_content = "This is the start of the prompt."
        print(SystemMessage(system_message_content))
        prompt_elements.append(SystemMessage(system_message_content))

        # Adding user message
        user_message_content = "hi!"
        prompt_elements.append(UserMessage(user_message_content))

        assistant_message_content = "Hello!"
        prompt_elements.append(AssistantMessage(assistant_message_content))

        return prompt_elements

    render_options = {"token_limit": 1000, "tokenizer": "cl100k_base"}
    # Test for break token
    do_not_break = await render(simple_message_prompt(), render_options)
    to_tokens = await prompt_to_tokens(do_not_break["prompt"], "cl100k_base")
    assert len(to_tokens) == do_not_break["token_count"]


@pytest.mark.asyncio
async def test_empty_message_content_to_tokens():
    def empty_message_prompt(props: PromptProps = None) -> PromptElement:
        """
        Create a prompt with empty message contents.
        """
        # return [SystemMessage(""), UserMessage(""), AssistantMessage("")]
        messages = []
        messages.append(SystemMessage("Testing to see if this works"))
        messages.append(UserMessage(""))
        messages.append(AssistantMessage(""))
        return messages

    render_options = {"token_limit": 50, "tokenizer": "cl100k_base"}
    rendered_empty = await render(empty_message_prompt(), render_options)
    empty_tokens = await prompt_to_tokens(rendered_empty["prompt"], "cl100k_base")
    assert len(empty_tokens) == rendered_empty["token_count"]


@pytest.mark.asyncio
async def test_multi_message_content_to_tokens():
    def multi_message_prompt(props: PromptProps = None) -> PromptElement:
        """
        Create a prompt with empty message contents.
        """
        messages = []
        # System Message
        messages.append(SystemMessage("Testing to see if this works"))

        # Message Set 1 with Priority 5
        scoped_message = Scope(
            absolute_priority=5,
            children=[
                UserMessage("Hello Hello this is daddy"),
                AssistantMessage("Hello hello this is also your daddy but a much bigger one"),
            ],
        )
        messages.append(scoped_message)

        # Message Set 2 with Priority 10
        scoped_message = Scope(
            absolute_priority=10,
            children=[
                UserMessage("Oh mah gawh"),
                AssistantMessage("Betty look at that cahh"),
            ],
        )
        messages.append(scoped_message)
        return messages

    render_options = {"token_limit": 40, "tokenizer": "cl100k_base"}
    rendered_few = await render(multi_message_prompt(), render_options)
    assert rendered_few == {
        "prompt": {
            "type": "chat",
            "messages": [
                {"role": "system", "content": "Testing to see if this works"},
                {"role": "user", "content": "Oh mah gawh"},
                {"role": "assistant", "content": "Betty look at that cahh"},
            ],
        },
        "token_count": 30,
        "tokens_reserved": 0,
        "token_limit": 40,
        "tokenizer": "cl100k_base",
        "duration_ms": None,
        "output_handlers": [],
        "stream_handlers": [],
        "priority_cutoff": 10,
    }


@pytest.mark.asyncio
async def test_function_message():
    def test_function(props: PromptProps = None) -> PromptElement:
        # Create a function message
        function_message = Function(
            name="echo",
            description="Echo a message to the user.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo.",
                    },
                },
                "required": ["message"],
            },
        )
        # Create a user message
        user_message = UserMessage("say hi")
        return [function_message, user_message]

    render_options = {"token_limit": 1000, "tokenizer": "cl100k_base"}
    rendered = await render(test_function(), render_options)
    assert is_chat_prompt(rendered["prompt"]) is True
    assert prompt_has_functions(rendered["prompt"]) is True
    if not prompt_has_functions(rendered["prompt"]):
        return
    assert rendered["prompt"]["functions"] == [
        {
            "name": "echo",
            "description": "Echo a message to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo.",
                    },
                },
                "required": ["message"],
            },
        },
    ]


@pytest.mark.asyncio
async def test_all_messages():
    def test_all_messages(props: PromptProps = None) -> PromptElement:
        # Create a function message
        function_message = Function(
            name="echo",
            description="Echo a message to the user.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo.",
                    },
                },
                "required": ["message"],
            },
        )
        # Create other messages
        system_message = SystemMessage("System message")
        user_message = UserMessage("User message")
        assistant_message = AssistantMessage(
            function_call={
                "name": "echo",
                "arguments": '{"message": "this is a test echo"}',
            }
        )
        function_message_content = FunctionMessage(name="echo", children=["this is a test echo"])

        return [function_message, system_message, user_message, assistant_message, function_message_content]

    render_options = {"token_limit": 1000, "tokenizer": "cl100k_base"}
    rendered = await render(test_all_messages(), render_options)

    assert is_chat_prompt(rendered["prompt"]), "The prompt should be a chat prompt"
    assert prompt_has_functions(rendered["prompt"]), "The prompt should contain functions"

    # Check if the functions are correctly rendered
    expected_functions = [
        {
            "name": "echo",
            "description": "Echo a message to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo.",
                    },
                },
                "required": ["message"],
            },
        },
    ]
    assert rendered["prompt"]["functions"] == expected_functions, "The functions are not as expected"

    # Check if the messages are correctly rendered
    expected_messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "", "function_call": {"name": "echo", "arguments": '{"message": "this is a test echo"}'}},
        {"role": "function", "name": "echo", "content": "this is a test echo"},
    ]
    assert rendered["prompt"]["messages"] == expected_messages, "The messages are not as expected"
