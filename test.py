import pytest
from components import SystemMessage, UserMessage, AssistantMessage
from prompt_types import (
    PromptProps,
    PromptElement,
    Scope,
)
from lib import render, prompt_to_tokens


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
                {"role": "system", "name": None, "content": "Testing to see if this works"},
                {"role": "user", "name": None, "content": "Oh mah gawh"},
                {"role": "assistant", "content": "Betty look at that cahh", "function_call": None},
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
