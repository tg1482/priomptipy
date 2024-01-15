import pytest
from components import SystemMessage, UserMessage
from prompt_types import (
    ChatPrompt,
    ChatPromptUserSystemMessage,
    ChatPromptAssistantMessage,
    ChatPromptFunctionResultMessage,
    FunctionPrompt,
    ChatAndFunctionPromptFunction,
    ChatUserSystemMessage,
    PromptProps,
    PromptElement,
)
from lib import render, prompt_to_tokens


@pytest.mark.asyncio
async def test_prompt_to_tokens():
    def simple_message_prompt(props: PromptProps = None) -> PromptElement:
        """
        Create a prompt with system and user messages, and an optional break token.

        Args:
        - props: A dictionary containing

        Returns:
        - A PromptElement representing the created prompt.
        """

        prompt_elements = []

        # Adding system message with optional break tokens
        system_message_content = [
            "This is the start of the prompt.\n",
            "\nThis is the second part of the prompt.",
        ]
        prompt_elements.append(SystemMessage(system_message_content))

        # Adding user message
        user_message_content = "hi!"
        prompt_elements.append(UserMessage(user_message_content))

        return prompt_elements

    render_options = {"token_limit": 1000, "tokenizer": "cl100k_base"}
    # Test for break token
    do_not_break = await render(simple_message_prompt(), render_options)
    to_tokens = await prompt_to_tokens(do_not_break["prompt"], "cl100k_base")
    assert len(to_tokens) == len(do_not_break["token_count"])
