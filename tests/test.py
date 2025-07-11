import pytest

from priomptipy.components import SystemMessage, UserMessage, AssistantMessage, Function, FunctionMessage
from priomptipy.prompt_types import PromptProps, PromptElement, Scope, Isolate, First
from priomptipy.lib import render, prompt_to_tokens, is_chat_prompt, prompt_has_functions

pytestmark = pytest.mark.asyncio


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


@pytest.mark.asyncio
async def test_isolate():
    """
    Tests the isolate and scope against a small token window. We should see that when Isolated, the children
    of that group are guarenteed to be there - depending on the children's relative priority to each other.
    However, when not isolated, then the priority of group takes precedence and there are no guarantees on whether
    a certain group will be included or not.
    """

    def isolate_component(props: PromptProps) -> PromptElement:
        if props.get("isolate"):
            return Isolate(token_limit=props.get("token_limit"), children=props.get("children"))
        else:
            return Scope(relative_priority=props.get("relative_priority", -10), children=props.get("children"))

    def test_component(props: PromptProps) -> PromptElement:
        isolated_messages = [
            Scope(relative_priority=-i - 2000, children=f"This is an SHOULDBEINCLUDEDONLYIFISOLATED user message number {i}")
            for i in range(10)
        ]
        non_isolated_messages = [Scope(relative_priority=-i - 1000, children=f"This is user message number {i}") for i in range(10)]

        mixed_messages = [
            Scope(relative_priority=-i, children=f"{i},xl,x,,{'SHOULDBEINCLUDEDONLYIFNOTISOLATED' if i > 50 else ''}") for i in range(100)
        ]

        final_message = [
            "This is the start of the prompt.",
            isolate_component({"token_limit": 100, "isolate": props.get("isolate"), "children": isolated_messages}),
            non_isolated_messages,
            isolate_component({"token_limit": 100, "isolate": props.get("isolate"), "children": mixed_messages}),
        ]
        return final_message

    render_options = {"token_limit": 300, "tokenizer": "cl100k_base"}
    rendered_isolated = await render(test_component({"isolate": True}), render_options)
    rendered_non_isolated = await render(test_component({"isolate": False}), render_options)

    assert "SHOULDBEINCLUDEDONLYIFISOLATED" in rendered_isolated.get("prompt")
    assert "SHOULDBEINCLUDEDONLYIFNOTISOLATED" in rendered_non_isolated.get("prompt")


@pytest.mark.asyncio
async def test_first():
    """
    Tests the first and scope against a small token window. We should only see the first child of the group.
    Otherwise, we should see all children of the group.
    """

    def first_component(props: PromptProps) -> PromptElement:
        if props.get("first"):
            return First(children=props.get("children"))
        else:
            return Scope(relative_priority=props.get("relative_priority", -10), children=props.get("children"))

    def test_component(props: PromptProps) -> PromptElement:
        first_messages = [Scope(relative_priority=-i - 2000, children=f"This is a TESTFIRST user message number {i}") for i in range(10)]
        non_first_messages = [Scope(relative_priority=-i - 1000, children=f"This is user message number {i}") for i in range(10)]

        final_message = [
            "This is the start of the prompt.",
            first_component({"first": props.get("first"), "children": first_messages}),
            non_first_messages,
        ]
        return final_message

    render_options = {"token_limit": 300, "tokenizer": "cl100k_base"}
    rendered_first = await render(test_component({"first": True}), render_options)
    rendered_regular = await render(test_component({"first": False}), render_options)

    assert rendered_first.get("prompt").count("TESTFIRST") == 1
    assert rendered_regular.get("prompt").count("TESTFIRST") == 10


@pytest.mark.asyncio
async def test_isolate_with_null_messages():
    """
    Tests the Isolate component with a very small token limit that might result in null messages.
    This ensures that we can handle cases where single messages can't be added due to token limits.
    """

    def test_component() -> PromptElement:
        messages = [
            SystemMessage("You are Quarkle, an AI Developmental Editor"),
            Isolate(
                token_limit=10,
                children=[
                    Scope(
                        [UserMessage("Hello Quarkle, how are you?"), AssistantMessage("Hello, I am doing well. How can I help you")],
                        absolute_priority=5,
                    ),
                ],
            ),
            UserMessage("Give me a story title"),
        ]
        return messages

    render_options = {"token_limit": 300, "tokenizer": "cl100k_base"}
    rendered = await render(test_component(), render_options)

    # Check if the system message and final user message are present
    assert any(
        msg["role"] == "system" and msg["content"] == "You are Quarkle, an AI Developmental Editor"
        for msg in rendered["prompt"]["messages"]
    )
    assert any(msg["role"] == "user" and msg["content"] == "Give me a story title" for msg in rendered["prompt"]["messages"])

    # Check that no content from the Isolate block is present
    isolate_content = ["Hello Quarkle", "Hello, I am doing well"]
    assert all(not any(content in msg.get("content", "") for msg in rendered["prompt"]["messages"]) for content in isolate_content)


@pytest.mark.asyncio
async def test_with_gpt4_model():
    def test_component() -> PromptElement:
        messages = [
            SystemMessage("You are a helpful assistant"),
            UserMessage("Hello, how are you?"),
            AssistantMessage("I'm doing well, thank you. How can I assist you today?"),
        ]
        return messages

    render_options = {"model": "gpt-4o"}
    rendered = await render(test_component(), render_options)

    assert rendered["prompt"]["type"] == "chat"
    assert len(rendered["prompt"]["messages"]) == 3
    assert rendered["prompt"]["messages"][0]["role"] == "system"
    assert rendered["prompt"]["messages"][1]["role"] == "user"
    assert rendered["prompt"]["messages"][2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_with_unknown_model():
    def test_component() -> PromptElement:
        messages = [
            SystemMessage("You are a helpful assistant"),
            UserMessage("Hello, how are you?"),
        ]
        return messages

    render_options = {"model": "unknown-model"}

    # Instead of expecting an exception, let's capture the response
    with pytest.raises(ValueError) as excinfo:
        await render(test_component(), render_options)

    assert "Must specify model or token_limit" in str(excinfo.value)
