# PriomptiPy

PriomptiPy (priority + prompt + python) is a Python-based prompting library that brings prioritized prompting from the Anysphere team's JavaScript library [Priompt](https://github.com/anysphere/priompt). Adapted by the [Quarkle](https://quarkle.ai) dev team, PriomptiPy integrates priority-based context management into Python applications, especially useful for AI-enabled agent and chatbot development.

## Motivation

It aims to simplify the process of managing and rendering prompts in AI-driven interactions, making it easier to focus on the most relevant context when hydrating the prompts from multiple sources.

## Installation

To install PriomptiPy, simply use pip:

```
pip install priomptipy
```

## Examples

Consider a scenario where you want to manage a conversation with a user:

![Priomptipy Example](public/Priomptipy.gif)

```
from priomptipy import SystemMessage, UserMessage, AssistantMessage, Scope, Empty, render

messages = [
    SystemMessage("You are Quarkle, an AI Developmental Editor"),
    Scope([
        UserMessage("Hello Quarkle, how are you?"),
        AssistantMessage("Hello, I am doing well. How can I help you?")
    ], absolute_priority=5),
    Scope([
        UserMessage("Write me a haiku on the number 17"),
        AssistantMessage("Seventeen whispers, In life's mosaic unseen, Quiet steps of time.")
    ], absolute_priority=10),
    UserMessage("Okay nice, now give me a title for it"),
    Empty(token_count=10)
]

render_options = {"token_limit": 80, "tokenizer": "cl100k_base"}
result = await render(messages, render_options)
print(result['prompt'])
```

In this dummy example, SystemMessage, UserMessage, and AssistantMessage are used to structure the conversation. Scope allows prioritizing certain parts of the conversation, ensuring the most relevant messages are included within the token limit. We always include the SystemMessage and the final UserMessage. And we reserve some tokens in the end for an LLM response.

## Principles

PriomptiPy operates on the principle of prioritized content rendering. Each element in a prompt can be assigned a priority using Scope, determining its importance in the overall context. This system allows for dynamic and efficient management of conversation flow, particularly in RAG applications where context space is limited but text abound.

## Components

These are logical components of PriomptiPy. They work the same way as in the original library.

- **Scope**: Groups messages and assigns priorities, dictating which messages should be rendered first.
- **Empty**: Reserves space in the prompt, useful for ensuring there's room for AI-generated content.
- **Isolate**: A section of the prompt with its own token limit. Useful when you want to include limited information from multiple sources.
- **First**: Sufficiently high child is selected for inclusion, while subsequent children are excluded. This feature is beneficial for creating fallback mechanisms, such as displaying a message like "(result omitted)" when the output exceeds a certain length.
- **Capture**: Capture the output and parse it right within the prompt. _Implemented but not functional yet._

And these are the message components that are used to build the content to send to AI models -

- **SystemMessage**: Represents system-level information.
- **UserMessage/AssistantMessage**: Denotes messages from the user or the AI assistant.
- **Function**: Encapsulates a callable function within the prompt. _The callable feature isn't fully supported yet._

## Caveats

While PriomptiPy enhances prompt management, it requires careful consideration of priorities to avoid overcomplicating prompts. It's crucial to balance the use of priorities to maintain efficient and cache-friendly prompts.

- Runnable function calling and capturing isn't supported yet. Will look to support this in the future.
- Just like the JS library, we haven't solved for cacheing yet. Would benefit from some help here.
- Will add more examples shortly, in the meantime, kindly take a look at the Tests for some sample usage.
- Would appreciate any support with maintaining and developing this library and keeping it in sync with the awesome Priompt library. There could be bugs in our current implementation as well, so use with caution.

## Contributions

This library was co-authored by Tanmay Gupta and Samarth Makhija, the founders of Quarkle. We warmly welcome contributions to PriomptiPy. The project is open-source under the MIT license, encouraging a collaborative and innovative community.
