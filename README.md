# PriomptiPy

PriomptiPy (priority + prompt + python) is a Python-based prompting library that brings prioritized prompting from the Anysphere team's JavaScript library [Priompt](https://github.com/anysphere/priompt). Adapted by the [Quarkle](https://quarkle.ai) dev team, PriomptiPy integrates priority-based context management into Python applications, especially useful for AI-enabled agent and chatbot development.

## Motivation

It aims to simplify the process of managing and rendering prompts in AI-driven interactions, making it easier to focus on the most relevant context when hydrating the prompts from multiple sources. This is particularly valuable in RAG applications, chatbots, and AI agents where context space is limited but available content is abundant.

## Installation

To install PriomptiPy, simply use pip:

```bash
pip install priomptipy
```

For development, we recommend using [UV](https://github.com/astral-sh/uv):

```bash
# Clone the repository
git clone https://github.com/tg1482/priomptipy.git
cd priomptipy

# Install development environment
make install

# Run tests
make test
```

## Quick Start

Here's a simple example showing how PriomptiPy handles prioritized content:

```python
import asyncio
from priomptipy import SystemMessage, UserMessage, AssistantMessage, Scope, Empty, render

async def main():
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
        Empty(token_count=10)  # Reserve space for response
    ]

    render_options = {"token_limit": 80, "tokenizer": "cl100k_base"}
    result = await render(messages, render_options)
    print(result['prompt'])

# Run the example
asyncio.run(main())
```

In this example, when the token limit is tight, PriomptiPy will:

1. Always include the SystemMessage and final UserMessage (highest priority)
2. Include the haiku conversation (priority=10) before the greeting (priority=5)
3. Reserve 10 tokens for the AI response
4. Gracefully truncate lower-priority content if needed

## Core Principles

PriomptiPy operates on the principle of **prioritized content rendering**. Each element in a prompt can be assigned a priority using Scope, determining its importance in the overall context. This system allows for:

- **Dynamic content management** based on available token budget
- **Graceful degradation** when content exceeds limits
- **Efficient conversation flow** in multi-turn interactions
- **Flexible context injection** from multiple sources

## Components

### Message Components

- **SystemMessage**: Represents system-level information and instructions
- **UserMessage**: Denotes messages from the user
- **AssistantMessage**: Represents AI assistant responses
- **FunctionMessage**: Handles function call results

### Control Components

- **Scope**: Groups messages and assigns priorities, dictating rendering order
- **Empty**: Reserves token space, useful for ensuring room for AI responses
- **Isolate**: A section with its own token limit - gracefully truncates content that exceeds its budget
- **First**: Selects the first child that fits within token limits, useful for fallback mechanisms
- **Capture**: Captures and processes AI output (experimental)

### Priority System

- **Absolute Priority**: Direct priority value (higher = more important)
- **Relative Priority**: Priority relative to parent scope
- **Graceful Truncation**: Lower priority content is excluded when tokens are limited

## Recent Improvements (v0.19.0)

### 🎯 Isolate Component Fix

The Isolate component now gracefully handles content that exceeds its token limit, truncating based on priorities instead of throwing errors. This enables robust memory management and content injection patterns.

```python
from priomptipy import Isolate, Scope

# This now works gracefully even if memories exceed 500 tokens
Isolate(
    token_limit=500,
    children=[
        Scope([...large_memory_content...], absolute_priority=i)
        for i in range(100)  # Lots of memories
    ]
)
```

### 🚀 Modern Development Setup

- **UV-based package management** for faster dependency resolution
- **Makefile commands** for common development tasks
- **Modern pyproject.toml** configuration
- **Improved testing** with pytest-asyncio

## Development

This project uses [UV](https://github.com/astral-sh/uv) for package management and development:

```bash
# Available commands
make help       # Show all available commands
make install    # Install development environment
make test       # Run test suite
make build      # Build distribution packages
make clean      # Clean build artifacts
make publish    # Publish to PyPI (requires PYPI_TOKEN)
```

### Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make test` to ensure tests pass
5. Submit a pull request

## Advanced Usage

### Memory Management with Isolate

```python
from priomptipy import Isolate, Scope, SystemMessage, UserMessage

# Manage large memory banks with token budgets
memories = Isolate(
    token_limit=900,  # Budget for memories
    children=[
        Scope([
            UserMessage(f"Context: {memory_text}"),
        ], absolute_priority=memory_importance)
        for memory_text, memory_importance in memory_bank
    ]
)

messages = [
    SystemMessage("You are a helpful assistant with access to memories."),
    memories,  # Will gracefully fit within 900 tokens
    UserMessage("Help me with my current task.")
]
```

### Fallback Patterns with First

```python
from priomptipy import First, Scope

# Try detailed response first, fall back to summary if tokens are tight
First([
    Scope([detailed_response], absolute_priority=1),
    Scope([summary_response], absolute_priority=0),
    Scope(["(Response truncated due to length)"], absolute_priority=-1)
])
```

## Caveats and Future Work

- **Function calling**: Basic support implemented, full execution capabilities planned
- **Caching**: Not yet implemented, would benefit from community contributions
- **Streaming**: Capture component is experimental
- **Performance**: Optimizations planned for very large prompt trees

## License

This project is open-source under the MIT license. Originally adapted from the excellent [Priompt](https://github.com/anysphere/priompt) library by Anysphere.

## Contributors

- **Tanmay Gupta** - Co-founder, Quarkle
- **Samarth Makhija** - Co-founder, Quarkle

We warmly welcome contributions from the community!
