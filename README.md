# PriomptiPy

Priomptipy (priority + prompt + python) is a python-based prompting library. It uses priorities to decide what to include in the context window. This is a python version of [Priompt](https://github.com/anysphere/priompt), which was generously open-sourced by Anysphere's team.

# Installation

```
pip install priomptipy
```

# Examples

```
    messages = []
    # System Message
    messages.append(SystemMessage("Testing to see if this works"))

    # Message Set 1 with Priority 5
    scoped_message = Scope(
        absolute_priority=5,
        children=[
            UserMessage("Hello Quarkle"),
            AssistantMessage("Hello, how can I help you?"),
        ],
    )
    messages.append(scoped_message)

    # Message Set 2 with Priority 10
    scoped_message = Scope(
        absolute_priority=10,
        children=[
            UserMessage("Write me a haiku on the number 17"),
            AssistantMessage("Seventeen whispers, In life's mosaic unseen, Quiet steps of time."),
        ],
    )
    messages.append(scoped_message)

    messages.append(UserMessage("Okay nice, now give me a title for it"))

    return messages
```

In the example above, we always include the system message and the latest user message, and are including as many messages from the history as possible, where later messages are prioritized over earlier messages.
