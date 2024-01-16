# PriomptiPy

Priomptipy (priority + prompt + python) is a python-based prompting library. It uses priorities to decide what to include in the context window. This is a python version of [Priompt](https://github.com/anysphere/priompt), which was generously open-sourced by Anysphere's team.

# Installation

```
pip install priomptipy
```

# Examples

```
messages = [
    SystemMessage("You are Quarkle, an AI Developmental Editor"),
    Scope([
        UserMessage("Hello Quarkle, how are you?"),
        AssistantMessage("Hello, I am doing well. How can I help you")
    ], absolute_priority=5),
    Scope(children=[
            UserMessage("Write me a haiku on the number 17"),
            AssistantMessage("Seventeen whispers, In life's mosaic unseen, Quiet steps of time.")
    ], absolute_priority=10),
    UserMessage("Okay nice, now give me a title for it"),
    Empty(token_count=10)
]
```

This creates a list of messages like so -

```
[ChatUserSystemMessage(type='chat',
                       role='system',
                       children=['You are Quarkle, an AI Developmental Editor'],
                       name=None),
 Scope(children=[ChatUserSystemMessage(type='chat',
                                       role='user',
                                       children=['Hello Quarkle, how are you?'],
                                       name=None),
                 ChatAssistantMessage(type='chat',
                                      role='assistant',
                                      children=['Hello, I am doing well. How '
                                                'can I help you'],
                                      function_call=None)],
       type='scope',
       absolute_priority=5,
       relative_priority=None,
       on_eject=None,
       on_include=None),
 Scope(children=[ChatUserSystemMessage(type='chat',
                                       role='user',
                                       children=['Write me a haiku on the '
                                                 'number 17'],
                                       name=None),
                 ChatAssistantMessage(type='chat',
                                      role='assistant',
                                      children=["Seventeen whispers, In life's "
                                                'mosaic unseen, Quiet steps of '
                                                'time.'],
                                      function_call=None)],
       type='scope',
       absolute_priority=10,
       relative_priority=None,
       on_eject=None,
       on_include=None),
 ChatUserSystemMessage(type='chat',
                       role='user',
                       children=['Okay nice, now give me a title for it'],
                       name=None),
 Empty(token_count=10, type='empty')]
```

The magic happens when we call render with a token_limit.

This will then ensure that we do not exceed the token limit by picking the parts of the message based on priority.

```
render_options = {"token_limit": 80, "tokenizer": "cl100k_base"}
await render(messages, render_options)
```

Output:

```
{'duration_ms': None,
 'output_handlers': [],
 'priority_cutoff': 10,
 'prompt': {'messages': [{'content': 'You are Quarkle, an AI Developmental '
                                     'Editor',
                          'role': 'system'},
                         {'content': 'Write me a haiku on the number 17',
                          'role': 'user'},
                         {'content': "Seventeen whispers, In life's mosaic "
                                     'unseen, Quiet steps of time.',
                          'role': 'assistant'},
                         {'content': 'Okay nice, now give me a title for it',
                          'role': 'user'}],
            'type': 'chat'},
 'stream_handlers': [],
 'token_count': 62,
 'token_limit': 80,
 'tokenizer': 'cl100k_base',
 'tokens_reserved': 10}
```

In the example above, we always include the system message and the latest user message, and are including as many messages from the history as can fit based on priority and token limits.
