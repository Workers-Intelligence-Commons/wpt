"""Answer as the assistant: learn to write the assistant's turns of the
corpus's conversations tagged
`answer-as-assistant` (docs/training.md, "The answer-as-assistant
step"), under by-part at a constant learning rate of
1e-4, without weight decay, with its early stop and no
average of its last states. The routers' balancing
biases stay as they are."""

METHOD = "assistant-tokens"
ENVIRONMENT = "conversations"
OPTIMIZER = {"name": "by-part", "settings": {"lr": 1e-4, "weight_decay": 0.0, "schedule": "constant", "patience": 10,
                                         "average_states": 0}}
LOCKED = ("balancing",)
SETTINGS = {"tag": "answer-as-assistant"}
