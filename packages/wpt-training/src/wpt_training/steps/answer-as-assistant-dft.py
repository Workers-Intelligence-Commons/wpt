"""Answer as the assistant with dynamic fine-tuning: the answer-as-assistant
step whose loss weighs each assistant token by the probability the model
gives it (docs/training.md, "The answer-as-assistant-dft step"), with the
same environment, optimizer and locked parts."""

METHOD = "assistant-tokens-dft"
ENVIRONMENT = "conversations"
KINDS = ("conversation",)
OPTIMIZER = {"name": "by-part", "settings": {"lr": 1e-4, "weight_decay": 0.0, "schedule": "constant", "patience": 10,
                                         "average_states": 0}}
LOCKED = ("balancing",)
SETTINGS = {"tag": "answer-as-assistant"}
