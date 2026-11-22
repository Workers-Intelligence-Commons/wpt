"""Answer as the assistant with noisy embeddings (NEFTune): the
answer-as-assistant step whose training passes add uniform noise to the
input embeddings (docs/training.md, "The answer-as-assistant-neftune
step"), with the same environment, optimizer and locked parts."""

METHOD = "assistant-tokens-neftune"
ENVIRONMENT = "conversations"
KINDS = ("conversation",)
OPTIMIZER = {"name": "by-part", "settings": {"lr": 1e-4, "weight_decay": 0.0, "schedule": "constant", "patience": 10,
                                         "average_states": 0}}
LOCKED = ("balancing",)
SETTINGS = {"tag": "answer-as-assistant", "alpha": 5.0}


def check(settings):
    if not settings["alpha"] >= 0:
        raise ValueError("alpha: 0 or more")
