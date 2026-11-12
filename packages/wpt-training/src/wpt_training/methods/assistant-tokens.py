"""Learning to write the assistant's tokens of conversations (docs/training.md,
"The answer-as-assistant step").

Each conversation is read whole, and the loss is the next-token loss of
the tokens the assistant wrote only (thinking, text, tool calls and the
closing <|im_end|>): the other turns are context the model reads and does
not learn to write. The loss of an update is the mean over every such
token of its conversations, on every process. The multi-token prediction block is not used, so it does not change.

A loss that is not finite stops the step (StepFailed): lower the
optimizer's lr.
"""

import math

import torch.nn.functional as F

from wpt_model import MULTI_TOKEN_PREDICTION
from wpt_training import Method, StepFailed

READS = "conversations"


class AssistantTokens(Method):
    unreached = (MULTI_TOKEN_PREDICTION,)

    def learn(self, model, wrapped, examples, passes, strategy, segment, quiet):
        """examples: (inputs, targets, learned): targets -100 where nothing
        is learned (everywhere in a pass whose process had no conversation
        left), learned the number of tokens to learn in the whole update
        over every process."""
        inputs, targets, learned = examples
        with quiet(True):
            logits = wrapped(inputs)
            share = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                                    reduction="sum") / learned
            # the processes' gradients are averaged: each process's share counts world_size times
            (share * strategy.world_size).backward()
        return share.detach()

    def update_loss(self, losses, strategy):
        loss = super().update_loss(losses, strategy)
        if not math.isfinite(loss):  # the same in every process, which all stop
            raise StepFailed(f"the loss is {loss}; lower the optimizer's lr")
        return loss


def start(settings):
    return AssistantTokens()
