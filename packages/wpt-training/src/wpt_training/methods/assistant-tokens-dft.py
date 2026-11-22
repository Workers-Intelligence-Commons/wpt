"""Learning to write the assistant's tokens of conversations with dynamic
fine-tuning (docs/training.md, "The answer-as-assistant-dft step").

The tokens learned are those of assistant-tokens: the tokens the assistant
wrote only. Each one's loss is -p * log p, p the probability the model
gives it, the weight p taken as a number (detached): no gradient goes
through it, so the gradient is the next-token loss's scaled by p. The loss
of an update is the sum over every such token of its conversations, on
every process, divided by their number, as assistant-tokens reduces it.
The multi-token prediction block is not used, so it does not change.

A loss that is not finite stops the step (StepFailed): lower the
optimizer's lr.
"""

import math

import torch.nn.functional as F

from wpt_model import MULTI_TOKEN_PREDICTION
from wpt_training import Method, StepFailed

READS = "conversations"


def token_losses(logits, targets):
    """(B, T) losses of the target tokens, -p.detach() * log p, 0 where the
    target is -100 (nothing learned)."""
    log_p = -F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                             reduction="none").reshape(targets.shape)
    return -log_p.detach().exp() * log_p


class AssistantTokensDft(Method):
    unreached = (MULTI_TOKEN_PREDICTION,)

    def learn(self, model, wrapped, examples, passes, strategy, segment, quiet):
        """examples: (inputs, targets, learned), as assistant-tokens reads them."""
        inputs, targets, learned = examples
        with quiet(True):
            share = token_losses(wrapped(inputs), targets).sum() / learned
            # the processes' gradients are averaged: each process's share counts world_size times
            (share * strategy.world_size).backward()
        return share.detach()

    def update_loss(self, losses, strategy):
        loss = super().update_loss(losses, strategy)
        if not math.isfinite(loss):  # the same in every process, which all stop
            raise StepFailed(f"the loss is {loss}; lower the optimizer's lr")
        return loss


def start(settings):
    return AssistantTokensDft()
