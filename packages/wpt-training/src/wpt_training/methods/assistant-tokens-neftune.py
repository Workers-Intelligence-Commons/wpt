"""Learning to write the assistant's tokens of conversations with noisy
embeddings, NEFTune (docs/training.md, "The answer-as-assistant-neftune
step").

The loss is assistant-tokens': the next-token loss of the tokens the
assistant wrote only, the mean over every such token of the update's
conversations, on every process. During its passes, while the model is in
training mode with gradients on, the input embeddings (the model's
`embedding` part) receive uniform noise in [-1, 1] times alpha / sqrt(L *
d), L the pass's padded length and d the model's width (Jain et al.,
"NEFTune: Noisy Embeddings Improve Instruction Finetuning"). The noise
comes from PyTorch's generator of the device, which the runner seeds for
each pass. Evaluation and generation run outside these passes, so they
never see noise. The multi-token prediction block is not used, so it does
not change.

A loss that is not finite stops the step (StepFailed): lower the
optimizer's lr.
"""

import math

import torch
import torch.nn.functional as F

from wpt_model import MULTI_TOKEN_PREDICTION
from wpt_training import Method, StepFailed

READS = "conversations"
SETTINGS = {"alpha": 5.0}


def noisy(alpha):
    """A forward hook for an embedding module: its (B, L, d) output plus
    uniform noise in [-1, 1] times alpha / sqrt(L * d), only in training
    mode with gradients on."""

    def hook(module, inputs, output):
        if not (module.training and torch.is_grad_enabled()):
            return output
        scale = alpha / math.sqrt(output.size(-2) * output.size(-1))
        return output + torch.empty_like(output).uniform_(-1, 1) * scale

    return hook


class AssistantTokensNeftune(Method):
    unreached = (MULTI_TOKEN_PREDICTION,)

    def __init__(self, alpha):
        self.alpha = alpha

    def learn(self, model, wrapped, examples, passes, strategy, segment, quiet):
        """examples: (inputs, targets, learned), as assistant-tokens reads them."""
        inputs, targets, learned = examples
        handle = model.embedding.register_forward_hook(noisy(self.alpha))  # on for this pass only
        try:
            with quiet(True):
                logits = wrapped(inputs)
                share = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                                        reduction="sum") / learned
                # the processes' gradients are averaged: each process's share counts world_size times
                (share * strategy.world_size).backward()
        finally:
            handle.remove()
        return share.detach()

    def update_loss(self, losses, strategy):
        loss = super().update_loss(losses, strategy)
        if not math.isfinite(loss):  # the same in every process, which all stop
            raise StepFailed(f"the loss is {loss}; lower the optimizer's lr")
        return loss


def start(settings):
    return AssistantTokensNeftune(settings["alpha"])
