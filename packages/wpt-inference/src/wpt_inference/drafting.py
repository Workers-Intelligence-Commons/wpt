"""Training a model's drafter (docs/model.md, "The drafter";
wpt_model.Drafter). A model and its drafter are one unit: wpt train trains
one at the end of every run, after the order's last step, against the
final weights. The model is frozen while it does: only the drafter learns.

The prompts are the beginnings of the dataset's training documents: of
every document read once, `prompts` drawn at random (a seeded reservoir),
each the document's first `prompt_tokens` tokens after the document start
(the END token, as wpt chat begins a text). The model alone writes each
answer, sampling at `temperature` (and `top_k`) with no drafter, up to a
stop token or `answer_tokens`: the rest of each document, the text people
wrote, is never kept, and the drafter learns to propose what the model
writes (DSpark's and DFlash's regenerated responses). A share of the
prompts (`held_out`, at least one) is kept apart.

The answers are written side by side, `rows` prompts per pass of the
model (_answers), each prompt with its own random stream, seeded from
`seed` and the prompt's place: each answer is then what the model writes
for its prompt alone with that stream (_answer), token for token on the
CPU (test_drafting.py), and within rounding on a GPU, whose passes over
several rows may round otherwise than over one. A row reads its prompt
in the first pass, then one token per pass, and leaves the batch once it
wrote a stop token or `answer_tokens` tokens. `rows` is fixed rather than
measured: at most ROWS, and at most PROMPT_READ tokens of prompts read
in one pass, so the batch's largest pass, its prompts' first, reads at
most PROMPT_READ tokens without gradients, and its scores, (rows, prompt
tokens, vocabulary) in FP32, take at most PROMPT_READ x vocabulary x 4
bytes: 512 MiB at a vocabulary of 65,536. Every later pass reads one
token per row, and the cache holds each row's keys and values of its
exact layers, a few MiB.

Each answer is read whole by the model, which gives its hidden states at
the drafter's feature layers and its distribution of each next token. From
random anchors in the answer (`anchors` at most per answer, drawn again
each epoch, as DFlash), the drafter learns DSpark's losses: cross-entropy on
the answer's tokens, the total variation distance to the model's
distribution, and the confidence head's prediction of the chance each
draft is kept (Drafter.losses). The model's update rule moves its weights
(wpt_model.NorMuonAdamW over Drafter.optimizer_roles): NorMuon for its
matrices, AdamW for the rest, both with cautious weight decay. Gradients are clipped, and the learning rate follows a
cosine schedule after a warm-up (DFlash's training values). The experts'
routing biases follow Quantile Balancing after each update, as the
model's.

On the held-out prompts, the drafter then drafts for the model, every draft
verified, and each round records each draft's confidence and whether the
model kept it (a draft after a rejected one counts as not kept): Sequential
Temperature Scaling sets each position's temperature on those real
outcomes (drafter.calibrate), and the same rounds measure how many drafts
the model keeps.

The settings come from the model order's `drafter` section (settings);
what it leaves out keeps the defaults below. Every failure (too few
prompts, a loss that is not finite) raises DrafterFailed: the run that
asked fails and writes nothing.
"""

import math
import random
from dataclasses import asdict, dataclass, fields

import torch

from wpt_model import NorMuonAdamW
from wpt_model.drafter import Drafter, DrafterConfig, calibrate, temperature_grid, training_blocks

from . import Writer, probabilities

CHUNK_SCORES = 2 ** 26  # the most scores (blocks x places x vocabulary) one pass of the drafter's loss holds
ROWS = 32  # the most answers written side by side
PROMPT_READ = 2048  # the most prompt tokens the first pass of a batch of answers reads


class DrafterFailed(Exception):
    """The drafter could not be trained; the reason says why."""


@dataclass(frozen=True)
class Settings:
    """How a drafter is trained: the model order's `drafter` section, but
    its shape (DrafterConfig: features, layers, block, markov_rank)."""
    epochs: int = 10            # passes over the answers (DSpark trains each drafter for 10 epochs)
    learning_rate: float = 6e-4  # DFlash's
    weight_decay: float = 0.0
    warm_up: float = 0.04       # share of the updates the learning rate rises over, then a cosine decay to 0 (DFlash)
    clip: float = 1.0           # gradient norm clipping (DFlash)
    anchors: int = 512          # anchors drawn from each answer per epoch, at most one per token (DFlash)
    batch: int = 8              # answers per update
    prompts: int = 256          # training documents whose beginnings are prompts
    prompt_tokens: int = 64     # the most tokens of a document's beginning in its prompt
    answer_tokens: int = 256    # the most tokens of each answer the model writes
    temperature: float = 1.0    # the answers come from the model's own probabilities, which the drafter learns
    top_k: int | None = None
    held_out: float = 0.1       # share of the prompts kept apart for calibration and the measure, at least one
    ece_bins: int = 15          # equal-width bins of the expected calibration error
    seed: int = 0               # the drafter's starting weights, the prompts and their split, the anchors, the answers


SHAPE = ("features", "layers", "block", "markov_rank")  # the order's drafter settings that are DrafterConfig's


def settings(model_config, section) -> tuple[DrafterConfig, Settings]:
    """(the drafter's shape, how it is trained) from a model order's
    `drafter` section; ValueError for a setting neither knows."""
    known = SHAPE + tuple(field.name for field in fields(Settings))
    unknown = sorted(set(section) - set(known))
    if unknown:
        raise ValueError(f"drafter settings: {', '.join(unknown)} unknown; known: {', '.join(known)}")
    shape = {name: section[name] for name in SHAPE if name in section}
    return (DrafterConfig.for_model(model_config, **shape),
            Settings(**{name: value for name, value in section.items() if name not in SHAPE}))


def beginnings(documents, count, length, end, seed):
    """The prompts: of documents (token sequences, each ending with end),
    count drawn at random with a seeded reservoir, each [end] then its
    first `length` tokens before its own end; documents with no token
    before their end are passed over. Only those first tokens are kept."""
    chosen, seen, draw = [], 0, random.Random(seed)
    for tokens in documents:
        text = min(length, len(tokens) - 1)
        if text < 1:
            continue
        prompt = [end, *(int(token) for token in tokens[:text])]
        seen += 1
        if len(chosen) < count:
            chosen.append(prompt)
        else:
            place = draw.randrange(seen)
            if place < count:
                chosen[place] = prompt
    return chosen


def train_drafter(model, documents, config: DrafterConfig, settings: Settings, *, end, stops, device, telemetry):
    """A drafter of that shape trained for model on the beginnings of
    documents (module docstring), in eval mode, with its provenance (what
    made it; the caller adds the weights it was trained against). end: the
    document-start token; stops: the tokens that end an answer. The model
    is left as it was, frozen meanwhile. Raises DrafterFailed."""
    prompts = beginnings(documents, settings.prompts, settings.prompt_tokens, end, settings.seed)
    if len(prompts) < 2:
        raise DrafterFailed(f"{len(prompts)} prompt: the drafter needs at least 2 documents with text, to learn from "
                            "one and calibrate on another")
    random.Random(settings.seed).shuffle(prompts)
    held = max(1, round(settings.held_out * len(prompts)))
    if held >= len(prompts):
        raise DrafterFailed(f"held_out {settings.held_out} keeps every one of the {len(prompts)} prompts apart")
    training_mode = model.training
    frozen = [p for p in model.parameters() if p.requires_grad]
    for p in frozen:
        p.requires_grad_(False)
    model.eval()
    try:
        return _train(model, prompts, held, config, settings, stops, device, telemetry)
    finally:
        for p in frozen:
            p.requires_grad_(True)
        model.train(training_mode)


def rows(settings) -> int:
    """How many answers are written side by side (module docstring)."""
    return max(1, min(ROWS, PROMPT_READ // settings.prompt_tokens))


def streams(settings, count, device) -> list:
    """The random stream of each of count prompts, in their order: a
    generator on device seeded from the settings' seed and the prompt's
    place."""
    draw = random.Random(f"answers {settings.seed}")
    return [torch.Generator(device=device).manual_seed(draw.getrandbits(63)) for _ in range(count)]


def _train(model, prompts, held, config, settings, stops, device, telemetry):
    together = rows(settings)
    telemetry.log(f"drafter: the model writes its answers to {len(prompts)} document beginnings, at most "
                  f"{settings.answer_tokens} tokens each and {together} side by side, for its drafter to learn from")
    with telemetry.span("drafter.answers", prompts=len(prompts), rows=together):
        answers, drawn = [], streams(settings, len(prompts), device)
        for first in range(0, len(prompts), together):
            answers += _answers(model, prompts[first:first + together], settings, stops,
                                drawn[first:first + together])
            if len(answers) < len(prompts):
                telemetry.log(f"drafter: {len(answers)} of {len(prompts)} answers written")
    sampling = torch.Generator(device=device).manual_seed(settings.seed)  # the calibration's drafts
    training, held_out = answers[held:], answers[:held]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings.seed)
        drafter = Drafter(model.config, config).to(device)
    shape = {**asdict(config), "features": list(config.features)}
    updates = settings.epochs * math.ceil(len(training) / settings.batch)
    telemetry.event("drafter.settings", settings=asdict(settings), drafter=shape, prompts=len(prompts),
                    training=len(training), held=len(held_out), tokens=sum(len(ids) - start for ids, start in answers))
    optimizer = NorMuonAdamW(drafter.optimizer_roles(), settings.learning_rate, settings.weight_decay)
    warm = max(1, math.ceil(settings.warm_up * updates))
    rate = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: (u + 1) / warm if u < warm else
                                             0.5 * (1 + math.cos(math.pi * (u - warm) / max(1, updates - warm))))
    anchors = torch.Generator().manual_seed(settings.seed)
    order = random.Random(settings.seed)
    losses = []
    telemetry.log(f"drafter: training, {settings.epochs} epochs of {updates // settings.epochs} updates on "
                  f"{len(training)} answers")
    drafter.train()
    for epoch in range(1, settings.epochs + 1):
        chosen = list(range(len(training)))
        order.shuffle(chosen)
        totals = []
        for first in range(0, len(chosen), settings.batch):
            batch = [training[i] for i in chosen[first:first + settings.batch]]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _learn(model, drafter, batch, settings.anchors, anchors, device)
            if not math.isfinite(loss):
                raise DrafterFailed(f"epoch {epoch}: the loss is {loss}; lower the drafter's learning_rate")
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), settings.clip)
            optimizer.step()
            rate.step()
            drafter.after_update(lambda tensor, op: tensor)  # one process
            totals.append(loss)
            telemetry.event("drafter.update", values={"loss": loss, **parts}, epoch=epoch, update=len(totals),
                            answers=len(batch))
        losses.append(sum(totals) / len(totals))
        telemetry.log(f"drafter epoch {epoch}/{settings.epochs}: mean loss {losses[-1]:.4f} over {len(training)} "
                      "answers")
    drafter.eval()

    telemetry.log(f"drafter: calibrating its confidence on {len(held_out)} held-out prompts")
    with telemetry.span("drafter.calibration"):
        outcomes, measure = _outcomes(model, drafter, [ids[:start] for ids, start in held_out], settings, stops,
                                      sampling)
        logits, kept, valid = _table(outcomes, drafter.block, device)
        if not len(logits):
            raise DrafterFailed(f"no draft was verified on the {len(held_out)} held-out prompts: nothing to "
                                "calibrate on (raise answer_tokens or prompts)")
        temperatures, before, after = calibrate(logits, kept, valid, settings.ece_bins, temperature_grid())
        drafter.temperatures.copy_(temperatures)
        telemetry.event("drafter.calibration", temperatures=temperatures.tolist(), ece_before=before, ece_after=after,
                        rounds=len(logits))
    telemetry.event("drafter.acceptance", **measure)
    telemetry.log(f"drafter: {measure['accepted']} of {measure['verified']} drafts kept on {len(held_out)} held-out "
                  f"prompts, {measure['tokens_per_round']:.2f} tokens per round of the model")
    drafter.provenance = {
        "drafter": shape,
        "settings": asdict(settings),
        "prompts": {"count": len(prompts), "training": len(training), "held_out": len(held_out),
                    "answer_tokens": sum(len(ids) - start for ids, start in answers)},
        "losses": losses,  # each epoch's mean
        "calibration": {"temperatures": temperatures.tolist(), "ece_before": before, "ece_after": after,
                        "rounds": len(logits)},
        "acceptance": measure,
    }
    return drafter


def _answer(model, prompt, settings, stops, generator):
    """(the prompt and the model's answer to it, where the answer starts):
    the model alone writes it, until a stop token or answer_tokens. What
    each row of _answers gives, one prompt at a time."""
    writer = Writer(model, prompt, temperature=settings.temperature, top_k=settings.top_k, generator=generator,
                    stops=stops)
    answer = []
    while len(answer) < settings.answer_tokens and not (answer and answer[-1] in stops):
        answer += writer.write({}, {0: settings.answer_tokens - len(answer)})[0]
    return prompt + answer, len(prompt)


@torch.no_grad()
def _answers(model, prompts, settings, stops, generators):
    """_answer for each of prompts, with generators[i] for prompts[i], the
    rows written side by side through one generation cache: each row reads
    what _answer's writer reads, in the same passes (its prompt but its
    last token, then one token per pass), and draws from its own
    generator; a row that wrote a stop token or answer_tokens leaves the
    cache."""
    device = next(model.parameters()).device
    cache = model.new_cache(len(prompts))
    heads = [prompt[:-1] for prompt in prompts]
    width = max(len(head) for head in heads)
    if width:
        model.extend(torch.tensor([head + [0] * (width - len(head)) for head in heads], device=device), cache,
                     [len(head) for head in heads])
    last = [prompt[-1] for prompt in prompts]
    written = [[] for _ in prompts]
    active, stops = list(range(len(prompts))), set(stops)
    while active:
        scores = model.extend(torch.tensor([[last[row]] for row in active], device=device), cache)[0][:, -1]
        p = probabilities(scores, settings.temperature, settings.top_k)
        drawn = torch.cat([torch.multinomial(p[i:i + 1], 1, generator=generators[row])[:, 0]
                           for i, row in enumerate(active)]).tolist()
        going = []
        for i, (row, token) in enumerate(zip(active, drawn)):
            written[row].append(token)
            last[row] = token
            if token not in stops and len(written[row]) < settings.answer_tokens:
                going.append(i)
        if len(going) < len(active):
            active = [active[i] for i in going]
            if active:
                cache = cache.select(going)
    return [(prompt + answer, len(prompt)) for prompt, answer in zip(prompts, written)]


@torch.no_grad()
def _model_reads(model, drafter, ids, device):
    scores, features = model(torch.tensor([ids], device=device), features=drafter.features)
    return scores[0], features[0]


def _learn(model, drafter, answers, most, generator, device):
    """The backward pass of Drafter.losses over every block of the answers
    ((ids, start) each), as one pass over all of them would give it, made an
    answer at a time and, within one, CHUNK_SCORES scores at a time: each
    chunk's loss weighs its share of the blocks, since the loss averages
    over blocks. Returns (the loss, {its parts}), as floats."""
    total = sum(min(most, len(ids) - start) for ids, start in answers)
    chunk = max(1, CHUNK_SCORES // (drafter.block * model.config.vocab_size))
    loss, parts = 0.0, {}
    for ids, start in answers:
        blocks = _blocks(model, drafter, ids, start, most, generator, device)
        for first in range(0, len(blocks[0]), chunk):
            part_loss, part_parts = drafter.losses(model, *[t[first:first + chunk] for t in blocks])
            share = len(blocks[0][first:first + chunk]) / total
            (part_loss * share).backward()
            loss += part_loss.item() * share
            for name, value in part_parts.items():
                parts[name] = parts.get(name, 0.0) + value.item() * share
    return loss, parts


def _blocks(model, drafter, ids, start, most, generator, device):
    """The blocks of one answer for Drafter.losses: anchors drawn among the
    tokens whose next token is the answer's, at most `most`."""
    scores, features = _model_reads(model, drafter, ids, device)
    places = len(ids) - start  # anchors start - 1 .. len(ids) - 2
    chosen = torch.randperm(places, generator=generator)[:most] + start - 1
    return training_blocks(features, torch.tensor(ids, device=device), scores, chosen.to(device), drafter.block)


def _outcomes(model, drafter, prompts, settings, stops, generator):
    """The drafter drafting for the model on prompts, every draft verified,
    its temperatures at 1: each round's (confidences, drafts kept), and
    the drafts verified and kept, the rounds, and tokens written per round."""
    drafter.temperatures.fill_(1.0)  # the confidences are then the head's own, which calibrate scales
    outcomes, verified, accepted, rounds = [], 0, 0, 0
    for prompt in prompts:
        writer = Writer(model, prompt, temperature=settings.temperature, top_k=settings.top_k, generator=generator,
                        stops=stops, drafter=drafter, draft=drafter.block, outcomes=outcomes)
        written = []
        while len(written) < settings.answer_tokens and not (written and written[-1] in stops):
            written += writer.write({}, {0: settings.answer_tokens - len(written)})[0]
        verified, accepted, rounds = verified + writer.verified, accepted + writer.accepted, rounds + writer.rounds
    return outcomes, {"verified": verified, "accepted": accepted, "rounds": rounds,
                      "tokens_per_round": (accepted + rounds) / max(1, rounds)}


def _table(outcomes, block, device):
    """(logits, kept, valid) (N, block) of the rounds that verified a draft."""
    rows = [(confidences, kept) for confidences, kept in outcomes if confidences]
    logits = torch.zeros(len(rows), block, device=device)
    kept = torch.zeros_like(logits)
    valid = torch.zeros_like(logits)
    for i, (confidences, count) in enumerate(rows):
        n = len(confidences)
        logits[i, :n] = torch.logit(torch.tensor(confidences, dtype=torch.float64), eps=1e-7).float()
        kept[i, :count] = 1
        valid[i, :n] = 1
    return logits, kept, valid
