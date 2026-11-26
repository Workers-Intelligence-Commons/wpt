import dataclasses
import json
import random
from contextlib import contextmanager

import pytest
import torch

from wpt_inference import DRAFT, costs_file, costs_setup, measure_costs, replayed_costs
from wpt_inference import drafting
from wpt_inference.drafting import DrafterFailed, Settings, settings, train_drafter
from wpt_model import WPT, DrafterConfig, ModelConfig
from wpt_utils.device import processor_name

CONFIG = ModelConfig(vocab_size=50, block_size=16, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=16,
                     attention_heads=2, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8, latent_width=8,
                     shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4,
                     engram_table_size=31, residual_groups=3)
END = 0
SMALL = Settings(epochs=1, prompts=4, prompt_tokens=5, answer_tokens=6, batch=2, anchors=4, held_out=0.25)


class Quiet:
    """A run record that keeps nothing."""

    def event(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    @contextmanager
    def span(self, *args, **kwargs):
        yield type("Span", (), {"set": lambda self, **_: None})()


class Beginning:
    """A training document whose text past its first `readable` tokens
    must never be read: reading it fails the test."""

    def __init__(self, tokens, readable):
        self.tokens, self.readable = tokens, readable

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        stop = index.stop if isinstance(index, slice) else index + 1
        if stop is None or stop > self.readable:
            raise AssertionError(f"the document was read up to {stop}, past its beginning of {self.readable}")
        return self.tokens[index]

    def __iter__(self):
        raise AssertionError("the whole document was read")


def documents(count=6, length=30):
    draw = torch.Generator().manual_seed(3)
    return [torch.randint(1, CONFIG.vocab_size, (length - 1,), generator=draw).tolist() + [END] for _ in range(count)]


def model():
    torch.manual_seed(0)
    return WPT(CONFIG).eval()


def test_written_answers_never_reach_the_drafter_s_training(monkeypatch):
    # the prompts are the beginnings of
    # the training documents, and the model alone writes the answers; the
    # rest of each document, text people wrote, is never read, and every
    # sequence the drafter learns from or is calibrated on is a prompt and
    # what the model wrote after it
    texts = documents()
    answers, learned, calibrated = [], [], []
    answer, blocks, outcomes = drafting._answers, drafting._blocks, drafting._outcomes

    def answering(model, prompts, *args):
        written = answer(model, prompts, *args)
        answers.extend(written)
        return written

    def learning(model, drafter, ids, start, *args):
        learned.append((list(ids), start))
        return blocks(model, drafter, ids, start, *args)

    def calibrating(model, drafter, prompts, *args):
        calibrated.extend(prompts)
        return outcomes(model, drafter, prompts, *args)

    monkeypatch.setattr(drafting, "_answers", answering)
    monkeypatch.setattr(drafting, "_blocks", learning)
    monkeypatch.setattr(drafting, "_outcomes", calibrating)
    target = model()
    before = {k: v.clone() for k, v in target.state_dict().items()}
    drafter = train_drafter(target, (Beginning(text, SMALL.prompt_tokens) for text in texts),
                            DrafterConfig.for_model(CONFIG), SMALL, end=END, stops=(END,), device=torch.device("cpu"),
                            telemetry=Quiet())

    beginnings = [[END, *text[:SMALL.prompt_tokens]] for text in texts]
    assert len(answers) == SMALL.prompts
    for ids, start in answers:
        assert ids[:start] in beginnings  # the prompt: a document's beginning, nothing more
        assert 1 <= len(ids) - start <= SMALL.answer_tokens  # the model's answer
    assert learned and all((ids, start) in answers for ids, start in learned)
    held = {tuple(ids[:start]) for ids, start in answers} - {tuple(ids[:start]) for ids, start in learned}
    assert calibrated and {tuple(prompt) for prompt in calibrated} == held  # kept apart, and only the prompts
    assert all(torch.equal(before[k], v) for k, v in target.state_dict().items())  # the model learned nothing
    assert all(p.requires_grad for p in target.parameters())  # and is left as it was
    assert drafter.provenance["prompts"] == {"count": 4, "training": 3, "held_out": 1,
                                             "answer_tokens": sum(len(ids) - start for ids, start in answers)}
    assert drafter.provenance["acceptance"]["verified"] > 0


@pytest.mark.parametrize("top_k", [None, 5])
def test_answers_written_side_by_side_are_those_each_prompt_gets_alone_with_its_stream(top_k):
    # module docstring of wpt_inference.drafting: prompts of different lengths, rows that stop early (a stop
    # token or answer_tokens) and leave the batch; each row's tokens are those its prompt gets written alone
    # by a Writer with its own stream, token for token on the CPU
    target = model()
    settings = dataclasses.replace(SMALL, answer_tokens=40, top_k=top_k)
    draw = random.Random(1)
    prompts = [[END] + [draw.randrange(1, CONFIG.vocab_size) for _ in range(draw.randrange(1, 9))]
               for _ in range(12)]
    stops = (END, 7)
    alone = [drafting._answer(target, prompt, settings, stops, stream)
             for prompt, stream in zip(prompts, drafting.streams(settings, len(prompts), "cpu"))]
    together = drafting._answers(target, prompts, settings, stops, drafting.streams(settings, len(prompts), "cpu"))
    assert together == alone
    lengths = {len(ids) - start for ids, start in together}
    assert settings.answer_tokens in lengths and min(lengths) < settings.answer_tokens  # some rows left early
    # the streams follow the seed and the place, never the batch
    assert [s.initial_seed() for s in drafting.streams(settings, 3, "cpu")] == \
        [s.initial_seed() for s in drafting.streams(settings, 12, "cpu")][:3]
    assert drafting.rows(settings) == 32 and drafting.rows(dataclasses.replace(SMALL, prompt_tokens=1024)) == 2


def test_the_drafter_s_update_is_the_same_one_block_at_a_time(monkeypatch):
    # chunks bound the memory of an update: they change neither its loss nor its gradients
    target = model()
    answers = [(ids[:20], 8) for ids in documents(3, 20)]

    def update(chunk_scores):
        monkeypatch.setattr(drafting, "CHUNK_SCORES", chunk_scores)
        torch.manual_seed(1)
        drafter = drafting.Drafter(CONFIG, DrafterConfig.for_model(CONFIG))
        loss, parts = drafting._learn(target, drafter, answers, 5, torch.Generator().manual_seed(2), "cpu")
        return loss, parts, [p.grad.clone() for p in drafter.parameters() if p.grad is not None]

    whole, one = update(2 ** 40), update(1)
    assert whole[0] == pytest.approx(one[0], rel=1e-5) and whole[1] == pytest.approx(one[1], rel=1e-5)
    assert len(whole[2]) == len(one[2]) > 0
    for a, b in zip(whole[2], one[2]):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_too_few_prompts_fail_the_drafter():
    # one document with text cannot be learned from and calibrated on apart
    with pytest.raises(DrafterFailed, match="at least 2 documents"):
        train_drafter(model(), [[5, 6, END], [END]], DrafterConfig.for_model(CONFIG), SMALL, end=END, stops=(END,),
                      device=torch.device("cpu"), telemetry=Quiet())


def test_the_order_s_drafter_section_gives_the_shape_and_the_training():
    # what it leaves out keeps the defaults; a setting neither knows is refused
    shape, training = settings(CONFIG, {"block": 3, "epochs": 2})
    assert shape == DrafterConfig.for_model(CONFIG, block=3) and training == Settings(epochs=2)
    with pytest.raises(ValueError, match="drafter settings: epoch unknown"):
        settings(CONFIG, {"epoch": 2})
    # feature layers the model does not have are refused with the settings,
    # before any training
    assert settings(CONFIG, {"features": [1, 4]})[0].features == (1, 4)
    with pytest.raises(ValueError, match=r"features \[2, 6\]: 6 is not a layer of a 5-layer model \(layers 1 to 5\)"):
        settings(CONFIG, {"features": [2, 6]})
    with pytest.raises(ValueError, match="0 is not a layer"):
        settings(CONFIG, {"features": [0]})


def test_the_scheduler_s_pass_times_are_replayed_only_on_the_setup_they_were_measured_on():
    # measured once per run, written into its record with the model config,
    # rows, drafter block and device they were measured on, the CPU by its
    # processor's name; a replay on another setup, another processor
    # included, is refused, naming each difference
    costs = measure_costs(model(), 1, DRAFT, context=8, repeats=1)
    assert len(costs) == DRAFT + 1 and all(seconds > 0 for seconds in costs)
    setup = costs_setup(CONFIG, 1, DRAFT, "cpu")
    here = f"cpu ({processor_name()})"
    assert processor_name().strip() and setup == {"model": json.loads(json.dumps(dataclasses.asdict(CONFIG))),
                                                  "rows": 1, "draft": DRAFT, "device": here}
    recorded = costs_file(costs, setup)
    assert replayed_costs(recorded, costs_setup(CONFIG, 1, DRAFT, torch.device("cpu"))) == costs
    with pytest.raises(ValueError, match=r"another setup: draft 5, here 4$"):
        replayed_costs(recorded, costs_setup(CONFIG, 1, 4, "cpu"))
    wider = dataclasses.replace(CONFIG, d_model=32)
    with pytest.raises(ValueError, match=r"another setup: model d_model 16, here 32; rows 1, here 2$"):
        replayed_costs(recorded, costs_setup(wider, 2, DRAFT, "cpu"))
    with pytest.raises(ValueError, match=r"another setup: device 'cpu \(Other Processor 9\)', here ") as refused:
        replayed_costs(costs_file(costs, {**setup, "device": "cpu (Other Processor 9)"}), setup)
    assert str(refused.value).endswith(f"here {here!r}")
    with pytest.raises(ValueError, match=r"another setup: device 'cpu \(.+\)', here 'cuda \(GPU\)'$"):
        replayed_costs(recorded, {**setup, "device": "cuda (GPU)"})
    with pytest.raises(ValueError, match="name no setup"):
        replayed_costs(json.dumps({"rows": 1, "draft": DRAFT, "seconds": costs}).encode(), setup)