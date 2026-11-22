import importlib.util
import shutil
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from wpt_model import BALANCING, MULTI_TOKEN_PREDICTION, PARTS, ROUTERS, WPT, ModelConfig, Routes
from wpt_model import attention
from wpt_model.experts import Experts
from wpt_model.layers import expert_forward, grouped_experts
from wpt_model import model as model_module
from wpt_model.model import _fused_loss, fused_loss_failure
from wpt_utils.device import select_device

TINY = dict(vocab_size=50, block_size=16, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=16,
            attention_heads=2, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8, latent_width=8,
            shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31,
            residual_groups=3)


@pytest.fixture
def few_chosen(monkeypatch):
    """Exact layers pick 16 tokens instead of 1,024, so that the choice
    matters within a few hundred tokens."""
    monkeypatch.setattr(attention, "TOP_K", 16)


def test_a_token_never_changes_the_scores_before_it(few_chosen):
    # a language model must not see ahead: changing token 150 leaves every
    # score at positions 0 .. 149 as it was, through the window layers, the
    # exact layers (their choice of tokens too), the sparse layers, Engram,
    # the experts and Attention Residuals; 300 tokens let the window slide
    # and the exact layers choose
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "block_size": 300})).eval()
    ids = torch.randint(50, (2, 300))
    changed = ids.clone()
    changed[:, 150] = (ids[:, 150] + 1) % 50
    with torch.no_grad():
        before, after = model(ids), model(changed)
    assert torch.allclose(before[:, :150], after[:, :150], atol=1e-5)
    assert not torch.allclose(before[:, 150], after[:, 150])


def test_quantile_balancing_gives_each_expert_its_share():
    # a router that favours some experts ends up sending each one about k/n
    # of the tokens, with no auxiliary loss
    torch.manual_seed(0)
    experts = Experts(ModelConfig(**{**TINY, "n_experts": 8}), hashed=False)
    with torch.no_grad():
        experts.router.weight[:3] += 0.5  # with inputs of positive mean, experts 0, 1 and 2 score higher
    x, ids = torch.randn(4, 256, 16) + 0.5, torch.zeros(4, 256, dtype=torch.long)

    def shares():
        chosen, _ = experts.route(x.reshape(-1, 16), ids.reshape(-1))
        return torch.bincount(chosen.reshape(-1), minlength=8) / chosen.numel()

    assert shares().max() > 0.3
    experts.train()
    for _ in range(40):  # the same batch each step: the biases settle
        experts(x, ids)
        experts.balance(lambda tensor, op: tensor)
    assert torch.allclose(shares(), torch.full((8,), 1 / 8), atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped matrix multiplications need a CUDA device")
def test_grouped_experts_equal_each_expert_on_its_own_tokens_with_an_expert_that_has_none():
    # the CUDA path of Experts._routed (BF16 tokens) must give what the loop
    # over experts gives, in outputs and in every weight's gradient, the
    # gradient of an expert no token chose being zero; on a GPU where
    # grouped.py's kernels run, the experts' tokens span several of their
    # tiles of 64 rows, the last one part empty
    torch.manual_seed(0)
    n, E, L = 4, 16, 8
    sizes = [5, 0, 150, 64]  # expert 1 has no token
    weights = [torch.randn(n, E, L, device="cuda") for _ in range(2)] + [torch.randn(n, L, E, device="cuda")]
    x = torch.randn(sum(sizes), L, device="cuda").bfloat16()

    def run(compute):
        for w in weights:
            w.grad = None
        out = compute()
        out.float().square().sum().backward()
        return out, [w.grad for w in weights]

    ends = torch.tensor(sizes, device="cuda").cumsum(0).to(torch.int32)
    for w in weights:
        w.requires_grad_()
    grouped, grouped_grads = run(lambda: grouped_experts(x, ends, *weights))
    loop, loop_grads = run(lambda: torch.cat([expert_forward(part, *(w[j].bfloat16() for w in weights))
                                              for j, part in enumerate(x.split(sizes))]))
    torch.testing.assert_close(grouped, loop)
    for a, b in zip(grouped_grads, loop_grads):
        torch.testing.assert_close(a, b)
        assert not a[1].any()


def test_a_replayed_pass_routes_each_token_through_the_experts_recorded_while_writing_it():
    # R3 (docs/training.md): routes recorded by growing prefixes, as
    # sampling reads them, route a later pass over the whole sequence the
    # same way, even when the routers would now choose other experts
    torch.manual_seed(0)
    model = WPT(ModelConfig(**TINY)).eval()
    ids = torch.randint(50, (1, 12))
    routes = Routes()
    with torch.no_grad():
        for end in range(5, 13):  # a prompt of 5 tokens, then one more token at a time
            model(ids[:, :end], record=routes)
        written = model(ids)
        routed = [layer.experts for layer in model.layers if not layer.experts.hashed]
        for experts in routed:
            experts.bias.copy_(torch.tensor([0.0, 0.0, 10.0, 10.0]))  # experts 2 and 3 for every token now
        assert not torch.allclose(model(ids), written, atol=1e-4)
        assert torch.allclose(model(ids, replay=routes), written, atol=1e-5)
    assert [len(routes.chosen[experts, 0]) for experts in routed] == [12, 12]


def test_recomputing_layers_trains_the_same_and_counts_each_token_once(few_chosen, monkeypatch):
    # --recompute only trades time for memory: the same gradients bit for
    # bit, the exact layers' keys and values reaching the sparse layers
    # included, and the layers' second run adds nothing to balancing; the
    # exact and sparse layers' attention runs once, never again for the
    # backward pass
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "block_size": 256})).train()
    ids, after = torch.randint(50, (2, 200)), torch.randint(50, (2, 200))
    runs = []
    for name in ("exact_attention", "sparse_attention"):
        def counted(*args, kernel=getattr(attention, name), name=name):
            runs.append(name)
            return kernel(*args)
        monkeypatch.setattr(attention, name, counted)
    results = []
    for recompute in (False, True):
        model.recompute = recompute
        model.zero_grad()
        runs.clear()
        logits, ahead = model(ids, next_tokens=after)
        (logits.logsumexp(-1).mean() + ahead.logsumexp(-1).mean()).backward()
        experts = model.layers[3].experts
        results.append(([p.grad.clone() for p in model.parameters()], experts.histogram.clone(), sorted(runs)))
        experts.histogram = None
    (grads, counts, ran), (grads_again, counts_again, ran_again) = results
    assert all(torch.equal(a, b) for a, b in zip(grads, grads_again))
    assert torch.equal(counts, counts_again)
    assert ran == ran_again == ["exact_attention", "exact_attention", "sparse_attention"]


def training_pass(model, ids, after):
    """One forward and backward pass as text windows train: (scores, gradients
    by name, the last layer's balancing counts)."""
    model.zero_grad()
    logits, ahead = model(ids, next_tokens=after)
    (logits.logsumexp(-1).mean() + ahead.logsumexp(-1).mean()).backward()
    experts = model.layers[3].experts
    found = logits.detach(), {name: p.grad.clone() for name, p in model.named_parameters()}, experts.histogram.clone()
    experts.histogram = None
    return found


@pytest.mark.skipif(shutil.which("g++") is None, reason="torch.compile needs a C++ compiler on the CPU")
@pytest.mark.parametrize("recompute", [False, True], ids=["kept", "recomputed"])
def test_a_compiled_training_pass_trains_as_the_uncompiled_one_under_the_same_names(recompute):
    # docs/training.md, "Compiling the passes": the same scores and gradients
    # within rounding, balancing counting each token once, and weights
    # that load in either mode; evaluation and a window in parts stay uncompiled
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "block_size": 256})).train()
    model.recompute = recompute
    ids, after = torch.randint(50, (2, 128)), torch.randint(50, (2, 128))
    names = list(model.state_dict())
    scores, grads, counts = training_pass(model, ids, after)
    with torch.no_grad():
        carried = model(ids[:, :64], sequence=model.carried(0, {}, collect=False)), model.eval()(ids)
    model.train()

    model.compile_training()
    scores_again, grads_again, counts_again = training_pass(model, ids, after)
    assert torch.allclose(scores, scores_again, atol=2e-3)
    assert grads.keys() == grads_again.keys()
    assert all(torch.allclose(grads[name], grads_again[name], atol=2e-3) for name in grads)
    assert counts.sum() == counts_again.sum()

    assert list(model.state_dict()) == names
    other = WPT(ModelConfig(**{**TINY, "block_size": 256}))
    other.load_state_dict(model.state_dict())

    with torch.no_grad():
        model.train()
        assert torch.equal(model(ids[:, :64], sequence=model.carried(0, {}, collect=False)), carried[0])
        assert torch.equal(model.eval()(ids), carried[1])


@pytest.mark.skipif(select_device().type != "cuda" or not torch.version.cuda,
                    reason="needs an NVIDIA GPU, where torch.compile writes Triton code (on the AMD Radeon 860M the "
                           "model's uncompiled pass already stops with a segmentation fault)")
def test_a_compiled_training_pass_on_a_gpu_trains_as_the_uncompiled_one():
    # docs/training.md, "Compiling the passes": on a GPU, through the
    # attention kernels installed there and over a window in which In-Place
    # TTT completes a chunk of 1,024 tokens, the same scores and gradients
    # within BF16 rounding. The compiled pass replays the uncompiled one's
    # routes: rounding alone may route a token another way, which changes
    # it by more than rounding
    device = select_device()
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "block_size": 1280})).to(device).train()
    ids, after = torch.randint(50, (1, 1280)).to(device), torch.randint(50, (1, 1280)).to(device)
    routes = Routes()

    def run(**route):
        model.zero_grad()
        logits, ahead = model(ids, next_tokens=after, **route)
        (logits.logsumexp(-1).mean() + ahead.logsumexp(-1).mean()).backward()
        for layer in model.layers:  # the routed layers' balancing counts
            if hasattr(layer.experts, "histogram"):
                layer.experts.histogram = None
        return logits.detach().float(), {name: p.grad.clone() for name, p in model.named_parameters()}

    scores, grads = run(record=routes)
    model.compile_training()
    scores_again, grads_again = run(replay=routes)
    # The bounds are BF16 rounding's, measured on an NVIDIA GPU with the same routes: the compiled pass rounds
    # elsewhere than the uncompiled one (it keeps fused intermediates in FP32), and the two differ by about 1% in the
    # scores and up to 6% in one parameter's gradient (Engram's tables, the first layers' attention norms), each
    # within the 5% to 7% by which either lies from the same pass in FP32; in FP32, autocast off, they differ by
    # under 1e-4. Per parameter, so that a wrong gradient of a small one (the residual mixes, the TTT conv and
    # target) does not hide behind the embedding's and the head's, the test catches a gradient that is zeroed, of
    # the wrong sign, missing a term worth more than a tenth of it, or scaled by a factor off 1 by more than a tenth; an
    # error of a few percent hides inside BF16 rounding
    assert (scores - scores_again).norm() <= 2e-2 * scores.norm()
    for name, grad in grads.items():
        assert (grad - grads_again[name]).norm() <= 1e-1 * grad.norm() + 1e-6, name


@pytest.mark.skipif(select_device().type != "cuda" or not torch.version.cuda,
                    reason="needs an NVIDIA GPU, where torch.compile writes Triton code")
def test_a_recorded_training_pass_trains_as_the_same_pass_run_as_it_is():
    # docs/training.md, "Compiling the passes": the compiled pass over whole
    # windows, recorded as CUDA graphs and replayed, gives what the same
    # compiled pass gives without them, over several windows whose
    # gradients add up in one update, and in the updates after it, which
    # start with no gradients and no balancing counts: the same losses, kept
    # after the later windows' replays, the same balancing counts, and the
    # same gradients within the rounding by which two runs of the sparse
    # layers' backward pass differ (their atomic additions)
    device = select_device()
    torch.manual_seed(0)
    config = ModelConfig(**{**TINY, "block_size": 1280})
    recorded, unrecorded = WPT(config).to(device).train(), WPT(config).to(device).train()
    recorded.compile_training()
    unrecorded.compile_training()
    unrecorded.graphs = None
    generator = torch.Generator().manual_seed(1)
    windows = [torch.randint(50, (2, 2, 1280), generator=generator).to(device) for _ in range(3)]
    for update in range(2):
        unrecorded.load_state_dict(recorded.state_dict())  # the same weights and balancing biases
        losses = {}
        for model in (recorded, unrecorded):
            kept = []
            for x, y in windows:
                first, after = model(x, next_tokens=y, targets=(y, y[:, 1:]))
                ((first + 0.3 * after) / 3).backward()
                kept.append((first.detach(), after.detach()))
            losses[model] = torch.stack([torch.stack(pair) for pair in kept])
        assert torch.allclose(losses[recorded], losses[unrecorded], rtol=1e-3)
        for (name, mine), theirs in zip(recorded.named_parameters(), unrecorded.parameters()):
            assert (mine.grad - theirs.grad).norm() <= 1e-1 * theirs.grad.norm() + 1e-6, (update, name)
        for mine, theirs in zip(recorded.layers, unrecorded.layers):
            if not mine.experts.hashed:
                assert torch.equal(mine.experts.histogram, theirs.experts.histogram)
                assert mine.experts.histogram.sum() == len(windows) * 2 * 1280 * mine.experts.n
        for model in (recorded, unrecorded):
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter -= 1e-2 * parameter.grad
            model.zero_grad(set_to_none=True)
            model.after_update(lambda tensor, op: tensor)
    assert recorded.graphs.key is not None  # the passes did replay recorded graphs


@pytest.mark.skipif(select_device().type != "cuda" or not _fused_loss(),
                    reason="needs a GPU where the fused head and loss imports")
def test_the_fused_head_and_loss_give_the_checkpointed_form_s_loss_and_gradients(monkeypatch):
    # docs/model.md, "Training": the summed loss and its gradients, scaled by the incoming gradient
    # as training scales them, are the checkpointed chunks' within BF16 rounding
    device = select_device()
    monkeypatch.setattr(model_module, "CHUNK", 8)  # several chunks in a few tokens
    ids = torch.randint(0, 50, (3, 2, 16), generator=torch.Generator().manual_seed(0)).to(device)

    def run(fused):
        torch.manual_seed(0)
        model = WPT(ModelConfig(**TINY)).to(device).train()
        monkeypatch.setattr(model_module, "_fused_loss", _fused_loss if fused else lambda: False)
        next_sum, ahead_sum = model(ids[0], next_tokens=ids[1], targets=(ids[1], ids[2, :, :15]))
        (next_sum / 3 + 0.3 * ahead_sum / 7).backward()
        return next_sum.item(), ahead_sum.item(), {n: p.grad for n, p in model.named_parameters()}

    *losses, grads = run(True)
    *expected, expected_grads = run(False)
    assert losses == pytest.approx(expected, rel=2e-3)
    # the head, the norm before it and the embedding: deeper weights differ between two identical runs
    # already (atomic additions in the kernels), whichever form computes the loss
    for name in ("head.weight", "final_norm.weight", "embedding.weight"):
        assert (grads[name] - expected_grads[name]).abs().max() <= 0.05 * expected_grads[name].abs().max(), name


@pytest.mark.skipif(select_device().type != "cuda" or not _fused_loss(),
                    reason="needs a GPU where the fused head and loss imports")
def test_the_fused_head_and_loss_follow_the_cross_entropy_over_rows_of_several_blocks():
    # head_loss.py: a vocabulary of more than one block of the kernel and a part block, and the
    # ignored target -100, give F.cross_entropy's summed loss and gradients within BF16 rounding
    from wpt_model import head_loss as module

    device = select_device()
    generator = torch.Generator(device).manual_seed(0)
    V = 2 * module.BLOCK + 1000
    hidden = torch.randn(64, 32, device=device, generator=generator).bfloat16().requires_grad_()
    weight = (torch.randn(V, 32, device=device, generator=generator) * 0.5).bfloat16().requires_grad_()
    targets = torch.randint(0, V, (64,), device=device, generator=generator)
    targets[::7] = -100
    loss = module.head_loss(hidden, targets, weight)
    (2 * loss).backward()
    hidden_f, weight_f = hidden.detach().float().requires_grad_(), weight.detach().float().requires_grad_()
    expected = F.cross_entropy(hidden_f @ weight_f.t(), targets, reduction="sum")
    (2 * expected).backward()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-3)
    for got, want in ((hidden.grad, hidden_f.grad), (weight.grad, weight_f.grad)):
        assert (got.float() - want).abs().max() <= 2e-2 * want.abs().max()


@pytest.mark.skipif(select_device().type != "cuda" or not _fused_loss(),
                    reason="needs a GPU where the fused head and loss imports")
def test_the_fused_head_and_loss_in_chunks_is_as_close_to_the_exact_loss_as_the_chunks_under_autocast():
    # head_loss.py: an FP32 weight cast once, its chunks' BF16 gradients added up in one FP32 sum; against
    # the loss in FP64 on the same inputs, the loss and both gradients are within twice the distance of
    # the checkpointed form's math (autocast casting the weight for each chunk, autograd adding the
    # chunks' gradients), plus 1e-3 of the norm: the rounding that form itself makes
    from wpt_model import head_loss as module

    device = select_device()
    generator = torch.Generator(device).manual_seed(0)
    V, chunk = module.BLOCK + 3000, 16
    hidden = torch.randn(70, 64, device=device, generator=generator).bfloat16()
    weight = torch.randn(V, 64, device=device, generator=generator) * 0.3
    targets = torch.randint(0, V, (70,), device=device, generator=generator)

    def gradients(compute):
        h, w = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
        loss = compute(h, w)
        (loss * 0.25).backward()
        return loss.detach().double(), h.grad.double(), w.grad.double(), w.grad.dtype

    def by_autocast(h, w):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return sum(F.cross_entropy(F.linear(h[a:a + chunk], w).float(), targets[a:a + chunk], reduction="sum")
                       for a in range(0, len(targets), chunk))

    fused = gradients(lambda h, w: module.head_loss(h, targets, w, chunk))
    plain = gradients(by_autocast)
    exact = gradients(lambda h, w: F.cross_entropy(h.double() @ w.double().t(), targets, reduction="sum"))
    assert fused[3] == torch.float32
    for got, pytorch, reference in zip(fused[:3], plain[:3], exact[:3]):
        gap = float((got - reference).norm() / reference.norm())
        assert gap <= 2 * float((pytorch - reference).norm() / reference.norm()) + 1e-3


def test_the_same_pass_gives_the_same_gradients_every_time(few_chosen):
    # a resumed run trains exactly as a run never interrupted only if every
    # gradient adds up in a fixed order: the sparse layers' gradients added
    # into the exact layer's keys and values at the tokens they read among
    # them
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "block_size": 256}))
    ids = torch.randint(50, (2, 257), generator=torch.Generator().manual_seed(0))

    def gradients():
        model.zero_grad()
        scores, ahead = model(ids[:, :-1], next_tokens=ids[:, 1:])
        (scores.float().logsumexp(-1).mean() + ahead.float().logsumexp(-1).mean()).backward()
        return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    first, second = gradients(), gradients()
    assert first.keys() == second.keys() and all(torch.equal(first[k], second[k]) for k in first)


def test_every_parameter_belongs_to_exactly_one_named_part():
    # a training step locks the model's parts by these names (docs/model.md,
    # "Parts"): a parameter outside them could never be locked, and one in
    # two would be locked by either name
    model = WPT(ModelConfig(**TINY))
    parts = model.parts()
    assert list(parts) == list(PARTS)
    found = [id(parameter) for named in parts.values() for parameter in named]
    assert sorted(found) == sorted(id(parameter) for parameter in model.parameters())  # each once
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    assert sorted(names[id(p)] for p in parts[ROUTERS]) == sorted(
        name for name in names.values() if ".experts.router." in name)
    assert parts[MULTI_TOKEN_PREDICTION] and all(names[id(p)].startswith("mtp") for p in parts[MULTI_TOKEN_PREDICTION])
    assert parts[BALANCING] == []  # the balancing biases are set after each update, never by a gradient


def test_a_failed_import_of_the_fused_head_and_loss_keeps_its_reason(monkeypatch):
    # the reason is what tells a run that the output head and its loss run
    # in checkpointed chunks instead of the fused kernel, and why
    monkeypatch.setattr(model_module, "_fused", None)
    monkeypatch.setitem(sys.modules, "wpt_model.head_loss", None)  # an import of it then fails
    assert _fused_loss() is False
    assert fused_loss_failure() == "ModuleNotFoundError: import of wpt_model.head_loss halted; None in sys.modules"

    monkeypatch.setattr(model_module, "_fused", None)
    loss = types.ModuleType("wpt_model.head_loss")  # the real one needs Triton
    loss.head_loss = object()
    monkeypatch.setitem(sys.modules, "wpt_model.head_loss", loss)
    assert _fused_loss() is loss.head_loss and fused_loss_failure() is None


def test_a_model_with_one_vocabulary_table_reads_its_embeddings_from_the_output_head():
    # tied_tables: the embedding is the output head's rows times a fixed scale, so the model holds one
    # table, in the output head's part, which a loss reaches through both of its uses
    torch.manual_seed(0)
    tied, plain = WPT(ModelConfig(**TINY, tied_tables=True)), WPT(ModelConfig(**TINY))
    assert sum(p.numel() for p in plain.parameters()) - sum(p.numel() for p in tied.parameters()) == 50 * 16
    assert "embedding.weight" not in tied.state_dict() and "head.weight" in tied.state_dict()
    parts = tied.parts()
    assert parts["embedding"] == [] and [id(p) for p in parts["output-head"]] == [id(tied.head.weight)]
    assert [role for name, _, role, _ in tied.optimizer_roles() if name == "head.weight"] == ["vector"]
    ids = torch.randint(50, (2, 16))
    assert torch.equal(tied.embedding(ids), tied.head.weight[ids] * 48 ** 0.5)
    # the layers' input keeps the scale an untied model's embedding has
    assert 0.8 < tied.embedding(torch.arange(50)).std() < 1.2 and 0.8 < plain.embedding.weight.std() < 1.2
    tied.embedding(ids).sum().backward()
    as_input = tied.head.weight.grad.clone()
    assert as_input.abs().sum() > 0
    tied.zero_grad()
    F.cross_entropy(tied(ids).flatten(0, 1), ids.flatten()).backward()
    assert torch.isfinite(tied.head.weight.grad).all() and tied.head.weight.grad.abs().sum() > 0


def situ_by_formula(gate, up):
    return 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * (25 * torch.tanh(up / 25))


@pytest.mark.skipif(select_device().type != "cuda" or importlib.util.find_spec("triton") is None,
                    reason="needs a GPU where Triton runs")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_situ_glu_kernel_is_as_close_to_the_formula_as_the_formula_in_pytorch(dtype):
    # situ_kernels.py: against the formula in FP64 on the same inputs, its output and both gradients are
    # within twice the distance of the formula computed in PyTorch in the same dtype, plus 1e-6 of the
    # norm: the rounding the PyTorch form makes, one rounding per operation, against one at the end
    from wpt_model.situ_kernels import situ_glu as fused

    device = select_device()
    generator = torch.Generator(device).manual_seed(0)
    gate = (torch.randn(77, 300, device=device, generator=generator) * 6).to(dtype)
    up = (torch.randn(77, 300, device=device, generator=generator) * 30).to(dtype)
    direction = torch.randn(77, 300, device=device, generator=generator).to(dtype)

    def run(compute, cast):
        g, u = cast(gate).requires_grad_(), cast(up).requires_grad_()
        out = compute(g, u)
        out.backward(cast(direction))
        return [x.double() for x in (out.detach(), g.grad, u.grad)]

    exact = run(situ_by_formula, lambda x: x.double())
    for got, plain, reference in zip(run(fused, torch.clone), run(situ_by_formula, torch.clone), exact):
        gap = float((got - reference).norm() / reference.norm())
        assert gap <= 2 * float((plain - reference).norm() / reference.norm()) + 1e-6


def test_off_a_gpu_or_without_triton_situ_glu_is_the_formula(monkeypatch):
    # layers.situ_glu: the PyTorch form on the CPU, and on a GPU where the kernel does not import
    from wpt_model import layers

    gate, up = torch.randn(5, 7) * 4, torch.randn(5, 7) * 30
    assert torch.equal(layers.situ_glu(gate, up), situ_by_formula(gate, up))
    monkeypatch.setattr(layers, "_situ", None)
    monkeypatch.setitem(sys.modules, "wpt_model.situ_kernels", None)  # its import fails
    assert layers._fused_situ() is None and layers._situ is False


def test_the_next_token_loss_alone_is_the_first_of_the_two_and_runs_no_multi_token_prediction(monkeypatch):
    # what validation reads: the same sum bit for bit as the first of the
    # two losses, without the multi-token prediction block's work
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "block_size": 256})).eval()
    ids, after = torch.randint(50, (2, 200)), torch.randint(50, (2, 200))
    with torch.no_grad():
        both = model(ids, next_tokens=after, targets=(after, after[:, 1:]))
        monkeypatch.setattr(model.mtp, "forward", lambda *args, **kwargs: pytest.fail("the block ran"))
        alone = model(ids, targets=(after,))
    assert torch.equal(alone, both[0])


def test_the_passes_compile_only_where_a_cuda_device_triton_and_a_c_compiler_are(monkeypatch):
    # docs/training.md, "Compiling the passes": what wpt train checks before it compiles
    import torch._inductor.cpp_builder as builder
    import torch.utils._triton as triton_check
    from wpt_model.model import compile_failure

    assert "this process runs on cpu" in compile_failure(torch.device("cpu"))
    monkeypatch.setattr(triton_check, "has_triton", lambda: False)
    assert "Triton does not import" in compile_failure(torch.device("cuda"))
    monkeypatch.setattr(triton_check, "has_triton", lambda: True)

    def none():
        raise RuntimeError("no g++\nmore")

    monkeypatch.setattr(builder, "get_cpp_compiler", none)
    assert compile_failure(torch.device("cuda")).endswith("none works here (RuntimeError: no g++)")
    monkeypatch.setattr(builder, "get_cpp_compiler", lambda: "g++")
    assert compile_failure(torch.device("cuda")) is None
