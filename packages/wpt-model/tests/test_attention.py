"""The three kinds of attention layer (docs/model.md, "Attention") against
HySparse2's formulas, one token at a time."""

import pytest
import torch

from wpt_model import WPT, ModelConfig
from wpt_model import attention
from wpt_model.attention import ROPE_DIMS, WINDOW, ExactAttention, SparseAttention, WindowAttention

CONFIG = ModelConfig(vocab_size=50, block_size=300, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=32,
                     attention_heads=4, attention_head_dim=64 + 16, n_experts=4, top_k=2, expert_width=8,
                     latent_width=8, shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4,
                     engram_table_size=31, residual_groups=3)


def rms(x, weight):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt() * weight


def rope(x, position):
    """Rotary encoding of the last ROPE_DIMS dimensions of x (..., d), in
    adjacent pairs, base 10000, at position."""
    rest, turning = x[..., :-ROPE_DIMS], x[..., -ROPE_DIMS:]
    pairs = torch.arange(ROPE_DIMS // 2, dtype=torch.float32)
    angle = position / 10000 ** (2 * pairs / ROPE_DIMS)
    even, odd = turning[..., 0::2], turning[..., 1::2]
    turned = torch.stack([even * angle.cos() - odd * angle.sin(), even * angle.sin() + odd * angle.cos()], -1)
    return torch.cat([rest, turned.flatten(-2)], -1)


def heads(layer, x):
    """Each query head of one token, normalized: (H, c)."""
    return rms((layer.q.weight @ x).view(layer.heads, layer.head_dim), layer.q_norm.weight)


def key(layer, x):
    return rms(layer.k.weight @ x, layer.k_norm.weight)


def attend(q, keys, values, sink, scale):
    """softmax over the keys, each head's sink in the denominator (none when sink is None)."""
    z = (q @ keys.T) * scale
    denominator = z.exp().sum(-1) + (0 if sink is None else sink.exp())
    return (z.exp() / denominator[:, None]) @ values


def gated_output(layer, o, x):
    return layer.out.weight @ (o.flatten() * torch.sigmoid(layer.gate.weight @ x))


def randomized(layer):
    """The layer with its sinks and norms away from their starting values, so they count."""
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if name == "sink" or name.endswith("norm.weight"):
                p.normal_(mean=1 if name.endswith("norm.weight") else 0, std=0.5)
    return layer


def test_a_window_layer_gives_the_formula_s_result():
    # QK-norm, rotary encoding of the last 64 dimensions of each query and
    # key, the last 128 tokens, a sink per head and a sigmoid gate; 300
    # tokens, so the window crosses its blocks of 128
    torch.manual_seed(0)
    layer = randomized(WindowAttention(CONFIG))
    x = torch.randn(1, 300, CONFIG.d_model)
    scale = layer.head_dim ** -0.5
    keys = torch.stack([rope(key(layer, x[0, t]), t) for t in range(300)])
    values = (layer.v.weight @ x[0].T).T
    expected = []
    for t in range(300):
        seen = slice(max(0, t - WINDOW + 1), t + 1)
        o = attend(rope(heads(layer, x[0, t]), t), keys[seen], values[seen], layer.sink, scale)
        expected.append(gated_output(layer, o, x[0, t]))
    with torch.no_grad():
        assert torch.allclose(layer(x)[0], torch.stack(expected), atol=1e-5)


def test_exact_and_sparse_layers_give_the_formula_s_result(monkeypatch):
    # the exact layer: QK-norm, no position encoding, every token up to
    # itself, keys and values from the input it is given (here another
    # than its queries', as in the second half); it picks for each query
    # the tokens it weighs most (the largest weight over its heads) among
    # those at least the window back. The sparse layer reads, through its
    # own queries, those tokens and its window, with a sink and a gate.
    # A window of 8 and 4 picks, so that 40 tokens exercise the choice
    monkeypatch.setattr(attention, "WINDOW", 8)
    monkeypatch.setattr(attention, "TOP_K", 4)
    torch.manual_seed(0)
    exact, sparse = ExactAttention(CONFIG), randomized(SparseAttention(CONFIG))
    x, source, later = (torch.randn(1, 40, CONFIG.d_model) for _ in range(3))
    scale = exact.head_dim ** -0.5
    keys = torch.stack([key(exact, source[0, t]) for t in range(40)])
    values = (exact.v.weight @ source[0].T).T
    expected_exact, expected_sparse = [], []
    for t in range(40):
        q = heads(exact, x[0, t])
        expected_exact.append(exact.out.weight @ attend(q, keys[:t + 1], values[:t + 1], None, scale).flatten())
        weights = torch.softmax((q @ keys[:t + 1].T) * scale, -1).amax(0)
        far = weights[:max(0, t + 1 - 8)]
        rows = set(far.topk(min(4, len(far))).indices.tolist()) | set(range(max(0, t - 7), t + 1))
        rows = sorted(rows)
        o = attend(heads(sparse, later[0, t]), keys[rows], values[rows], sparse.sink, scale)
        expected_sparse.append(gated_output(sparse, o, later[0, t]))
    with torch.no_grad():
        out, handoff = exact(x, source)
        assert torch.allclose(out[0], torch.stack(expected_exact), atol=1e-5)
        assert torch.allclose(sparse(later, handoff)[0], torch.stack(expected_sparse), atol=1e-5)


def test_the_layers_follow_the_layout_numbers():
    # window layers on each side of an exact layer, then groups of one
    # exact layer and its sparse layers
    config = ModelConfig(**{**CONFIG.__dict__, "window_layers": 2, "sparse_groups": 2, "sparse_layers": 3,
                            "residual_groups": 3})
    kinds = [type(layer.attention) for layer in WPT(config).layers]
    window, exact, sparse = WindowAttention, ExactAttention, SparseAttention
    assert kinds == [window, window, exact, window, window, exact, sparse, sparse, sparse, exact, sparse, sparse,
                     sparse]
    assert config.n_layer == 13 and config.first_half == 5


def test_only_an_exact_layer_sparse_layers_follow_picks_tokens(monkeypatch):
    # the first half's exact layer hands its choice to no sparse layer
    monkeypatch.setattr(attention, "WINDOW", 8)
    model = WPT(CONFIG)
    first, second = [layer.attention for layer in model.layers if isinstance(layer.attention, ExactAttention)]
    x = torch.randn(1, 40, CONFIG.d_model)
    with torch.no_grad():
        assert first(x, x)[1][2].shape[-1] == 0
        assert second(x, x)[1][2].shape[-1] == 40 - 8


def test_the_second_half_reads_the_first_half_s_output_for_its_keys_and_values():
    # changing a second-half exact layer's own input, its queries' input,
    # leaves the keys and values it hands on as they were: they come from
    # the first half's output
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**CONFIG.__dict__, "sparse_groups": 2})).eval()
    handed = []
    second = model.layers[CONFIG.first_half + CONFIG.sparse_layers + 1]  # the exact layer of the second group
    assert isinstance(second.attention, ExactAttention)
    second.attention.register_forward_hook(lambda module, args, out: handed.append(out[1][:2]))
    ids = torch.randint(50, (1, 20), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        model(ids)
        torch.nn.init.normal_(model.layers[CONFIG.first_half + 1].experts_norm.weight)  # changes later inputs only
        model(ids)
    (k1, v1), (k2, v2) = handed
    assert torch.equal(k1, k2) and torch.equal(v1, v2)


def test_the_mtp_block_sees_only_the_last_128_tokens():
    # its attention is a window layer's: a change at position 10 reaches
    # positions 10 .. 137 and no other
    torch.manual_seed(0)
    mtp = WPT(CONFIG).mtp.eval()
    hidden, following = torch.randn(1, 300, CONFIG.d_model), torch.randn(1, 300, CONFIG.d_model)
    changed = hidden.clone()
    changed[:, 10] += 1
    with torch.no_grad():
        before, after = mtp(hidden, following), mtp(changed, following)
    differs = (before - after).abs().amax(-1)[0] > 1e-5
    assert differs[10:10 + WINDOW].all() and not differs[:10].any() and not differs[10 + WINDOW:].any()


@pytest.mark.parametrize("bad", [{"attention_head_dim": 32}, {"attention_head_dim": 65}, {"sparse_groups": 0}])
def test_a_config_the_layers_cannot_have_is_refused(bad):
    # a window layer rotates 64 dimensions in pairs, and the second half
    # holds at least one group
    with pytest.raises(ValueError, match="model config"):
        ModelConfig(**{**CONFIG.__dict__, **bad})
