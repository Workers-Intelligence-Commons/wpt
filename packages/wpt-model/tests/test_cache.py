"""The generation cache (docs/model.md, "Generation"): reading tokens
through it gives the scores the whole text gives without it, within
rounding, for each kind of layer and the whole model, past block_size; and
so do a prompt shared by several rows, rows read together at different
places, and undoing rejected draft tokens."""

import pytest
import torch

from wpt_model import WPT, ModelConfig
from wpt_model import attention
from wpt_model.attention import ExactAttention, SparseAttention, WindowAttention
from wpt_model.cache import Rows
from wpt_model.engram import Engram
from wpt_model.sequence import WHOLE, Carried
from wpt_utils.device import select_device

CONFIG = ModelConfig(vocab_size=50, block_size=16, window_layers=1, sparse_groups=1, sparse_layers=2, d_model=32,
                     attention_heads=4, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8, latent_width=8,
                     shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4,
                     engram_table_size=31, residual_groups=3)
# a prompt, single tokens, a few, a long piece and the rest: 300 tokens, past block_size (16) and the window
# (128), with pieces that end inside and across the window's blocks
PIECES = [37, 1, 1, 5, 100, 1, 155]


@pytest.fixture(autouse=True)
def few_chosen(monkeypatch):
    """Exact layers pick 16 tokens instead of 1,024, so that the choice
    matters within these 300 tokens."""
    monkeypatch.setattr(attention, "TOP_K", 16)


def close(got, expected, share=1e-3):
    """Equal within share of the expected values' largest magnitude: the
    same computation in another order."""
    return (got - expected).abs().max() <= share * expected.abs().max()


def in_pieces(read, x, pieces=PIECES):
    """read(piece, start, reading) over x (B, T, ...) piece by piece, each
    piece read through a cache's rows (cache.Reading); the outputs joined."""
    rows, out, start = Rows(x.shape[0], x.device), [], 0
    for size in pieces:
        reading = rows.reading(torch.full((x.shape[0],), size, device=x.device), undoable=False)
        out.append(read(x[:, start:start + size], start, reading))
        reading.finish()
        start += size
    assert start == x.shape[1]
    return torch.cat(out, 1)


def randomized(module):
    """module with its zero-started parameters (sinks, Engram's convolution) made to count."""
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.abs().max() == 0:
                p.normal_(std=0.5)
    return module


# on a GPU, through the attention's form there, in BF16, within a wider share
DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(select_device().type != "cuda", reason="needs a GPU"))]


def on(device):
    """(the device, the share of rounding expected there, autocast as the model runs there)."""
    if device == "cpu":
        return torch.device("cpu"), 1e-3, torch.autocast("cpu", enabled=False)
    return select_device(), 3e-2, torch.autocast("cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("device", DEVICES)
def test_a_window_layer_reads_through_its_last_keys_and_values_as_through_the_whole_text(device):
    device, share, autocast = on(device)
    torch.manual_seed(0)
    layer = randomized(WindowAttention(CONFIG)).eval().to(device)
    x = torch.randn(2, 300, CONFIG.d_model, device=device)
    with torch.no_grad(), autocast:
        got = in_pieces(lambda piece, _, reading: layer(piece, reading), x)
        assert close(got.float(), layer(x).float(), share)


@pytest.mark.parametrize("device", DEVICES)
def test_exact_and_sparse_layers_read_through_the_exact_layer_s_keys_and_values_as_through_the_whole_text(device):
    # the exact layer keeps every token's key and value; the sparse layer
    # reads them, at the tokens the exact layer chose and its last 128,
    # and keeps nothing of its own
    device, share, autocast = on(device)
    torch.manual_seed(0)
    exact, sparse = (randomized(kind(CONFIG)).eval().to(device) for kind in (ExactAttention, SparseAttention))
    x = torch.randn(2, 300, CONFIG.d_model, device=device)

    def both(piece, place):
        out, handoff = exact(piece, piece, place)
        return torch.cat([out, sparse(piece, handoff, place)], -1)

    with torch.no_grad(), autocast:
        got = in_pieces(lambda piece, _, reading: both(piece, reading), x)
        assert close(got.float(), both(x, WHOLE).float(), share)


def test_engram_reads_through_its_last_ids_and_convolution_inputs_as_through_the_whole_text():
    torch.manual_seed(0)
    engram = randomized(Engram(CONFIG)).eval()
    hidden = torch.randn(2, 300, CONFIG.d_model)
    canonical = torch.randint(50, (2, 300))
    with torch.no_grad():
        got = in_pieces(lambda piece, start, reading: engram(piece, canonical[:, start:start + piece.shape[1]],
                                                             reading), hidden)
        assert close(got, engram(hidden, canonical, WHOLE))


def model():
    torch.manual_seed(0)
    return randomized(WPT(CONFIG)).eval()


def test_the_model_reads_through_its_cache_as_through_the_whole_text_past_block_size(routing):
    # with the multi-token prediction block's window, which reads the
    # (hidden state, next token) pairs as they come. On a GPU, BF16's
    # rounding sends some tokens of this random model to other experts, so
    # the whole model is compared on the CPU only. On the CPU, the cache's
    # routes differ from the whole text's at near ties only, and a row's scores are compared before the
    # first position that went to other experts
    wpt = model()
    ids = torch.randint(50, (2, 301), generator=torch.Generator().manual_seed(0))
    cache = wpt.new_cache(2, routes=True)
    with torch.no_grad():
        scores, ahead = wpt(ids[:, :300], next_tokens=ids[:, 1:])
        got, got_ahead, start = [], [], 0
        for size in PIECES:
            piece, hidden = wpt.extend(ids[:, start:start + size], cache)
            got.append(piece)
            got_ahead.append(wpt.extend_mtp(hidden, ids[:, start + 1:start + size + 1], cache))
            start += size
    got, got_ahead = torch.cat(got, 1), torch.cat(got_ahead, 1)
    for row in range(2):
        first = routing.first_difference(wpt, ids[row, :300].tolist(), cache.routes[row])
        n = 300 if first is None else first
        assert close(got[row, :n], scores[row, :n]) and close(got_ahead[row, :n], ahead[row, :n])
    assert cache.lengths.tolist() == [300, 300] and cache.mtp.lengths.tolist() == [300, 300]


def test_segments_and_the_cache_carry_exactly_what_each_module_says_it_passes_on():
    # one description per module
    # (passes_on, carry.py) of what it passes on to later tokens; a segment
    # of a window and the generation cache carry those values and no other
    wpt = model()
    described = {(id(module), name) for module in wpt.modules() if hasattr(module, "passes_on")
                 for name in module.passes_on()}
    assert {name for module in wpt.modules() if hasattr(module, "passes_on") for name in module.passes_on()} == {
        "inputs", "ids", "window", "entries", "tokens", "state"}
    ids = torch.randint(50, (2, 41), generator=torch.Generator().manual_seed(0))
    segment = Carried(0, {})
    with torch.no_grad():
        wpt(ids[:, :40], next_tokens=ids[:, 1:], sequence=segment)
    assert set(segment.passed) | set(segment.added) == {(*key, 0) for key in described}  # its one pass's
    cache = wpt.new_cache(2)
    with torch.no_grad():
        _, hidden = wpt.extend(ids[:, :40], cache)
        wpt.extend_mtp(hidden, ids[:, 1:], cache)
    assert set(cache.text.values) | set(cache.mtp.values) == described


def by_itself(wpt, ids):
    """The scores of one text, read whole without a cache."""
    with torch.no_grad():
        return wpt(torch.tensor([ids]))[0]


def test_a_shared_prompt_and_rows_read_together_give_each_row_s_own_scores(routing):
    # one prompt read once and copied into 3 rows; then the rows go on with
    # their own tokens, in one batch, each its own number per call (the
    # rest of its row padding), so they end up at different places; each
    # row's scores are compared before the first position whose routes
    # differ from the whole text's (at near ties only)
    wpt = model()
    generator = torch.Generator().manual_seed(0)
    prompt = torch.randint(50, (140,), generator=generator).tolist()
    texts = [prompt + torch.randint(50, (n,), generator=generator).tolist() for n in (170, 10, 61)]
    shared = wpt.new_cache(1, routes=True)
    with torch.no_grad():
        wpt.extend(torch.tensor([prompt]), shared)
        cache = shared.select([0, 0, 0])
        got = [[] for _ in texts]
        place = [len(prompt)] * 3
        for counts in ([1, 1, 1], [40, 0, 7], [3, 8, 1], [126, 1, 52]):
            width = max(counts)
            ids = torch.zeros(3, width, dtype=torch.long)
            for row, count in enumerate(counts):
                ids[row, :count] = torch.tensor(texts[row][place[row]:place[row] + count])
            scores, _ = wpt.extend(ids, cache, counts)
            for row, count in enumerate(counts):
                got[row].append(scores[row, :count])
                place[row] += count
    assert place == [len(text) for text in texts]
    for row, text in enumerate(texts):
        first = routing.first_difference(wpt, text, cache.routes[row])
        n = len(text) if first is None else first
        assert n > len(prompt) and close(torch.cat(got[row])[:n - len(prompt)], by_itself(wpt, text)[len(prompt):n])
    assert shared.lengths.tolist() == [len(prompt)]  # the copies changed apart from it


def test_undoing_rejected_draft_tokens_leaves_the_cache_as_if_only_the_kept_ones_were_read(routing):
    # each round, every row reads 4 draft tokens and keeps a different
    # number of them; the kept ones are the text's own, so every score
    # (each round's reads what the undone rounds left) must be the text's
    # read whole, and the routes recorded those of the kept tokens alone;
    # the scores are compared before the first position whose routes differ
    # from the whole text's (at near ties only)
    wpt = model()
    generator = torch.Generator().manual_seed(0)
    texts = [torch.randint(50, (330,), generator=generator).tolist() for _ in range(2)]
    cache = wpt.new_cache(2, routes=True)
    with torch.no_grad():
        wpt.extend(torch.tensor([text[:120] for text in texts]), cache)
        place, got = [120, 120], [[], []]
        for kept in ([4, 0], [1, 4], [0, 2], [3, 3], [2, 1], [4, 4], [0, 0], [1, 3]) * 3:
            ids = torch.tensor([texts[row][place[row]:place[row] + 4] for row in range(2)])
            drafts = ids.clone()
            for row in range(2):  # past the kept ones, drafts the text does not hold
                drafts[row, kept[row]:] = (ids[row, kept[row]:] + 1 + row) % 50
            scores, _ = wpt.extend(drafts, cache, undoable=True)
            wpt.undo(cache, kept)
            for row in range(2):
                got[row].append(scores[row, :kept[row]])
                place[row] += kept[row]
        assert cache.lengths.tolist() == place
        for row in range(2):
            whole = by_itself(wpt, texts[row][:place[row]])
            first = routing.first_difference(wpt, texts[row][:place[row]], cache.routes[row])
            n = place[row] if first is None else first
            assert n > 120 and close(torch.cat(got[row])[:n - 120], whole[120:n])
            routed = wpt(torch.tensor([texts[row][:place[row]]]), replay=cache.routes[row])
            assert close(routed[0, :n], whole[:n])
            assert all(len(chosen) == place[row] for chosen in cache.routes[row].chosen.values())
