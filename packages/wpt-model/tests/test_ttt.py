"""In-Place TTT (docs/model.md, "In-Place TTT"): the shared experts' down
projection changes chunk by chunk as the paper's apply-then-update loop
says, returns to its trained value at each document, and segments and the
generation cache give what the whole text gives."""

import pytest
import torch

from wpt_model import WPT, ModelConfig
from wpt_model import ttt
from wpt_model.cache import Rows
from wpt_model.experts import Experts
from wpt_model.layers import situ_glu
from wpt_model.sequence import WHOLE, Carried
from wpt_model.ttt import END_OF_DOCUMENT, RATE, InPlaceTTT

CHUNK = 8
D, F = 6, 10


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    """Chunks of 8 tokens instead of 1,024, so that a few dozen tokens hold several."""
    monkeypatch.setattr(ttt, "CHUNK", CHUNK)


def module():
    torch.manual_seed(0)
    found = InPlaceTTT(D)
    with torch.no_grad():  # the convolution starts at zero: made to count
        found.conv.weight.normal_()
    return found


def inputs(T=45, rows=2):
    """z, embeddings and ids of `rows` texts of T tokens: documents end in
    the middle of a chunk, at a chunk's last token, and twice in one
    chunk."""
    generator = torch.Generator().manual_seed(1)
    z = torch.randn(rows, T, F, generator=generator, requires_grad=True)
    embedding = torch.randn(rows, T, D, generator=generator, requires_grad=True)
    ids = torch.randint(END_OF_DOCUMENT, (rows, T), generator=generator)
    ids[0, [11, 23, 33, 36]] = END_OF_DOCUMENT
    ids[1, [7, 30]] = END_OF_DOCUMENT
    return z, embedding, ids


def paper(layer, z, embedding, ids):
    """RATE z S^T token by token, as the paper's loop says: each chunk reads
    the matrix the chunks before it left, a token after an end of document
    in its chunk reads the trained one, and each whole chunk then takes one
    step, v = W_target conv(e) over its tokens after its last end, with S
    starting over at a chunk that holds an end."""
    B, T, _ = z.shape
    w = layer.conv.weight[:, 0]  # (D, KERNEL), centred
    reach = w.shape[1] // 2
    out = []
    for b in range(B):
        S, rows = torch.zeros(D, F), []
        for a in range(0, T, CHUNK):
            chunk = range(a, min(a + CHUNK, T))
            ended = False
            for t in chunk:
                rows.append(torch.zeros(D) if ended else RATE * (S @ z[b, t]))
                ended = ended or ids[b, t] == END_OF_DOCUMENT
            if len(chunk) < CHUNK:
                break
            last = max([i for i, t in enumerate(chunk) if ids[b, t] == END_OF_DOCUMENT], default=-1)
            e = [embedding[b, a + i] if i > last else torch.zeros(D) for i in range(CHUNK)]
            update = torch.zeros(D, F)
            for i in range(last + 1, CHUNK):
                conv = sum(w[:, k] * e[i + k - reach] for k in range(w.shape[1]) if 0 <= i + k - reach < CHUNK)
                update = update + torch.outer(layer.target(conv), z[b, a + i])
            S = (torch.zeros(D, F) if last >= 0 else S) + update
        out.append(torch.stack(rows))
    return torch.stack(out)


def test_the_matrix_changes_chunk_by_chunk_and_starts_over_at_each_document():
    layer = module()
    z, embedding, ids = inputs()
    got = layer(z, embedding, ids, WHOLE)
    expected = paper(layer, z, embedding, ids)
    assert torch.allclose(got, expected, atol=1e-5)
    assert got[0, CHUNK:].abs().max() > 0  # the change counts
    weights = torch.randn_like(got)
    gradients = torch.autograd.grad((got * weights).sum(), [z, embedding, layer.conv.weight, layer.target.weight])
    expected_gradients = torch.autograd.grad((expected * weights).sum(),
                                             [z, embedding, layer.conv.weight, layer.target.weight])
    for found, wanted in zip(gradients, expected_gradients):  # the parts that build the target learn
        assert torch.allclose(found, wanted, atol=1e-4)


def test_segments_carry_the_unfinished_chunk_and_the_sum_of_updates():
    layer = module()
    z, embedding, ids = inputs()
    with torch.no_grad():
        expected = layer(z, embedding, ids, WHOLE)
        carried, out, start = {}, [], 0
        for size in (5, 13, 1, 16, 10):
            place = Carried(start, carried)
            out.append(layer(z[:, start:start + size], embedding[:, start:start + size], ids[:, start:start + size],
                             place))
            carried, start = place.passed, start + size
    assert torch.allclose(torch.cat(out, 1), expected, atol=1e-5)


def test_the_cache_reads_rows_at_their_own_places_and_takes_back_tokens():
    layer = module()
    z, embedding, ids = inputs()
    with torch.no_grad():
        expected = layer(z, embedding, ids, WHOLE)
        rows, start = Rows(2, z.device), torch.zeros(2, dtype=torch.long)
        got = torch.zeros_like(expected)
        # row 0 reads 3 then 9 tokens per call, row 1 7 then 4: they reach chunk ends at different calls
        for step, counts in enumerate([(3, 7), (9, 4)] * 4 + [(9, 9)]):
            counts = torch.tensor(counts)
            counts = torch.minimum(counts, 45 - start)
            n = int(counts.max())
            index = (start[:, None] + torch.arange(n)).clamp(max=44)
            reading = rows.reading(counts, undoable=step == 3)
            out = layer(*(t[torch.arange(2)[:, None], index] for t in (z, embedding, ids)), reading)
            reading.finish()
            if step == 3:  # a draft of 9 and 4 tokens, of which 2 and 0 are kept
                kept = torch.tensor([2, 0])
                rows.undo(kept)
                counts = kept
            for b in range(2):
                got[b, start[b]:start[b] + counts[b]] = out[b, :counts[b]]
            start = start + counts
    assert (start == 45).all()
    assert torch.allclose(got, expected, atol=1e-5)


def test_every_sixth_layer_from_the_first_changes_its_shared_experts_down_projection():
    config = ModelConfig(vocab_size=300, block_size=16, window_layers=2, sparse_groups=1, sparse_layers=5, d_model=16,
                         attention_heads=2, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8,
                         latent_width=8, shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2,
                         engram_dim=4, engram_table_size=31, residual_groups=22)
    model = WPT(config)
    assert [n for n, layer in enumerate(model.layers, 1) if layer.experts.ttt is not None] == [1, 7]
    roles = {name: role for name, _, role, _ in model.optimizer_roles()}
    assert roles["layers.0.experts.ttt.target.weight"] == "matrix"  # NorMuon
    assert roles["layers.0.experts.ttt.conv.weight"] == "vector"   # AdamW, as the other convolutions


def test_the_shared_experts_output_is_the_trained_matrix_plus_the_change():
    config = ModelConfig(vocab_size=300, block_size=16, window_layers=1, sparse_groups=1, sparse_layers=1, d_model=D,
                         attention_heads=1, attention_head_dim=64, n_experts=4, top_k=2, expert_width=8,
                         latent_width=8, shared_width=F // 2, engram_layers=[], engram_order=3, engram_heads=1,
                         engram_dim=4, engram_table_size=31, residual_groups=1)
    torch.manual_seed(0)
    changing, plain = Experts(config, hashed=True, ttt=True), Experts(config, hashed=True)
    plain.load_state_dict(changing.state_dict(), strict=False)
    with torch.no_grad():
        changing.ttt.conv.weight.normal_()
    x = torch.randn(2, 45, D)
    _, embedding, ids = inputs()
    with torch.no_grad():
        z = torch.cat([situ_glu(expert.gate(x), expert.up(x)) for expert in changing.shared], -1)
        change = changing.ttt(z, embedding, ids, WHOLE)
        assert torch.allclose(changing(x, ids, embedding=embedding), plain(x, ids) + change, atol=1e-5)


def test_the_id_that_ends_a_document_is_the_tokenizer_s():
    wpt_tokenizer = pytest.importorskip("wpt_tokenizer")
    assert END_OF_DOCUMENT == wpt_tokenizer.END
