"""The PyTorch form of the two attention operations whose cost grows with
the text (kernels.py): the reference that the GPU kernels of
triton_kernels.py and tilelang_kernels.py are tested against, and what runs
where no kernels do.

Both operations read a table of every token so far, one key-value head
shared by every query head (MQA), where token p sits at row p, and a
query at position p sees the rows up to p:

    exact_attention   every query attends to every row it sees, and also
                      gives the top_k rows it weighs most among those at
                      least `window` tokens back (HySparse2's oracle
                      selection, arXiv 2609.26368)
    sparse_attention  every query attends to the rows its indices name plus
                      its last `window` tokens, with a learned sink logit
                      per head added to the softmax's denominator

The contract every kernel module keeps (NAME, exact_attention,
sparse_attention): q (B, T, H, c) and the table's k and v (B, N, c) in one
dtype; positions (B, T) int64, each query's position, which is also its own
row; sink (H,) FP32; indices (B, T, K) int64, -1 for none. A query of a
generation cache's padding may sit past the table's last row: it reads
the rows up to the last. Each operation is differentiable in its
floating-point inputs and returns its output in q's dtype.

A row's weight for a query is its softmax probability in one head, and a
row's score is its largest weight over the heads: HySparse's group-wise
maximum, where MQA makes every head one group. Selection compares the
scores of one query only, so each query picks its own rows. Equal scores
go to whichever row topk returns first.

This form computes in FP32 from inputs in any dtype, one block of queries
at a time, in the forward and the backward pass, so memory holds one
block's scores (about BLOCK_VALUES values) instead of every query's: it
grows with the text, never with its square. A block reads only the rows
its last query sees.
"""

import torch

NAME = "PyTorch"
BLOCK_VALUES = 1 << 24  # scores or gathered values per block of queries: 64 MiB in FP32


def _rows(batch, *sizes):
    """Queries per block, so one block's largest tensor holds about BLOCK_VALUES values."""
    values = batch
    for size in sizes:
        values *= max(1, size)
    return max(1, BLOCK_VALUES // values)


def _blocks(positions, rows, N):
    """(the block's slice, the rows of the table its queries see) for each
    block of `rows` queries, of a table of N rows (a padding query of a
    generation cache may sit past them)."""
    ends = positions.amax(0)  # (T,): the last row any query of that column sees
    for start in range(0, positions.shape[1], rows):
        part = slice(start, start + rows)
        yield part, min(int(ends[part].max()) + 1, N)


def _scores(q, k, positions, scale, seen):
    """z (B, R, H, seen): the scaled dot products, -inf past each query."""
    z = torch.einsum("brhc,bnc->brhn", q, k[:, :seen]) * scale
    later = torch.arange(seen, device=q.device) > positions[..., None]  # (B, R, seen)
    return z.masked_fill(later[:, :, None], float("-inf"))


def chosen_columns(top_k, window, N):
    """The columns of the chosen rows, for a table of N rows: top_k, or the
    rows at least window tokens before the table's last when fewer. It
    reads no tensor, so no kernel waits on the GPU for it."""
    return max(0, min(top_k, N - window))


class _Exact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, positions, scale, top_k, window):
        B, T, H, c = q.shape
        given, dtype = (q, k, v), q.dtype
        q, k, v = q.float(), k.float(), v.float()
        out, lse = q.new_empty(B, T, H, c), q.new_empty(B, T, H)
        K = chosen_columns(top_k, window, k.shape[1])
        chosen = torch.full((B, T, K), -1, dtype=torch.long, device=q.device)
        for part, seen in _blocks(positions, _rows(B, H, k.shape[1]), k.shape[1]):
            z = _scores(q[:, part], k, positions[:, part], scale, seen)
            lse[:, part] = torch.logsumexp(z, -1)
            log_p = z - lse[:, part, :, None]
            out[:, part] = torch.einsum("brhn,bnc->brhc", log_p.exp(), v[:, :seen])
            if K:
                score = log_p.amax(2)  # (B, R, seen): each row's largest log-weight over the heads
                near = torch.arange(seen, device=q.device) > positions[:, part, None] - window
                score = score.masked_fill(near, float("-inf"))
                best, index = score.topk(min(K, seen), -1)
                chosen[:, part, :index.shape[-1]] = index.masked_fill(best == float("-inf"), -1)
        ctx.save_for_backward(*given, positions, out, lse)  # the inputs as given: no FP32 copy is kept
        ctx.scale = scale
        ctx.mark_non_differentiable(chosen)
        return out.to(dtype), chosen

    @staticmethod
    def backward(ctx, d_out, _d_chosen):
        *given, positions, out, lse = ctx.saved_tensors
        q, k, v = (x.float() for x in given)
        B, T, H, c = q.shape
        d_out = d_out.float()
        delta = (d_out * out).sum(-1)  # (B, T, H): sum_j p_j dp_j
        dq, dk, dv = torch.empty_like(q), torch.zeros_like(k), torch.zeros_like(v)
        for part, seen in _blocks(positions, _rows(B, H, k.shape[1]), k.shape[1]):
            z = _scores(q[:, part], k, positions[:, part], ctx.scale, seen)
            p = (z - lse[:, part, :, None]).exp()
            dp = torch.einsum("brhc,bnc->brhn", d_out[:, part], v[:, :seen])
            dz = p * (dp - delta[:, part, :, None]) * ctx.scale
            dq[:, part] = torch.einsum("brhn,bnc->brhc", dz, k[:, :seen])
            dk[:, :seen] += torch.einsum("brhn,brhc->bnc", dz, q[:, part])
            dv[:, :seen] += torch.einsum("brhn,brhc->bnc", p, d_out[:, part])
        return *(grad.to(x.dtype) for grad, x in zip((dq, dk, dv), given)), None, None, None, None


def exact_attention(q, k, v, positions, scale, top_k, window):
    """(output (B, T, H, c), the chosen rows (B, T, K)): K =
    chosen_columns(top_k, window, N), -1 where a query has fewer rows
    `window` tokens back. The rows are in no particular order."""
    return _Exact.apply(q, k, v, positions, scale, top_k, window)


def _window(positions, window, N):
    """(B, T, window): each query's last window rows, -1 before the text
    and past the table's N rows."""
    rows = positions[..., None] - window + 1 + torch.arange(window, device=positions.device)
    return rows.masked_fill((rows < 0) | (rows >= N), -1)


def _gather(table, idx):
    """table (B, N, c), idx (B, R, K) -> (B, R, K, c). An index of -1 reads
    row 0, which the caller masks."""
    B, N, c = table.shape
    flat = idx.clamp(min=0) + torch.arange(B, device=idx.device)[:, None, None] * N
    return table.reshape(B * N, c)[flat]


def _scatter_add(table_grad, idx, values):
    """The reverse of _gather: adds values (B, R, K, c) into table_grad (B, N, c)."""
    B, N, c = table_grad.shape
    flat = idx.clamp(min=0) + torch.arange(B, device=idx.device)[:, None, None] * N
    table_grad.view(B * N, c).index_add_(0, flat.reshape(-1), values.reshape(-1, c))


class _Sparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sink, idx, scale):
        B, T, H, c = q.shape
        given, dtype = (q, k, v, sink), q.dtype
        q, k, v, sink = q.float(), k.float(), v.float(), sink.float()
        out, lse = q.new_empty(B, T, H, c), q.new_empty(B, T, H)
        rows = _rows(B, idx.shape[-1], max(c, H))
        for start in range(0, T, rows):
            part = slice(start, start + rows)
            z = torch.einsum("brhc,brkc->brhk", q[:, part], _gather(k, idx[:, part])) * scale
            z = z.masked_fill(idx[:, part, None, :] < 0, float("-inf"))
            lse[:, part] = torch.logaddexp(torch.logsumexp(z, -1), sink)  # the sink joins the denominator
            out[:, part] = torch.einsum("brhk,brkc->brhc", (z - lse[:, part, :, None]).exp(), _gather(v, idx[:, part]))
        ctx.save_for_backward(*given, idx, out, lse)
        ctx.scale = scale
        return out.to(dtype)

    @staticmethod
    def backward(ctx, d_out):
        *given, idx, out, lse = ctx.saved_tensors
        q, k, v, sink = (x.float() for x in given)
        B, T, H, c = q.shape
        d_out = d_out.float()
        delta = (d_out * out).sum(-1)  # (B, T, H): sum_j p_j dp_j
        dq, dk, dv = torch.empty_like(q), torch.zeros_like(k), torch.zeros_like(v)
        rows = _rows(B, idx.shape[-1], max(c, H))
        for start in range(0, T, rows):
            part = slice(start, start + rows)
            keys, values = _gather(k, idx[:, part]), _gather(v, idx[:, part])
            z = torch.einsum("brhc,brkc->brhk", q[:, part], keys) * ctx.scale
            z = z.masked_fill(idx[:, part, None, :] < 0, float("-inf"))
            p = (z - lse[:, part, :, None]).exp()
            dp = torch.einsum("brhc,brkc->brhk", d_out[:, part], values)
            dz = p * (dp - delta[:, part, :, None]) * ctx.scale
            dq[:, part] = torch.einsum("brhk,brkc->brhc", dz, keys)
            _scatter_add(dk, idx[:, part], torch.einsum("brhk,brhc->brkc", dz, q[:, part]))
            _scatter_add(dv, idx[:, part], torch.einsum("brhk,brhc->brkc", p, d_out[:, part]))
        d_sink = -((sink - lse).exp() * delta).sum((0, 1))  # the sink carries no value
        return *(grad.to(x.dtype) for grad, x in zip((dq, dk, dv, d_sink), given)), None, None


def sparse_attention(q, k, v, sink, idx, positions, scale, window):
    """The output (B, T, H, c) over each query's rows idx (B, T, K) and its
    last window tokens."""
    rows = torch.cat([idx, _window(positions, window, k.shape[1])], -1)
    return _Sparse.apply(q, k, v, sink, rows, scale)
