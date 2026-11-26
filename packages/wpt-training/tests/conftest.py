import pytest
import torch

from dock import api
from wpt_corpus.files import DockFiles
from wpt_corpus.shards import write_shard


@pytest.fixture
def dataset(tmp_path):
    """A dock with 2 token files of 300 random tokens, as training reads its
    files (wpt_corpus.files); and the corpus's record."""
    dock = (tmp_path / "dock").as_uri()
    generator = torch.Generator().manual_seed(0)

    def shard(name, part):
        path = tmp_path / f"{name}.parquet"
        digest = write_shard(path, [(torch.randint(50, (300,), generator=generator).tolist(), "s", f"s/{name}",
                                     part, 0)])
        stored = api.put_from(dock, f"corpora/d/{digest}.parquet", path)
        return {"key": stored["key"], "sha256": stored["sha256"], "token_sha256": digest, "rows": 1, "tokens": 300}

    record = {"shards": {"plain": [shard("a", "p"), shard("b", "p")]}, "total_tokens": 600}
    return DockFiles(dock, tmp_path / "reader"), record
