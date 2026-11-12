from dataclasses import asdict

import pytest
import torch

from wpt_store import LocalStore
from wpt_store.shards import write_shard


@pytest.fixture
def dataset(tmp_path):
    """A store with 2 token files of 300 random tokens; and the corpus's record."""
    store = LocalStore(tmp_path / "store")
    generator = torch.Generator().manual_seed(0)

    def shard(name, part):
        path = tmp_path / f"{name}.parquet"
        digest = write_shard(path, [(torch.randint(50, (300,), generator=generator).tolist(), "s", f"s/{name}",
                                     part, 0)])
        return {**asdict(store.put_named(f"corpora/d/{digest}.parquet", path)), "token_sha256": digest, "rows": 1,
                "tokens": 300}

    record = {"shards": [shard("a", "p"), shard("b", "p")], "total_tokens": 600}
    return store, record
