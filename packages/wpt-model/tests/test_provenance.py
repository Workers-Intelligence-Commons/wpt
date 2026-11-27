"""What a model folder's provenance.json says of its corpus, in any of its
three shapes (wpt_model.provenance)."""

from wpt_model.provenance import corpus_id, corpus_label, deliveries, on_snapshot, order, order_name

def test_a_new_provenance_names_its_snapshot_change_digest_depot_order_and_commit():
    new = {"data": {"depot": "https://depot.example.org", "snapshot": "2026-11", "change": 1234, "digest": "ab" * 32},
           "order_name": "small", "corpus": {"build_key": "cd" * 32, "rules": 8},
           "repository_commit": "0123456789abcdef0123456789abcdef01234567", "repository_dirty": True}
    assert on_snapshot(new) and corpus_id(new) == "cd" * 32  # a folder that names no corpus id: its build key
    label = (f"snapshot 2026-11 (change 1234, digest {'ab' * 32}) of the depot https://depot.example.org, order "
             "small, code at commit 0123456789abcdef0123456789abcdef01234567 with uncommitted changes")
    assert corpus_label(new) == label
    # a folder written before the key order_name names it recipe_name, and reads the same
    older = {"recipe_name" if key == "order_name" else key: value for key, value in new.items()}
    assert corpus_label(older) == label and order_name(older) == order_name(new) == "small"


def test_a_provenance_that_names_its_family_and_size_says_them():
    sized = {"data": {"depot": "https://depot.example.org", "snapshot": "2026-11", "change": 1234, "digest": "ab" * 32},
             "order_name": "pocket:64", "family": "pocket", "size": "64", "repository_commit": None}
    assert corpus_label(sized) == (f"snapshot 2026-11 (change 1234, digest {'ab' * 32}) of the depot "
                                   "https://depot.example.org, family pocket, size 64, code outside a git checkout")
    branch = {**sized, "order_name": "pocket:64:branch-1", "run": "branch-1"}
    assert ", family pocket, size 64, run branch-1, " in corpus_label(branch)


def test_the_order_and_its_name_read_under_either_key():
    trained = {"data": {"depot": "fox"}, "steps": []}
    assert order({"order": trained}) == order({"recipe": trained}) == trained and order({}) == {}
    assert order_name({"command": "wpt train", "order_name": "chat"}) == "chat"
    assert order_name({"command": "wpt train", "recipe_name": "chat"}) == "chat"  # a drafter's older trained_by


def test_an_old_provenance_names_its_corpus_dataset_and_depot():
    old = {"recipe": {"data": {"depot": "fox", "dataset": "classics", "total_tokens": 10}},
           "corpus": {"id": "ef" * 32, "depot": "fox", "dataset": "classics"}, "repository_commit": None}
    assert not on_snapshot(old) and corpus_id(old) == "ef" * 32
    assert corpus_label(old) == f"corpus {'ef' * 32}, built from dataset classics of the depot fox"


def test_the_model_card_says_when_the_corpus_was_built_at_another_size_than_the_order_s():
    from wpt_model.notice import _corpus_lines
    data = {"depot": "https://depot.example.org", "snapshot": "2026-11", "change": 1234, "digest": "ab" * 32,
            "total_tokens": 1_000}
    same = {"data": data, "order_name": "small", "repository_commit": None,
            "corpus": {"id": "cd" * 32, "build_key": "ef" * 32, "rules": 8, "data": data}}
    assert _corpus_lines(same)[1] == f"- Corpus: `{'cd' * 32}` (a cache: the same snapshot, order and code build it again)"
    larger = {**same, "corpus": {**same["corpus"], "data": {**data, "total_tokens": 2_000}}}
    assert _corpus_lines(larger)[1] == (f"- Corpus: `{'cd' * 32}` (a cache: built at 2,000 tokens, not the order's "
                                        "1,000: the same snapshot and code build it again from its own data section, "
                                        "`corpus.data` in provenance.json)")


def test_a_provenance_of_deliveries_names_both_and_the_model_card_says_so():
    from wpt_model.notice import _corpus_lines
    data = {"depot": "fox", "snapshot": "2026-11", "change": 1234, "digest": "ab" * 32, "total_tokens": 1_000,
            "vocabulary": {"size": 600, "sample_bytes": 5_000}}
    named = {"vocabulary": {"key": "01" * 32, "id": "02" * 32,
                            "preparation": {"name": "wpt-vocabulary", "version": "1", "settings": {}}},
             "corpus": {"key": "03" * 32, "id": "04" * 32,
                        "preparation": {"name": "wpt-tokens", "version": "1", "settings": {}}}}
    request = {"selection": {"size": {"unit": "tokens", "amount": 1_000}}}
    provenance = {"data": data, "deliveries": named, "order_name": "small", "repository_commit": None,
                  "corpus": {"id": "04" * 32, "delivery": "03" * 32, "rules": 8, "request": request}}
    assert on_snapshot(provenance) and corpus_id(provenance) == "04" * 32 and deliveries(provenance) == named
    assert deliveries({"corpus": {"id": "x"}}) == {}
    lines = _corpus_lines(provenance)
    assert lines[1] == (f"- Corpus: `{'04' * 32}`, delivery `{'03' * 32}` (a cache: the same request to the freight "
                        "makes it again)")
    assert lines[2] == f"- Vocabulary: delivery `{'01' * 32}` (made by wpt-vocabulary)"
