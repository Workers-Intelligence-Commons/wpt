"""What a model folder's provenance.json says of its corpus, in any of its
three shapes (wpt_model.provenance)."""

from wpt_model.provenance import corpus_id, corpus_label, deliveries, on_snapshot

def test_a_new_provenance_names_its_snapshot_change_digest_depot_recipe_and_commit():
    new = {"data": {"depot": "https://depot.example.org", "snapshot": "2026-11", "change": 1234, "digest": "ab" * 32},
           "recipe_name": "small", "corpus": {"build_key": "cd" * 32, "rules": 8},
           "repository_commit": "0123456789abcdef0123456789abcdef01234567", "repository_dirty": True}
    assert on_snapshot(new) and corpus_id(new) == "cd" * 32  # a folder that names no corpus id: its build key
    assert corpus_label(new) == (f"snapshot 2026-11 (change 1234, digest {'ab' * 32}) of the depot "
                                 "https://depot.example.org, recipe small, code at commit "
                                 "0123456789abcdef0123456789abcdef01234567 with uncommitted changes")


def test_an_old_provenance_names_its_corpus_dataset_and_depot():
    old = {"recipe": {"data": {"depot": "fox", "dataset": "classics", "total_tokens": 10}},
           "corpus": {"id": "ef" * 32, "depot": "fox", "dataset": "classics"}, "repository_commit": None}
    assert not on_snapshot(old) and corpus_id(old) == "ef" * 32
    assert corpus_label(old) == f"corpus {'ef' * 32}, built from dataset classics of the depot fox"


def test_the_model_card_says_when_the_corpus_was_built_at_another_size_than_the_recipe_s():
    from wpt_model.notice import _corpus_lines
    data = {"depot": "https://depot.example.org", "snapshot": "2026-11", "change": 1234, "digest": "ab" * 32,
            "total_tokens": 1_000}
    same = {"data": data, "recipe_name": "small", "repository_commit": None,
            "corpus": {"id": "cd" * 32, "build_key": "ef" * 32, "rules": 8, "data": data}}
    assert _corpus_lines(same)[1] == f"- Corpus: `{'cd' * 32}` (a cache: the same snapshot, recipe and code build it again)"
    larger = {**same, "corpus": {**same["corpus"], "data": {**data, "total_tokens": 2_000}}}
    assert _corpus_lines(larger)[1] == (f"- Corpus: `{'cd' * 32}` (a cache: built at 2,000 tokens, not the recipe's "
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
    provenance = {"data": data, "deliveries": named, "recipe_name": "small", "repository_commit": None,
                  "corpus": {"id": "04" * 32, "delivery": "03" * 32, "rules": 8, "request": request}}
    assert on_snapshot(provenance) and corpus_id(provenance) == "04" * 32 and deliveries(provenance) == named
    assert deliveries({"corpus": {"id": "x"}}) == {}
    lines = _corpus_lines(provenance)
    assert lines[1] == (f"- Corpus: `{'04' * 32}`, delivery `{'03' * 32}` (a cache: the same request to the freight "
                        "makes it again)")
    assert lines[2] == f"- Vocabulary: delivery `{'01' * 32}` (made by wpt-vocabulary)"
