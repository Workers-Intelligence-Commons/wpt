"""WPT's preparations of a corpus: the programs the freight runs on the
items it chose (docs/api/preparation.md), and the token file format
training reads (shards).

    wpt-tokens   the token files by kind and tasks.jsonl of a corpus (programs.wpt_tokens)
    wpt-count    the tokens a corpus would hold, writing nothing (programs.wpt_count)

encoding turns one item into its tokens, as both programs and an estimate
count them; channel reads and writes the messages of the exchange.
"""
