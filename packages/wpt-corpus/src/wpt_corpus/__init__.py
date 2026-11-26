"""WPT's preparations: the programs the freight runs on the items it
chose (docs/api/preparation.md), and the token file format training reads
(shards).

    wpt-vocabulary  a vocabulary learned from plain texts (vocabulary.wpt_vocabulary)
    wpt-tokens      the token files by kind and tasks.jsonl of a corpus (programs.wpt_tokens)
    wpt-count       the tokens a corpus would hold, writing nothing (programs.wpt_count)

encoding turns one item into its tokens, as both programs and an estimate
count them; channel reads and writes the messages of the exchange. roles
gives the portions of a corpus their roles (held_out, pretrain, finetune)
and their rules.
corpora reads the corpora the depot built before the freight, which the
dock keeps under corpora/.
"""
