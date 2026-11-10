## What the change does

<!-- What it does and why, in a few lines. Link the issue it answers. -->

## Part it touches

<!-- One or more: sources and data rules, tokenizer, dataset build, training,
evaluation, generation and chat, web interface, model architecture,
documentation. -->

## Checks

- [ ] `uv run ci/all.py` passed
- [ ] Every commit is signed off (`git commit -s`)
- [ ] The documentation is updated in the same change as the code
- [ ] A new package has its row in `docs/licenses.md`
- [ ] Nothing personal or secret is in the change, its commits or its logs

## The model

<!-- Only for a pull request that changes the model or its training. -->

Measurements (the run record's folder, the eval results, before and after):

Do earlier weights still load, and give the same output? If not, what must be
trained again:
