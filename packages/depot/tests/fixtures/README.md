# Fixture depots

This folder holds two small synthetic depots, one of format 11 and one of format 12, each with the dock of its
corpora. The upgrade tests and the legacy API's tests read them. Nothing in them is anyone's writing: every text is
made of words drawn from a fixed list with a fixed seed, and every decision is declared by "Ada Lovelace",
ada@example.org.

| File | What it is |
| --- | --- |
| `format-11.sqlite` | a depot of format 11, written by the depot of commit d0d454b22b, the last at format 11 |
| `format-12.sqlite` | a depot of format 12, written by the depot of commit 6b470328c7 |
| `format-11-dock/`, `format-12-dock/` | the dock of each, holding two corpora of the dataset ds-two |
| `make_fixtures.py` | the script that writes a depot and its dock; its docstring lists every step |
| `../test_fixtures.py` | checks, in plain SQL, that each depot still holds the cases below |

Each depot is called `fixture` and follows an upstream called `up`. Its about row names the upstream as
`http://127.0.0.1:8531`, the address the script served the upstream on, which no test reaches. Each depot holds:

- the texts of both depots: 70 of the upstream's, received by a clone and a pull, and 248 of its own (plain texts,
  conversations, tasks). In format 12, two conversations quote a plain text with an excerpt.
- the upstream's two locked libraries, `up-a` and `up-b`.
- its own libraries: `lib-a` and `lib-b`, locked at different changes; `lib-c`, open; `lib-d`, an open fork of
  `lib-a`. A text that `lib-a` pinned has a newer content version. `lib-a` and `lib-c` each have a reported
  problem.
- the dataset `ds-two`, which reads `lib-a` and `lib-b` with shares by tag, at revision 3. It publishes the
  vocabulary of the recipe `tok`.
- the dataset `ds-mixed`, which reads `lib-a` at a change above its lock, `lib-c`, and the upstream's `up-b`.
- the tokenizer recipe `tok`: version 1 trained, version 2 not. The recipe `tok-mixed` is not trained.
- two erasure requests: one admitted and carried out on a text `lib-a` pinned, and one under review.

Each dock holds one corpus built from revision 2 of `ds-two` at 2,000 tokens, and one built from revision 3 at
6,000 tokens.

## Making them again

Making them again gives new ids for the corpora and new times, so every test that reads them has to be checked
again, and so do the golden answers of `apps/depot/tests/legacy-golden/`.

1. Extract each commit into its own folder, outside the repository, without touching git state:
   `git archive d0d454b22b | tar -x -C EXTRACT-11`, and the same with 6b470328c7 into `EXTRACT-12`.
2. In each extract, make an environment of its own:
   `UV_PROJECT_ENVIRONMENT=VENV-N uv sync --frozen`.
3. From each extract, run the script into an empty folder OUT. Port 8531 must be free.
   `UV_PROJECT_ENVIRONMENT=VENV-N uv run --frozen --no-sync python -I PATH/make_fixtures.py OUT`
4. Copy `OUT/format-N.sqlite` and `OUT/format-N-dock/` here.
