# Contributing

Thank you for helping. WPT trains only on content whose license is clear.
The repository holds code, recipes and documentation; the texts live in
depots, and most of this page is about adding texts to a depot and
deciding on them.

## Every commit: sign off (DCO)

Every commit carries a `Signed-off-by` line with your name and email, added
by `git commit -s`. By it you certify the
[Developer Certificate of Origin](https://developercertificate.org/): you
have the right to submit the change under the project's license. CI refuses
commits without it.

## Issues and pull requests

Open an issue from one of the forms in
[.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE): **Bug**, **Enhancement**
(both cover the tooling and the model), **Model behavior**, **Source
proposal** and **Erasure request**. Each form asks for the part concerned.
The Bug and Enhancement forms also ask for the commit, the exact command,
the recipe, the corpus id and the weights' sha256, the run record's
folder, and your system, GPU and PyTorch build. The other forms ask for
what fits their kind. A blank issue stays possible. For an
erasure request that should not be public, or for a security problem, use
the repository's **Security** tab, **Report a vulnerability**, and never
post personal details in a public issue.

A pull request starts from
[.github/pull_request_template.md](.github/pull_request_template.md): what
the change does, the part it touches, and the checks the template lists.

### Labels

Every form adds the label of its kind: `bug`, `enhancement`,
`model behavior`, `source proposal` or `erasure request`. Every form also
asks for the part concerned, and the workflow
[.github/workflows/area-label.yml](.github/workflows/area-label.yml) adds
the matching area label when the issue is opened:

- `area: sources and data rules`
- `area: tokenizer`
- `area: dataset build`
- `area: training`
- `area: evaluation`
- `area: generation and chat`
- `area: web interface`
- `area: model architecture`
- `area: documentation`

The three labels of kind (`model behavior`, `source proposal` and
`erasure request`) must exist too, like the 9 area labels. A maintainer
creates every label before its first use. The words of the dropdown and of
the workflow are the area label names without `area: `: change them
together.

## Adding a text

A text is one piece of final training text, such as an article, a book or
a code file ([docs/data-selection.md](docs/data-selection.md), "Texts").
Each file of a repository or a collection is its own text.

1. Check the license against [docs/licenses.md](docs/licenses.md). It must
   be on the allowed list, clearly; its "What to check" column says where
   each kind of license hides a trap. CC BY-SA 4.0, the one grey-zone
   license, is allowed until legal review says otherwise. Public domain
   must hold under the law of the country where the work was first
   published.
2. Read the license of the text, look for a "no AI training" wish from the
   owner (a statement, or a machine-readable text-and-data-mining
   reservation), and check that the text holds no personal data
   ([docs/licenses.md](docs/licenses.md), "Rules").
3. The depot must hold each license the text names, by its SPDX id
   (`wpt depot sql "SELECT id, name FROM licenses"`). A text's licenses are
   a choice: one allowed license is enough.
4. Write the text's notice into a file: the license statement exactly as
   the source gives it, with its copyright line, followed for Apache-2.0 by
   the source's `NOTICE` content; for public domain, the proof and its
   sources ([docs/licenses.md](docs/licenses.md), "Licenses and notices").
   Every model trained on the text repeats it.
5. Add it:

   ```sh
   uv run wpt depot add FILE --license L [--license L2] --notice NOTICE_FILE \
       --tag TAG ... --title TITLE --author NAME ... --address ADDRESS \
       --name "LEGAL NAME" --email EMAIL --accept
   ```

   It joins the depot under review, with your declaration as its first
   decision ("Decisions", below). Take tags only from the list of
   [docs/data-selection.md](docs/data-selection.md), "Tags". The "Add"
   page of `uv run wpt depot web` does the same.

The depot refuses a text that names a license it does not hold, and
nothing is written.

## Adding your own texts and items

Text you write for WPT is added the same way, under a license you choose
(for example `CC0-1.0`), with yourself as its author. Conversations,
tasks, ratings and probes are items: their content is JSON that matches
their kind's schema, added with `--kind KIND`
([docs/data-selection.md](docs/data-selection.md), "Items"):

- a conversation, in the chat format, for the `answer-as-assistant` step
  ([docs/chat-format.md](docs/chat-format.md));
- a task, for the `finish-tasks` steps
  ([docs/training.md](docs/training.md), "Tasks");
- a rating, your pass or fail on one attempt of a `wpt eval` run, a
  measure only that no training step reads;
- a probe, a question to ask a model and the patterns that flag a reply
  as alarming, a measure only too.

Your declaration on every decision, your text's included, records your
legal name and email, and both are public. Add only text you wrote, that
holds no other person's personal data and no output of another company's
model.

No program admits or refuses a text. A script may help you prepare a text
and its metadata: its output is not a source of truth, a person checks it,
and only a person admits, after reviewing it. Tools such as ScanCode may
help you find license texts; they never decide.

## Decisions

Every decision is a row of the depot with one of three statuses,
`under review`, `admitted` or `refused`, and the decider's declaration: a
legal name, an email, and the confirmation (`--accept`, or a confirming
button shown beside the confirmation's words) that they read the requirements, accept
them, and that what they submit meets them. The declaration is public and
not verified ([docs/licenses.md](docs/licenses.md), "Who decides").

```sh
uv run wpt depot decide admit "list_contains(tags, 'gutenberg')" \
    --name "LEGAL NAME" --email EMAIL --accept
```

One decision applies to every text the SQL condition over the `texts`
table matches. Anyone's decisions are appended. A text is admitted when at
least one decision admits it and none refuses it, and only an admitted
text joins a library and trains. A refused text stays in the depot as a
reference. A decision never changes a text a locked library holds.

## Libraries and datasets

A library groups admitted texts, and a dataset says which libraries to
train on, with optional shares by tag. `wpt depot library new`, `put`,
`lock`, `fork` and `report` make and fill libraries, and report a
problem with one; `wpt depot dataset new`,
`export` and `import` make datasets and write them out as remotes;
`wpt depot build DATASET` builds a dataset's corpus into the store
([docs/data-selection.md](docs/data-selection.md)).

## Changing a text

A text never changes. A newer or corrected version is another text, added
under review; the older one stays where it is until someone refuses it or
asks for its erasure. A locked library keeps the texts it holds.

## Erasure requests

Every erasure request is honored. Ask through `wpt depot erase` or the
page of the text in `wpt depot web`, open a public issue, or use the
repository's **Security** tab, **Report a vulnerability**, to ask
privately. See [docs/erasure.md](docs/erasure.md).

## Models and generated text

No output of another company's model is accepted, in data or in
training, and no synthetic data that a WPT model did not make. WPT's
own models may be used in training, their output included
([docs/data-selection.md](docs/data-selection.md), "Models in training").
A contribution that uses a WPT model (to produce data, or in any training
role) must say exactly what that model did. A conversation's `assistant`
message that started as a WPT model's draft records it in its
`metadata.draft` ([docs/chat-format.md](docs/chat-format.md), "Model
drafts"), and the `finish-tasks` step's reward on a task's one-line form
compares an attempt with the model's own attempt at the full form
([docs/training.md](docs/training.md), "Rewards").

An AI assistant (a coding agent helping you) never reads or writes the
words of people's texts and items: it works on their metadata only
(licenses, notices' ids, tags, credits, decisions, libraries), except that
it may read a text to choose its tags, and then changes only the tags.
People write and add their texts themselves. When a person asks an
assistant only to change the form of a text they wrote (line endings,
spacing, splitting into files), the assistant may move it with scripts
that transform whole files mechanically and report only its shape, never
its words; it never reads or writes the words itself.

## Training records

Demonstrations and environment recordings are items of a depot, like any
other text ([docs/training.md](docs/training.md), "Where records live"):
their licenses, credits, notice and decisions are the item's. Their shape
is not settled yet; the code reads conversations and tasks, and the depot
also holds ratings and probes.

## Writing about the project

The repository describes only its current state: the documentation, the
skills (`.agents/skills/`), code comments and docstrings, and recipe
notes say what is true now and why, never when it was decided or what it
was before. A change may break compatibility with what came before; the
text then describes the new state alone, and the old one stays only in
git's history. No text holds a decision's date, a "(maintainer's
decision)" tag (only "maintainer's decision alone" stays, for a license
choice with no outside ruling) or an account of earlier versions. Rules
are general: they name no source and no example, so they hold for any
number of texts. Dates that belong to the content stay, such as an
author's death date or a lawyer's answer in `LEGAL-QUESTIONS.html`.

## Code

```sh
uv run scripts/setup.py      # once per machine: installs the right PyTorch build
uv run ci/all.py             # the checks your change touches, before you push
uv run ci/all.py --all       # every check, whatever changed
uv run ci/all.py --cpu       # with the software conditions of CI: every GPU hidden, Triton blocked
uv run ci/tests.py           # one part: lock, tests, rust, data or commits
```

The tests prove that things work, not that they are correct. `ci/all.py`
is what CI runs too: the tests, the Rust tests, the recipes' check, and
the checks on your commits. Only what your change touches runs:
pytest-testmon runs the tests whose source changed, and the other parts
run when their files changed since your upstream branch, committed or not.
The commits check always runs, on what a push would send. There is no
local hook. A green run here is no promise of a green CI: `--cpu` matches
CI's software, and only CI shows the timing of its 4 cores.
`docs/architecture.md`, "Checks", says how each part chooses what it runs,
and what each takes.

A test checks that the code works, not that it works well: it runs without crashing, its computation is valid
(finite numbers, values in their range, tokens come out, training changes
the weights it reaches), and two routes the code says are the same give the
same result within rounding. Whether a method is efficient or a good idea
is a question for an evaluation (`wpt eval`, docs/evaluation.md), never for
a test: a test that asks for a quality, such as a trained part beating a
simple one, fails on one processor and passes on another, by rounding and
chance, not by a bug.
