# Output notice and attribution

The text in force is [OUTPUT-NOTICE](../OUTPUT-NOTICE), pending legal
review: it was written without a lawyer, and its wording may change after
review. This page explains it.

## Principle

- The license text (AGPL-3.0-or-later) stays **intact**. The notice lives in
  its own file, OUTPUT-NOTICE, which refers to the license.
- GPL and AGPL forbid adding further restrictions on recipients. Each clause
  therefore stays a **permission, a recommendation or a statement of fact**,
  never a condition.
- A mandatory "ethical use" clause (RAIL licenses) would make the model
  incompatible: there is none.

## What there is to say about outputs

- AGPL section 2: "The output from running a covered work is covered by this
  License only if the output, given its content, constitutes a covered
  work." A model generating new text does not automatically pass its license
  on to its outputs, but for a model the line with a copy of what it learned
  is blurry: a clear statement helps.
- One can only give rights one has. In many countries, generated text
  without real human creation is not protected by copyright, so there may be
  nothing to give. Whatever rights the authors of WPT might hold, they waive.
- Liability: the license already disclaims warranty and limits liability
  (sections 15 and 16).
- The gap no clause fills: a model can reproduce pieces of its training data.
  If the piece comes from a source that requires attribution, whoever
  publishes the output without knowing it may breach that obligation toward
  the author. A notice can move responsibility, not remove it.

## Clause 1: output license (CC0-1.0)

The authors of WPT waive their rights in outputs through CC0 1.0 Universal
rather than a plain "we claim no copyright" sentence:

- CC0 is a known, irrevocable tool, so readers and courts know what it
  grants.
- Where rights cannot be waived (in France, moral rights), CC0 falls back to
  a permission without conditions.
- The waiver is the authors' own statement about their rights in outputs,
  not an "additional permission" of the AGPL: AGPL section 7 lets anyone who
  passes the work on remove additional permissions, and a waiver of the
  authors' own rights cannot be removed that way.

It covers only rights the authors hold, so only what the model itself
generates. Material from anywhere else keeps its owner's rights and license
when it appears in an output: pieces of the training data the model
reproduces, and anything placed in its context (the prompt, documents, web
pages or files it reads, tool results). A model that reads a proprietary
document and repeats it does not make that document public domain. The
notice says so in its own paragraph, so readers do not have to infer it
from "rights the authors hold".

## Clause 2: attribution recommendation (non-binding)

- It says "to help" and not "to comply": the link does not guarantee
  compliance. Attribution obligations arise when a piece of a source is
  actually reproduced, and fall on whoever republishes it.
- Some licenses accept attribution through a link to a page holding the
  information (CC BY 4.0 does: its section 3(a)(2) accepts "a URI or
  hyperlink to a resource"); others may not. The notice
  never says that the mention makes anyone compliant.

## Clause 3: transparency (statement of fact)

This clause recommends nothing and requires nothing: it describes reality,
so it carries no license risk. It also serves people who have no intention
of republishing but want to know what is in the model.

## The address

Clauses 2 and 3 link to
https://github.com/Workers-Intelligence-Commons/wpt, the repository
under the project's organization. This link gets copied into outputs and
can never be corrected there: if the repository moves again or a project
domain is chosen, the notice changes for new outputs only. GitHub redirects a moved
repository, but the old address can later be taken by someone else.

## Where to show it

Almost nobody reads a license file: the information must appear where people
look.

- **The repository README**: links to OUTPUT-NOTICE.
- **The repository NOTICE**: says where to find the notice of a given model.
- **The model folder's CREDITS**: every text the model learned from, one
  line per text, with its credit and licenses, then the notices and
  license texts those lines name.
- **The model folder's NOTICE**: the licenses, the corpus the model learned
  from, the content of CREDITS, and the full output notice.
- **Model card** (the model folder's README.md): the licenses at the top,
  not the bottom.
- **Weight file metadata**: each safetensors header carries `license`,
  `output_license` and `notice` (the folder's NOTICE and the project's
  address), so the licenses travel with a weight file even alone. Other
  formats (GGUF key-value pairs) need the same when weights are converted.
- **The tool serving the model**: `wpt chat` names the licenses and the
  notice on its start line, and `wpt chat --about` prints the notice.

## One source of truth

Each text of a depot holds its credits (title, authors, address), its
licenses and its notice ([licenses.md](licenses.md), "Licenses and
notices"). A corpus copies them into its own tables when it is built
([data-selection.md](data-selection.md), "Corpora"), so it never depends on
the depot afterwards. Training writes the model folder's CREDITS from the
corpus's tables alone, one line per text the model's steps read
(`wpt_depot.notice`), and records the corpus in `provenance.json`; the
model folder's NOTICE, model card and weight metadata are generated from
the folder's own files, so they cannot contradict each other. They are
generated when a model is trained, not checked by CI, because the
repository holds no model. No public web page lists every text.

## Before publishing weights

- [ ] Have OUTPUT-NOTICE reviewed by a lawyer
- [ ] Choose a stable domain name for the attribution address, or keep the
      repository address
- [ ] Measure memorization on the model to publish (`wpt eval` holds no such
      measure, see [evaluation.md](evaluation.md))
- [x] Check section 2, sections 15 and 16, and license compatibility in the
      official AGPL text (section 7 also matters, see clause 1)
- [x] Decide where the clauses live: OUTPUT-NOTICE
- [x] Generate CREDITS, NOTICE, the model card block and the metadata from
      the corpus's tables
- [x] Show the notice in `wpt chat` (`--about`)
