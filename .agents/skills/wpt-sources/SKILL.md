---
name: wpt-sources
description: How to gather new training data for WPT, from finding candidate works to the maintainer's admission. Use it when asked to find, add or review texts for a depot (books, collections, repositories, datasets), to check whether a work is in the public domain, or to admit texts. It covers where the rules live, when to ask the maintainer, adding texts to the depot, and the explicit yes that admission needs.
---

# Gathering texts

This skill is the order of work. It holds no rule about what a text may
contain: every rule lives in the documentation, and the documentation is
the only place to take one from.

## 0. Read the documentation first

Before each batch, read, in full:

- `docs/licenses.md`: "What the corpus accepts" (the license list and what
  to check for each), "Rules" (every rule), "Who decides";
- `CONTRIBUTING.md`, "Adding a text": what each text needs and the
  command;
- `docs/data-selection.md` (texts, items, tags) and `docs/chat-format.md`
  (conversations);
- the `wpt` skill, for the tools.

Prepare each text from these documents. A case they do not settle is a
question for the maintainer, never a rule you make up: no exclusion, no
filter, no default that the documentation does not state. When the
maintainer answers, write the answer into the document that covers it, in
the same change, as a general rule about the current state only: no
date, no history, and no source named (CONTRIBUTING.md, "Writing about
the project").

## 1. Candidates

The maintainer names what to look for (an author, a collection, a kind of
text, a list of repositories). Note for each candidate what the
documentation asks you to prove (for public domain: every person whose
words are in the text, the edition, the first publication).

## 2. Prove the license, and ask about every doubt

Prove the license from the source's own statements, as `docs/licenses.md`
says for that license. Cite where each fact comes from. When a fact cannot
be found, or the evidence disagrees, **do not settle it yourself in either
direction**: ask the maintainer about that candidate right away, with the
doubt, the evidence found and what would remove it. Without their answer
the candidate stays out: a doubt stays under review and is not used, and
`docs/licenses.md` says where a question the maintainer cannot settle
alone goes.

Candidates without doubt need no question yet; they wait for step 5.

## 3. Prepare the texts

Everything the documentation lets in comes in: leave a file out only for a
reason a document gives (a license that is not allowed, a doubt, a
maintainer's decision). Draft licenses, notices, credits and tags from
the source's own files with a script, never from memory, and show the
maintainer one full text's metadata as the pattern for the batch before
preparing the rest. The depot must hold each license a text names
(`wpt depot sql "SELECT id, name FROM licenses"`).

## 4. Add the texts under review

Adding a text records a person's declaration: a legal name, an email and
the confirmation (`--accept`). Ask the maintainer whose declaration goes
on the texts before adding any, and never give `--accept` for a person
who did not confirm it. Then add each text with `wpt depot add`, as
`CONTRIBUTING.md` says; it joins the depot under review. Report to the
maintainer what was added: the ids, and for each its licenses and tags.

## 5. Admission needs the maintainer's explicit yes

Never admit a text without it (`docs/licenses.md`, "Who decides"). Ask
once for the batch: the question lists every text or group of texts with
its one-line license reason, and the candidates the maintainer already
discussed in step 2 with what was decided. An answer that approves only
some admits only those.

An admission carries the maintainer's own declaration and confirmation,
so the maintainer makes it: give them the exact
`wpt depot decide admit WHERE` command, with the SQL condition that
matches exactly the approved texts, and check with `wpt depot sql` that
the condition matches those texts and no other.
