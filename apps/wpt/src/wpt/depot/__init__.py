"""wpt depot: the depot WPT_DEPOT names (environment or .env), its texts
and items, the decisions on them, its libraries, datasets and erasures.

    wpt depot new ADDRESS --name N --maintainer M --takedown-contact C
                                          an empty depot in a new folder
    wpt depot clone ADDRESS TO --name N --maintainer M --takedown-contact C
                                          a copy of the depot at ADDRESS in the new folder TO, following it,
                                          with a name of its own (it begins the ids of the rows it adds)
    wpt depot backup TO                   a new full copy of the depot, every table with its log and its
                                          kept older states, as of its settled change, in a new folder under
                                          TO (a folder or a bucket) named NAME-UTCTIME-changeN; the app never
                                          cuts, erases or removes a backup
    wpt depot restore BACKUP TO           a copy of the backup BACKUP, as it is, at TO, which must hold
                                          nothing; erasures carried out after the backup are not carried out
    wpt depot pull                        fetch the updates of the depot this one follows, admitted
                                          erasures included
    wpt depot sql "SELECT ..."            any read query over the depot's tables; one JSON row per line:
                                          texts, libraries, datasets, about and erasures are the current
                                          state (texts with their current content and the libraries holding
                                          them now), <name>_revisions every revision of each, contents every
                                          content version of each text, and memberships, pins and pulls as
                                          they are
    wpt depot add FILE [--kind KIND] --license L [...] [--notice FILE] [--tag T ...] [--title T]
                [--author A ...] [--address A] DECLARATION
                                          a plain text (FILE, UTF-8) or an item of KIND (FILE, its JSON),
                                          under review
    wpt depot license add ID NAME FILE    a license the depot holds: its official text read from FILE (UTF-8);
                                          ID may be a license with an exception ('Apache-2.0 WITH
                                          LLVM-exception') or licenses that apply together ('MIT AND
                                          CC-BY-4.0')
    wpt depot show ID                     one text or item as it is now, with its decisions, libraries, how
                                          many revisions it has and the revision each locked library
                                          holding it pinned, as JSON
    wpt depot decide admit|refuse WHERE DECLARATION [--reason TEXT]
                                          one decision on every text the SQL condition WHERE matches;
                                          admitting is refused when a text has no allowed license; the
                                          optional reason is public, and never starts with 'licenses
                                          changed from ' or 'content changed from version ', the depot's
                                          own records of a license or content change
    wpt depot relicense ID [...] --license L [...] DECLARATION
                                          change the licenses of the texts ID as a new revision, their ids
                                          kept: the depot records it as a decision under review with the
                                          reason 'licenses changed from OLD to NEW'; the decisions before it
                                          no longer count, so each text is under review again and leaves
                                          every open library; a locked library keeps the revision it pinned
    wpt depot content ID FILE DECLARATION
                                          change the content of the text ID to FILE (UTF-8: its plain text, or
                                          an item's JSON, which must match its kind) as a new content version
                                          and a new revision, its id kept: the depot records it as a decision
                                          under review with the reason 'content changed from version OLD to
                                          version NEW'; the decisions before it no longer count, so the text is
                                          under review again and leaves every open library, to be put back
                                          after its admission; a locked library keeps the content of the
                                          revision it pinned; a content that changes nothing writes nothing
    wpt depot credits ID [--title T] [--author A ...] [--tag T ...] [--address A] [--notice FILE]
                [--clear CREDIT ...] DECLARATION
                                          change the credits of the text ID as a new revision, its id kept:
                                          the depot records it as a decision under review whose reason
                                          names each credit's old and new value; the decisions before it
                                          keep counting, so the text keeps its status and its libraries, and
                                          a locked library keeps the revision it pinned; an option given
                                          replaces that credit (--tag and --author: the whole list), --clear
                                          empties it (title, authors, tags, address, notice)
    wpt depot decide admit|refuse --erasure ID DECLARATION [--reason TEXT]
                                          one decision on the erasure request ID; once it is admitted,
                                          the depot carries it out at once and marks stale every corpus
                                          of the store that holds a text it cut or removed
    wpt depot library new NAME            a new open library
    wpt depot library put LIBRARY WHERE   every admitted text WHERE matches joins the open library
    wpt depot library lock LIBRARY        lock the library for good: it pins the current revision of
                                          each text it holds and reads those for good
    wpt depot library rename LIBRARY NAME a new name for the library, open or locked
    wpt depot library fork LIBRARY NAME [--without WHERE]
                                          a new open library holding the same texts, less those the SQL
                                          condition WHERE matches
    wpt depot library report LIBRARY PROBLEM [WHERE] DECLARATION
                                          a problem with the library, locked or not: the texts of the
                                          library WHERE matches (none without WHERE) and the problem
                                          in words; no text and no membership changes
    wpt depot dataset new NAME --tokenizer RECIPE --library [ADDRESS:]ID [...] [--share TAG=SHARE ...]
                                          a dataset reading those libraries (ADDRESS: another depot's),
                                          each at its depot's change number now; shares in order
    wpt depot dataset edit NAME [--new-name N] [--library [ADDRESS:]ID ...] [--share TAG=SHARE ... | --no-shares]
                                          change the dataset as a new revision, its id and tokenizer kept:
                                          an option given replaces that part (--library: every library); a
                                          library it read already keeps its change number, a new one is read
                                          at its depot's change number now; a corpus built before keeps its
                                          own copy
    wpt depot dataset export NAME         the dataset as a remote, JSON on standard output
    wpt depot dataset import FILE         a dataset made from a remote
    wpt depot tokenizer RECIPE            train the vocabulary of a tokenizer recipe on its dataset's
                                          texts in its depot, put it in the store WPT_STORE names, and
                                          write the recipe's lock
    wpt depot build DATASET               the dataset's corpus, built into the store WPT_STORE names with
                                          the vocabulary its tokenizer recipe locked
    wpt depot erase ID [START END] [--field POINTER] --reason TEXT DECLARATION
                                          ask to erase characters [START, END) of a text, or of the string
                                          at the JSON Pointer POINTER in an item's content, or the whole
                                          text or item (no START, END or POINTER)
    wpt depot erasures                    every erasure request; one JSON row per line
    wpt depot about [--maintainer M] [--takedown-contact C]
                                          change who maintains the depot, or where to ask for a takedown;
                                          its name never changes, since it begins every id
    wpt depot web [--port N]              the same operations as pages, on http://127.0.0.1:N/

DECLARATION is --name LEGAL_NAME --email EMAIL --accept: who decides, and
the confirmation every decision needs (an add, an erasure request and a
reported problem are decisions 'under review'). The name and email are recorded on the decision
and are public. The requirements are docs/requirements.md in the WPT repository.

Every subcommand is a controller over wpt_depot, as every route of `wpt
depot web` is: the depot applies its rules, and a refused change writes
nothing. A refusal prints its kind and message and ends with exit code 2.
In a clone, every change to a row its upstream owns is refused
(BelongsToUpstream): only the upstream changes it, and pull brings it.
"""

import argparse
import json
import sys
from pathlib import Path

from wpt_depot import About, Declaration, Depot, NewText, backup, clone, restore
from wpt_depot import remote as remotes
from wpt_depot.errors import DepotError
from wpt_depot.schema import ADMITTED, REFUSED
from wpt_depot.settings import depot_address
from wpt_recipes import RecipeError
from wpt_store import StoreError
from wpt_utils.lines import ERROR, say

from .shared import (CONFIRMATION, NoSample, NoVocabulary, TokenizerLocked, bucket, build_corpus, decide_erasures,
                     env_file, jsonable, opened, pulled, train_vocabulary, viewed)

STATUSES = {"admit": ADMITTED, "refuse": REFUSED}


def cmd_new(args):
    depot = Depot.new(args.address, _about(args), bucket())
    say(f"a new depot in {depot.address}")
    return 0


def cmd_clone(args):
    depot = clone(args.address, args.to, _about(args), bucket())
    say(f"{depot.address}: a clone of {args.address}, at its change {depot.change()}")
    return 0


def cmd_backup(args):
    depot = viewed(depot_address(env_file=env_file()))
    found = backup(depot, args.to, bucket())
    say(f"{found.address}: a backup of {depot.address}, at its change {found.journal.settled()}")
    return 0


def cmd_restore(args):
    found = restore(args.backup, args.to, bucket())
    say(f"{found.address}: a copy of the backup {args.backup}, at its change {found.journal.settled()}")
    return 0


def cmd_pull(args):
    depot = opened()
    brought = pulled(depot, say)
    say(f"{depot.address}: pulled from {depot.about()['upstream']}, now at its change {depot.change()}"
        + (f"; carried out the erasures {', '.join(brought)}" if brought else ""))
    return 0


def cmd_sql(args):
    _rows(opened().sql(args.query).to_pylist())
    return 0


def cmd_add(args):
    content = Path(args.file).read_text(encoding="utf-8")
    notice = None if args.notice is None else Path(args.notice).read_text(encoding="utf-8")
    one = NewText(text=None if args.kind else content, kind=args.kind, content=content if args.kind else None,
                  licenses=tuple(args.license), notice=notice, title=args.title,
                  authors=tuple(args.author) if args.author else None, tags=tuple(args.tag), address=args.address)
    [added] = opened().add([one], _declaration(args))
    say(f"added {'an item of kind ' + args.kind if args.kind else 'a text'}: {added}, under review")
    return 0


def cmd_show(args):
    sys.stdout.write(json.dumps(jsonable(opened().show(args.id)), ensure_ascii=False, indent=2) + "\n")
    return 0


def cmd_license_add(args):
    opened().add_license(args.id, args.name, Path(args.file).read_text(encoding="utf-8"))
    say(f"license {args.id}: {args.name}")
    return 0


def cmd_decide(args):
    if args.erasure is not None:
        carried_out = decide_erasures(opened(), [args.erasure], STATUSES[args.status], _declaration(args),
                                      args.reason, say)[args.erasure]
        say(f"erasure {args.erasure}: {STATUSES[args.status]}" + (", carried out" if carried_out else ""))
        return 0
    decision, texts = opened().decide(STATUSES[args.status], args.where, _declaration(args), args.reason)
    say(f"decision {decision}: {STATUSES[args.status]}, on {len(texts):,} texts")
    return 0


def cmd_relicense(args):
    changed, skipped = opened().relicense(args.ids, args.license, _declaration(args))
    for decision, texts in changed:
        say(f"decision {decision}: licenses changed to {', '.join(dict.fromkeys(args.license))}, on "
            f"{len(texts):,} texts, now under review")
    if skipped:
        say(f"left out, naming these licenses already: {', '.join(skipped)}")
    return 0


def cmd_content(args):
    depot = opened()
    new = Path(args.file).read_text(encoding="utf-8")
    item = depot.show(args.id)["text"]["kind"] is not None
    changed = depot.change_content(args.id, None if item else new, new if item else None, _declaration(args))
    if changed is None:
        say(f"{args.id}: its content is this already, nothing written")
        return 0
    decision, version = changed
    say(f"decision {decision}: under review, content version {version}; the text left every open library, to be put "
        "back once it is admitted")
    return 0


def cmd_credits(args):
    changes = {name: None if name != "tags" else [] for name in args.clear}
    given = {"title": args.title, "authors": args.author or None, "tags": args.tag or None, "address": args.address,
             "notice": None if args.notice is None else Path(args.notice).read_text(encoding="utf-8")}
    changes.update({name: value for name, value in given.items() if value is not None})
    decision, reason = opened().edit_credits(args.id, changes, _declaration(args))
    say(f"decision {decision}: under review, {reason}")
    return 0


def cmd_about(args):
    depot = opened()
    depot.edit_about(args.maintainer, args.takedown_contact)
    about = depot.about()
    say(f"{about['name']}: maintained by {about['maintainer']}, takedown contact {about['takedown_contact']}")
    return 0


def cmd_library_new(args):
    say(f"library {opened().new_library(args.name)}: {args.name}, open")
    return 0


def cmd_library_rename(args):
    opened().rename_library(args.library, args.name)
    say(f"library {args.library}: {args.name}")
    return 0


def cmd_library_put(args):
    joined, left_out = opened().put(args.library, args.where)
    say(f"library {args.library}: {len(joined):,} texts joined it"
        + (f"; {len(left_out):,} matched and left out, not admitted" if left_out else ""))
    return 0


def cmd_library_lock(args):
    opened().lock(args.library)
    say(f"library {args.library}: locked")
    return 0


def cmd_library_fork(args):
    library = opened().fork(args.library, args.name, args.without)
    say(f"library {library}: {args.name}, a fork of library {args.library}"
        + (f", less the texts {args.without} matches" if args.without else ""))
    return 0


def cmd_library_report(args):
    decision, texts = opened().report(args.library, args.problem, args.where, _declaration(args))
    say(f"library {args.library}: problem reported by decision {decision}"
        + (f", on {len(texts):,} texts" if args.where else ""))
    return 0


def cmd_dataset_new(args):
    shares = [(tag, share) for tag, share in args.share] if args.share else None
    say(f"dataset {opened().new_dataset(args.name, args.tokenizer, args.library, shares)}: {args.name}")
    return 0


def cmd_dataset_edit(args):
    depot = opened()
    row = depot.dataset(args.name)
    libraries = args.library or [(one["depot"], one["library"]) for one in row["libraries"]]
    shares = (None if args.no_shares else [(tag, share) for tag, share in args.share] if args.share
              else None if row["shares"] is None else [(one["tag"], one["share"]) for one in row["shares"]])
    name = args.new_name or row["name"]
    depot.edit_dataset(row["id"], name, libraries, shares)
    say(f"dataset {row['id']}: {name}, changed")
    return 0


def cmd_dataset_export(args):
    sys.stdout.write(remotes.to_json(opened().export_dataset(args.name)))
    return 0


def cmd_dataset_import(args):
    found = remotes.from_json(Path(args.file).read_text(encoding="utf-8"))
    say(f"dataset {opened().import_dataset(found)}: {found['name']}")
    return 0


def cmd_build(args):
    record = build_corpus(opened(), args.dataset, say)
    say(f"corpus {record['id']}")
    return 0


def cmd_tokenizer(args):
    lock = train_vocabulary(args.recipe, say)
    say(f"vocabulary {lock['vocabulary_sha256']}, trained on {len(lock['texts']):,} texts")
    return 0


def cmd_erase(args):
    request = opened().ask_erasure(args.reason, [(args.id, args.start, args.end, args.field)], _declaration(args))
    say(f"erasure {request}: asked, under review")
    return 0


def cmd_erasures(args):
    _rows(opened().erasures().to_pylist())
    return 0


def cmd_web(args):
    from .web import serve
    return serve(args.port)


def _rows(found: list[dict]):
    for row in found:
        sys.stdout.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def _about(args) -> About:
    return About(args.name, args.maintainer, args.takedown_contact)


def _declaration(args) -> Declaration:
    return Declaration(args.name, args.email, args.accept)


def _declares(parser):
    parser.add_argument("--name", required=True, help="your legal name, recorded on the decision and public")
    parser.add_argument("--email", required=True, help="your contact email, recorded on the decision and public")
    parser.add_argument("--accept", action="store_true", help=CONFIRMATION)


def _library(value: str) -> tuple[str | None, str]:
    """[ADDRESS:]ID: (the depot's address, None for this one, and the library's id)."""
    address, _, library = value.rpartition(":")
    if not library:
        raise argparse.ArgumentTypeError(f"{value!r}: a library is [ADDRESS:]ID")
    return address or None, library


def _share(value: str) -> tuple[str, float]:
    tag, _, share = value.rpartition("=")
    try:
        return tag, float(share)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: a share is TAG=SHARE, SHARE a number (0.3 = 30 %)") from None


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(prog="wpt depot", description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = found.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="an empty depot in a new folder")
    new.add_argument("address", help="the new depot's folder, a path or a file:// address")
    new.add_argument("--name", required=True, help="the depot's name")
    new.add_argument("--maintainer", required=True, help="who maintains it")
    new.add_argument("--takedown-contact", required=True, help="where to ask for a takedown")
    new.set_defaults(run=cmd_new)
    cloning = commands.add_parser("clone", help="a copy of a depot, following it")
    cloning.add_argument("address", help="the depot to copy")
    cloning.add_argument("to", help="the new depot's folder")
    cloning.add_argument("--name", required=True, help="the clone's own name, not the upstream's")
    cloning.add_argument("--maintainer", required=True, help="who maintains the clone")
    cloning.add_argument("--takedown-contact", required=True, help="where to ask for a takedown of the clone")
    cloning.set_defaults(run=cmd_clone)
    backing = commands.add_parser("backup", help="a new full copy of the depot, as of its settled change")
    backing.add_argument("to", help="where the new backup folder goes: a folder or a bucket")
    backing.set_defaults(run=cmd_backup)
    restoring = commands.add_parser("restore", help="a copy of a backup, as it is, at a new address")
    restoring.add_argument("backup", help="the backup's folder")
    restoring.add_argument("to", help="the new depot's folder, which must hold nothing")
    restoring.set_defaults(run=cmd_restore)
    commands.add_parser("pull", help="fetch the updates of the depot this one follows").set_defaults(run=cmd_pull)
    query = commands.add_parser("sql", help="any read query over the depot's tables")
    query.add_argument("query", help="one SELECT statement")
    query.set_defaults(run=cmd_sql)

    adding = commands.add_parser("add", help="a text or an item, under review")
    adding.add_argument("file", help="the plain text (UTF-8), or with --kind the item's JSON")
    adding.add_argument("--kind", help="the kind of the item (wpt depot sql \"SELECT id, meaning FROM kinds\")")
    adding.add_argument("--license", action="append", required=True,
                        help="a license the depot holds, by its id; several are a choice: one is enough")
    adding.add_argument("--notice", help="a file holding the copyright notice exactly as the source gives it, or "
                                         "why the work is public domain")
    adding.add_argument("--tag", action="append", default=[], help="what the text is: wikipedia, en, code")
    adding.add_argument("--title")
    adding.add_argument("--author", action="append", default=[])
    adding.add_argument("--address", help="where it came from")
    _declares(adding)
    adding.set_defaults(run=cmd_add)
    showing = commands.add_parser("show", help="one text or item, with its decisions, libraries and revisions")
    showing.add_argument("id")
    showing.set_defaults(run=cmd_show)
    deciding = commands.add_parser("decide", help="one decision on every text a condition matches")
    deciding.add_argument("status", choices=list(STATUSES))
    deciding.add_argument("where", nargs="?", help="an SQL condition over the texts table: list_contains(tags, "
                                                   "'en')")
    deciding.add_argument("--erasure", help="decide this erasure request instead of texts")
    deciding.add_argument("--reason", help="why, public (optional); never starting with 'licenses changed from ' or "
                                           "'content changed from version ', the forms the depot keeps for its "
                                           "records of a license or content change")
    _declares(deciding)
    deciding.set_defaults(run=cmd_decide)
    relicensing = commands.add_parser("relicense", help="change the licenses of texts as a new revision; their "
                                                        "decisions start over")
    relicensing.add_argument("ids", nargs="+", metavar="ID", help="a text or item of the depot")
    relicensing.add_argument("--license", action="append", required=True,
                             help="a license the depot holds, by its id; several are a choice: one is enough")
    _declares(relicensing)
    relicensing.set_defaults(run=cmd_relicense)

    contenting = commands.add_parser("content", help="change a text's content as a new content version; its "
                                                     "decisions start over")
    contenting.add_argument("id", help="the text or item")
    contenting.add_argument("file", help="the new plain text (UTF-8), or for an item its new JSON")
    _declares(contenting)
    contenting.set_defaults(run=cmd_content)

    crediting = commands.add_parser("credits", help="change a text's credits as a new revision, recorded as a "
                                                    "decision under review")
    crediting.add_argument("id", help="the text or item")
    crediting.add_argument("--title", help="its new title")
    crediting.add_argument("--author", action="append", help="its authors, in order: the whole new list")
    crediting.add_argument("--tag", action="append", help="its tags: the whole new list")
    crediting.add_argument("--address", help="where it came from")
    crediting.add_argument("--notice", help="a file holding its new copyright notice")
    crediting.add_argument("--clear", action="append", default=[], choices=["title", "authors", "tags", "address",
                                                                            "notice"],
                           help="empty that credit")
    _declares(crediting)
    crediting.set_defaults(run=cmd_credits)

    licensing = commands.add_parser("license", help="the licenses the depot holds")
    licenses = licensing.add_subparsers(dest="action", required=True)
    adding_license = licenses.add_parser("add", help="a license with its official text")
    adding_license.add_argument("id", help="its SPDX name: CC-BY-SA-4.0, LicenseRef-public-domain")
    adding_license.add_argument("name", help="its full name")
    adding_license.add_argument("file", help="the official text (UTF-8)")
    adding_license.set_defaults(run=cmd_license_add)

    library = commands.add_parser("library", help="make, fill, lock and fork libraries")
    libraries = library.add_subparsers(dest="action", required=True)
    making = libraries.add_parser("new", help="a new open library")
    making.add_argument("name")
    making.set_defaults(run=cmd_library_new)
    renaming = libraries.add_parser("rename", help="a new name for a library, open or locked")
    renaming.add_argument("library")
    renaming.add_argument("name")
    renaming.set_defaults(run=cmd_library_rename)
    putting = libraries.add_parser("put", help="every admitted text a condition matches joins an open library")
    putting.add_argument("library")
    putting.add_argument("where", help="an SQL condition over the texts table")
    putting.set_defaults(run=cmd_library_put)
    locking = libraries.add_parser("lock", help="lock a library for good")
    locking.add_argument("library")
    locking.set_defaults(run=cmd_library_lock)
    forking = libraries.add_parser("fork", help="a new open library holding the same texts")
    forking.add_argument("library")
    forking.add_argument("name")
    forking.add_argument("--without", metavar="WHERE", help="an SQL condition over the texts table: the texts it "
                                                            "matches are left out of the fork")
    forking.set_defaults(run=cmd_library_fork)
    reporting = libraries.add_parser("report", help="a problem with a library, locked or not")
    reporting.add_argument("library")
    reporting.add_argument("problem", help="the problem in words")
    reporting.add_argument("where", nargs="?", help="an SQL condition over the texts table: the texts of the "
                                                    "library it matches are the ones concerned")
    _declares(reporting)
    reporting.set_defaults(run=cmd_library_report)

    dataset = commands.add_parser("dataset", help="make, export and import datasets")
    datasets = dataset.add_subparsers(dest="action", required=True)
    making = datasets.add_parser("new", help="a dataset reading libraries, at their depots' change number now")
    making.add_argument("name")
    making.add_argument("--tokenizer", required=True, help="the tokenizer recipe whose vocabulary its corpus uses")
    making.add_argument("--library", action="append", required=True, type=_library,
                        help="[ADDRESS:]ID: a library of this depot, or of the depot at ADDRESS")
    making.add_argument("--share", action="append", type=_share,
                        help="TAG=SHARE, in order: a plain text counts for the first share whose tag it has, and "
                             "one no share takes is left out; the build trims the shares to their proportions")
    making.set_defaults(run=cmd_dataset_new)
    editing = datasets.add_parser("edit", help="change a dataset's name, libraries or shares as a new revision")
    editing.add_argument("name", help="the dataset's name now")
    editing.add_argument("--new-name", help="its new name")
    editing.add_argument("--library", action="append", type=_library,
                         help="[ADDRESS:]ID, every library it reads: a library it read already keeps its change "
                              "number")
    sharing = editing.add_mutually_exclusive_group()
    sharing.add_argument("--share", action="append", type=_share, help="TAG=SHARE, in order: every share")
    sharing.add_argument("--no-shares", action="store_true", help="no shares: every text")
    editing.set_defaults(run=cmd_dataset_edit)
    exporting = datasets.add_parser("export", help="the dataset as a remote, JSON on standard output")
    exporting.add_argument("name")
    exporting.set_defaults(run=cmd_dataset_export)
    importing = datasets.add_parser("import", help="a dataset made from a remote")
    importing.add_argument("file")
    importing.set_defaults(run=cmd_dataset_import)
    training = commands.add_parser("tokenizer", help="train a tokenizer recipe's vocabulary and lock it")
    training.add_argument("recipe", help="the name of a tokenizer recipe (data/recipes/tokenizers)")
    training.set_defaults(run=cmd_tokenizer)
    building = commands.add_parser("build", help="a dataset's corpus, built into the store")
    building.add_argument("dataset")
    building.set_defaults(run=cmd_build)

    erasing = commands.add_parser("erase", help="ask to erase a passage of a text, or a whole text or item")
    erasing.add_argument("id", help="the text or item")
    erasing.add_argument("start", type=int, nargs="?", help="the first character to erase, from 0")
    erasing.add_argument("end", type=int, nargs="?", help="the character after the last one to erase")
    erasing.add_argument("--field", help="for an item: the JSON Pointer (RFC 6901) of the string of its content "
                                         "that START and END count in, such as /messages/0/content/0/text")
    erasing.add_argument("--reason", required=True)
    _declares(erasing)
    erasing.set_defaults(run=cmd_erase)
    commands.add_parser("erasures", help="every erasure request").set_defaults(run=cmd_erasures)
    abouting = commands.add_parser("about", help="change who maintains the depot or its takedown contact")
    abouting.add_argument("--maintainer", help="who maintains it")
    abouting.add_argument("--takedown-contact", help="where to ask for a takedown")
    abouting.set_defaults(run=cmd_about)
    serving = commands.add_parser("web", help="the same operations as pages, on 127.0.0.1")
    serving.add_argument("--port", type=int, default=8410)
    serving.set_defaults(run=cmd_web)
    return found


def main(argv=None) -> int:
    found = parser()
    args = found.parse_args(argv)
    if args.run is cmd_decide and (args.where is None) == (args.erasure is None):
        found.error("decide takes a condition WHERE for texts, or --erasure ID for an erasure request, not both")
    if args.run is cmd_erase and (args.start is None) != (args.end is None):
        found.error("erase takes START and END together, or neither for the whole text")
    if args.run is cmd_erase and args.field is not None and args.start is None:
        found.error("erase --field takes START and END")
    if args.run is cmd_credits:
        given = {"title": args.title, "authors": args.author, "tags": args.tag, "address": args.address,
                 "notice": args.notice}
        both = [name for name in args.clear if given[name] is not None]
        if both:
            found.error(f"credits takes --clear {both[0]} or a new value for it, not both")
    try:
        return args.run(args)
    except (DepotError, RecipeError, StoreError, NoVocabulary, TokenizerLocked, NoSample, OSError, UnicodeDecodeError) as error:
        say(f"{type(error).__name__}: {error}", ERROR)
        return 2


if __name__ == "__main__":
    sys.exit(main())
