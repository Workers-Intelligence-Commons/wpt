"""depot: the depot DEPOT names (environment or .env), its texts
and items, the decisions on them, its snapshots and erasures.

    depot new FILE --name N --maintainer M --takedown-contact C
                                          an empty depot in a new SQLite file
    depot clone URL TO --name N --maintainer M --takedown-contact C
                                          a copy of the depot whose web app (depot web) is at the web
                                          address URL, in the new file TO, following it, with a name of its
                                          own (it begins the ids of the rows it adds); run again after it was
                                          cut short, it goes on where it stopped
    depot backup [--full]             send the change files waiting beside the depot's file onto the
                                          dock DOCK names, under backups/NAME/, each deleted here once
                                          stored; the first run, one after a clone, an upgrade or a restore,
                                          and --full also send a full copy of the file; the app never removes
                                          a backup, and a timer outside it says how often it runs
    depot backup --off                nothing backs the depot up: its writes record no change file and
                                          the waiting ones are deleted, until its next backup, which sends
                                          a full copy
    depot restore TO [--at MOMENT] [--depot NAME]
                                          a new depot file TO, which must not exist, made from the dock's
                                          backups of the depot NAME (the only one backed up, by default) as
                                          of MOMENT, a change number or a UTC time (the latest change sent,
                                          by default): the last full copy at or before it and every change
                                          after it, in order; refused on any change that does not apply
    depot pull                        fetch the updates of the depot this one follows, admitted
                                          erasures included, over the upstream's web app
    depot upgrade [--check | --yes] [--to FILE] [--upstream URL]
                                          what bringing a depot of an older format up to the format this wpt
                                          reads would do, read without writing its file: the steps, what each
                                          locked library and dataset revision becomes (a snapshot; the dataset
                                          revisions the corpora of DOCK were built from included), the
                                          check of every revision each snapshot reads against the libraries'
                                          reads, the names made snapshots' names, an order data section for
                                          each dataset revision, the cost (time, disk, WAL, the next backup's
                                          full copy) and what to do before; --yes runs that check again and,
                                          when it passes, upgrades: a folder runs each step up to format 3 in
                                          it, recorded as a change, then a new SQLite file (FILE, by default
                                          the folder's path with .sqlite added) written from it, which leaves
                                          the folder at format 3; a clone takes its upstream's web address
                                          (URL); every other subcommand, the web pages, clone and pull refuse
                                          a depot of another format; run it again after a crash, it goes on
    depot sql "SELECT ..."            any read query over the depot's tables, in SQLite's SQL; one JSON
                                          row per line: texts, about and erasures are the current state
                                          (texts with their current content), <name>_revisions every
                                          revision of each, contents every content version of each text,
                                          content_lists each version as the list of its parts, and parts,
                                          pulls and changes as they are; libraries_revisions, memberships,
                                          pins, library_moves and datasets_revisions hold what a depot of an
                                          older format held, which nothing writes any more; a list is JSON
                                          text: EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'en')
    depot add FILE [--kind KIND] --license L [...] [--notice FILE] [--tag T ...] [--title T]
                [--author A ...] [--address A] DECLARATION
                                          a text of KIND, fixed for good (default plain: FILE is its
                                          text, UTF-8; any other kind: FILE is its JSON), under review
    depot license add ID NAME FILE    a license the depot holds: its official text read from FILE (UTF-8);
                                          ID may be a license with an exception ('Apache-2.0 WITH
                                          LLVM-exception') or licenses that apply together ('MIT AND
                                          CC-BY-4.0')
    depot show ID                     one text or item as it is now (its scores included), with its
                                          decisions and how many revisions it has, as JSON
    depot decide admit|refuse|back-to-review WHERE DECLARATION [--reason TEXT]
                                          one decision on every text the SQL condition WHERE matches;
                                          admitting is refused when a text has no allowed license;
                                          back-to-review undoes an admission or a refusal: the depot
                                          records it as a decision under review with the reason 'put back
                                          under review' (then ': ' and the reason given), the decisions
                                          before it no longer count, and a text under review already is
                                          left out; the optional reason is public, and never starts with
                                          'licenses changed from ', 'content changed from version ' or 'put
                                          back under review', the depot's own records
    depot reason DECISION REASON DECLARATION
                                          give the admission or refusal DECISION its public reason, or
                                          another one in place of the one it has; only the person who made it
                                          (the same name and email) gives it, and the decision keeps its
                                          status, texts, name, email and time
    depot relicense ID [...] --license L [...] DECLARATION
                                          change the licenses of the texts ID as a new revision, their ids
                                          kept: the depot records it as a decision under review with the
                                          reason 'licenses changed from OLD to NEW'; the decisions before it
                                          no longer count, so each text is under review again; a snapshot
                                          keeps the revision it holds
    depot content ID FILE [--kind KIND] DECLARATION
                                          change the content of the text ID to FILE (UTF-8: its plain text, or
                                          an item's JSON, which must match its kind; a text's kind is fixed,
                                          so --kind, when given, must be its own) as a new content version
                                          and a new revision, its id kept: the depot records it as a decision
                                          under review with the reason 'content changed from version OLD to
                                          version NEW'; the decisions before it no longer count, so the text is
                                          under review again; a snapshot keeps the content of the revision it
                                          holds; a content that changes nothing writes nothing
    depot content-file FILE DECLARATION
                                          change the content of many plain texts at once, each as wpt
                                          depot content changes one: FILE (UTF-8) holds one JSON object per
                                          line, one line per text, {"id": ID, "text": T, "was": V}: T its
                                          new plain text, V (optional) the content version the line was
                                          written from, the text changing only while that is still its
                                          current one; at most 350,000 lines; the depot records one decision
                                          under review with the reason 'content changed from version OLD to
                                          version NEW' for each pair of versions, on the texts it moves; the
                                          decisions before it no longer count, so each text is under review
                                          again; a snapshot keeps the content of the revision it holds; a
                                          text whose content is that
                                          already, or whose was no longer holds, is left out; an item is
                                          refused; prints each decision and how many texts changed and were
                                          left out
    depot credits ID [--title T] [--author A ...] [--tag T ...] [--address A] [--notice FILE]
                [--clear CREDIT ...] DECLARATION
                                          change the credits of the text ID as a new revision, its id kept,
                                          tags alone as rows of taggings and no revision:
                                          the depot records it as a decision under review whose reason
                                          names each credit's old and new value; the decisions before it
                                          keep counting, so the text keeps its status, and a snapshot keeps
                                          the revision it holds and its tags as of it; an option given
                                          replaces that credit (--tag and --author: the whole list), --clear
                                          empties it (title, authors, tags, address, notice)
    depot scores FILE DECLARATION     change the scores of many texts at once, each as a new revision:
                                          FILE (UTF-8) holds one JSON object per line, one line per text,
                                          {"id": ID, "scores": [{"name": N, "value": V, "content_version":
                                          C}, ...], "remove": [N, ...]}: each score set (a name the text has
                                          already is replaced; V a finite number, C the content version of
                                          the text it was made from), each name removed; the depot records
                                          it as one decision under review on every text it changes, whose
                                          reason names the scores set and removed; the decisions before it
                                          keep counting, so the texts keep their status, and a snapshot
                                          keeps the revision it holds; a text whose scores
                                          stay as they are is left out; prints how many texts changed and
                                          how many were left out
    depot score NAME (--value V | --remove) WHERE DECLARATION
                                          set the score NAME to V, made from each text's current content
                                          version, on every text the SQL condition WHERE matches, or remove it
                                          from them, as depot scores records a change: one decision under
                                          review on every text it changes; a text whose scores stay as they
                                          are is left out
    depot measure WHERE --out FOLDER [--lines N]
                                          measure every plain text the SQL condition WHERE matches, from its
                                          text alone (depot.measures): symbols, the share of its
                                          characters that are neither letters, marks, digits nor white
                                          space, and running-text, the share that stands in ordinary lines
                                          (not a table row or a heading, image placeholders and link
                                          targets left out); writes them as scores files of at most N lines
                                          (1,000,000 by default) into the new folder FOLDER, each made from
                                          the text's current content version, for depot scores to
                                          record; reads the depot only, and leaves items out
    depot score-names                 every name of a score a text has now, with how many texts have it;
                                          one JSON row per line, from the depot's counts
    depot score-values NAME           the values the score NAME has across every current text, read at
                                          once from the depot's counts, each to three significant digits: how
                                          many have it, its smallest and largest, its 24 most frequent; one
                                          JSON row
    depot credits-file FILE DECLARATION
                                          change the credits of many texts at once, each as a new
                                          revision: FILE (UTF-8) holds one JSON object per line, one line
                                          per text, {"id": ID, "title": T, "authors": [A, ...], "tags": [T,
                                          ...], "address": A, "notice": N}, each credit optional: one given
                                          replaces the text's, null clears it, one left out stays; a line
                                          may add "was": {CREDIT: VALUE, ...}, the values its writer read,
                                          and changes the text only while they still hold; at most
                                          350,000 lines; the depot records it as one decision under
                                          review on every text it changes, whose reason names the credits
                                          changed; the decisions before it keep counting, so the texts keep
                                          their status, and a snapshot keeps the revision it holds; a text
                                          whose credits stay as they are, or
                                          whose was no longer holds, is left out; prints how many texts
                                          changed and how many were left out
    depot decide admit|refuse --erasure ID DECLARATION [--reason TEXT]
                                          one decision on the erasure request ID; once it is admitted,
                                          the depot carries it out at once; it marks nothing on the dock,
                                          `freight check` marks stale what the erasure reached there
    depot snapshot new [--note TEXT]  a snapshot of the texts admitted now, named by the UTC time it is
                                          made (YYYY-MM-DDTHH-MM-SSZ): the change number it reads, its texts,
                                          their digest and its summary
    depot snapshot list               every snapshot of the depot, one JSON row per line
    depot snapshot show NAME [--owner DEPOT]
                                          one snapshot (of this depot, or of DEPOT, the upstream a clone
                                          follows): its row, its summary and what an erasure did to it since
    depot jobs [cancel ID]            the web app's jobs, one JSON row per line; cancel: a queued job
                                          never runs, a running one is stopped
    depot erase ID [START END] [--field POINTER] --reason TEXT DECLARATION
                                          ask to erase characters [START, END) of a text, or of the string
                                          at the JSON Pointer POINTER in an item's content, or the whole
                                          text or item (no START, END or POINTER)
    depot erasures                    every erasure request; one JSON row per line
    depot recount                     make the depot's counts (how many texts have each status, tag
                                          and license and each decision is on) again
                                          from its rows, as one change, after copying again onto each
                                          revision the kind, length and bytes of its content version and
                                          onto each current one the copy of its text's tags; the
                                          writes keep them right, so it repairs only what a failure outside
                                          the app broke
    depot about [--maintainer M] [--takedown-contact C]
                                          change who maintains the depot, or where to ask for a takedown;
                                          its name never changes, since it begins every id
    depot web [--port N]              the same operations as pages, on http://127.0.0.1:N/, but
                                          new, clone, pull, upgrade, backup, restore, about's change, jobs
                                          cancel and recount; a snapshot runs as a job of its heavy lane, one
                                          at a time, and the writes over many texts as jobs of its write
                                          lane, beside them

DECLARATION is --name LEGAL_NAME --email EMAIL --accept: who decides, and
the confirmation every decision needs (an add and an erasure request are
decisions 'under review'). The name and email are recorded on the decision
and are public. The requirements are docs/requirements.md in the WPT repository.

Every subcommand is a controller over depot, as every route of `wpt
depot web` is: the depot applies its rules, and a refused change writes
nothing. A refusal prints its kind and message and ends with exit code 2.
In a clone, every change to a row its upstream owns is refused
(BelongsToUpstream): only the upstream changes it, and pull brings it.
"""

import argparse
import json
import sys
from pathlib import Path

from depot import About, Declaration, Depot, NewText, backup, clone, pull, restore
from depot.errors import DepotError
from depot.filters import literal
from depot.kinds import PLAIN
from depot.parts import FileText
from depot.schema import ADMITTED, FORMAT, REFUSED, UNDER_REVIEW
from depot.settings import depot_address
from depot import measures, snapshots, upgrade
from dock import api as dock_api
from dock.api import DockError, DockSettingError
from wic_utils.lines import ERROR, say

from . import corpora
from .jobs import Jobs
from .shared import CONFIRMATION, env_file, jsonable, opened, public_url, viewed

PASSAGE_CHARACTERS = 200_000  # the characters depot show --from prints unless told

STATUSES = {"admit": ADMITTED, "refuse": REFUSED, "back-to-review": UNDER_REVIEW}
BACK = "back-to-review"  # the status of decide that puts texts back under review, never an erasure request


class DockBackups:
    """Where `depot backup` and `depot restore` keep the depot's backups:
    the calls depot.backup makes of its `store`, each on the dock named by
    its URL, through dock.api (depot.backup knows nothing of the dock)."""

    def __init__(self, dock: str):
        self.dock = dock

    def put_from(self, key: str, path) -> dict:
        return dock_api.put_from(self.dock, key, path)

    def put_json(self, key: str, record: dict) -> dict:
        return dock_api.put_json(self.dock, key, record)

    def get_json(self, key: str) -> dict | None:
        return dock_api.get_json(self.dock, key)

    def folders(self, prefix: str) -> list[str]:
        return dock_api.folders(self.dock, prefix)

    def fetch_into(self, reference: dict, path) -> dict:
        return dock_api.fetch_into(self.dock, reference["key"], path, reference["sha256"])


def cmd_new(args):
    depot = Depot.new(args.address, _about(args))
    say(f"a new depot in {depot.address}")
    return 0


def cmd_clone(args):
    depot = clone(args.address, args.to, _about(args))
    say(f"{depot.address}: a clone of {args.address}, at its change {depot.change()}")
    return 0


def cmd_backup(args):
    depot = opened()
    if args.off:
        depot.db.backup_off()
        say(f"{depot.address}: its writes record no change file until its next backup, which sends a full copy; "
            "its waiting change files are deleted")
        return 0
    record = backup(depot, DockBackups(dock_api.url(env_file())), full=args.full)
    full = record["full"]
    say(f"{depot.address}: sent {len(record['changes'])} change files"
        + (f" and a full copy at its change {full['change']}" if full else "")
        + f" to the dock, under backups/{record['depot']}/")
    return 0


def cmd_restore(args):
    found = restore(DockBackups(dock_api.url(env_file())), args.to, args.at, args.depot)
    say(f"{found.address}: restored from the dock's backups of {found.name()}, at its change {found.change()}")
    return 0


def cmd_pull(args):
    depot = opened()
    brought = pull(depot)
    say(f"{depot.address}: pulled from {depot.about()['upstream']}, now at its change {depot.change()}"
        + (f"; carried out the erasures {', '.join(brought)}" if brought else ""))
    return 0


def cmd_upgrade(args):
    address = depot_address(env_file=env_file())
    try:
        records = corpora.records(dock_api.url(env_file()))
    except DockSettingError:
        records = None  # the check says only the current dataset revisions convert
    public = public_url()
    found = upgrade.check(address, records, public, log=say, save=not args.yes)
    for line in found.lines:
        say(line)
    if not found.needed:
        return 0
    if not found.passed:
        say("Nothing was written.", ERROR)
        return 2
    if not args.yes:
        say("Nothing was written: run `depot upgrade --yes` to upgrade, which runs this check again first.")
        return 0
    depot, start = Depot.upgrade(address, args.to, args.upstream, records=records, public=public, checked=found)
    say(f"{depot.address}: format {start} of {address} brought up to format {FORMAT}, at its change "
        f"{depot.change()}; set DEPOT to {depot.address}")
    return 0


def cmd_sql(args):
    columns, found = opened().answer(args.query)
    _rows(dict(zip(columns, one)) for one in found)
    return 0


def cmd_add(args):
    # a plain text is read as a stream, so a file of any size is added; an item's JSON is read whole
    plain = args.kind == PLAIN
    content = None if plain else Path(args.file).read_text(encoding="utf-8")
    notice = None if args.notice is None else Path(args.notice).read_text(encoding="utf-8")
    one = NewText(text=FileText(Path(args.file)) if plain else None, kind=args.kind, content=content,
                  licenses=tuple(args.license), notice=notice, title=args.title,
                  authors=tuple(args.author) if args.author else None, tags=tuple(args.tag), address=args.address)
    [added] = opened().add([one], _declaration(args))
    say(f"added {'a plain text' if plain else 'an item of kind ' + args.kind}: {added}, under review")
    return 0


def cmd_show(args):
    if args.start is not None or args.characters is not None:
        shown = opened().passage(args.id, args.start or 0, args.characters if args.characters is not None
                                 else PASSAGE_CHARACTERS, args.version)
    else:
        shown = opened().show(args.id)
    sys.stdout.write(json.dumps(jsonable(shown), ensure_ascii=False, indent=2) + "\n")
    return 0


def cmd_license_add(args):
    opened().add_license(args.id, args.name, Path(args.file).read_text(encoding="utf-8"))
    say(f"license {args.id}: {args.name}")
    return 0


def cmd_decide(args):
    if args.erasure is not None:
        carried_out = opened().decide_erasure(args.erasure, STATUSES[args.status], _declaration(args), args.reason)
        say(f"erasure {args.erasure}: {STATUSES[args.status]}" + (", carried out" if carried_out else ""))
        return 0
    decision, texts, left = opened().decide(STATUSES[args.status], args.where, _declaration(args), args.reason,
                                            progress=_stage)
    say(f"decision {decision}: {'put back under review' if args.status == BACK else STATUSES[args.status]}, on "
        f"{texts:,} texts" + "".join(f"; {count:,} left out, {reason}" for reason, count in left.items()))
    return 0


def cmd_reason(args):
    opened().give_reason(args.decision, args.reason, _declaration(args))
    say(f"decision {args.decision}: its reason is now {args.reason.strip()!r}")
    return 0


def cmd_relicense(args):
    changed, skipped = opened().relicense(args.ids, args.license, _declaration(args), progress=_stage)
    for decision, texts in changed:
        say(f"decision {decision}: licenses changed to {', '.join(dict.fromkeys(args.license))}, on "
            f"{texts:,} texts, now under review")
    if skipped:
        say(f"{skipped:,} texts left out, naming these licenses already")
    return 0


def cmd_content(args):
    depot = opened()
    item = [one["kind"] for one in depot.sql(f"SELECT kind FROM texts WHERE id = {literal(args.id)}")] not in (
        [PLAIN], [])
    new = Path(args.file).read_text(encoding="utf-8") if item else FileText(Path(args.file))  # a plain text as a stream
    changed = depot.change_content(args.id, None if item else new, new if item else None, _declaration(args),
                                   args.kind)
    if changed is None:
        say(f"{args.id}: its content is this already, nothing written")
        return 0
    decision, version = changed
    say(f"decision {decision}: under review, content version {version}")
    return 0


def cmd_content_file(args):
    if not Path(args.file).is_file():
        raise FileNotFoundError(f"no file {args.file}")
    done = opened().change_content_many(Path(args.file), _declaration(args))
    for decision, reason, texts in done.decisions:
        say(f"decision {decision}: under review, {reason}, on {texts:,} texts")
    say(f"{done.changed:,} texts changed, under review again"
        + (f"; {done.unchanged:,} left out, their content this already" if done.unchanged else "")
        + (f"; {done.stale:,} left out, their content version no longer the one their line's was names (the "
           f"first: {', '.join(done.stale_ids)})" if done.stale else ""))
    return 0


def cmd_credits(args):
    changes = {name: None if name != "tags" else [] for name in args.clear}
    given = {"title": args.title, "authors": args.author or None, "tags": args.tag or None, "address": args.address,
             "notice": None if args.notice is None else Path(args.notice).read_text(encoding="utf-8")}
    changes.update({name: value for name, value in given.items() if value is not None})
    decision, reason = opened().edit_credits(args.id, changes, _declaration(args))
    say(f"decision {decision}: under review, {reason}")
    return 0


def cmd_scores(args):
    if not Path(args.file).is_file():
        raise FileNotFoundError(f"no file {args.file}")
    decision, reason, changed, left_out = opened().edit_scores(Path(args.file), _declaration(args))
    say(f"decision {decision}: under review, {reason}, on {changed:,} texts"
        + (f"; {left_out:,} texts left out, their scores unchanged" if left_out else ""))
    return 0


def cmd_score(args):
    decision, changed, left_out = opened().score(args.where, args.score, None if args.remove else args.value,
                                                 _declaration(args), progress=_stage)
    done = f"removed {args.score}" if args.remove else f"set {args.score} to {args.value:g}"
    if decision is None:
        say(f"nothing written: the scores of the {left_out:,} texts are those already")
    else:
        say(f"decision {decision}: under review, {done}, on {changed:,} texts"
            + (f"; {left_out:,} texts left out, their scores unchanged" if left_out else ""))
    return 0


def cmd_retag(args):
    depot = opened()
    add, remove = args.add or [], args.remove or []
    found = depot.retag_estimate(args.where, add, remove)
    say(f"{found['changed']:,} of the {found['texts']:,} texts the condition matches would change: "
        f"{found['written']:,} rows written, about {found['bytes'] / 1e6:,.1f} MB added to the depot's file and "
        f"{found['change_bytes'] / 1e6:,.1f} MB to its next backup, about {found['seconds']:,.0f} seconds")
    if not args.yes:
        say("Nothing was written: run it again with --yes to change the tags.")
        return 0
    decision, changed, left_out = depot.retag(args.where, add, remove, _declaration(args), progress=_stage)
    if decision is None:
        say(f"nothing written: the {left_out:,} texts have those tags already")
    else:
        say(f"decision {decision}: under review, tags changed on {changed:,} texts"
            + (f"; {left_out:,} texts left out, their tags unchanged" if left_out else ""))
    return 0


def cmd_measure(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists already: the scores files go into a new folder")
    if args.lines < 1:
        raise ValueError("--lines is 1 at least")
    out.mkdir(parents=True)
    walked, files, written, held, said = measures.Walked(), 0, 0, None, [0]

    def progress(state):
        if state["texts"] // 100_000 > said[0]:
            said[0] = state["texts"] // 100_000
            say(f"{state['texts']:,} texts measured")
    try:
        for line in measures.score_lines(viewed(depot_address(env_file=env_file())), args.where, walked, progress):
            if held is None or written == args.lines:
                if held is not None:
                    held.close()
                files += 1
                held, written = open(out / f"scores-{files:05d}.jsonl", "w", encoding="utf-8"), 0
            held.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1
    finally:
        if held is not None:
            held.close()
    say(f"{walked.measured:,} texts measured into {files:,} files in {out}"
        + (f"; {walked.items:,} items left out" if walked.items else "")
        + ("" if files else ": no plain text matches"))
    return 0


def cmd_score_names(args):
    _rows(opened().score_names())
    return 0


def cmd_score_values(args):
    _rows([opened().score_values(args.name)])
    return 0


def cmd_credits_file(args):
    if not Path(args.file).is_file():
        raise FileNotFoundError(f"no file {args.file}")
    decision, reason, changed, left_out = opened().edit_credits_many(Path(args.file), _declaration(args))
    say(f"decision {decision}: under review, {reason}, on {changed:,} texts"
        + (f"; {left_out:,} texts left out, their credits unchanged" if left_out else ""))
    return 0


def cmd_about(args):
    depot = opened()
    depot.edit_about(args.maintainer, args.takedown_contact)
    about = depot.about()
    say(f"{about['name']}: maintained by {about['maintainer']}, takedown contact {about['takedown_contact']}")
    return 0


def _stage(stage: str):
    """What a write over many texts says as it starts each stage."""
    say(f"{stage}...")


def cmd_recount(args):
    opened().recount()
    say("the depot's counts are made again from its rows")
    return 0


def cmd_snapshot_new(args):
    row = opened().create_snapshot(args.note, progress=lambda stage: say(f"  {stage}"))
    say(f"snapshot {row['id']}: {row['name']}, {row['texts']:,} texts at change {row['at_change']}, digest "
        f"{row['digest']}")
    return 0


def cmd_snapshot_list(args):
    for row in opened().snapshots():
        print(json.dumps(jsonable(row), ensure_ascii=False))
    return 0


def cmd_snapshot_show(args):
    depot = opened()
    row = depot.snapshot(args.name, args.owner)
    print(json.dumps(jsonable({**row, "erased_since": snapshots.erased_since(depot, row["id"])}), ensure_ascii=False,
                     indent=1))
    return 0


def cmd_jobs(args):
    for job in Jobs(opened().address).listed():
        print(json.dumps({key: job[key] for key in ("id", "kind", "title", "state", "progress", "problem", "created",
                                                     "started", "ended")}, ensure_ascii=False))
    return 0


def cmd_jobs_cancel(args):
    job = Jobs(opened().address).cancel(args.id)
    say(f"job {job['id']} ({job['title']}): cancelled, it was {job['state']}")
    return 0


def cmd_erase(args):
    request = opened().ask_erasure(args.reason, [(args.id, args.start, args.end, args.field)], _declaration(args))
    say(f"erasure {request}: asked, under review")
    return 0


def cmd_erasures(args):
    _rows(opened().erasures())
    return 0


def cmd_web(args):
    from .web import serve
    return serve(args.port)


def _rows(found):
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


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(prog="depot", description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = found.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="an empty depot in a new SQLite file")
    new.add_argument("address", help="the new depot's file, a path or a file:// address")
    new.add_argument("--name", required=True, help="the depot's name")
    new.add_argument("--maintainer", required=True, help="who maintains it")
    new.add_argument("--takedown-contact", required=True, help="where to ask for a takedown")
    new.set_defaults(run=cmd_new)
    cloning = commands.add_parser("clone", help="a copy of a depot, following it")
    cloning.add_argument("address", help="the web address of the web app of the depot to copy (https://...)")
    cloning.add_argument("to", help="the new depot's file")
    cloning.add_argument("--name", required=True, help="the clone's own name, not the upstream's")
    cloning.add_argument("--maintainer", required=True, help="who maintains the clone")
    cloning.add_argument("--takedown-contact", required=True, help="where to ask for a takedown of the clone")
    cloning.set_defaults(run=cmd_clone)
    backing = commands.add_parser("backup", help="send the depot's waiting change files, and a full copy when one "
                                                 "is due, onto the dock")
    backing.add_argument("--full", action="store_true", help="send a new full copy of the file too")
    backing.add_argument("--off", action="store_true",
                         help="nothing backs this depot up: its writes record no change file and the waiting ones "
                              "are deleted, until the next backup, which sends a full copy")
    backing.set_defaults(run=cmd_backup)
    restoring = commands.add_parser("restore", help="a new depot file made from the dock's backups")
    restoring.add_argument("to", help="the new depot's file, which must not exist")
    restoring.add_argument("--at", metavar="MOMENT", help="a change number or a UTC time (ISO 8601); by default the "
                                                          "latest change sent")
    restoring.add_argument("--depot", metavar="NAME", help="the name of the depot whose backups to read; by default "
                                                           "the only one the dock holds backups of")
    restoring.set_defaults(run=cmd_restore)
    commands.add_parser("pull", help="fetch the updates of the depot this one follows").set_defaults(run=cmd_pull)
    upgrading = commands.add_parser("upgrade", help="what bringing a depot of an older format up to the format "
                                                    "this wpt reads would do and cost; with --yes, do it")
    upgrading.add_argument("--to", metavar="FILE", help="the new file; by default the folder's path with .sqlite added")
    upgrading.add_argument("--upstream", metavar="URL", help="for a clone: the web address of its upstream's web app")
    acting = upgrading.add_mutually_exclusive_group()
    acting.add_argument("--check", action="store_true", help="only say what the upgrade would do and cost, reading "
                                                             "the depot's file without writing it (the default)")
    acting.add_argument("--yes", action="store_true", help="run the check again, then upgrade when it passes")
    upgrading.set_defaults(run=cmd_upgrade)
    query =commands.add_parser("sql", help="any read query over the depot's tables")
    query.add_argument("query", help="one SELECT statement")
    query.set_defaults(run=cmd_sql)

    adding = commands.add_parser("add", help="a text or an item, under review")
    adding.add_argument("file", help="the plain text (UTF-8), or with --kind another than plain the item's JSON")
    adding.add_argument("--kind", default=PLAIN, help="the text's kind, fixed for good (default plain; depot sql "
                                                      "\"SELECT id, meaning FROM kinds\")")
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
    showing = commands.add_parser("show", help="one text or item, with its decisions and revisions")
    showing.add_argument("id")
    showing.add_argument("--from", dest="start", type=int, help="only a passage of a plain text, from this "
                                                                "character (from 0), reading only the parts it reaches")
    showing.add_argument("--characters", type=int, help=f"the passage's characters at most (default "
                                                        f"{PASSAGE_CHARACTERS:,})")
    showing.add_argument("--version", type=int, help="the passage's content version (default: the current one)")
    showing.set_defaults(run=cmd_show)
    deciding = commands.add_parser("decide", help="one decision on every text a condition matches")
    deciding.add_argument("status", choices=list(STATUSES))
    deciding.add_argument("where", nargs="?", help="an SQL condition over the texts table, in SQLite's SQL: "
                                                   "EXISTS (SELECT 1 FROM json_each(tags) WHERE value = 'en')")
    deciding.add_argument("--erasure", help="decide this erasure request instead of texts")
    deciding.add_argument("--reason", help="why, public (optional); never starting with 'licenses changed from ', "
                                           "'content changed from version ' or 'put back under review', the forms "
                                           "the depot keeps for its records of a license or content change and of "
                                           "a text put back under review")
    _declares(deciding)
    deciding.set_defaults(run=cmd_decide)
    reasoning = commands.add_parser("reason", help="give an admission or a refusal you made its public reason")
    reasoning.add_argument("decision", help="the decision's id")
    reasoning.add_argument("reason", help="why, public; never starting with the forms the depot keeps for its own "
                                          "records (see decide)")
    _declares(reasoning)
    reasoning.set_defaults(run=cmd_reason)
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
    contenting.add_argument("--kind", help="the kind the new content is of: refused unless it is the text's own, "
                                           "fixed when the text was made")
    _declares(contenting)
    contenting.set_defaults(run=cmd_content)

    contenting_file = commands.add_parser("content-file", help="change the content of many plain texts as new "
                                                               "content versions; their decisions start over")
    contenting_file.add_argument("file", help="one JSON object per line: {\"id\": ID, \"text\": T, \"was\": V}, T "
                                              "the new plain text, V (optional) the content version the line was "
                                              "written from; at most 350,000 lines")
    _declares(contenting_file)
    contenting_file.set_defaults(run=cmd_content_file)

    crediting = commands.add_parser("credits", help="change a text's credits as a new revision (tags alone: no "
                                                    "revision), recorded as a decision under review")
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

    crediting_file = commands.add_parser("credits-file", help="change the credits of many texts as new revisions, "
                                                              "recorded as one decision under review")
    crediting_file.add_argument("file", help="one JSON object per line: {\"id\": ID, \"title\": T, \"authors\": [A], "
                                             "\"tags\": [T], \"address\": A, \"notice\": N}, each credit optional, "
                                             "null clearing it, and optionally \"was\": {CREDIT: VALUE}, the "
                                             "values it was written from; at most 350,000 lines")
    _declares(crediting_file)
    crediting_file.set_defaults(run=cmd_credits_file)

    scoring = commands.add_parser("scores", help="change the scores of many texts as new revisions, recorded as one "
                                                 "decision under review")
    scoring.add_argument("file", help="one JSON object per line: {\"id\": ID, \"scores\": [{\"name\": N, "
                                      "\"value\": V, \"content_version\": C}], \"remove\": [N]}")
    _declares(scoring)
    scoring.set_defaults(run=cmd_scores)

    scoring_one = commands.add_parser("score", help="set one score on, or remove it from, every text a condition "
                                                    "matches, recorded as one decision under review")
    scoring_one.add_argument("score", metavar="NAME", help="the score's name")
    given = scoring_one.add_mutually_exclusive_group(required=True)
    given.add_argument("--value", type=float, help="its value, a finite number")
    given.add_argument("--remove", action="store_true", help="remove the score from the texts")
    scoring_one.add_argument("where", help="an SQL condition over the texts table, in SQLite's SQL")
    _declares(scoring_one)
    scoring_one.set_defaults(run=cmd_score)
    retagging = commands.add_parser("retag", help="add tags to, and take tags from, every text a condition "
                                                  "matches, recorded as one decision under review; says what it "
                                                  "would write, and writes it only with --yes")
    retagging.add_argument("where", help="an SQL condition over the texts table, in SQLite's SQL")
    retagging.add_argument("--add", action="append", metavar="TAG", help="a tag to give each text (repeatable)")
    retagging.add_argument("--remove", action="append", metavar="TAG", help="a tag to take from each text "
                                                                            "(repeatable)")
    retagging.add_argument("--yes", action="store_true", help="write the change; without it, only say its cost")
    _declares(retagging)
    retagging.set_defaults(run=cmd_retag)
    measuring = commands.add_parser("measure", help="measure how much of each plain text a condition matches is "
                                                    "writing, as scores files for depot scores")
    measuring.add_argument("where", help="an SQL condition over the texts table, in SQLite's SQL")
    measuring.add_argument("--out", required=True, help="the new folder the scores files go into")
    measuring.add_argument("--lines", type=int, default=1_000_000, help="the most lines of one file (1,000,000)")
    measuring.set_defaults(run=cmd_measure)
    naming_scores = commands.add_parser("score-names", help="every name of a score a text has, with how many texts "
                                                            "have it")
    naming_scores.set_defaults(run=cmd_score_names)
    valuing = commands.add_parser("score-values", help="the values a score has across every current text: how many "
                                                       "have it, its smallest and largest, its most frequent")
    valuing.add_argument("name", help="the score's name")
    valuing.set_defaults(run=cmd_score_values)

    licensing = commands.add_parser("license", help="the licenses the depot holds")
    licenses = licensing.add_subparsers(dest="action", required=True)
    adding_license = licenses.add_parser("add", help="a license with its official text")
    adding_license.add_argument("id", help="its SPDX name: CC-BY-SA-4.0, LicenseRef-public-domain")
    adding_license.add_argument("name", help="its full name")
    adding_license.add_argument("file", help="the official text (UTF-8)")
    adding_license.set_defaults(run=cmd_license_add)

    snapshotting = commands.add_parser("snapshot", help="snapshots: the texts admitted at a change number")
    snapshot_actions = snapshotting.add_subparsers(dest="action", required=True)
    making_snapshot = snapshot_actions.add_parser("new", help="a snapshot of the texts admitted now, named by the "
                                                             "UTC time it is made")
    making_snapshot.add_argument("--note")
    making_snapshot.set_defaults(run=cmd_snapshot_new)
    listing_snapshots = snapshot_actions.add_parser("list", help="every snapshot of the depot")
    listing_snapshots.set_defaults(run=cmd_snapshot_list)
    showing_snapshot = snapshot_actions.add_parser("show", help="one snapshot, with its summary")
    showing_snapshot.add_argument("name")
    showing_snapshot.add_argument("--owner", help="the depot whose snapshot it is (default: this one)")
    showing_snapshot.set_defaults(run=cmd_snapshot_show)
    job = commands.add_parser("jobs", help="the web app's jobs: list them, or cancel one")
    job.set_defaults(run=cmd_jobs)
    job_actions = job.add_subparsers(dest="action")
    cancelling = job_actions.add_parser("cancel", help="cancel a queued or running job")
    cancelling.add_argument("id")
    cancelling.set_defaults(run=cmd_jobs_cancel)

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
    commands.add_parser("recount", help="make the depot's counts again from its rows").set_defaults(run=cmd_recount)
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
    if args.run is cmd_decide and args.erasure is not None and args.status == BACK:
        found.error("decide back-to-review puts texts back under review; an erasure request is admitted or refused")
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
    except (DepotError, DockError, DockSettingError, OSError, UnicodeDecodeError, ValueError) as error:
        say(f"{type(error).__name__}: {error}", ERROR)
        return 2


def entry():
    sys.exit(main())


if __name__ == "__main__":
    entry()
