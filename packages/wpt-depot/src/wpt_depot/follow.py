"""A depot that follows another, its upstream, as a Git clone follows its
remote: clone copies a depot, and pull fetches its upstream's updates,
admitted erasures included.

A clone holds every row of every table of the upstream but its journal and
its about rows: every revision with the change number it was made at,
every text's content, every membership and pin. Its about row is its own:
its own name, which begins the ids of the rows it adds (the upstream's rows
keep the upstream's ids, so the two never collide), its maintainer and
takedown contact, and the upstream's address, at a change number past
every one it received; its own changes number on from there. Its pulls
table records the clone as its first pull (wpt_depot.schema). It only
reads the upstream: whatever the upstream's journal holds undone, only its
own app finishes. A clone is written at the code's format
(wpt_depot.schema), and clone and pull read only an upstream of that
format: one of another format is refused (OtherFormat), so both upgrade.

A clone is read-only on the rows its upstream owns, those whose id does
not carry the clone's name (is not exactly that name, a hyphen and digits,
wpt_depot.ids): every operation that would change one is
refused (wpt_depot.depot), so only the upstream changes them and a pull
never meets a clone's revision of one. The clone's own rows stay fully
changeable, and its own libraries may hold upstream texts.

The parts of the texts' content versions (wpt_depot.parts) are read and
written a few at a time, never whole in memory. A part never changes once
stored (its key is its text's id and the hash of its bytes), so a pull
adds only the parts that the upstream's rows of contents the depot does
not hold as they are list and the depot lacks, before those rows, and, for
each text an erasure it brings cut, drops every part no content version of
the text lists any longer, as carrying the erasure out here would.

A pull puts every other row the upstream holds into the depot: a row of the same
key (an id and a revision or a content version, a license's or a kind's name, a membership's
library, text and joining change) is replaced by the upstream's, which
brings what the upstream changed in place (a content version an erasure cut, a
membership that ended), and a row only the depot holds stays. The texts an
erasure the upstream carried out removed are removed here too, with every
revision, membership and pin, the clone's own libraries' included, and
when the pull brings an erasure carried out that the depot had not, every
older state of the depot's storage is dropped, as carrying it out here
would.

A text the upstream refused, or started over (its licenses or its content
changed, wpt_depot.policy), since the depot
last received it leaves every open library of the depot's own that holds
it, as such a decision made here would (wpt_depot.depot): the pull ends
those memberships at its own change number, so its own libraries read at
an earlier change still hold the text. Its locked libraries keep their
pins. When the pull begins, its change records the depot's own open
libraries and the decisions the depot holds on each upstream text they
hold; applying it compares those against the texts and decisions rows it
actually read (at the versions it names, or the upstream's current state),
so applying it again, even after the upstream dropped those versions,
ends the memberships of every text those rows refuse or start over.

A pull is a change, so it goes through the journal, numbered past every
change number of the upstream it brings. The rows it brings keep the
upstream's change numbers, so it records in pulls its own change number
and the upstream's change number it brought the depot up to: the
upstream's settled change number (wpt_depot.journal), read before the
storage versions of its tables, so every upstream change numbered up to it
was written whole, and the rows of a change the upstream was writing, or
holds undone after a crash, stay past it until a later pull records a
higher one. A clone records its first number the same way. Its change
carries that number, so applying it again records the same (when an
erasure upstream dropped the versions it names, it reads the upstream
again as a pull begins and records the higher number). A read of the
depot's own library at a past change sees the upstream's rows only up to
what the last pull at or before that change brought (Depot.bounds). Its change names the
upstream, the storage version of each of its tables it read, and the
storage version of the depot's own erasures when it began; applied
again, it reads those versions, or the upstream's current state when an
erasure there dropped them. Whether it brings an erasure is decided from
the erasure rows it actually read, against the depot's erasures when it
began, so an erasure the upstream carries out while the pull reads, or
before a pull cut short is applied again, still drops the older states.
"""

import pyarrow as pa
import pyarrow.compute as pc

from . import ids, policy
from .depot import About, Depot, location
from .errors import DepotExists, NoUpstream, Refused, VersionGone
from .filters import literal
from .journal import Journal
from .schema import FORMAT, REFUSED, TABLES, rows
from .tables import Tables

# parts first, so a content version a pull brings finds every part it lists
COPIED = ("parts", *(name for name in TABLES if name not in ("about", "journal", "parts")))


def clone(address: str, to: str, about: About, settings: dict | None = None) -> Depot:
    """A new depot at `to` holding what the depot at address holds, following
    it, with about as its own name, maintainer and takedown contact (its
    upstream is address). DepotExists when `to` holds tables; Refused when
    the name is the upstream's or cannot begin an id. Its about row is
    written last, so a clone cut short is no depot."""
    ids.check_name(about.name)
    upstream = Depot.view(address, settings)
    if about.name == upstream.name():
        raise Refused(f"a clone has a name of its own: {about.name!r} is the upstream's, and ids carry the name")
    target = Tables(location(to), upstream.settings)
    if target.exist():
        raise DepotExists(f"{target.root} already holds a depot's tables")
    target.create()
    received = upstream.journal.settled()  # before any row is read: what is numbered up to it is whole
    for name in COPIED:
        if name == "parts":
            target.append_stream(name, upstream.tables.stream(name))
        else:
            target.append(name, upstream.tables.read(name))
    number = Depot(to, upstream.settings).change() + 1  # past every change number it received
    target.append("pulls", rows("pulls", [{"depot": about.name, "change": number, "upstream": upstream.name(),
                                            "upstream_change": received}]))
    target.append("about", rows("about", [{"revision": 1, "change": number, **vars(about),
                                            "upstream": upstream.address, "format": FORMAT}]))
    return Depot.open(to, upstream.settings)


def pull(depot: Depot) -> list[str]:
    """Fetch the upstream's updates into depot: the ids of the erasures the
    upstream carried out that the depot now has carried out too.
    NoUpstream when it follows none."""
    address = depot.about()["upstream"]
    if address is None:
        raise NoUpstream(f"{depot.address} follows no upstream")
    upstream = Depot.view(address, depot.settings)
    with depot._writes():
        before = _carried_out(depot.tables.read("erasures"))
        settled, versions = _capture(upstream.tables)
        depot._change({"kind": "pull", "upstream": upstream.address, "own_erasures": depot.tables.version("erasures"),
                       "versions": versions, "leave": _held(depot),
                       "pull": {"upstream": upstream.name(), "upstream_change": settled}},
                      upstream.change())  # past every change number it brings
        return sorted(_carried_out(depot.tables.read("erasures")) - before)


def _capture(upstream: Tables) -> tuple[int, dict[str, int]]:
    """(the upstream's settled change number, wpt_depot.journal, then the
    storage version of each table a pull reads): read in that order, so the
    rows read at those versions hold every row numbered up to that number,
    and a change the upstream had not written whole stays past it."""
    settled = Journal(upstream).settled()
    return settled, {name: upstream.version(name) for name in COPIED}


def _held(depot: Depot) -> dict:
    """What a pull may take out of the depot's libraries: {libraries: the
    depot's own open libraries, held: {each upstream text they hold: the
    decisions the depot holds on it}}."""
    libraries = depot.tables.query(f"SELECT id FROM libraries WHERE locked_at IS NULL AND "
                                   f"{ids.carries_sql(depot.name())}")["id"].to_pylist()
    held = {one["id"]: sorted(one["decisions"]) for one in depot.tables.query(
        f"SELECT id, decisions FROM texts WHERE list_has_any(libraries, ?) AND {depot.upstream_sql()}",
        [libraries]).to_pylist()} if libraries else {}
    return {"libraries": libraries, "held": held}


def _leaving(held: dict, texts, decisions) -> list[str]:
    """The texts of held ({text: the decisions the depot held on it}) that
    the rows read of the upstream's texts and decisions tables refuse or
    start over (wpt_depot.policy) by a decision not among those held."""
    if not held:
        return []
    current: dict[str, dict] = {}
    for one in texts.filter(pc.is_in(texts["id"], pa.array(list(held), pa.string()))).to_pylist():
        if one["id"] not in current or one["revision"] > current[one["id"]]["revision"]:
            current[one["id"]] = one
    added = {text: [decision for decision in one["decisions"] if decision not in held[text]]
             for text, one in current.items()}
    wanted = sorted({decision for found in added.values() for decision in found})
    ending = {one["id"] for one in decisions.filter(pc.is_in(decisions["id"], pa.array(wanted, pa.string())))
              .to_pylist() if one["status"] == REFUSED or policy.starts_over(one["reason"])}
    return sorted(text for text, found in added.items() if ending.intersection(found))


def apply_pull(depot: Depot, change: dict, number: int):
    """Apply a pull's change (Depot._apply), numbered number. The storage
    history is dropped when the erasures it read hold one carried out that
    the depot had not carried out when the pull began (its own erasures as
    of `own_erasures`; when that state is no longer kept, the history is
    dropped anyway). Of the upstream texts its change names, those that the
    texts and decisions rows it read refuse or start over by a decision the
    depot did not hold when the pull began leave the libraries it names at
    that number. It records the pull in pulls: its number, and the upstream
    change number its change carries; when a version it names is gone, it
    reads every table again as a pull begins (_capture) and records the
    higher of the two. OtherFormat when the upstream's format is no longer
    the code's (wpt_depot.schema.FORMAT)."""
    upstream = Depot.view(change["upstream"], depot.settings).tables
    try:
        before = _carried_out(depot.tables.read("erasures", change["own_erasures"]))
    except VersionGone:
        before = None
    brought, versions = change["pull"]["upstream_change"], change["versions"]
    while True:
        try:
            fresh, read = _bring(depot, upstream, versions, before)
            break
        except VersionGone:
            settled, versions = _capture(upstream)
            brought = max(brought, settled)
    leave = change["leave"]
    depot._leave(_leaving(leave["held"], read["texts"], read["decisions"]), leave["libraries"], number)
    depot.tables.upsert("pulls", rows("pulls", [{"depot": depot.name(), "change": number,
                                                "upstream": change["pull"]["upstream"], "upstream_change": brought}]))
    if fresh:
        depot.tables.drop_history()


def _bring_parts(depot: Depot, upstream: Tables, version: int, contents: pa.Table):
    """Add to depot every part that a row of the upstream's contents
    (contents, read at the version the pull reads) lists and the depot does
    not hold, read from the upstream's parts at version a few at a time:
    only the rows of contents the depot does not hold as they are can list
    one, whatever change numbers the parts carry, so a clone of a clone gets
    them too."""
    con = depot.tables.bare()
    try:
        con.register("theirs", contents.select(["id", "version", "parts"]))
        con.register("ours", depot.tables.read("contents", columns=["id", "version", "parts"]))
        listed = con.sql("SELECT DISTINCT id, unnest(parts).hash AS hash FROM theirs t WHERE NOT EXISTS "
                         "(SELECT 1 FROM ours o WHERE o.id = t.id AND o.version = t.version AND o.parts = t.parts)"
                         ).to_arrow_table().to_pylist()
    finally:
        con.close()
    texts = sorted({one["id"] for one in listed})
    if not texts:
        return
    named = ", ".join(map(literal, texts))
    held = {(one["id"], one["hash"]) for one in depot.tables.query(
        f"SELECT id, hash FROM parts WHERE id IN ({named})").to_pylist()}
    wanted = {(one["id"], one["hash"]) for one in listed} - held
    if not wanted:
        return

    def missing(found):
        for batch in found:
            keep = [(one, key) in wanted for one, key in zip(batch.column("id").to_pylist(),
                                                           batch.column("hash").to_pylist())]
            if any(keep):
                yield batch.filter(pa.array(keep))
    named = ", ".join(map(literal, sorted({text for text, _ in wanted})))
    depot.tables.append_stream("parts", missing(upstream.stream("parts", version, f"id IN ({named})")))


def _bring(depot: Depot, upstream: Tables, versions: dict[str, int], before: set[str] | None) -> tuple[bool, dict]:
    """Put into depot every row of the upstream's tables at versions (of
    parts, those its rows of contents list that it lacks, before those rows),
    removing the texts the erasures read removed and the parts no content
    version lists any longer of the texts they cut: (whether the erasures
    read hold one carried out that is not in before, True when before is
    None; {texts, decisions: the rows read of those tables}). VersionGone
    when a version is no longer kept; bringing again is harmless."""
    fresh, read = before is None, {}
    for name in COPIED:
        if name == "parts":
            continue  # brought with contents, before its rows
        found = upstream.read(name, versions[name])
        if name == "contents":
            _bring_parts(depot, upstream, versions["parts"], found)
        if name in ("texts", "decisions"):
            read[name] = found
        depot.tables.upsert(name, found)
        if name == "erasures":
            fresh = fresh or bool(_carried_out(found) - before)
            carried = [one for one in found.to_pylist() if one["erased"] is not None]
            gone = sorted({needle["text"] for one in carried for needle in one["needles"] if needle["start"] is None})
            depot._remove_texts(gone)
            cut = sorted({needle["text"] for one in carried if before is None or one["id"] not in before
                          for needle in one["needles"] if needle["start"] is not None} - set(gone))
            if cut:
                depot._drop_unlisted(cut)
    return fresh, read


def _carried_out(erasures) -> set[str]:
    """The ids of the erasures carried out among rows of the erasures table."""
    return {one["id"] for one in erasures.select(["id", "erased"]).to_pylist() if one["erased"] is not None}
