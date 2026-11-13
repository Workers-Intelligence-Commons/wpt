"""What the depot's operations can fail with. Every failure leaves the
depot as it was before the operation: a change is one transaction."""


class DepotError(Exception):
    """The base of every failure below."""


class NotADepot(DepotError):
    """The address holds no depot, or a depot missing one of its tables."""


class OtherFormat(DepotError):
    """The depot's layout is of another format than the one this code
    reads (wpt_depot.schema.FORMAT): an older one, which `wpt depot
    upgrade` brings up to it, or a newer one, which only a newer wpt reads.
    The message names the depot's address, its format and the code's."""


class DepotExists(DepotError):
    """A new depot, a clone, an upgrade's new file or a restore was asked
    for at an address that already holds a file."""


class UnsupportedAddress(DepotError):
    """The address is neither a path nor a file:// address (a depot), or not
    an http:// or https:// address (an upstream's web app)."""


class NoDepotSetting(DepotError):
    """WPT_DEPOT is set neither in the environment nor in .env."""


class NotFound(DepotError):
    """No text, item, decision, library, erasure, dataset, license or kind with that id or name."""


class Refused(DepotError):
    """A change the rules forbid, named in the message: a declaration not
    accepted or empty, a person's reason in the form the depot keeps for
    license changes, a license or kind the
    depot does not hold, a license change to the licenses a text names
    already, a name already taken, a bad needle or share, a
    problem reported with no text of the library it names."""


class Locked(Refused):
    """The change would add a text to a locked library. A text a locked
    library holds still takes decisions and changes of its licenses and
    credits, as new revisions the library does not read."""


class BelongsToUpstream(Refused):
    """The change would change a row the upstream owns, in a depot that
    follows an upstream (a clone): a clone only reads its upstream's rows.
    The message names those rows and says to ask the upstream."""


class BadContent(Refused):
    """An item's content is not JSON, or does not match its kind's schema."""


class BadQuery(DepotError):
    """A query that is not one statement that reads, or that SQLite
    refuses, a write, an attached database or a pragma included
    (wpt_depot.database)."""


class TimeLimitReached(DepotError):
    """A person's query ran past its time limit and was stopped."""


class BadRemote(DepotError):
    """A remote (an exported dataset) is not in the shape export writes."""


class BackupRefused(DepotError):
    """A backup or a restore cannot be made: no backup in the store, a
    change file missing between a full copy and the moment asked, or a
    change file that does not apply to the depot (a conflict). A restore
    refused leaves nothing at its address."""


class UpstreamUnreachable(DepotError):
    """The upstream's web app did not answer a clone or a pull, or answered
    with a failure; nothing was written."""


class NoUpstream(DepotError):
    """A pull from a depot that follows no upstream."""
