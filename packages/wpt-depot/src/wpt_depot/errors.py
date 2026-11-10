"""What the depot's operations can fail with. Every failure but
JournalUnfinished leaves the depot as it was before the operation."""


class DepotError(Exception):
    """The base of every failure below."""


class NotADepot(DepotError):
    """The address holds no depot, or a depot missing one of its tables."""


class DepotExists(DepotError):
    """A new depot or a clone was asked for at an address that already holds tables."""


class UnsupportedAddress(DepotError):
    """The address is neither a folder, a file:// address nor an s3:// address."""


class NoDepotSetting(DepotError):
    """WPT_DEPOT is set neither in the environment nor in .env."""


class NoBucketSetting(DepotError):
    """A depot in a bucket needs WPT_DEPOT_ACCESS_KEY_ID and
    WPT_DEPOT_SECRET_ACCESS_KEY, set neither in the environment nor in .env."""


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
    """A query that is not one read (one SELECT statement), or that DuckDB refuses."""


class BadRemote(DepotError):
    """A remote (an exported dataset) is not in the shape export writes."""


class VersionGone(DepotError):
    """A storage version of a table that an erasure dropped (wpt_depot.tables);
    a pull that names one reads the current state instead."""


class BackupCut(DepotError):
    """A backup or a restore did not finish its copy: an erasure dropped
    the older storage states a backup was copying, or a file could not be
    copied. The folder it began holds no about table, so it is no depot."""


class NoUpstream(DepotError):
    """A pull from a depot that follows no upstream."""


class JournalUnfinished(DepotError):
    """A change was written to the journal and could not be applied to
    every table; the depot's next write or next opening finishes it. The cause is
    chained."""
