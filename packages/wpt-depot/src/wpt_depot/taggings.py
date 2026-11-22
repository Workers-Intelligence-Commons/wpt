"""A text's tags, as rows of taggings (wpt_depot.schema): each row a text
and a tag, with the change the text joined the tag at (joined_change) and
the change it left it at (left_change, null while it has it), and the
change that last wrote the row (change: joined_change, then left_change),
by which a pull picks the rows it lacks (wpt_depot.follow). A change of
tags writes or ends one row for each text and tag, never a revision of the
text, so a tag costs a small row and its listing (text_tags); what a text
held at any change is read from the rows that cover it (schema.tags_at),
which is how a snapshot reads its texts' tags as they were when it was made.

This module writes the rows, over a temporary table of (text, tag) pairs
filled by the caller, and sets again the copy of its tags each text it
changes holds on its current revision (schema.tags_sql); the caller keeps
the counts and the listing text_tags right (wpt_depot.counts) around it. A tag a change gives and takes back in
the same change leaves no row, and a tag it takes and gives back keeps the
row it had.

    PAIRS                                   the temporary table of pairs
    pairs(con)                              that table, made empty
    give(con, number)                       the pairs' texts join their tags at change number
    take(con, number)                       the pairs' texts leave their tags at change number
    put(con, number, given)                 each text of given has exactly its list of tags
    copy(con, texts)                        the copy of its tags on each text's current revision set again
"""

from .schema import tags_sql

PAIRS = "temp.tag_pairs"
# the pairs a text has now, each a row of taggings it has not left
_HELD = "SELECT 1 FROM taggings g WHERE g.text = p.text AND g.tag = p.tag AND g.left_change IS NULL"


def pairs(con) -> str:
    """The temporary table PAIRS (text, tag), made empty: what give and take read."""
    con.execute("CREATE TEMP TABLE IF NOT EXISTS tag_pairs (text TEXT NOT NULL, tag TEXT NOT NULL, "
                "PRIMARY KEY (text, tag)) WITHOUT ROWID")
    con.execute(f"DELETE FROM {PAIRS}")
    return PAIRS


def give(con, number: int):
    """Each text of PAIRS joins its tag at the change number, unless it has
    it already: a row it left at that same change is taken up again, else a
    new row."""
    _give(con, number)
    copy(con, f"SELECT text FROM {PAIRS}")


def _give(con, number: int):
    con.execute(f"UPDATE taggings SET left_change = NULL, change = :number WHERE left_change = :number "
                f"AND (text, tag) IN (SELECT text, tag FROM {PAIRS})", {"number": number})
    con.execute(f"INSERT INTO taggings (text, tag, joined_change, left_change, change) SELECT p.text, p.tag, :number, "
                f"NULL, :number FROM {PAIRS} p WHERE NOT EXISTS ({_HELD})", {"number": number})


def take(con, number: int):
    """Each text of PAIRS leaves its tag at the change number, when it has
    it: a row it joined at that same change goes, else its row ends."""
    _take(con, number)
    copy(con, f"SELECT text FROM {PAIRS}")


def _take(con, number: int):
    con.execute(f"DELETE FROM taggings WHERE joined_change = :number AND left_change IS NULL "
                f"AND (text, tag) IN (SELECT text, tag FROM {PAIRS})", {"number": number})
    con.execute(f"UPDATE taggings SET left_change = :number, change = :number WHERE left_change IS NULL "
                f"AND (text, tag) IN (SELECT text, tag FROM {PAIRS})", {"number": number})


def put(con, number: int, given: str):
    """Each text of the query given (columns id, and tags: a JSON array)
    has exactly those tags from the change number on: it leaves those it
    has that the array does not name and joins those it names that it
    lacks."""
    pairs(con)
    con.execute(f"INSERT INTO {PAIRS} (text, tag) SELECT g.text, g.tag FROM ({given}) s "
                "CROSS JOIN taggings g ON g.text = s.id AND g.left_change IS NULL "
                "WHERE g.tag NOT IN (SELECT value FROM json_each(s.tags))")
    _take(con, number)
    pairs(con)
    con.execute(f"INSERT OR IGNORE INTO {PAIRS} (text, tag) SELECT s.id, j.value FROM ({given}) s, json_each(s.tags) j")
    _give(con, number)
    con.execute(f"DELETE FROM {PAIRS}")
    copy(con, f"SELECT id FROM ({given})")


def copy(con, texts: str):
    """The copy of its tags on the current revision of each text the query
    texts gives (its first column) set again from its rows of taggings,
    where it differs (schema.tags_sql)."""
    con.execute(tags_sql(texts))
