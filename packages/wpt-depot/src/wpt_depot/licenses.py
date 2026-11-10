"""The allowed licenses (docs/licenses.md, "What the corpus accepts"): the
app admits a text only when one of its licenses is allowed. A text's
licenses are a choice, so one allowed license is enough.

A license of a depot's `licenses` table is allowed when it is on the list
below, or is one such license with an exception that adds permissions
(`<license> WITH <exception>`), or is several of these that apply together
(`<license> AND <license>`, each part allowed on its own: a combined row is
allowed only if every part is).
"""

ALLOWED = frozenset({
    "LicenseRef-public-domain", "CC0-1.0", "Unlicense",
    "MIT", "MIT-0", "0BSD", "BSD-1-Clause", "BSD-2-Clause", "BSD-2-Clause-Patent", "BSD-3-Clause", "ISC", "X11",
    "NCSA", "PostgreSQL", "UPL-1.0", "HPND", "W3C", "curl", "Apache-2.0", "Zlib", "BSL-1.0", "WTFPL", "Python-2.0",
    "PSF-2.0", "Unicode-3.0", "Unicode-DFS-2016", "Artistic-2.0", "BlueOak-1.0.0", "FSFAP",
    "BSD-3-Clause-No-Nuclear-Warranty",
    "GPL-3.0-only", "GPL-3.0-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later", "AGPL-3.0-only", "AGPL-3.0-or-later",
    "GPL-2.0-or-later", "LGPL-2.1-only", "LGPL-2.1-or-later", "MPL-2.0", "EUPL-1.2", "CECILL-2.1",
    "CC-BY-4.0", "CC-BY-SA-4.0",
})
EXCEPTIONS = frozenset({"Classpath-exception-2.0", "LLVM-exception", "GCC-exception-3.1"})
TOGETHER = " AND "
WITH = " WITH "


def allowed(license_id: str) -> bool:
    """Whether the license of that id is allowed (see above)."""
    return all(_part_allowed(part) for part in license_id.split(TOGETHER))


def any_allowed(license_ids) -> bool:
    """Whether one of the licenses, a choice, is allowed."""
    return any(allowed(one) for one in license_ids)


def _part_allowed(part: str) -> bool:
    base, with_, exception = part.partition(WITH)
    return base in ALLOWED and (not with_ or exception in EXCEPTIONS)
