"""Where training reads a corpus's files from. Two readers give them alike:
DockFiles, the dock named by its URL, on the machine whose settings reach
it (the wpt manager), and wpt_worker.job.LinkDock, the link to the wpt
manager, on a gpu worker. wpt_corpus.shards and the environments of
wpt_training take either.

    path(reference)       the file as a local file of the reader's own folder, checked against its sha256,
                          kept until release(reference)
    get_bytes(reference)  the file's bytes, checked against its sha256
    release(reference)    the local file path() made removed (the file stays where it was)
    scratch               a local folder for files made from them, such as token files

A reference is what a record names a file by: {"key", "sha256"}.
"""

from pathlib import Path

from dock import api


class DockFiles:
    """The files of the dock named by its URL (dock.api), local copies in
    folder/files and files made from them in folder/scratch: folder is the
    caller's own, and removing it removes every copy."""

    def __init__(self, dock: str, folder):
        self.dock, self.folder = dock, Path(folder)
        self.copies, self.scratch = self.folder / "files", self.folder / "scratch"

    def __repr__(self):
        return f"the dock {self.dock}"

    def _local(self, reference) -> Path:
        return self.copies / reference["sha256"][:2] / reference["sha256"]

    def path(self, reference) -> Path:
        local = self._local(reference)
        if not local.is_file():
            api.fetch_into(self.dock, reference["key"], local, reference["sha256"])
        return local

    def get_bytes(self, reference) -> bytes:
        return api.fetch(self.dock, reference["key"], reference["sha256"])

    def release(self, reference):
        self._local(reference).unlink(missing_ok=True)
