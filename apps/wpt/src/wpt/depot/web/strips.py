"""The maps of the libraries read as one strip (wpt_depot.strips), made
by `wpt depot web` in a thread of its own and kept in its jobs' folder
(<depot file>.work/strips/), one file per library, so a restart reads them
again instead of walking the library.

A map is made for one state of its library (strips.state). Asking for the
map of a library that has none for its state now starts a walk of it in
the background, one walk at a time over every library (the others wait
their turn), and answers how far it went until it ends. Meanwhile the
library's last map, made for an earlier state, still answers: its
positions are off by what moved, but a jump still lands near them. A walk
reads no text, only the library's rows in the strip's order, a statement
of strips.WALKED rows at a time. A walk that failed is not tried again
for RETRY seconds, so a library that cannot be walked is not walked on
every request; its failure is answered until then.
"""

import json
import os
import threading
import time
from pathlib import Path

from wpt_depot import Depot, strips
from wpt_utils.lines import ERROR, say

RETRY = 600  # seconds before a failed walk of the same state is tried again


class Maps:
    def __init__(self, depot: Depot, folder: Path):
        self.depot, self.folder = depot, folder
        self.lock = threading.Lock()  # guards found and walking
        self.one_walk = threading.Lock()  # one walk at a time
        self.found: dict[str, strips.Map] = {}  # library: its latest map, of any state
        self.walking: dict[str, dict] = {}  # library: {state, walked, failed, failed_at, running}

    def _file(self, library: str) -> Path:
        return self.folder / f"{library}.json"

    def _kept(self, library: str) -> strips.Map | None:
        try:
            return strips.Map.of_json(json.loads(self._file(library).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def get(self, library: str) -> tuple[strips.Map | None, dict]:
        """(the library's latest map, None when it has none yet; {fresh:
        whether that map is of the library's state now, walked: the texts
        the walk for the state now read so far, failed: why that walk
        failed, else None}), a walk started when the state now has no map
        and none runs or failed less than RETRY seconds ago. NotFound
        without such a library."""
        state = strips.state(self.depot, library)
        with self.lock:
            found = self.found.get(library)
        if found is None:
            found = self._kept(library)
            if found is not None:
                with self.lock:
                    found = self.found.setdefault(library, found)
        if found is not None and found.state == state:
            return found, {"fresh": True, "walked": found.texts, "failed": None}
        with self.lock:
            walking = self.walking.get(library)
            again = (walking is None or walking["state"] != state
                     or (not walking["running"] and walking["failed"] is not None
                         and time.monotonic() - walking["failed_at"] > RETRY))
            if again and not (walking and walking["running"]):
                walking = {"state": state, "walked": 0, "failed": None, "failed_at": None, "running": True}
                self.walking[library] = walking
                threading.Thread(target=self._walk, args=(library, walking), daemon=True,
                                 name=f"strip map {library}").start()
            answer = {"fresh": False, "walked": walking["walked"],
                      "failed": walking["failed"] if walking["state"] == state else None}
        return found, answer

    def _walk(self, library: str, walking: dict):
        def progress(walked: int):
            walking["walked"] = walked
        try:
            with self.one_walk:
                found = strips.build(self.depot, library, progress)
            self.folder.mkdir(parents=True, exist_ok=True)
            partial = self._file(library).with_suffix(".partial")
            partial.write_text(json.dumps(found.as_json()), encoding="utf-8")
            os.replace(partial, self._file(library))
            with self.lock:
                self.found[library] = found
                walking["walked"] = found.texts
        except Exception as error:  # noqa: BLE001 - a failed walk is answered, and the server goes on
            say(f"wpt depot web: the map of {library} failed: {type(error).__name__}: {error}", ERROR)
            with self.lock:
                walking["failed"], walking["failed_at"] = f"{type(error).__name__}: {error}", time.monotonic()
        finally:
            with self.lock:
                walking["running"] = False
