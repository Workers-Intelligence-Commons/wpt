"""How a part of a corpus fills its budget a
text at a time in the corpus's order (Fill), and the two numbers that close
a part: FULL and SKIPS (docs/data-selection.md, "Filling a part")."""

FULL = 1000  # a pool closes once what is left of its budget is at most this fraction of it (1 / FULL)
SKIPS = 1000  # or once this many of its texts in a row did not fit


class Fill:
    """What each pool takes of its budget ({pool: the most it takes, None
    for no limit}), a text at a time in the corpus's order: a text that
    fits what is left of its pool is taken, one that would pass it is left
    out, and the next is tried. A pool closes, taking nothing more, once
    what is left of it is at most a FULL-th of its budget (at once for a
    budget of 0), or once SKIPS of its texts in a row did not fit; a pool
    of no limit never closes. state() is what a Fill is made again from."""

    def __init__(self, budget: dict, state: dict | None = None):
        state = state or {}
        self.budget = budget
        self.used = {pool: state.get("used", {}).get(str(pool), 0) for pool in budget}
        self.run = {pool: state.get("run", {}).get(str(pool), 0) for pool in budget}
        self.closed = set(state.get("closed", [])) | {pool for pool in budget if self._full(pool)}

    def _full(self, pool) -> bool:
        limit = self.budget[pool]
        return limit is not None and limit - self.used[pool] <= limit // FULL

    def take(self, pool, size: int) -> bool:
        """Whether the pool takes a text of that size, which counts it in."""
        if pool in self.closed:
            return False
        if self.budget[pool] is not None and self.used[pool] + size > self.budget[pool]:
            self.run[pool] += 1
            if self.run[pool] >= SKIPS:
                self.closed.add(pool)
            return False
        self.used[pool] += size
        self.run[pool] = 0
        if self._full(pool):
            self.closed.add(pool)
        return True

    def done(self) -> bool:
        """Whether every pool is closed."""
        return len(self.closed) == len(self.budget)

    def state(self) -> dict:
        return {"used": {str(pool): value for pool, value in self.used.items()},
                "run": {str(pool): value for pool, value in self.run.items()}, "closed": sorted(self.closed)}

    def copy(self) -> "Fill":
        return Fill(self.budget, self.state())
