// The counts of the texts a filter matches, made in the background a step at
// a time through POST /api/texts/counts: each step counts one slice of the
// depot's texts (a filter that names its texts is counted in one step), and
// the page adds the steps up, so the numbers grow as the steps answer, with
// how many texts are counted out of how many. One step of the whole page
// runs at a time, the counts open on the page taking turns, and a step
// starts at least PACE after the one before it, so counting never takes
// more than half of what the proxy in front of the server lets one address
// send to the routes that run SQL. A step the proxy refuses (429), or that
// found no free reader on the server (NoReaderFree), is sent again after a
// wait that doubles each time. A step over a slice that the server's time
// limit stopped (TimeLimitReached) is sent again over a slice half as big,
// down to SMALLEST texts, and a step that answers within QUICK lets the
// next one go over twice as many, up to STEP; a filter that names its texts
// has no slice, so its step is not sent again. A count ends when its page
// no longer shows it: its component leaves, or its filter or version
// changes. A selection (filter.tsx) is counted as a write takes it: less its
// unticked texts and those changed since its change number.

import { useEffect, useState } from "react";
import { post } from "./api";
import { sent, type Selection, type TextFilter } from "./filter";
import { plural, STATUSES, type Status } from "./parts";

// blocked: how many an admission cannot cover, by the depot's reason (blocked.tsx)
export type Counts = {
  statuses: Record<Status, number>; locked: number; blocked: Record<string, number>;
  tags: { tag: string; texts: number }[]; licenses: { license: string; texts: number }[];
};
// counted: texts gone over so far; total: how many there are to go over, known from the first step
export type Counting = Counts & { counted: number; total: number | null; done: boolean; problems: string[] | null };

type Step = Counts & { texts: number; total: number | null; next: string | null };

const PACE = 2000; // milliseconds from the start of a step to the start of the next one, at the least
const WAIT = 2000; // milliseconds before a step refused for the rate is sent again, doubled each time
const LONGEST_WAIT = 60_000;
const STEP = 1_000_000; // texts a step goes over at first, the most the server takes
const SMALLEST = 1000; // texts of the smallest step tried after the time limit stopped bigger ones
const QUICK = 15_000; // milliseconds: a quarter of the server's limit of 60 seconds a request

type Job = {
  selection: Selection; start: string | null; size: number; wait: number; users: number;
  state: Counting; listeners: Set<(c: Counting) => void>;
};

const jobs = new Map<string, Job>();
let turn = 0; // where the next turn starts among the jobs
let running = false;
let lastStart = 0;

const EMPTY: Counting = {
  statuses: { "under review": 0, admitted: 0, refused: 0 }, locked: 0, blocked: {}, tags: [], licenses: [],
  counted: 0, total: null, done: false, problems: null,
};

// The sum of two lists of named counts, the most first, then by name.
function added<K extends string>(key: K, a: ({ [k in K]: string } & { texts: number })[], b: ({ [k in K]: string } & { texts: number })[]) {
  const sums = new Map<string, number>();
  for (const one of [...a, ...b]) sums.set(one[key], (sums.get(one[key]) ?? 0) + one.texts);
  return [...sums].map(([name, texts]) => ({ [key]: name, texts }) as { [k in K]: string } & { texts: number })
    .sort((x, y) => y.texts - x.texts || (x[key] < y[key] ? -1 : 1));
}

function publish(job: Job, state: Counting) {
  job.state = state;
  job.listeners.forEach((listen) => listen(state));
}

const pause = (ms: number) => new Promise((done) => window.setTimeout(done, ms));

// Runs the steps of every job, one at a time, until every job is done or gone.
async function run() {
  if (running) return;
  running = true;
  try {
    for (;;) {
      const waiting = [...jobs.entries()].filter(([, job]) => !job.state.done);
      if (waiting.length === 0) return;
      const [key, job] = waiting[turn++ % waiting.length];
      await pause(Math.max(0, lastStart + PACE - Date.now()));
      if (jobs.get(key) !== job) continue; // it ended while waiting
      lastStart = Date.now();
      const size = job.size;
      const a = await post<Step>("/api/texts/counts", { ...sent(job.selection), size: job.size, ...(job.start !== null ? { start: job.start } : {}) });
      if (jobs.get(key) !== job) continue; // it ended while its step ran: the answer is dropped
      const refused = (name: string) => !a.ok && a.status === 422 && a.problems.some((p) => p.startsWith(`${name}: `));
      if (!a.ok && (a.status === 429 || refused("NoReaderFree"))) {
        await pause(job.wait);
        job.wait = Math.min(job.wait * 2, LONGEST_WAIT);
        continue;
      }
      job.wait = WAIT;
      if (refused("TimeLimitReached") && !Array.isArray(job.selection.filter.ids) && size / 2 >= SMALLEST) {
        job.size = Math.floor(size / 2);
        continue;
      }
      if (!a.ok) { publish(job, { ...job.state, done: true, problems: a.problems }); continue; }
      const s = job.state, step = a.data;
      job.start = step.next;
      if (Date.now() - lastStart < QUICK) job.size = Math.min(size * 2, STEP);
      publish(job, {
        statuses: Object.fromEntries(STATUSES.map((st) => [st, s.statuses[st] + step.statuses[st]])) as Record<Status, number>,
        locked: s.locked + step.locked, tags: added("tag", s.tags, step.tags), licenses: added("license", s.licenses, step.licenses),
        blocked: Object.fromEntries([...new Set([...Object.keys(s.blocked), ...Object.keys(step.blocked)])]
          .map((reason) => [reason, (s.blocked[reason] ?? 0) + (step.blocked[reason] ?? 0)])),
        counted: s.counted + step.texts, total: s.total ?? step.total, done: step.next === null, problems: null,
      });
    }
  } finally {
    running = false;
  }
}

// The counts of the texts a filter or a selection matches as they grow, null
// without one; version: raised after a change, so the count starts again.
export function useCounts(what: TextFilter | Selection | null, version = 0): Counting | null {
  const selection: Selection | null = what === null ? null : "filter" in what ? what : { filter: what };
  const key = selection === null ? null : JSON.stringify([selection, version]);
  const [state, setState] = useState<{ key: string | null; counting: Counting | null }>({ key: null, counting: null });
  useEffect(() => {
    if (key === null) return;
    let job = jobs.get(key);
    if (!job) {
      job = { selection: JSON.parse(key)[0], start: null, size: STEP, wait: WAIT, users: 0, state: EMPTY, listeners: new Set() };
      jobs.set(key, job);
    }
    const mine = job;
    const listen = (counting: Counting) => setState({ key, counting });
    mine.users += 1;
    mine.listeners.add(listen);
    setState({ key, counting: mine.state });
    run();
    return () => {
      mine.listeners.delete(listen);
      mine.users -= 1;
      if (mine.users === 0) jobs.delete(key);
    };
  }, [key]);
  return state.key === key ? state.counting : key === null ? null : EMPTY;
}

// How far a count has gone, while it goes on: nothing once it is done.
export function CountingNote({ counting }: { counting: Counting | null }) {
  if (!counting) return null;
  if (counting.problems) return <span className="aside">The count stopped: {counting.problems.join(" ")}</span>;
  if (counting.done) return null;
  return (
    <span className="aside" role="status">
      counting: {counting.total === null ? "starting" : `${counting.counted.toLocaleString("en")} of ${plural(counting.total, "text")}`}
    </span>
  );
}

// How many texts a count found so far.
export const matched = (c: Counts) => STATUSES.reduce((n, s) => n + c.statuses[s], 0);

// How many texts a filter matches as its count grows: the number once done,
// "at least" with the progress while it goes on.
export function MatchedCount({ counting }: { counting: Counting | null }) {
  if (!counting) return null;
  if (counting.done && !counting.problems) return <>{plural(matched(counting), "text")}</>;
  return <>{counting.counted > 0 && <>at least {plural(matched(counting), "text")}; </>}<CountingNote counting={counting} /></>;
}
