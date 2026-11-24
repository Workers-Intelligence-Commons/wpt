// Jobs: the server's queue (GET /api/jobs), read again every few seconds, in
// its three lanes, each running one job at a time: the writes over a
// selection of texts, the heavy work (making a snapshot, training a
// vocabulary, building a corpus), and the reads that order a snapshot by
// size for its strip, so a long build never holds up a write or a strip. Each lane
// lists every job queued or running, in the order they run, then the latest
// that ended, each as a sentence: what it does, linked to what it works on
// (the snapshot, the corpus, the tokenizer recipe, the decision), its
// state with how long it waited or ran, and what it gave. Anyone starts a
// snapshot, a vocabulary training or a corpus build, each after its estimate
// and a confirmation, from the Snapshots, Tokenizers and Build pages; only
// the command line cancels a job (wpt depot jobs cancel ID). This file also
// keeps what the other pages show of a job: JobProgress, its progress in
// words; JobLine, one job followed from its start to its end; and
// RunningJobs, the mark in the navigation while a job waits or runs.

import { Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState, type ReactNode } from "react";
import { get, waiting, type Answer, type Job, type Lane } from "../api";
import { leftOutWords, PageHead, plainProblem, plural, Problems, Section, takenWords, type LeftOut } from "../parts";
import { href } from "../route";
import { pipelineChanged, snapshotHref } from "../pipeline";
import type { Page } from "./index";

const EVERY_SECONDS = 3; // seconds between two readings of the queue on its page
export const MARK_SECONDS = 10; // seconds between two readings of the queue for the mark in the navigation
const RETRY_MOST_SECONDS = 30; // the longest wait before reading a followed job again after a failed reading

export const capital = (text: string) => (text ? `${text[0].toUpperCase()}${text.slice(1)}` : text);
const when = (seconds: number | null) => (seconds ? new Date(seconds * 1000).toISOString().replace("T", " ").slice(0, 16) : "");
// How long ago a moment was, short: "just now", "5 min ago", "3 h ago", "2 days ago".
const ago = (seconds: number, now: number) => {
  const s = Math.max(0, now - seconds);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${plural(Math.floor(s / 86400), "day")} ago`;
};


// A length of time, short: 1 s, 45 s, 3 min 20 s, 2 h 5 min.
export function lasted(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ""}`;
  return `${Math.floor(s / 3600)} h${Math.round((s % 3600) / 60) ? ` ${Math.round((s % 3600) / 60)} min` : ""}`;
}

// The stages a vocabulary training, a snapshot and a corpus build report
// (wpt.depot.shared, wpt_depot.snapshots, wpt_depot.corpus), in words.
const STAGES: Record<string, string> = {
  "reading texts": "reading the admitted texts",
  "catching up": "reading the texts decided while it read",
  writing: "writing the snapshot's sets and summary",
  "finding long texts": "finding the texts longer than a million characters, which come as pieces",
  "counting items": "counting the tokens of the conversations and tasks",
  choosing: "choosing each share's texts, its part split over the tags of its texts by the square root of what each holds",
  reading: "tokenizing the texts",
  finishing: "writing the corpus's tables",
  "reading the sample": "reading the sample of texts",
  merging: "learning the vocabulary",
  keeping: "saving the vocabulary",
};
export const stageWords = (stage: string) => STAGES[stage] ?? stage;

// The numbers a stage gave, in words: "of" is the end the stage moves toward.
function numbersWords(stage: unknown, numbers: Record<string, unknown>): string {
  const n = (key: string) => (typeof numbers[key] === "number" ? (numbers[key] as number).toLocaleString("en") : null);
  const of = n("of");
  if (stage === "reading" && n("position") !== null && of) {
    const rest = Object.entries(numbers).filter(([k]) => !["position", "of"].includes(k));
    return [`text ${((numbers.position as number) + 1).toLocaleString("en")} of ${of}`, ...rest.map(([k, v]) => `${typeof v === "number" ? v.toLocaleString("en") : String(v)} ${k === "texts" ? "taken" : k}`)].join(", ");
  }
  if (stage === "merging" && n("merges") !== null && of) return `${n("merges")} of ${of} tokens`;
  if (stage === "reading texts" && n("texts") !== null && of) return `${n("texts")} of ${of} texts`;
  if (stage === "reading the sample" && typeof numbers.bytes === "number" && typeof numbers.of === "number") return `${n("texts")} texts, ${bytesShort(numbers.bytes)} of ${bytesShort(numbers.of)}`;
  return Object.entries(numbers).map(([k, v]) => (typeof v === "number" ? `${v.toLocaleString("en")} ${k}` : `${k} ${String(v)}`)).join(", ");
}
const bytesShort = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : b >= 1e3 ? `${Math.round(b / 1e3)} KB` : `${b} bytes`);

// A job's progress in words: its stage and the numbers it gave ("tokenizing the texts: text 120 of 525, 5,000 tokens").
export function JobProgress({ job }: { job: Job<unknown> }) {
  const p = job.progress;
  if (!p) return <span className="aside">{job.state === "queued" ? "waiting its turn" : stageWords(job.stages.at(-1)?.stage ?? "starting")}</span>;
  const { stage, ...numbers } = p as Record<string, unknown>;
  const said = numbersWords(stage, numbers);
  return (
    <span>
      {stage ? <span className="cue">{stageWords(String(stage))}</span> : null}{stage && said ? ": " : null}<span className="aside">{said}</span>
    </span>
  );
}

// Where a job's result is: the snapshot it made (the list while it runs),
// the corpus of the build key it builds, the tokenizer recipe it trains,
// and for a write that ended well, the decision it recorded.
function resultLink(job: Job<unknown>): string | null {
  const a = (job.state === "done" ? job.answer ?? {} : {}) as Record<string, unknown>;
  if (job.kind === "snapshot") return typeof a.id === "string" ? snapshotHref({ id: a.id }) : href(["snapshots"]);
  if (job.kind === "corpus" && job.key) return href(["build"], { key: job.key });
  if (job.kind === "corpus") return href(["build"]);
  if (job.kind === "vocabulary" && job.recipe) return href(["tokenizers", job.recipe]);
  if (job.kind === "sizes" && job.key) return href(["snapshots", job.key, "strip"]);
  if (typeof a.decision === "string") return href(["decisions", a.decision]);
  return null;
}

// Whether a link leads to the page shown now.
const shownNow = (link: string) => window.location.hash.split("?")[0] === link.split("?")[0];

// What the depot left out of a write, by reason.
const leftWords = (left: unknown) => {
  if (!left || typeof left !== "object") return "";
  const parts = Object.entries(left as Record<string, number>).filter(([, n]) => n > 0).map(([why, n]) => `${n.toLocaleString("en")} ${why}`);
  return parts.length ? `; left out: ${parts.join(", ")}` : "";
};

// What a job that ended well gave, in words, with a link to its result
// when that is not the page shown.
function answered(job: Job<unknown>): ReactNode {
  const a = (job.answer ?? {}) as Record<string, unknown>;
  const n = (key: string) => (typeof a[key] === "number" ? (a[key] as number) : 0);
  const link = resultLink(job);
  const open = (words: string) => (link && !shownNow(link) ? <>: <a href={link}>{words}</a></> : null);
  if (job.kind === "snapshot") return <>{plural(n("texts"), "text")} as of change {n("at_change").toLocaleString("en")}{open("open it")}</>;
  if (job.kind === "corpus") return <>{takenWords(n("texts"), n("conversations"), n("tasks"))}, {plural(n("tokens") + n("item_tokens"), "token")} in all{n("item_tokens") ? ` (${n("tokens").toLocaleString("en")} of plain text, ${n("item_tokens").toLocaleString("en")} of conversations and tasks)` : ""}{leftOutWords(a.left_out as LeftOut | undefined) ? `; left out: ${leftOutWords(a.left_out as LeftOut | undefined)}` : ""}{open("download it")}</>;
  if (job.kind === "vocabulary") return link && !shownNow(link) ? <>Trained: <a href={link}>see the vocabulary</a></> : null;
  if (job.kind === "sizes") return <>{plural(n("texts"), "text")} ordered by size{open("read it whole")}</>;
  if ("changed" in a && typeof a.decision === "string") return <>{plural(n("changed"), "text")} changed{leftWords(a.left)}: <a href={href(["decisions", a.decision])} title={`decision ${a.decision}`}>see the decision and its texts</a></>;
  if ("decisions" in a && Array.isArray(a.decisions)) return <>{plural((a.decisions as { texts: number }[]).reduce((t, d) => t + d.texts, 0), "text")} relicensed{leftWords(a.left)}</>;
  if (typeof a.decision === "string") return <>{plural(n("texts"), "text")}{leftWords(a.left)}: <a href={href(["decisions", a.decision])} title={`decision ${a.decision}`}>see the decision and its texts</a></>;
  return null;
}

// A job's state in words: waiting, running for how long, or how it ended
// and after how long, with the time it waited for the jobs before it once
// that reaches a second.
export function JobState({ job, now }: { job: Job<unknown>; now: number }) {
  if (job.state === "queued") return <span>waiting for the job before it ({lasted(now - job.created)})</span>;
  const waited = job.started !== null && job.started - job.created >= 1 ? `, after waiting ${lasted(job.started - job.created)} for the job before it` : "";
  if (job.state === "running") return <span className="cue">running ({lasted(now - (job.started ?? job.created))}){waited}</span>;
  const took = job.ended && job.started ? ` in ${lasted(job.ended - job.started)}` : "";
  if (job.state === "done") return <span className="cue">done{took}{waited}</span>;
  return <span className="not-allowed">{job.state}{took}{waited}</span>;
}

// The time now, in seconds, again every second while a job waits or runs.
export function useNow(ticking: boolean): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!ticking) return;
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, [ticking]);
  return now;
}

// One job, followed from the answer that queued it (GET /api/jobs/ID every
// second until it ends): what it does, its state, its progress, and once
// ended what it gave with the link to its result, or its problem. A reading
// that fails is said under the line and tried again, each wait twice the one
// before, up to RETRY_MOST_SECONDS. onEnded: once, when it ends, with how it
// ended.
export function JobLine({ job: started, onEnded }: { job: Job<unknown>; onEnded?: (job: Job<unknown>) => void }) {
  const [job, setJob] = useState(started);
  const [failed, setFailed] = useState<{ times: number; problems: string[] } | null>(null);
  useEffect(() => { setJob(started); setFailed(null); }, [started]);
  const wait = failed ? Math.min(2 ** failed.times, RETRY_MOST_SECONDS) : 1;
  useEffect(() => {
    if (!waiting(job.state)) return;
    let live = true;
    const timer = window.setTimeout(() => {
      get<Job<unknown>>(`/api/jobs/${encodeURIComponent(job.id)}`).then((a) => {
        if (!live) return;
        if (!a.ok) { setFailed((f) => ({ times: (f?.times ?? 0) + 1, problems: a.problems })); return; }
        setFailed(null);
        setJob(a.data);
        if (!waiting(a.data.state)) { pipelineChanged(); onEnded?.(a.data); }
      });
    }, wait * 1000);
    return () => { live = false; window.clearTimeout(timer); };
  }, [job, failed, onEnded]);
  const now = useNow(waiting(job.state));
  const gave = job.state === "done" ? answered(job) : null;
  return (
    <>
      {job.state === "done" ? (
        // a short headline, then what it gave and the next step on a line of their own
        <div className="done" role="status">
          <p className="cue">{capital(job.title)}: <JobState job={job} now={now} />.</p>
          {gave && <p>{gave}.</p>}
        </div>
      ) : (
        <p className="job-line" role="status">
          {capital(job.title)}: <JobState job={job} now={now} />
          {job.state === "running" && <>: <JobProgress job={job} /></>}
          {job.problem && <span className="not-allowed">. {plainProblem(job.problem)}</span>}
          {waiting(job.state) && <>. <a href={href(["jobs"])}>See the queue</a></>}
        </p>
      )}
      {failed && <Problems problems={[`The job's state could not be read, so the line above may be out of date; trying again in ${lasted(wait)}.`, ...failed.problems]} />}
    </>
  );
}

// The readings of the queue, one for each period in seconds, shared by
// every part of the page that asks for that period, so two of them never
// read the queue twice.
type Reading = { latest: Answer<Job<unknown>[]> | null; listeners: Set<(a: Answer<Job<unknown>[]>) => void>; stop: () => void };
const readings = new Map<number, Reading>();

// Read the queue now and again every few seconds while the page is shown.
export function useJobs(seconds = EVERY_SECONDS): Answer<Job<unknown>[]> | null {
  const [jobs, setJobs] = useState<Answer<Job<unknown>[]> | null>(() => readings.get(seconds)?.latest ?? null);
  useEffect(() => {
    let reading = readings.get(seconds);
    if (!reading) {
      const made: Reading = { latest: null, listeners: new Set(), stop: () => {} };
      const read = () => { if (!document.hidden) get<Job<unknown>[]>("/api/jobs").then((a) => { made.latest = a; made.listeners.forEach((l) => l(a)); }); };
      read();
      const timer = window.setInterval(read, seconds * 1000);
      const shown = () => { if (!document.hidden) read(); };
      document.addEventListener("visibilitychange", shown);
      made.stop = () => { window.clearInterval(timer); document.removeEventListener("visibilitychange", shown); };
      readings.set(seconds, made);
      reading = made;
    } else if (reading.latest) setJobs(reading.latest);
    const shared = reading;
    shared.listeners.add(setJobs);
    return () => {
      shared.listeners.delete(setJobs);
      if (shared.listeners.size === 0) { shared.stop(); readings.delete(seconds); }
    };
  }, [seconds]);
  return jobs;
}

// The mark in the navigation while a job waits or runs: how many, linked to the queue.
export function RunningJobs() {
  const jobs = useJobs(MARK_SECONDS);
  const open = jobs?.ok ? jobs.data.filter((j) => waiting(j.state)) : [];
  if (open.length === 0) return null;
  const running = open.find((j) => j.state === "running");
  return (
    <a className="running-mark small wic-focus" href={href(["jobs"])} title={open.map((j) => `${j.title}: ${j.state}`).join("\n")}>
      <span className="running-dot" aria-hidden="true" />
      {running ? `${plural(open.length, "job")} running` : `${plural(open.length, "job")} waiting`}
    </a>
  );
}

const LANES: { lane: Lane; title: string; help: string }[] = [
  { lane: "heavy", title: "Snapshots, vocabularies and corpora", help: "Making a snapshot, training a vocabulary and building a corpus, one at a time." },
  { lane: "write", title: "Writes over many texts", help: "A decision, a change of tags or licenses over a selection, one at a time; they never wait for the heavy lane." },
  { lane: "read", title: "Snapshots ordered by size", help: "Ordering a snapshot by the size of its texts, once for good, the first time its strip or its size map opens; it never waits for the other lanes." },
];

// One lane's jobs as a table, or a line when it has none.
function LaneTable({ list, now, label }: { list: Job<unknown>[]; now: number; label: string }) {
  if (list.length === 0) return <p className="small aside">None.</p>;
  return (
    <div className="table-scroll cards">
      <table className="ledger-table" aria-label={label}>
        <thead><tr><th scope="col">job</th><th scope="col">state</th><th scope="col">what it gave, or its progress</th><th scope="col">queued</th></tr></thead>
        <tbody>
          {list.map((j) => {
            const link = resultLink(j);
            return (
              <tr key={j.id}>
                <td className="wrap-cell" data-label="job">{link ? <a href={link}>{capital(j.title)}</a> : capital(j.title)}</td>
                <td className="nowrap" data-label="state"><JobState job={j} now={now} /></td>
                <td className="wrap-cell small" data-label="gave">{waiting(j.state) ? <JobProgress job={j} />
                  : j.problem ? <span className="not-allowed">{plainProblem(j.problem)}</span>
                  : j.state === "done" ? answered(j) : null}</td>
                <td className="nowrap" data-label="queued" title={`queued ${when(j.created)} UTC${j.ended ? `, ended ${when(j.ended)} UTC` : ""}`}>{ago(j.created, now)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// A job's lane: the one the server gives, else (a queue read before lanes) by its kind.
export const laneOf = (j: Job<unknown>): Lane => j.lane ?? (j.kind === "write" ? "write" : "heavy");

function Jobs() {
  const jobs = useJobs();
  const list = jobs?.ok ? jobs.data : [];
  const now = useNow(true); // every second: the states running and the "ago" of each job
  return (
    <>
      <PageHead title="Jobs"
        meta={jobs?.ok && list.length > 0 ? <><span>{list.filter((j) => j.state === "running").length.toLocaleString("en")} running</span><span>{list.filter((j) => j.state === "queued").length.toLocaleString("en")} waiting</span></> : undefined}
        chain={<>The long work the server runs, in three lanes of one job at a time: making a <a href={href(["snapshots"])}>snapshot</a>, training a <a href={href(["tokenizers"])}>tokenizer recipe</a>'s vocabulary and <a href={href(["build"])}>building</a> a recipe's corpus in one; a write over many texts in another; ordering a snapshot by size, for its strip, in the third. A job that ended links to what it made. A job cannot be cancelled here; the command line can, on the depot's machine.</>} />
      {!jobs && <Skeleton lines={5} label="Reading the queue" />}
      {jobs && !jobs.ok && <Problems problems={jobs.problems} />}
      {jobs?.ok && LANES.map(({ lane, title, help }) => {
        const mine = list.filter((j) => laneOf(j) === lane);
        const open = mine.filter((j) => waiting(j.state)).length;
        return (
          <Section key={lane} title={<>{title}{open ? `: ${open} open` : ""}</>} help={help}>
            <LaneTable list={mine} now={now} label={title} />
          </Section>
        );
      })}
    </>
  );
}

export const jobs: Page = { path: "jobs", title: "Jobs", group: "data", Component: () => <Jobs /> };
