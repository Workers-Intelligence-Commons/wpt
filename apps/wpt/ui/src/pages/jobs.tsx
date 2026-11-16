// Jobs: the server's queue (GET /api/jobs), read again every few seconds:
// every job queued or running, in the order they run, then the latest that
// ended, each as a sentence: what it does, linked to what it works on, its
// state with how long it waited or ran, and what it gave, linked to its
// result (a corpus to download, a vocabulary, a decision, a library). Anyone
// starts a vocabulary training or a corpus build, each after its estimate
// and a confirmation, from the Tokenizers and Download pages; only the
// command line cancels a job (wpt depot jobs cancel ID). This file also
// keeps what the other pages show of a job: JobProgress, its progress in
// words; JobLine, one job followed from its start to its end; and
// RunningJobs, the mark in the navigation while a job waits or runs.

import { Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState, type ReactNode } from "react";
import { get, waiting, type Answer, type Job } from "../api";
import { PageHead, plainProblem, plural, Problems, Section, takenWords, useDatasets, useLibraries } from "../parts";
import { href } from "../route";
import { pipelineChanged, StageMark, standing, StepButton, useCorpora, useRecipesAndVocabularies, vocabularyNames } from "../pipeline";
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

// What the pages know that a job's answer names by id: the dataset that
// publishes each vocabulary (by its sha256), and each library's name.
export type Known = { published?: Map<string, { id: string; name: string }>; libraries?: Map<string, string>; vocabularies?: Map<string, string> };

// A length of time, short: 1 s, 45 s, 3 min 20 s, 2 h 5 min.
export function lasted(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ""}`;
  return `${Math.floor(s / 3600)} h${Math.round((s % 3600) / 60) ? ` ${Math.round((s % 3600) / 60)} min` : ""}`;
}

// The stages a vocabulary training and a corpus build report
// (wpt.depot.shared, wpt_depot.corpus), in words.
const STAGES: Record<string, string> = {
  ordering: "putting the texts in the build's order",
  "counting items": "counting the tokens of the conversations and tasks",
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

const sizePart = (n: number | null) => (n === null ? "all" : String(n));

// Where a job's result is: a corpus's page (its vocabulary by its name in
// names, else by its sha256), a recipe's page, and for a write that ended
// well, the library it changed, the decision it recorded or the library it made.
function resultLink(job: Job<unknown>, names?: Map<string, string>): string | null {
  const target = job.target;
  if (job.kind === "corpus" && job.dataset && target) return href(["download", job.dataset, names?.get(target.vocabulary) ?? target.vocabulary, sizePart(target.max_tokens)]);
  if (job.kind === "corpus" && job.dataset) return href(["download", job.dataset]);
  if (job.kind === "vocabulary" && job.recipe) return href(["tokenizers", job.recipe]);
  const a = (job.state === "done" ? job.answer ?? {} : {}) as Record<string, unknown>;
  if (typeof a.library === "string") return href(["texts"], { library: a.library });
  if (typeof a.decision === "string") return href(["decisions", a.decision]);
  if (typeof a.id === "string") return href(["texts"], { library: a.id });
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

// What a job that ended well gave, in words, with a link to its result.
// A vocabulary links to the next step from where it stands now: published
// (known.published), build a corpus; else publish it on its recipe's page.
function answered(job: Job<unknown>, known: Known = {}): ReactNode {
  const a = (job.answer ?? {}) as Record<string, unknown>;
  const n = (key: string) => (typeof a[key] === "number" ? (a[key] as number) : 0);
  const link = resultLink(job, known.vocabularies);
  if (job.kind === "corpus") return <>{takenWords(n("texts"), n("conversations"), n("tasks"))}, {plural(n("tokens") + n("item_tokens"), "token")} in all{n("item_tokens") ? ` (${n("tokens").toLocaleString("en")} of plain text, ${n("item_tokens").toLocaleString("en")} of conversations and tasks)` : ""}{link && <>: <a href={link}>download it</a></>}</>;
  if (job.kind === "vocabulary") {
    const on = typeof a.vocabulary === "string" ? known.published?.get(a.vocabulary) : undefined;
    if (on) return <>the vocabulary is trained and {on.name} publishes it: <a href={href(["download", on.id])}>build a corpus</a></>;
    return link && !shownNow(link) ? <>Trained: <a href={link}>publish it</a></> : null;
  }
  const library = typeof a.library === "string" ? a.library : null;
  const libraryLink = library ? <>: <a href={href(["texts"], { library })}>open {known.libraries?.get(library) ?? "the library"}</a></> : null;
  if ("joined" in a) return <>{plural(n("joined"), "text")} joined{n("left_out") ? `, ${n("left_out").toLocaleString("en")} not admitted left out` : ""}{leftWords(a.left)}{libraryLink}</>;
  if ("taken_out" in a) return <>{plural(n("taken_out"), "text")} taken out{n("not_held") ? `, ${n("not_held").toLocaleString("en")} it did not hold` : ""}{leftWords(a.left)}{libraryLink}</>;
  if ("decisions" in a && Array.isArray(a.decisions)) return <>{plural((a.decisions as { texts: number }[]).reduce((t, d) => t + d.texts, 0), "text")} relicensed{leftWords(a.left)}</>;
  if (typeof a.decision === "string") return <>{plural(n("texts"), "text")}{leftWords(a.left)}: <a href={href(["decisions", a.decision])} title={`decision ${a.decision}`}>see the decision and its texts</a></>;
  if (typeof a.id === "string") return <>the new library <a href={href(["texts"], { library: a.id })}>opens here</a>{leftWords(a.left)}</>;
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
export function JobLine({ job: started, onEnded, known }: { job: Job<unknown>; onEnded?: (job: Job<unknown>) => void; known?: Known }) {
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
  const { trained } = useRecipesAndVocabularies();
  const gave = job.state === "done" ? answered(job, { ...known, vocabularies: trained?.ok ? vocabularyNames(trained.data) : undefined }) : null;
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

// The datasets' published vocabularies and the libraries' names, for what the jobs' answers name by id.
export function useKnown(): Known {
  const { datasets } = useDatasets();
  const { libraries } = useLibraries();
  const { trained } = useRecipesAndVocabularies();
  return {
    vocabularies: trained?.ok ? vocabularyNames(trained.data) : undefined,
    published: new Map((datasets?.ok ? datasets.data : []).filter((d) => d.vocabulary).map((d) => [d.vocabulary!, { id: d.id, name: d.name }])),
    libraries: new Map(libraries.map((l) => [l.id, l.name])),
  };
}

// The next step of each dataset that has one, the first the page's main
// button: what a job could start from, or lead to (a recipe made first), apart
// from the jobs, each with its dataset and where it stands.
function NextSteps() {
  const { datasets } = useDatasets();
  const { recipes, trained } = useRecipesAndVocabularies();
  const corpora = useCorpora(datasets?.ok ? datasets.data.map((d) => d.id) : []);
  const steps = (datasets?.ok ? datasets.data : []).map((d) => ({ d, at: standing(d, recipes?.ok ? recipes.data : null, trained?.ok ? trained.data : null, corpora.get(d.id)) }))
    .filter((s) => s.at.next);
  if (!steps.length) return null;
  return (
    <Section title="Can start next">
      <ul className="plain next-steps">
        {steps.map(({ d, at }, i) => (
          <li key={d.id}><a className="cue" href={href(["datasets", d.id])}>{d.name}</a><StageMark at={at} /><StepButton at={at} primary={i === 0} /></li>
        ))}
      </ul>
    </Section>
  );
}

function Jobs() {
  const jobs = useJobs();
  const known = useKnown();
  const list = jobs?.ok ? jobs.data : [];
  const now = useNow(true); // every second: the states running and the "ago" of each job
  return (
    <>
      <PageHead title="Jobs"
        meta={jobs?.ok && list.length > 0 ? <><span>{list.filter((j) => j.state === "running").length.toLocaleString("en")} running</span><span>{list.filter((j) => j.state === "queued").length.toLocaleString("en")} waiting</span></> : undefined}
        chain={<>The long work the server runs, one job at a time: training a <a href={href(["tokenizers"])}>tokenizer recipe</a>'s vocabulary, building a <a href={href(["datasets"])}>dataset</a>'s corpus, and a write over many texts. A job that ended links to what it made. A job cannot be cancelled here; the command line can, on the depot's machine.</>} />
      {!jobs && <Skeleton lines={5} label="Reading the queue" />}
      {jobs && !jobs.ok && <Problems problems={jobs.problems} />}
      {jobs?.ok && (list.length === 0 ? (
        <p className="empty-state">No job yet.</p>
      ) : (
        <div className="table-scroll cards">
          <table className="ledger-table" aria-label="The jobs">
            <thead><tr><th scope="col">job</th><th scope="col">state</th><th scope="col">what it gave, or its progress</th><th scope="col">queued</th></tr></thead>
            <tbody>
              {list.map((j) => {
                const link = resultLink(j, known.vocabularies);
                return (
                  <tr key={j.id}>
                    <td className="wrap-cell" data-label="job">{link ? <a href={link}>{capital(j.title)}</a> : capital(j.title)}</td>
                    <td className="nowrap" data-label="state"><JobState job={j} now={now} /></td>
                    <td className="wrap-cell small" data-label="gave">{waiting(j.state) ? <JobProgress job={j} />
                      : j.problem ? <span className="not-allowed">{plainProblem(j.problem)}</span>
                      : j.state === "done" ? answered(j, known) : null}</td>
                    <td className="nowrap" data-label="queued" title={`queued ${when(j.created)} UTC${j.ended ? `, ended ${when(j.ended)} UTC` : ""}`}>{ago(j.created, now)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}
      {jobs?.ok && <NextSteps />}
    </>
  );
}

export const jobs: Page = { path: "jobs", title: "Jobs", group: "data", Component: () => <Jobs /> };
