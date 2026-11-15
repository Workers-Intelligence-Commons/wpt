// Jobs: the server's queue (GET /api/jobs), read again every few seconds:
// every job queued or running, in the order they run, then the latest that
// ended, each with what it does, its state, its progress and its answer or
// problem. Anyone starts a vocabulary training or a corpus build from the
// Tokenizers and Datasets pages; only the command line cancels a job (wpt
// depot jobs cancel ID). JobProgress says one job's progress in words, for
// the pages that show the job they started.

import { Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, type Answer, type Job } from "../api";
import { PageHead, Problems } from "../parts";
import { href } from "../route";
import type { Page } from "./index";

const EVERY_SECONDS = 3; // seconds between two readings of the queue

const when = (seconds: number | null) => (seconds ? new Date(seconds * 1000).toISOString().replace("T", " ").slice(0, 19) : "");

// A job's progress in words: its stage and the numbers it gave.
export function JobProgress({ job }: { job: Job<unknown> }) {
  const p = job.progress;
  if (!p) return <span className="aside">{job.state === "queued" ? "waiting its turn" : job.stages.at(-1)?.stage ?? "started"}</span>;
  const { stage, ...numbers } = p as Record<string, unknown>;
  return (
    <span>
      <span className="cue">{String(stage ?? "")}</span>{" "}
      <span className="aside">{Object.entries(numbers).map(([k, v]) => `${k} ${typeof v === "number" ? v.toLocaleString("en") : String(v)}`).join(", ")}</span>
    </span>
  );
}

function link(job: Job<unknown>) {
  if (job.kind === "corpus" && job.dataset) return <a href={href(["download", job.dataset])}>{job.title}</a>;
  if (job.kind === "vocabulary" && job.recipe) return <a href={href(["tokenizers", job.recipe])}>{job.title}</a>;
  return <span>{job.title}</span>;
}

// Read the queue now and again every few seconds.
export function useJobs(): Answer<Job<unknown>[]> | null {
  const [jobs, setJobs] = useState<Answer<Job<unknown>[]> | null>(null);
  useEffect(() => {
    let alive = true;
    const read = () => get<Job<unknown>[]>("/api/jobs").then((a) => { if (alive) setJobs(a); });
    read();
    const timer = window.setInterval(read, EVERY_SECONDS * 1000);
    return () => { alive = false; window.clearInterval(timer); };
  }, []);
  return jobs;
}

function Jobs() {
  const jobs = useJobs();
  return (
    <>
      <PageHead title="Jobs"
        chain={<>The long work the server runs for anyone, one job at a time: training a <a href={href(["tokenizers"])}>tokenizer recipe</a>'s vocabulary, building a <a href={href(["datasets"])}>dataset</a>'s corpus, and a write over many texts. Only <span className="ledger">wpt depot jobs cancel ID</span>, on the machine of the depot, cancels one.</>} />
      {!jobs && <Skeleton lines={5} label="Reading the queue" />}
      {jobs && !jobs.ok && <Problems problems={jobs.problems} />}
      {jobs?.ok && (jobs.data.length === 0 ? <p>No job yet.</p> : (
        <div className="table-scroll">
          <table className="ledger-table" aria-label="The jobs">
            <thead><tr><th scope="col">job</th><th scope="col">state</th><th scope="col">progress, answer or problem</th><th scope="col">queued (UTC)</th><th scope="col">ended (UTC)</th><th scope="col">id</th></tr></thead>
            <tbody>
              {jobs.data.map((j) => (
                <tr key={j.id}>
                  <td className="wide-cell">{link(j)}{j.after && j.state === "queued" && <span className="aside"> after {j.after}</span>}</td>
                  <td className={j.state === "failed" || j.state === "refused" ? "not-allowed" : ""}>{j.state}</td>
                  <td className="wide-cell small">{j.state === "running" || j.state === "queued" ? <JobProgress job={j} />
                    : j.problem ? <span className="not-allowed">{j.problem}</span>
                    : j.answer ? <span className="ledger cut" title={JSON.stringify(j.answer)}>{JSON.stringify(j.answer)}</span> : null}</td>
                  <td className="ledger nowrap">{when(j.created)}</td>
                  <td className="ledger nowrap">{when(j.ended)}</td>
                  <td className="ledger nowrap">{j.id}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </>
  );
}

export const jobs: Page = { path: "jobs", title: "Jobs", group: "data", Component: () => <Jobs /> };
