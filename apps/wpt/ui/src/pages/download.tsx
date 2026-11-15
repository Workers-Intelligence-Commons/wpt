// Download: #/download/ID is a dataset's download page, at an address that
// never changes (GET /api/datasets/ID/download). It gives the latest fresh
// corpus built from the dataset's current revision: its token files and
// tables, each with its size and sha256, streamed from the store, and its
// texts as numbered parts, each in JSONL compressed with zstd and in
// Parquet, made when asked and the same bytes every time, with each part's
// size once it was sent whole. Without such a corpus it says why, with
// nothing to download: being built (the job's progress, read again every few
// seconds), stale (an erasure reached one of its texts; nothing builds it
// again by itself) or not built, with the button that queues the build.

import { Button, Skeleton } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, part, post, type Answer, type Job } from "../api";
import { Outcome, PageHead, plural, Problems, Section, type Dataset } from "../parts";
import { href } from "../route";
import { JobProgress } from "./jobs";
import type { Page } from "./index";

type File = { name: string; bytes: number | null; sha256: string };
type Form = { name: string; bytes: number | null; sha256: string | null };
type TextPart = { part: number; texts: number | null; text_bytes: number | null; forms: Record<string, Form> };
type Shown = {
  state: "built" | "checking an erasure" | "being built" | "stale" | "not built" | "no store"; erasures?: string[]; dataset: Dataset; job: Job<unknown> | null; before: Job<unknown> | null;
  corpus: { id: string; built_at: string; texts?: number; total_tokens?: number; stale?: { erasures: string[] }; deleted?: { why: string } } | null;
  files?: File[]; parts?: TextPart[]; problem?: string;
};

const size = (n: number | null) => (n === null ? "size known once sent" : plural(n, "byte"));

function Download({ id }: { id: string }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  const [answer, setAnswer] = useState<Answer<Job<unknown>> | null>(null);
  const load = useCallback(() => { get<Shown>(`/api/datasets/${part(id)}/download`).then(setShown); }, [id]);
  useEffect(load, [load]);
  useEffect(() => {  // a build or an erasure's marking under way: read again until it ends
    if (!shown?.ok || (shown.data.state !== "being built" && shown.data.state !== "checking an erasure")) return;
    const timer = window.setTimeout(load, 3000);
    return () => window.clearTimeout(timer);
  }, [shown, load]);
  if (!shown) return <Skeleton lines={6} label="Reading the dataset's download" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const { state, dataset, job, before, corpus } = shown.data;
  const base = `/api/datasets/${part(id)}/download`;
  // A deleted corpus is built again by its id, which the build checks; a stale one is built anew without the erased texts.
  const again = state === "not built" && corpus?.deleted ? { corpus: corpus.id } : {};
  const build = async () => { setAnswer(await post<Job<unknown>>(`/api/datasets/${part(id)}/build`, again)); load(); };
  return (
    <>
      <PageHead title={`Download ${dataset.name}`}
        meta={<><span>revision {dataset.revision}</span><span className="cue">{state}</span>{corpus && <span className="ledger" title={corpus.id}>corpus {corpus.id.slice(0, 12)}</span>}</>}
        chain={<>The corpus of the <a href={href(["datasets", dataset.id])}>dataset {dataset.name}</a> as it is now: this address stays the same whatever is built next.</>}
        actions={(state === "stale" || state === "not built") && <Button variant="primary" onClick={build}>{state === "stale" || corpus?.deleted ? "Build it again" : "Build it"}</Button>} />
      <Outcome answer={answer} done={(d) => `Queued: ${d.title}.`} />
      {state === "no store" && <Problems problems={[shown.data.problem ?? "the store cannot be read here"]} />}
      {state === "checking an erasure" && <p>Checking an erasure: the corpora holding a text of {shown.data.erasures?.join(", ")} are being marked stale. Nothing is sent until that ends.</p>}
      {state === "being built" && job && (
        <p>Being built: {before && before.state !== "done" && <>first {before.title}, {before.state}, <JobProgress job={before} />; then </>}{job.title}, {job.state}{job.state === "running" && <>, <JobProgress job={job} /></>}. Nothing to download until it ends.</p>
      )}
      {state === "stale" && <p>Stale: an admitted erasure reached a text of its latest corpus{corpus?.stale ? <> ({corpus.stale.erasures.join(", ")})</> : null}. Nothing to download until it is built again, which nothing starts by itself.</p>}
      {state === "not built" && <p>Not built{corpus?.deleted ? <>: its latest corpus was deleted ({corpus.deleted.why})</> : <> from its current revision</>}. Nothing to download.</p>}
      {state === "built" && corpus && (
        <>
          <p className="small">{plural(corpus.texts ?? 0, "text")}, {plural(corpus.total_tokens ?? 0, "token")} of plain text, built {corpus.built_at.replace("T", " ").slice(0, 16)} UTC.</p>
          <Section title="Tokens: the corpus's files">
            <div className="table-scroll">
              <table className="ledger-table" aria-label="The corpus's files">
                <thead><tr><th scope="col">file</th><th scope="col" className="number">size</th><th scope="col">sha256</th></tr></thead>
                <tbody>
                  {shown.data.files!.map((f) => (
                    <tr key={f.name}>
                      <td className="ledger"><a href={`${base}/files/${part(f.name)}`} download={f.name}>{f.name}</a></td>
                      <td className="number ledger">{size(f.bytes)}</td>
                      <td className="ledger small">{f.sha256}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
          <Section title="Text: its texts in the corpus's order, in parts">
            <p className="small aside">Each part is made as it is sent, the same bytes every time. At most two parts are sent at once, one to each visitor.</p>
            <div className="table-scroll">
              <table className="ledger-table" aria-label="The parts of its texts">
                <thead><tr><th scope="col" className="number">part</th><th scope="col" className="number">texts</th><th scope="col">JSONL, zstd</th><th scope="col">Parquet</th></tr></thead>
                <tbody>
                  {shown.data.parts!.map((p) => (
                    <tr key={p.part}>
                      <td className="number ledger">{p.part}</td>
                      <td className="number ledger">{(p.texts ?? 0).toLocaleString("en")}</td>
                      {["jsonl.zst", "parquet"].map((form) => (
                        <td key={form} className="ledger small">
                          <a href={`${base}/text/${part(p.forms[form].name)}`} download={p.forms[form].name}>{p.forms[form].name}</a>{" "}
                          <span className="aside">{size(p.forms[form].bytes)}</span>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </>
      )}
    </>
  );
}

export const download: Page = {
  path: "download", title: "Download", group: null,
  Component: ({ rest }) => (rest[0] ? <Download key={rest[0]} id={rest[0]} /> : <p>A dataset's download page is reached from <a href={href(["datasets"])}>its dataset</a>.</p>),
};
