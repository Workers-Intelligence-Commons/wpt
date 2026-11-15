// Download: the token corpora of a dataset. A corpus is built for a dataset,
// one vocabulary the depot holds (by its sha256) and a size (a number of
// tokens, or all). #/download/ID lists the corpora the store holds of the
// dataset (GET /api/datasets/ID/download), one row per vocabulary and size,
// each with its own page, and holds the form of a new build: a trained
// vocabulary (the one the dataset publishes filled in when it has one), a
// size (the dataset's max_tokens filled in when it has one),
// then "Estimate" (GET /api/datasets/ID/estimate: tokens, store bytes, time,
// whether it fits under the store's cap), and only once an estimate is shown
// "Start build", which restates the estimate and asks for a confirmation
// before it queues the build (POST /api/datasets/ID/build, confirmed).
// Changing the vocabulary or the size clears the estimate.
// #/download/ID/VOCABULARY/SIZE is one corpus's page, at an address that
// never changes (GET /api/datasets/ID/download/VOCABULARY/SIZE): the token
// files and tables of the latest fresh corpus built from the dataset's
// current revision (its max_tokens and published vocabulary aside, which a
// build never reads), each
// with its size and sha256, streamed from the store; without one it says why
// (being built, its progress read again every few seconds; stale; not built;
// checking an erasure) and starts nothing. The texts are on the Export page.

import { Button, Dialog, Select, Skeleton, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, part, post, type Answer, type Job } from "../api";
import { Outcome, PageHead, plural, Problems, required, Section, type Dataset } from "../parts";
import { href } from "../route";
import { JobProgress } from "./jobs";
import type { Page } from "./index";

type File = { name: string; bytes: number | null; sha256: string };
type Vocabulary = { sha256: string; recipe: string | null; version: number | null };
type Corpus = { id: string; built_at: string; texts?: number; total_tokens?: number; revision: number; stale?: { erasures: string[] } | null; deleted?: { why: string } | null };
type State = "built" | "checking an erasure" | "being built" | "stale" | "not built" | "no store";
type Row = { vocabulary: Vocabulary; max_tokens: number | null; size: string; link: string; state: State; corpus: Corpus | null };
type Listed = { state: "listed" | "checking an erasure" | "no store"; dataset: Dataset; job: Job<unknown> | null; corpora: Row[]; erasures?: string[]; problem?: string };
type Shown = {
  state: State; dataset: Dataset; vocabulary: Vocabulary; max_tokens: number | null; size: string; job: Job<unknown> | null;
  corpus: Corpus | null; files?: File[]; erasures?: string[]; problem?: string;
};
type Trained = { sha256: string; recipe: string; name: string | null; version: number; texts: number; made_at: string };
type Estimate = {
  tokens: number; plain_tokens: number; item_tokens: number; texts: number; bytes: number; token_bytes: number; table_bytes: number;
  seconds: number | null; workers: number; order_texts: number; cap: number | null; others: number | null; fits: boolean; evicts: number | null;
  sample: { texts: number; tokenized: number; bytes: number; seconds: number }; warnings: string[]; store: string | null;
};

const UNITS = ["bytes", "KB", "MB", "GB", "TB", "PB"];
export const bytes = (n: number) => {
  let value = n, unit = 0;
  while (value >= 1000 && unit < UNITS.length - 1) { value /= 1000; unit += 1; }
  return unit === 0 ? plural(n, "byte") : `${value.toFixed(value < 10 ? 1 : 0)} ${UNITS[unit]}`;
};
export const duration = (seconds: number) => {
  if (seconds < 60) return "under a minute";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return plural(minutes, "minute");
  const hours = Math.floor(minutes / 60);
  return `${plural(hours, "hour")}${minutes % 60 ? ` ${plural(minutes % 60, "minute")}` : ""}`;
};
const sizeWords = (maxTokens: number | null) => (maxTokens === null ? "every token" : `at most ${plural(maxTokens, "token")}`);
const vocabularyWords = (v: Vocabulary) => (v.recipe ? `${v.recipe} version ${v.version}` : "a vocabulary the depot does not hold");
const fileSize = (n: number | null) => (n === null ? "size unknown" : bytes(n));

// What an estimate says, in words, for the form and its confirmation.
function EstimateFacts({ e }: { e: Estimate }) {
  return (
    <dl className="facts">
      <dt className="aside">tokens</dt><dd>about {plural(e.tokens, "token")} ({e.plain_tokens.toLocaleString("en")} of plain text, {e.item_tokens.toLocaleString("en")} of conversations and tasks), about {plural(e.texts, "text")}</dd>
      <dt className="aside">store</dt><dd>about {bytes(e.bytes)} ({bytes(e.token_bytes)} of token files, {bytes(e.table_bytes)} of tables)</dd>
      <dt className="aside">time</dt><dd>{e.seconds === null ? "unknown: the sample tokenized nothing" : `about ${duration(e.seconds)} of tokenizing on this machine, with ${plural(e.workers, "tokenizing process", "tokenizing processes")}`}, plus the pass that orders the dataset's {plural(e.order_texts, "text")}, which every build makes first and which this estimate does not measure</dd>
      <dt className="aside">cap</dt><dd>{e.cap === null ? "no cap: the store keeps every corpus" : e.fits
        ? <>fits under the cap of {bytes(e.cap)}{e.others !== null && <>; the other corpora take {bytes(e.others)}{e.evicts ? <>, and the build deletes about {bytes(e.evicts)} of the corpora downloaded least recently as it needs room</> : null}</>}</>
        : <span className="not-allowed">more than the cap of {bytes(e.cap)} on its own: the build would be refused; choose a smaller size</span>}</dd>
      {e.store && <><dt className="aside">store</dt><dd className="not-allowed">{e.store}</dd></>}
      <dt className="aside">basis</dt><dd className="small">An estimate from a sample of {plural(e.sample.texts, "text")}, {plural(e.sample.tokenized, "text")} ({bytes(e.sample.bytes)}) tokenized in {e.sample.seconds} s.{e.warnings.map((w) => <span key={w}> {w}.</span>)}</dd>
    </dl>
  );
}

// The form of a new build: a vocabulary, a size, an estimate, then a confirmed start.
function NewBuild({ dataset, onStarted }: { dataset: Dataset; onStarted: () => void }) {
  const [trained, setTrained] = useState<Answer<Trained[]> | null>(null);
  const [vocabulary, setVocabulary] = useState("");
  const [all, setAll] = useState(dataset.max_tokens === null);
  const [limit, setLimit] = useState(dataset.max_tokens ? String(dataset.max_tokens) : "");
  // the last estimate, with the vocabulary and size it was asked for: shown only while they are the form's
  const [asked, setAsked] = useState<{ answer: Answer<Estimate>; vocabulary: string; size: string } | null>(null);
  const [estimating, setEstimating] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [started, setStarted] = useState<Answer<Job<unknown>> | null>(null);
  useEffect(() => {
    get<Trained[]>("/api/vocabularies").then((a) => {
      setTrained(a);
      if (a.ok && a.data.length) setVocabulary((a.data.find((v) => v.sha256 === dataset.vocabulary) ?? a.data.find((v) => v.name === dataset.tokenizer) ?? a.data[0]).sha256);
    });
  }, [dataset.tokenizer, dataset.vocabulary]);
  const size = all ? "all" : limit.trim();
  const valid = vocabulary !== "" && (all || (Number.isInteger(Number(limit)) && Number(limit) >= 1));
  const estimate = asked && asked.vocabulary === vocabulary && asked.size === size ? asked.answer : null;
  const clear = () => { setAsked(null); setStarted(null); setConfirming(false); };
  const ask = async () => {
    const wanted = { vocabulary, size };
    setEstimating(true);
    const answer = await get<Estimate>(`/api/datasets/${part(dataset.id)}/estimate?vocabulary=${part(wanted.vocabulary)}&max_tokens=${part(wanted.size)}`);
    setAsked({ answer, ...wanted });
    setEstimating(false);
  };
  const start = async () => {
    if (!estimate?.ok) return;
    const a = await post<Job<unknown>>(`/api/datasets/${part(dataset.id)}/build`, { vocabulary, max_tokens: all ? "all" : Number(limit), confirmed: true, estimate: estimate.data });
    setStarted(a);
    setConfirming(false);
    if (a.ok) onStarted();
  };
  if (!trained) return <Skeleton lines={3} label="Reading the vocabularies" />;
  if (!trained.ok) return <Problems problems={trained.problems} />;
  if (trained.data.length === 0) return <p className="small">No vocabulary is trained yet: train one on its <a href={href(["tokenizers"])}>tokenizer recipe</a>'s page first. A build never trains one.</p>;
  const chosen = trained.data.find((v) => v.sha256 === vocabulary);
  return (
    <form onSubmit={(e) => { e.preventDefault(); if (valid) ask(); }}>
      <div className="fields">
        <Select label={required("Vocabulary")} required value={vocabulary} onChange={(e) => { setVocabulary(e.target.value); clear(); }}
          hint={dataset.vocabulary ? "A vocabulary the depot holds, trained on its tokenizer recipe's page. The one the dataset publishes, which training takes, is filled in." : "A vocabulary the depot holds, trained on its tokenizer recipe's page. The dataset publishes none yet: its page sets the one training takes."}>
          {trained.data.map((v) => <option key={v.sha256} value={v.sha256}>{v.name ?? v.recipe} version {v.version}, {v.sha256.slice(0, 12)}</option>)}
        </Select>
        <Select label={required("Size")} value={all ? "all" : "limit"} onChange={(e) => { setAll(e.target.value === "all"); clear(); }}
          hint="Every token the dataset's rules take, or at most a number of tokens, conversations and tasks first.">
          <option value="limit">at most a number of tokens</option>
          <option value="all">all</option>
        </Select>
        {!all && <TextInput label={required("Most tokens")} required type="number" min="1" step="1" value={limit} onChange={(e) => { setLimit(e.target.value); clear(); }}
          hint={dataset.max_tokens ? `The dataset's max_tokens, ${dataset.max_tokens.toLocaleString("en")}, is filled in.` : "The dataset names no max_tokens."} />}
      </div>
      <Button type="submit" variant={estimate?.ok ? "quiet" : "primary"} disabled={!valid || estimating}>{estimating ? "Estimating" : "Estimate"}</Button>
      {estimate && !estimate.ok && <Problems problems={estimate.problems} />}
      {estimate?.ok && (
        <>
          <EstimateFacts e={estimate.data} />
          <Button variant="primary" disabled={!estimate.data.fits} onClick={() => setConfirming(true)}>Start build</Button>
        </>
      )}
      <Outcome answer={started} done={(d) => `Queued: ${d.title}.`} />
      <Dialog open={confirming} onClose={() => setConfirming(false)} title="Start this build?">
        {confirming && estimate?.ok && chosen && (
          <>
            <p>The corpus of the dataset {dataset.name} with the vocabulary {chosen.name ?? chosen.recipe} version {chosen.version} (<span className="ledger">{chosen.sha256.slice(0, 12)}</span>), {sizeWords(all ? null : Number(limit))}. It runs as a job of the <a href={href(["jobs"])}>queue</a>, one job at a time, and only the command line cancels it.</p>
            <EstimateFacts e={estimate.data} />
            <Button variant="primary" onClick={start}>Confirm and start the build</Button>{" "}
            <Button variant="quiet" onClick={() => setConfirming(false)}>Cancel</Button>
          </>
        )}
      </Dialog>
    </form>
  );
}

function Listing({ id }: { id: string }) {
  const [listed, setListed] = useState<Answer<Listed> | null>(null);
  const load = useCallback(() => { get<Listed>(`/api/datasets/${part(id)}/download`).then(setListed); }, [id]);
  useEffect(load, [load]);
  useEffect(() => {  // a build or an erasure's marking under way: read again until it ends
    if (!listed?.ok || (!listed.data.job && listed.data.state !== "checking an erasure")) return;
    const timer = window.setTimeout(load, 3000);
    return () => window.clearTimeout(timer);
  }, [listed, load]);
  if (!listed) return <Skeleton lines={6} label="Reading the dataset's corpora" />;
  if (!listed.ok) return <Problems problems={listed.problems} />;
  const { state, dataset, corpora, job } = listed.data;
  return (
    <>
      <PageHead title={`Download ${dataset.name}`} meta={<><span>revision {dataset.revision}</span><span>{plural(corpora.length, "corpus", "corpora")}</span></>}
        chain={<>The token corpora of the <a href={href(["datasets", dataset.id])}>dataset {dataset.name}</a>, one per vocabulary and size, each at an address that stays the same whatever is built next. Its texts, with no tokenizer, are on <a href={href(["export", dataset.id])}>its export page</a>.</>} />
      {state === "no store" && <Problems problems={[listed.data.problem ?? "the store cannot be read here"]} />}
      {state === "checking an erasure" && <p>Checking an erasure: the corpora holding a text of {listed.data.erasures?.join(", ")} are being marked stale. Nothing is sent until that ends.</p>}
      {job && <p className="small">Its corpus job: {job.title}, {job.state}{job.state === "running" && <>, <JobProgress job={job} /></>}.</p>}
      <Section title="Corpora built">
        {corpora.length === 0 ? <p className="small">No corpus of the store is built from it.</p> : (
          <div className="table-scroll">
            <table className="ledger-table" aria-label="The corpora of the dataset">
              <thead><tr><th scope="col">vocabulary</th><th scope="col">size</th><th scope="col">state</th><th scope="col" className="number">tokens of plain text</th><th scope="col">corpus</th></tr></thead>
              <tbody>
                {corpora.map((c) => (
                  <tr key={c.link}>
                    <td><a href={href(["download", dataset.id, c.vocabulary.sha256, c.size])}>{vocabularyWords(c.vocabulary)}</a> <span className="ledger aside">{c.vocabulary.sha256.slice(0, 12)}</span></td>
                    <td>{sizeWords(c.max_tokens)}</td>
                    <td className={c.state === "stale" ? "not-allowed" : ""}>{c.state}</td>
                    <td className="number ledger">{c.corpus?.total_tokens?.toLocaleString("en") ?? ""}</td>
                    <td className="ledger nowrap" title={c.corpus?.id}>{c.corpus?.id.slice(0, 12) ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
      <Section title="New build">
        <p className="small aside">A build names a trained vocabulary and a size. It starts only after its estimate is shown and confirmed.</p>
        <NewBuild dataset={dataset} onStarted={load} />
      </Section>
    </>
  );
}

function OneCorpus({ id, vocabulary, size }: { id: string; vocabulary: string; size: string }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  const base = `/api/datasets/${part(id)}/download/${part(vocabulary)}/${part(size)}`;
  const load = useCallback(() => { get<Shown>(base).then(setShown); }, [base]);
  useEffect(load, [load]);
  useEffect(() => {  // a build or an erasure's marking under way: read again until it ends
    if (!shown?.ok || (shown.data.state !== "being built" && shown.data.state !== "checking an erasure")) return;
    const timer = window.setTimeout(load, 3000);
    return () => window.clearTimeout(timer);
  }, [shown, load]);
  if (!shown) return <Skeleton lines={6} label="Reading the corpus" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const { state, dataset, job, corpus } = shown.data;
  return (
    <>
      <PageHead title={`Download ${dataset.name}`}
        meta={<><span>{vocabularyWords(shown.data.vocabulary)}</span><span>{sizeWords(shown.data.max_tokens)}</span><span className="cue">{state}</span>{corpus && <span className="ledger" title={corpus.id}>corpus {corpus.id.slice(0, 12)}</span>}</>}
        chain={<>The corpus of the <a href={href(["datasets", dataset.id])}>dataset {dataset.name}</a> as it is now, with the vocabulary <span className="ledger">{vocabulary.slice(0, 12)}</span>, {sizeWords(shown.data.max_tokens)}: this address stays the same whatever is built next. <a href={href(["download", dataset.id])}>Its other corpora, and a new build</a>.</>} />
      {state === "no store" && <Problems problems={[shown.data.problem ?? "the store cannot be read here"]} />}
      {state === "checking an erasure" && <p>Checking an erasure: the corpora holding a text of {shown.data.erasures?.join(", ")} are being marked stale. Nothing is sent until that ends.</p>}
      {state === "being built" && job && <p>Being built: {job.title}, {job.state}{job.state === "running" && <>, <JobProgress job={job} /></>}. Nothing to download until it ends.</p>}
      {state === "stale" && <p>Stale: an admitted erasure reached a text of its latest corpus{corpus?.stale ? <> ({corpus.stale.erasures.join(", ")})</> : null}. Nothing to download until it is built again, which nothing starts by itself.</p>}
      {state === "not built" && <p>Not built{corpus?.deleted ? <>: its latest corpus was deleted ({corpus.deleted.why})</> : <> from the dataset as it is now</>}. Nothing to download; <a href={href(["download", dataset.id])}>a new build</a> starts one, after its estimate.</p>}
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
                      <td className="number ledger">{fileSize(f.bytes)}</td>
                      <td className="ledger small">{f.sha256}</td>
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
  Component: ({ rest }) => (rest[0] && rest[1] && rest[2] ? <OneCorpus key={rest.join("/")} id={rest[0]} vocabulary={rest[1]} size={rest[2]} />
    : rest[0] ? <Listing key={rest[0]} id={rest[0]} /> : <p>A dataset's download page is reached from <a href={href(["datasets"])}>its dataset</a>.</p>),
};
