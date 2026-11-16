// Download: the token corpora of a dataset. A corpus is built for a dataset,
// one vocabulary the depot holds (by its sha256) and a size (a number of
// tokens, or all). #/download/ID lists the corpora the store holds of the
// dataset (GET /api/datasets/ID/download), one row per vocabulary and size,
// each with its own page, and holds the form of a new build: a trained
// vocabulary (the one the dataset publishes filled in when it has one), a
// size (the dataset's max_tokens filled in when it has one),
// then "Estimate" (GET /api/datasets/ID/estimate: tokens, store bytes, time,
// whether it fits under the store's cap), and only once an estimate is shown
// "Confirm and build", right under the estimate's warnings and above its
// details, which the page scrolls to, so it is in view without scrolling;
// it restates the estimate and queues the build (POST
// /api/datasets/ID/build, confirmed).
// Changing the vocabulary or the size clears the estimate. At a size, the
// estimate's tokens are a forecast that follows the build's order, where a
// share stops at its first text that does not fit (wpt_depot.estimate), so
// its headline, above "Confirm and build", says what will come out: the
// tokens in all, and each share that comes out more than UNEVEN off its
// proportion, with the smallest size where every share comes within it
// (even_size) and a link to change the shares.
// A build started here is followed at the top of the page.
// #/download/ID/VOCABULARY/SIZE is one corpus's page (VOCABULARY: its
// recipe and version, as mix-tok-v1, or its sha256), at an address that
// never changes (GET /api/datasets/ID/download/VOCABULARY/SIZE): the token
// files and tables of the latest fresh corpus built from the dataset's
// current revision (its max_tokens and published vocabulary aside, which a
// build never reads), each
// with its size and sha256, streamed from the store; without one it says why
// (being built, its progress read again every few seconds; stale; not built;
// checking an erasure) and starts nothing. A built corpus compares the
// shares asked with those it got, and when they are off asks the estimate
// of its own vocabulary and size for the size where they come even, the
// same number the build form offers, and each file downloads under the name
// the server gives it (dataset, vocabulary, size and role). The texts are on
// the Export page.

import { Button, ButtonLink, Select, Skeleton, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useRef, useState } from "react";
import { get, part, post, type Answer, type Job } from "../api";
import { PageHead, plural, Problems, required, Section, takenWords, type Dataset } from "../parts";
import { href } from "../route";
import { JobLine } from "./jobs";
import type { Page } from "./index";

type File = { name: string; download: string; bytes: number | null; sha256: string; role: string; tokens: number | null };
type Vocabulary = { sha256: string; recipe: string | null; version: number | null };
type Corpus = {
  id: string; built_at: string; texts?: number; conversations?: number; tasks?: number; total_tokens?: number; items_tokens?: number | null;
  shares_tokens?: { tag: string; tokens: number }[] | null; revision: number; stale?: { erasures: string[] } | null; deleted?: { why: string } | null;
};
type State = "built" | "checking an erasure" | "being built" | "stale" | "not built" | "no store";
type Row = { vocabulary: Vocabulary; max_tokens: number | null; size: string; link: string; state: State; corpus: Corpus | null };
type Listed = { state: "listed" | "checking an erasure" | "no store"; dataset: Dataset; job: Job<unknown> | null; corpora: Row[]; erasures?: string[]; problem?: string };
type Shown = {
  state: State; dataset: Dataset; vocabulary: Vocabulary; max_tokens: number | null; size: string; job: Job<unknown> | null;
  corpus: Corpus | null; files?: File[]; erasures?: string[]; problem?: string;
};
type Trained = { sha256: string; recipe: string; name: string | null; version: number; texts: number; made_at: string };
type ShareForecast = { tag: string; tokens: number; part: number };
type Estimate = {
  tokens: number; plain_tokens: number; item_tokens: number; item_share: number; share_tokens: ShareForecast[] | null; forecast: boolean | null;
  even_size: number | null; texts: number; plain_texts: number; item_texts: number; items_held: number; bytes: number; token_bytes: number; table_bytes: number;
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
  if (seconds < 60) return "less than 1 minute";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return plural(minutes, "minute");
  const hours = Math.floor(minutes / 60);
  return `${plural(hours, "hour")}${minutes % 60 ? ` ${plural(minutes % 60, "minute")}` : ""}`;
};
const sizeWords = (maxTokens: number | null) => (maxTokens === null ? "every token" : `at most ${plural(maxTokens, "token")}`);
const vocabularyWords = (v: Vocabulary) => (v.recipe ? `${v.recipe} version ${v.version}` : "a vocabulary the depot does not hold");
const percent = (n: number) => `${Math.round(n * 1000) / 10}\u00a0%`; // a no-break space: the number and % never part
const fileSize = (n: number | null) => (n === null ? "size unknown" : bytes(n));

// the tokens of a corpus's conversations and tasks
const itemTokens = (c: Corpus) => c.items_tokens ?? null;

const QUICK = 100_000; // tokens a new build is filled in with when the dataset names no size: a quick test

// UNEVEN: how far, in points, a share's part of the plain text may be from
// its proportion and still count as even (wpt_depot.estimate uses the same)
const UNEVEN = 0.1;

// The shares of a forecast that come out short: more than UNEVEN under their proportion of the plain text (the
// others then hold more than theirs, which follows from it and is said by the numbers).
function offShares(e: Estimate): ShareForecast[] {
  const shares = e.share_tokens ?? [];
  const parts = shares.reduce((t, s) => t + s.part, 0);
  if (!parts) return [];
  return shares.filter((s) => s.part > 0 && (!e.plain_tokens || s.tokens / e.plain_tokens < s.part / parts - UNEVEN));
}

// The headline of an estimate at a size, above "Confirm and build": what will come out, and when a share comes out
// off its proportion, why and what to do. onSize: asks for the estimate of another size.
function Forecast({ e, dataset, asked, onSize }: { e: Estimate; dataset: Dataset; asked: number | null; onSize: (tokens: number) => void }) {
  const off = asked === null ? [] : offShares(e);
  const items = e.item_tokens > 0 ? `conversations and tasks about ${e.item_tokens.toLocaleString("en")}` : null;
  const parts = [...(e.share_tokens ?? []).map((s) => `${s.tag} about ${s.tokens.toLocaleString("en")}`), ...(items ? [items] : [])];
  const head = asked === null ? `About ${plural(e.tokens, "token")}` : `About ${plural(e.tokens, "token")} of the ${asked.toLocaleString("en")} asked`;
  const empty = off.filter((s) => s.tokens === 0);
  return (
    <div className={off.length ? "warning" : undefined}>
      <p>
        <strong>{head}{off.length ? `, and ${off.map((s) => `${s.tag} will likely get ${s.tokens ? `only about ${s.tokens.toLocaleString("en")}` : "0"} tokens of its ${s.part.toLocaleString("en")}`).join(" and ")}` : ""}.</strong>
        {parts.length > 1 && <> <span className="aside">{parts.join(", ")}.</span></>}
      </p>
      {off.length > 0 && (
        <>
          <p className="small">Texts are never cut: a share stops at its first text that does not fit in what is left of its part, in the dataset's fixed order. Here {empty.length ? `an early ${empty.map((s) => s.tag).join(" or ")} text is` : "a long text comes"} too long for {off.length === 1 ? "its part" : "their parts"}, so the corpus holds fewer tokens than asked{e.item_tokens ? ", and the conversations take a bigger part of it" : ""}.</p>
          <ul className="plain small choices">
            {e.even_size !== null && asked !== null && e.even_size > asked && <li><Button variant="quiet" onClick={() => onSize(e.even_size!)}>Use {e.even_size.toLocaleString("en")} tokens</Button> the smallest size where every share's part of the plain text comes within {Math.round(UNEVEN * 100)} percentage points of the part it asks (a share of 60 % gets between 50 and 70 %), by the same forecast.</li>}
            {!dataset.upstream && <li><a href={href(["datasets", dataset.id], { edit: "1" })}>Change the shares</a> of the dataset, lowering {off.map((s) => s.tag).join(" and ")}: a new revision, so the estimate is asked again after.</li>}
            <li>Or keep this size: <q>Confirm and build</q> below builds this mix, and the corpus page shows the shares it got.</li>
          </ul>
        </>
      )}
    </div>
  );
}

// What an estimate says, in words: what the size holds of each kind (the
// conversations and tasks, then each share's forecast beside its part), the
// store's bytes, the time, the cap, and what it was estimated from.
function EstimateFacts({ e, limited }: { e: Estimate; limited: boolean }) {
  const items = e.item_share > 0;
  return (
    <dl className="facts">
      <dt className="aside">holds</dt>
      <dd>
        about {plural(e.tokens, "token")} from about {e.item_texts || e.items_held ? <>{plural(e.plain_texts, "plain text")} and {e.item_texts.toLocaleString("en")} of the dataset's {plural(e.items_held, "conversation or task", "conversations and tasks")}</> : plural(e.plain_texts, "plain text")}:
        <ul className="plain kinds">
          {items && <li><span>conversations and tasks{limited ? <span className="aside"> (their part: {percent(e.item_share)} of the size, as in the corpus of every token)</span> : <span className="aside"> (every one)</span>}</span> <span className="ledger">about {e.item_tokens.toLocaleString("en")}</span></li>}
          {e.share_tokens?.length ? e.share_tokens.map((s) => (
            <li key={s.tag}><span>plain text tagged {s.tag}{limited ? <span className="aside"> (its part: about {s.part.toLocaleString("en")})</span> : null}</span> <span className="ledger">about {s.tokens.toLocaleString("en")}</span></li>
          )) : <li><span>plain text</span> <span className="ledger">about {e.plain_tokens.toLocaleString("en")}</span></li>}
        </ul>
        {limited && e.forecast === false && <p className="small aside">The dataset is long, so past its first 50,000 texts in order a part counts at its share of the size, not as the build will take it.</p>}
      </dd>
      <dt className="aside">store</dt><dd>about {bytes(e.bytes)}: {bytes(e.token_bytes)} of token files, and {bytes(e.table_bytes)} for the list of its texts with their credits and licenses</dd>
      <dt className="aside">time</dt><dd>{e.seconds === null ? "unknown: the sample tokenized nothing" : `${e.seconds < 60 ? "" : "about "}${duration(e.seconds)} of tokenizing on this machine`}. Before it, the build reads the list of the dataset's {plural(e.order_texts, "text")} once to put them in order; that pass reads no text's content, and this estimate does not time it.</dd>
      <dt className="aside">cap</dt><dd>{e.cap === null ? "no cap: the store keeps every corpus" : e.fits
        ? <>fits under the store's cap of {bytes(e.cap)}{e.others !== null && <>; the other corpora take {bytes(e.others)}{e.evicts ? <>, and the build deletes about {bytes(e.evicts)} of the corpora downloaded least recently to make room</> : null}</>}</>
        : <span className="not-allowed">more than the store's cap of {bytes(e.cap)} on its own: the build would be refused; choose a smaller size</span>}</dd>
      {e.store && <><dt className="aside">store</dt><dd className="not-allowed">{e.store}</dd></>}
      <dt className="aside">basis</dt><dd className="small">Estimated from {plural(e.sample.tokenized, "text")} ({bytes(e.sample.bytes)}) tokenized in {e.sample.seconds} s, out of a sample of {plural(e.sample.texts, "text")}; the build's counts can differ a little.{e.warnings.map((w) => <span key={w}> {w}.</span>)}</dd>
    </dl>
  );
}

// The form of a new build: a vocabulary, a size, an estimate, then the confirmed start under it, followed to its end.
// onDataset: after the size is made the dataset's default, a new revision of it, to read it again.
function NewBuild({ dataset, open, startSize, onStarted, onDataset }: { dataset: Dataset; open: Job<unknown> | null; startSize: string | null; onStarted: (job: Job<unknown>) => void; onDataset: () => void }) {
  const [trained, setTrained] = useState<Answer<Trained[]> | null>(null);
  const [vocabulary, setVocabulary] = useState("");
  const [all, setAll] = useState(false);
  const given = startSize && Number.isInteger(Number(startSize)) && Number(startSize) >= 1 ? startSize : null;
  const [limit, setLimit] = useState(given ?? String(dataset.max_tokens ?? QUICK));
  // the last estimate, with the vocabulary and size it was asked for: shown only while they are the form's
  const [asked, setAsked] = useState<{ answer: Answer<Estimate>; vocabulary: string; size: string } | null>(null);
  const [estimating, setEstimating] = useState(false);
  const [started, setStarted] = useState<Answer<Job<unknown>> | null>(null);
  const [starting, setStarting] = useState(false); // the build asked for, until the server answers: one click queues one build
  const [defaulted, setDefaulted] = useState<Answer<unknown> | null>(null); // the answer to making the size the dataset's default
  const [defaulting, setDefaulting] = useState(false);
  // the size typed as the dataset's default size, a new revision of it, the rest of its row as read
  const makeDefault = async () => {
    setDefaulting(true);
    const a = await post(`/api/datasets/${part(dataset.id)}`, {
      name: dataset.name, libraries: dataset.libraries.map((r) => ({ depot: r.depot, library: r.library })),
      shares: dataset.shares, max_tokens: Number(limit), vocabulary: dataset.vocabulary, revision: dataset.revision,
    });
    setDefaulting(false);
    setDefaulted(a);
    if (a.ok) onDataset();
  };
  useEffect(() => {
    get<Trained[]>("/api/vocabularies").then((a) => {
      setTrained(a);
      if (a.ok && a.data.length) setVocabulary((a.data.find((v) => v.sha256 === dataset.vocabulary) ?? a.data.find((v) => v.name === dataset.tokenizer) ?? a.data[0]).sha256);
    });
  }, [dataset.tokenizer, dataset.vocabulary]);
  const size = all ? "all" : limit.trim();
  const estimatedGiven = useRef(false); // the size the address named is estimated once, as soon as a vocabulary is chosen
  useEffect(() => {
    if (!given || estimatedGiven.current || !vocabulary) return;
    estimatedGiven.current = true;
    ask(given);
  }, [given, vocabulary]);
  const valid = vocabulary !== "" && (all || (Number.isInteger(Number(limit)) && Number(limit) >= 1));
  const estimate = asked && asked.vocabulary === vocabulary && asked.size === size ? asked.answer : null;
  const clear = () => { setAsked(null); setStarted(null); };
  const estimateOf = (v: string, s: string) => get<Estimate>(`/api/datasets/${part(dataset.id)}/estimate?vocabulary=${part(v)}&max_tokens=${part(s)}`);
  const ask = async (s = size) => {
    const wanted = { vocabulary, size: s };
    setEstimating(true);
    const answer = await estimateOf(wanted.vocabulary, wanted.size);
    setAsked({ answer, ...wanted });
    setEstimating(false);
    window.setTimeout(() => document.querySelector(".estimate, .new-build .problems")?.scrollIntoView({ block: "start", behavior: "smooth" }), 0);
  };
  const start = async () => {
    if (!estimate?.ok || starting) return;
    setStarting(true);
    const a = await post<Job<unknown>>(`/api/datasets/${part(dataset.id)}/build`, { vocabulary, max_tokens: all ? "all" : Number(limit), confirmed: true, estimate: estimate.data });
    setStarting(false);
    setStarted(a);
    if (a.ok) { setAsked(null); onStarted(a.data); }
  };
  if (!trained) return <Skeleton lines={3} label="Reading the vocabularies" />;
  if (!trained.ok) return <Problems problems={trained.problems} />;
  if (trained.data.length === 0) return <p>No vocabulary is trained yet, and a build needs one. <a href={href(["tokenizers", dataset.tokenizer])}>Train the vocabulary of {dataset.tokenizer}</a> first; a build never trains one.</p>;
  const chosen = trained.data.find((v) => v.sha256 === vocabulary);
  const busy = Boolean(open);
  return (
    <form className="new-build" onSubmit={(e) => { e.preventDefault(); if (valid) ask(); }}>
      <div className="fields roomy">
        <Select label={required("Vocabulary")} required value={vocabulary} onChange={(e) => { setVocabulary(e.target.value); clear(); }}
          hint={dataset.vocabulary ? "The one the dataset publishes, which model training takes, is filled in." : "The dataset publishes no vocabulary yet; any trained one can build a corpus."}>
          {trained.data.map((v) => <option key={v.sha256} value={v.sha256}>{v.name ?? v.recipe} version {v.version}{v.sha256 === dataset.vocabulary ? " (published)" : ""}</option>)}
        </Select>
        <Select label={required("Size")} value={all ? "all" : "limit"} onChange={(e) => { setAll(e.target.value === "all"); clear(); }}
          hint="Every token the dataset's rules take, or at most a number of tokens.">
          <option value="limit">a number of tokens</option>
          <option value="all">every token</option>
        </Select>
        {!all && <TextInput label={required("Most tokens")} required type="number" min="1" step="1" value={limit} onChange={(e) => { setLimit(e.target.value); clear(); }}
          hint={given && limit === given ? "The size the corpus page offered is filled in." : dataset.max_tokens && limit === String(dataset.max_tokens) ? `The dataset's default size, ${dataset.max_tokens.toLocaleString("en")}, is filled in.` : !dataset.max_tokens && limit === String(QUICK) ? `The dataset names no default size; ${QUICK.toLocaleString("en")} tokens is a quick test.` : "The corpus never passes it."} />}
      </div>
      {!all && !dataset.upstream && valid && Number(limit) !== dataset.max_tokens && (
        <p className="small row tight">
          <span className="aside">A new corpus of {dataset.name} starts at {dataset.max_tokens ? plural(dataset.max_tokens, "token") : "no set size"}.</span>
          <Button variant="quiet" disabled={defaulting} onClick={makeDefault}>Make {Number(limit).toLocaleString("en")} tokens its default size</Button>
          <span className="aside">A new revision of the dataset; its corpora stay as they are.</span>
        </p>
      )}
      {defaulted && !defaulted.ok && <Problems problems={defaulted.problems} />}
      {defaulted?.ok && Number(limit) === dataset.max_tokens && <p className="small done" role="status">{dataset.name} now starts a new corpus at {plural(dataset.max_tokens, "token")}.</p>}
      <Button type="submit" variant={estimate?.ok ? "quiet" : "primary"} disabled={!valid || estimating}>{estimating ? "Estimating" : estimate?.ok ? "Estimate again" : "Estimate"}</Button>
      {estimate && !estimate.ok && <Problems problems={estimate.problems} />}
      {estimate?.ok && (
        <div className="estimate">
          <Forecast e={estimate.data} dataset={dataset} asked={all ? null : Number(limit)} onSize={(n) => { setLimit(String(n)); clear(); ask(String(n)); }} />
          <div className="row tight">
            <Button variant="primary" disabled={!estimate.data.fits || busy || starting} onClick={start}>
              Confirm and build{chosen ? ` with ${chosen.name ?? chosen.recipe} version ${chosen.version}` : ""}
            </Button>
            <span className="small aside">{busy ? "A build of this dataset waits or runs: one at a time." : `Queues the build of about ${plural(estimate.data.tokens, "token")}, ${bytes(estimate.data.bytes)} in the store. It cannot be cancelled here.`}</span>
          </div>
          <EstimateFacts e={estimate.data} limited={!all} />
        </div>
      )}
      {started && !started.ok && <Problems problems={started.problems} />}
    </form>
  );
}

// size: a size the address names (?size=N, from a corpus page whose shares came out uneven): the form takes it and
// estimates it at once, since an estimate writes nothing
function Listing({ id, size }: { id: string; size: string | null }) {
  const [listed, setListed] = useState<Answer<Listed> | null>(null);
  const [mine, setMine] = useState<Job<unknown> | null>(null); // the build the form below started, followed at the top
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
      <PageHead title={`Corpora of ${dataset.name}`} meta={<><span>{plural(corpora.filter((c) => c.state === "built").length, "corpus", "corpora")} built</span>{job && <span className="cue">1 {job.state === "queued" ? "waiting its turn" : "being built"}</span>}</>}
        chain={<>The token files of the <a href={href(["datasets", dataset.id])}>dataset {dataset.name}</a> for training, one corpus per vocabulary and size, each at an address that stays the same whatever is built next. Its texts, with no tokenizer, are on <a href={href(["export", dataset.id])}>its export page</a>.</>} />
      {state === "no store" && <Problems problems={[listed.data.problem ?? "the store cannot be read here"]} />}
      {state === "checking an erasure" && <p>Checking an erasure: the corpora holding a text of {listed.data.erasures?.join(", ")} are being marked stale. Nothing is sent until that ends.</p>}
      {mine && <JobLine key={mine.id} job={mine} onEnded={load} />}
      {job && job.id !== mine?.id && <JobLine key={job.id} job={job} onEnded={load} />}
      <Section title="Corpora built">
        {corpora.length === 0 ? <p className="small">No corpus is built yet: the form below builds the first.</p> : (
          <div className="table-scroll cards">
            <table className="ledger-table" aria-label="The corpora of the dataset">
              <thead><tr><th scope="col">vocabulary</th><th scope="col">size</th><th scope="col">state</th><th scope="col" className="number">texts</th><th scope="col" className="number">tokens of plain text</th><th scope="col" className="number">tokens of conversations and tasks</th><th scope="col" className="number">tokens in all</th><th scope="col">shares got</th><th scope="col"><span className="visually-hidden">download</span></th></tr></thead>
              <tbody>
                {corpora.map((c) => (
                  <tr key={c.link}>
                    <td data-label="vocabulary"><a href={corpusHref(dataset.id, c, corpora)}>{vocabularyWords(c.vocabulary)}</a></td>
                    <td data-label="size">{sizeWords(c.max_tokens)}</td>
                    <td data-label="state" className={c.state === "stale" ? "not-allowed" : ""}>{c.state}</td>
                    <td data-label="texts" className="small">{c.corpus?.texts !== undefined ? takenWords(c.corpus.texts, c.corpus.conversations, c.corpus.tasks) : ""}</td>
                    <td data-label="plain text" className="number ledger">{c.corpus?.total_tokens?.toLocaleString("en") ?? ""}</td>
                    <td data-label="conversations and tasks" className="number ledger" title={c.corpus?.conversations !== undefined ? plural(c.corpus.conversations, "conversation") : undefined}>{c.corpus ? itemTokens(c.corpus)?.toLocaleString("en") ?? "" : ""}</td>
                    <td data-label="in all" className="number ledger">{c.corpus ? ((c.corpus.total_tokens ?? 0) + (itemTokens(c.corpus) ?? 0)).toLocaleString("en") : ""}</td>
                    <td data-label="shares got" className="small wrap-cell">{sharesGotWords(dataset, c.corpus)}</td>
                    <td data-label="">{c.state === "built" && <ButtonLink variant="quiet" href={corpusHref(dataset.id, c, corpora)}>Download</ButtonLink>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
      <Section title="New build">
        <p className="small aside">A build names a trained vocabulary and a size. It starts only after its estimate is shown and confirmed.</p>
        <NewBuild dataset={dataset} open={job} startSize={size} onDataset={load} onStarted={(j) => { setMine(j); load(); window.scrollTo({ top: 0, behavior: "smooth" }); }} />
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
      <PageHead title={`Corpus of ${dataset.name}`}
        meta={<><span>{vocabularyWords(shown.data.vocabulary)}</span><span>{sizeWords(shown.data.max_tokens)}</span><span className="cue">{state}</span></>}
        chain={<>The corpus of the <a href={href(["datasets", dataset.id])}>dataset {dataset.name}</a> as it is now, with the vocabulary {vocabularyWords(shown.data.vocabulary)}, {sizeWords(shown.data.max_tokens)}. This address stays the same whatever is built next, so it can be shared. <a href={href(["download", dataset.id])}>Its other corpora, and a new build</a>.</>}
        actions={<CopyLink />} />
      {state === "no store" && <Problems problems={[shown.data.problem ?? "the store cannot be read here"]} />}
      {state === "checking an erasure" && <p>Checking an erasure: the corpora holding a text of {shown.data.erasures?.join(", ")} are being marked stale. Nothing is sent until that ends.</p>}
      {state === "being built" && job && <><JobLine key={job.id} job={job} onEnded={load} /><p className="small">Nothing to download until it ends.</p></>}
      {state === "stale" && <p>Stale: an admitted erasure reached a text of its latest corpus{corpus?.stale ? <> ({corpus.stale.erasures.join(", ")})</> : null}. Nothing to download until it is built again, which nothing starts by itself.</p>}
      {state === "not built" && <p>Not built{corpus?.deleted ? <>: its latest corpus was deleted ({corpus.deleted.why})</> : <> from the dataset as it is now</>}. Nothing to download; <a href={href(["download", dataset.id])}>a new build</a> starts one, after its estimate.</p>}
      {state === "built" && corpus && <Built corpus={corpus} files={shown.data.files!} base={base} dataset={dataset} asked={shown.data.max_tokens} vocabulary={vocabulary} />}
    </>
  );
}

// A button that copies the page's address, with what it did.
function CopyLink() {
  const [copied, setCopied] = useState<boolean | null>(null);
  const copy = async () => {
    try { await navigator.clipboard.writeText(window.location.href); setCopied(true); } catch { setCopied(false); }
  };
  return <Button variant="quiet" onClick={copy}>{copied === null ? "Copy its link" : copied ? "Link copied" : "Copying refused: copy the address bar"}</Button>;
}

// A corpus's address, its vocabulary named by its recipe and version (mix-tok-v1) when no other row's vocabulary
// bears that name, else by its sha256 (CorpusAt).
function corpusHref(id: string, c: Row, rows: Row[]): string {
  const name = c.vocabulary.recipe ? `${c.vocabulary.recipe}-v${c.vocabulary.version}` : null;
  const alone = name !== null && rows.every((r) => r.vocabulary.sha256 === c.vocabulary.sha256 || `${r.vocabulary.recipe}-v${r.vocabulary.version}` !== name);
  return href(["download", id, alone ? name! : c.vocabulary.sha256, c.size]);
}

// The shares a built corpus got, each as a part of its plain text, beside
// the share asked when it differs by more than UNEVEN ("code 0 % of 60 %").
function sharesGotWords(dataset: Dataset, corpus: Corpus | null): string {
  const got = corpus?.shares_tokens;
  const built = got?.reduce((t, s) => t + s.tokens, 0) ?? 0;
  if (!got?.length || !built) return "";
  const asked = dataset.shares ?? [];
  const whole = asked.reduce((t, s) => t + s.share, 0) || 1;
  return got.map((s) => {
    const share = s.tokens / built;
    const wanted = (asked.find((a) => a.tag === s.tag)?.share ?? 0) / whole;
    return Math.abs(share - wanted) > UNEVEN ? `${s.tag} ${percent(share)} of ${percent(wanted)} asked` : `${s.tag} ${percent(share)}`;
  }).join(", ");
}

// The shares a dataset asks for and those a built corpus got, each as a
// part of its plain text, and whether one is far off (UNEVEN, in points).
// When one is, it asks the estimate of the corpus's vocabulary and size for
// the size where every share comes even (even_size), the number the build
// form offers for that size.
function SharesGot({ asked, got, dataset, vocabulary, size }: { asked: { tag: string; share: number }[]; got: { tag: string; tokens: number }[]; dataset: Dataset; vocabulary: string; size: number | null }) {
  const whole = asked.reduce((t, s) => t + s.share, 0) || 1;
  const built = got.reduce((t, s) => t + s.tokens, 0);
  const rows = asked.map((s) => ({ tag: s.tag, asked: s.share / whole, got: (got.find((g) => g.tag === s.tag)?.tokens ?? 0) / built }));
  const off = rows.some((r) => Math.abs(r.asked - r.got) > UNEVEN);
  const [even, setEven] = useState<number | null>(null);
  useEffect(() => {
    if (!off || size === null) return;
    get<Estimate>(`/api/datasets/${part(dataset.id)}/estimate?vocabulary=${part(vocabulary)}&max_tokens=${size}`)
      .then((a) => setEven(a.ok && a.data.even_size !== null && a.data.even_size > size ? a.data.even_size : null));
  }, [off, size, dataset.id, vocabulary]);
  const rebuild = even;
  if (!built) return null;
  return (
    <div className={off ? "warning" : undefined}>
      <p className="small">
        {off ? <strong>Shares asked {rows.map((r) => percent(r.asked)).join(" / ")}, got {rows.map((r) => percent(r.got)).join(" / ")} </strong> : <>Shares asked {rows.map((r) => percent(r.asked)).join(" / ")}, got {rows.map((r) => percent(r.got)).join(" / ")} </>}
        ({rows.map((r) => r.tag).join(" / ")}).
        {off && <> Texts are never cut, and a share stops at its first text that does not fit in what is left of its part, so at a small size one long text can end a share early. A bigger size keeps closer to the shares.</>}
        {off && got.length > 0 && <> When the plain text comes out short, the conversations and tasks hold a bigger part of the corpus than in the estimate, since their tokens stay the same.</>}
      </p>
      {off && (
        <p className="row tight small">
          {rebuild !== null && <ButtonLink variant="quiet" href={href(["download", dataset.id], { size: String(rebuild) })}>Estimate a build at {plural(rebuild, "token")}</ButtonLink>}
          {rebuild !== null && <span className="aside">the smallest size where every share comes within {Math.round(UNEVEN * 100)} percentage points of the part it asks, by the build estimate's forecast;</span>}
          {!dataset.upstream && <a href={href(["datasets", dataset.id], { edit: "1" })}>change the shares</a>}
          <span className="aside">{dataset.upstream ? "" : "or "}keep this corpus as it is.</span>
        </p>
      )}
    </div>
  );
}

// What each file of a corpus is, for the person who downloads it.
const TABLES: Record<string, string> = {
  texts: "the list of texts it holds, with their credits, licenses and token counts",
  licenses: "the licenses its texts are under, with their texts",
  notices: "the notices its texts carry",
  dataset: "the dataset as it was when built",
};
function role(f: File, dataset: Dataset): { what: string; tokens: boolean } {
  const tags = (dataset.shares ?? []).map((s) => s.tag);
  if (f.role === "text tokens") return { what: `Tokens of the plain texts${tags.length ? ` (${tags.length > 1 ? `${tags.slice(0, -1).join(", ")} and ${tags[tags.length - 1]}` : tags[0]})` : ""}`, tokens: true };
  if (f.role === "conversation tokens") return { what: "Tokens of the conversations", tokens: true };
  if (f.role === "vocabulary") return { what: "The vocabulary it was tokenized with, as JSON", tokens: false };
  if (f.role === "tasks") return { what: "The agentic tasks, one JSON line each", tokens: false };
  if (f.role.startsWith("table ")) return { what: `A table: ${TABLES[f.role.slice(6)] ?? f.role.slice(6)}`, tokens: false };
  return { what: "Another file of the corpus", tokens: false };
}

// A built corpus: what it holds, then its files, the token files first, each named by what it is.
function Built({ corpus, files, base, dataset, asked, vocabulary }: { corpus: Corpus; files: File[]; base: string; dataset: Dataset; asked: number | null; vocabulary: string }) {
  const sorted = [...files].sort((a, b) => Number(role(b, dataset).tokens) - Number(role(a, dataset).tokens));
  const total = files.reduce((t, f) => t + (f.bytes ?? 0), 0);
  const plain = corpus.total_tokens ?? 0;
  const items = itemTokens(corpus);
  const inAll = plain + (items ?? 0);
  const tasks = files.some((f) => f.role === "tasks");
  return (
    <>
      <p>
        {takenWords(corpus.texts ?? 0, corpus.conversations, corpus.tasks)}: {plural(plain, "token")} of plain text
        {corpus.shares_tokens?.length ? <> ({corpus.shares_tokens.map((s) => `${s.tag} ${s.tokens.toLocaleString("en")}`).join(", ")})</> : null}
        {items !== null && items > 0 ? <>, and {plural(items, "token")} of conversations{tasks ? " and tasks" : ""}{inAll ? <> ({percent(items / inAll)} of its tokens)</> : null}</> : null}.
        {" "}<span className="aside">Built {corpus.built_at.replace("T", " ").slice(0, 16)} UTC; {bytes(total)} in all.</span>
      </p>
      {asked !== null && items !== null && <p className="small aside">{plural(inAll, "token")} in all, of the {plural(asked, "token")} at most asked for; the conversations and tasks take their share of the corpus of every token, the plain text the rest.</p>}
      {dataset.shares?.length && corpus.shares_tokens?.length ? <SharesGot asked={dataset.shares} got={corpus.shares_tokens} dataset={dataset} vocabulary={vocabulary} size={asked} /> : null}
      <Section title="Its files">
        <p className="small aside">The token files are what training reads; the tables say what the tokens came from. The token files and the tables are Parquet, the vocabulary is JSON, and the tasks are JSON lines.</p>
        <div className="table-scroll cards">
          <table className="ledger-table" aria-label="The corpus's files">
            <thead><tr><th scope="col">what it is</th><th scope="col" className="number">size</th><th scope="col"><span className="visually-hidden">download</span></th><th scope="col">saved as</th></tr></thead>
            <tbody>
              {sorted.map((f) => {
                const r = role(f, dataset);
                return (
                  <tr key={f.name}>
                    <td className="wrap-cell" data-label="">{r.tokens ? <strong>{r.what}</strong> : r.what}{f.tokens !== null && r.tokens ? <span className="aside">, {plural(f.tokens, "token")}</span> : null}</td>
                    <td className="number ledger" data-label="size">{fileSize(f.bytes)}</td>
                    <td data-label=""><ButtonLink variant={r.tokens ? "primary" : "quiet"} href={`${base}/files/${part(f.name)}`} download={f.download}>Download</ButtonLink></td>
                    <td className="ledger small aside file-name" data-label="saved as" title={`stored as ${f.name}, sha256 ${f.sha256}`}>{f.download}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}

// A corpus's address names its vocabulary by its recipe and version
// (mix-tok-v1) when no other trained vocabulary bears that name, else by its
// sha256, which the server's routes take. An address with the sha256 is
// shown with the name instead, without reading the page again.
const SHA256 = /^[0-9a-f]{64}$/;
const vocabularyName = (v: Trained) => `${v.name ?? v.recipe}-v${v.version}`;
function CorpusAt({ id, vocabulary, size }: { id: string; vocabulary: string; size: string }) {
  const [trained, setTrained] = useState<Answer<Trained[]> | null>(null);
  useEffect(() => { get<Trained[]>("/api/vocabularies").then(setTrained); }, []);
  const given = SHA256.test(vocabulary);
  const named = trained?.ok ? trained.data.filter((v) => vocabularyName(v) === vocabulary) : [];
  const sha = given ? vocabulary : named.length === 1 ? named[0].sha256 : null;
  useEffect(() => {
    if (!given || !trained?.ok) return;
    const found = trained.data.find((v) => v.sha256 === vocabulary);
    if (!found || trained.data.filter((v) => vocabularyName(v) === vocabularyName(found)).length !== 1) return;
    window.history.replaceState(null, "", href(["download", id, vocabularyName(found), size]));
  }, [given, trained, vocabulary, id, size]);
  if (sha) return <OneCorpus id={id} vocabulary={sha} size={size} />;
  if (!trained) return <Skeleton lines={6} label="Reading the vocabularies" />;
  if (!trained.ok) return <Problems problems={trained.problems} />;
  return <p>{named.length > 1 ? `Several trained vocabularies are named ${vocabulary}, so this address cannot tell which one.` : `No trained vocabulary is named ${vocabulary} here.`} <a href={href(["download", id])}>The dataset's corpora</a> list those built.</p>;
}

export const download: Page = {
  path: "download", title: "Corpora", group: null, under: "datasets",
  Component: ({ rest, query }) => (rest[0] && rest[1] && rest[2] ? <CorpusAt key={rest.join("/")} id={rest[0]} vocabulary={rest[1]} size={rest[2]} />
    : rest[0] ? <Listing key={`${rest[0]} ${query.get("size") ?? ""}`} id={rest[0]} size={query.get("size")} /> : <p>A dataset's download page is reached from <a href={href(["datasets"])}>its dataset</a>.</p>),
};
