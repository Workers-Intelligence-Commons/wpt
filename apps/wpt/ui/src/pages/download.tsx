// Download: the token corpora of a dataset. A corpus is built for a dataset,
// one vocabulary the depot holds (by its sha256) and a size (a number of
// tokens, or all). #/download/ID lists the corpora the store holds of the
// dataset (GET /api/datasets/ID/download), one row per vocabulary and size,
// each with its own page, under the dataset's stages (pipeline.tsx), and
// under them, folded, the corpora of the dataset's older revisions (GET
// /api/datasets/ID), and the form of a new build, folded once a corpus is built, and folded
// behind the dataset's next step while it has no vocabulary of its own: a
// trained vocabulary (the one the dataset publishes, else its own recipe's
// latest, filled in), a size (the dataset's max_tokens filled in when it has one),
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
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { get, part, post, type Answer, type Job } from "../api";
import { Cut, Help, PageHead, plural, Problems, required, Section, Stat, takenWords, type Dataset } from "../parts";
import { href } from "../route";
import { built, mixLine, MixBar, off, UNEVEN, whole, type Mixed } from "../mix";
import { trainedWords, useCorpora, usePipeline, useRecipesAndVocabularies, vocabularyName, type BuiltCorpus, type Trained } from "../pipeline";
import { DatasetStages } from "../stages";
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
type ShareForecast = { tag: string; tokens: number; part: number };
type Estimate = {
  tokens: number; plain_tokens: number; item_tokens: number; item_share: number; share_tokens: ShareForecast[] | null; forecast: boolean | null;
  even_size: number | null; texts: number; plain_texts: number; item_texts: number; items_held: number; bytes: number; token_bytes: number; table_bytes: number;
  seconds: number | null; workers: number; figures: boolean; read_again: boolean; cap: number | null; others: number | null; fits: boolean; evicts: number | null;
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
const fileSize = (n: number | null) => (n === null ? "size unknown" : bytes(n));

// the tokens of a corpus's conversations and tasks
const itemTokens = (c: Corpus) => c.items_tokens ?? null;

const QUICK = 100_000; // tokens a new build is filled in with when the dataset names no size: a quick test

// UNEVEN (mix.tsx): how far, in points, a share's part of the plain text may be from
// its proportion and still count as even (wpt_depot.estimate uses the same)

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
  // the mix said as every page says it (mix.tsx): the conversations' part of the whole, then each share's part of the plain text
  const items = e.item_tokens > 0 ? `conversations ~${whole(e.item_share)}` : null;
  const parts = [...(items ? [items] : []), ...(e.share_tokens ?? []).map((s) => `${s.tag} ~${whole(e.plain_tokens ? s.tokens / e.plain_tokens : 0)}`)];
  const head = asked === null ? `About ${plural(e.tokens, "token")}` : `About ${plural(e.tokens, "token")} of the ${asked.toLocaleString("en")} asked`;
  const empty = off.filter((s) => s.tokens === 0);
  return (
    <div className={off.length ? "warning" : undefined}>
      <p>
        <strong>{head}{off.length ? `, and ${off.map((s) => `${s.tag} will likely get ${s.tokens ? `only about ${s.tokens.toLocaleString("en")}` : "0"} tokens of its ${s.part.toLocaleString("en")}`).join(" and ")}` : ""}.</strong>
        {parts.length > 1 && <> <span className="aside">{parts.join(", ")}.</span></>}
      </p>
      {off.length > 0 && (
        <p className="row tight small">
          <Help label="uneven shares">Texts are never cut: a share stops at its first text that does not fit in what is left of its part, in the dataset's fixed order. Here {empty.length ? `an early ${empty.map((s) => s.tag).join(" or ")} text is` : "a long text comes"} too long for {off.length === 1 ? "its part" : "their parts"}, so the corpus holds fewer tokens than asked{e.item_tokens ? ", and the conversations take a bigger part of it" : ""}. The even size is the smallest where every share's part of the plain text comes within {Math.round(UNEVEN * 100)} percentage points of the part it asks (a share of 60 % gets between 50 and 70 %), by the same forecast. Changing the shares makes a new revision of the dataset. Keeping this size builds this mix as forecast.</Help>
          {e.even_size !== null && asked !== null && e.even_size > asked && <Button variant="quiet" onClick={() => onSize(e.even_size!)}>Use {e.even_size.toLocaleString("en")} tokens</Button>}
          {!dataset.upstream && <a href={href(["datasets", dataset.id], { edit: "1" })}>Change the shares</a>}
        </p>
      )}
    </div>
  );
}

// What an estimate says, in words: what the size holds of each kind (the
// conversations and tasks, then each share's forecast beside its part), the
// store's bytes, the time, the cap, and what it was estimated from.
function EstimateFacts({ e, limited, dataset }: { e: Estimate; limited: boolean; dataset: Dataset }) {
  const items = e.item_share > 0;
  return (
    <dl className="facts">
      <dt className="aside">holds</dt>
      <dd>
        about {plural(e.tokens, "token")}, from {e.item_texts || e.items_held ? <>{plural(e.plain_texts, "plain text")} and {e.item_texts.toLocaleString("en")} of {plural(e.items_held, "conversation or task", "conversations and tasks")}</> : plural(e.plain_texts, "plain text")}:
        <ul className="plain kinds">
          {items && <li><span>conversations and tasks{limited ? <span className="aside"> ~{whole(e.item_share)}</span> : null}<Help label="conversations and tasks">{limited ? `Their part: about ${whole(e.item_share)} of the size, as in the corpus of every token. A conversation is never cut, so at a small size a few long ones make the part swing.` : "Every one is taken."}</Help></span> <span className="ledger">about {e.item_tokens.toLocaleString("en")}</span></li>}
          {e.share_tokens?.length ? e.share_tokens.map((s) => (
            <li key={s.tag}><span title={limited ? `target ${s.part.toLocaleString("en")} tokens` : undefined}>{s.tag}{limited && e.plain_tokens ? <span className="aside"> ~{whole(s.tokens / e.plain_tokens)} of the rest</span> : null}</span> <span className="ledger">about {s.tokens.toLocaleString("en")}</span></li>
          )) : <li><span>plain text</span> <span className="ledger">about {e.plain_tokens.toLocaleString("en")}</span></li>}
        </ul>
        {limited && e.forecast === false && <p className="small aside">Rough: the forecast stopped after 50,000 texts.<Help label="the forecast">It follows the build's order through the first 50,000 texts of the dataset only. A part still open there is counted at its share of the size, not as the build will take it.</Help></p>}
      </dd>
      <dt className="aside">store</dt><dd>about {bytes(e.bytes)}<Help label="the store's bytes">{bytes(e.token_bytes)} of token files, and {bytes(e.table_bytes)} for the list of its texts with their credits and licenses.</Help></dd>
      <dt className="aside">time</dt><dd>{e.seconds === null ? "unknown: the sample tokenized nothing" : `${e.seconds < 60 ? "" : "about "}${duration(e.seconds)} on this machine`}<Help label="the time">{e.figures ? "Of tokenizing. The build reads the texts in a fixed shuffled order, the export's, and stops once the size is full." : "Of tokenizing. Before it, the build reads the size of every text of the dataset, since this revision's figures are not known; this estimate does not time that."}</Help></dd>
      <dt className="aside">cap</dt><dd>{e.cap === null ? "no cap" : e.fits
        ? <>fits under {bytes(e.cap)}{e.others !== null && <>; others take {bytes(e.others)}{e.evicts ? <>; deletes about {bytes(e.evicts)} of old corpora<Help label="the deletion">To make room, the build deletes the corpora downloaded least recently.</Help></> : null}</>}</>
        : <span className="not-allowed">over the cap of {bytes(e.cap)}: choose a smaller size</span>}</dd>
      {e.store && <><dt className="aside">store</dt><dd className="not-allowed">{e.store}</dd></>}
      <dt className="aside">basis</dt><dd className="small">a sample of {plural(e.sample.texts, "text")}<Help label="the basis">Estimated from {plural(e.sample.tokenized, "text")} ({bytes(e.sample.bytes)}) tokenized in {e.sample.seconds} s, out of a sample of {plural(e.sample.texts, "text")}; the build's counts can differ a little.</Help>{e.warnings.map((w) => <span key={w} className="not-allowed"> {w}.</span>)}{!e.figures && e.read_again && <> <a href={href(["datasets", dataset.id])}>Read again</a></>}</dd>
    </dl>
  );
}

// The form of a new build: a vocabulary, a size, an estimate, then the confirmed start under it, followed to its end.
// onDataset: after the size is made the dataset's default, a new revision of it, to read it again.
function NewBuild({ dataset, open, startSize, onStarted, onDataset, quiet = false }: { dataset: Dataset; open: Job<unknown> | null; startSize: string | null; onStarted: (job: Job<unknown>) => void; onDataset: () => void; quiet?: boolean }) {
  const { trained } = useRecipesAndVocabularies();
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
  // the vocabulary chosen first: the published one, else its own recipe's latest, else any; chosen again
  // when the dataset's recipe or published vocabulary changes, or the one chosen is gone
  const chosenFor = useRef("");
  useEffect(() => {
    if (!trained?.ok || !trained.data.length) return;
    const key = `${dataset.tokenizer}\u0000${dataset.vocabulary}`;
    if (chosenFor.current === key && trained.data.some((v) => v.sha256 === vocabulary)) return;
    chosenFor.current = key;
    setVocabulary((trained.data.find((v) => v.sha256 === dataset.vocabulary) ?? trained.data.filter((v) => v.name === dataset.tokenizer).at(-1) ?? trained.data[0]).sha256);
  }, [trained, dataset.tokenizer, dataset.vocabulary, vocabulary]);
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
  if (trained.data.length === 0) return <p>No vocabulary is trained here, and a build needs one. <a href={href(["tokenizers", dataset.tokenizer])}>Train the vocabulary of {dataset.tokenizer}</a> first; a build never trains one.</p>;
  const chosen = trained.data.find((v) => v.sha256 === vocabulary);
  // published first, then the dataset's own recipe's, then the others
  const rank = (v: Trained) => (v.sha256 === dataset.vocabulary ? 0 : v.name === dataset.tokenizer ? 1 : 2);
  const ordered = [...trained.data].sort((a, b) => rank(a) - rank(b));
  const busy = Boolean(open);
  return (
    <form className="new-build" onSubmit={(e) => { e.preventDefault(); if (valid) ask(); }}>
      <div className="fields roomy">
        <Select label={required("Vocabulary")} required value={vocabulary} onChange={(e) => { setVocabulary(e.target.value); clear(); }}>
          {ordered.map((v) => <option key={v.sha256} value={v.sha256}>{v.name ?? v.recipe} version {v.version}{v.sha256 === dataset.vocabulary ? " (published)" : v.name === dataset.tokenizer ? "" : ", another recipe's"}</option>)}
        </Select>
        <Select label={required("Size")} value={all ? "all" : "limit"} onChange={(e) => { setAll(e.target.value === "all"); clear(); }}>
          <option value="limit">a number of tokens</option>
          <option value="all">every token</option>
        </Select>
        {!all && <TextInput label={required("Most tokens")} required type="number" min="1" step="1" value={limit} onChange={(e) => { setLimit(e.target.value); clear(); }}
          hint={given && limit === given ? "From the corpus page." : dataset.max_tokens && limit === String(dataset.max_tokens) ? "The dataset's default." : !dataset.max_tokens && limit === String(QUICK) ? "A quick test." : undefined} />}
      </div>
      {!all && !dataset.upstream && valid && Number(limit) !== dataset.max_tokens && (
        <p className="small row tight">
          <button type="button" className="link-button wic-focus" disabled={defaulting} onClick={makeDefault}>Make it the default size</button>
          <Help label="the default size">A new corpus of {dataset.name} starts at {dataset.max_tokens ? plural(dataset.max_tokens, "token") : "no set size"} now. Making {Number(limit).toLocaleString("en")} tokens its default is a new revision of the dataset; its corpora stay as they are.</Help>
        </p>
      )}
      {defaulted && !defaulted.ok && <Problems problems={defaulted.problems} />}
      {defaulted?.ok && Number(limit) === dataset.max_tokens && <p className="small done" role="status">Default size: {plural(dataset.max_tokens, "token")}.</p>}
      <p className="row tight estimate-ask">
        <Button type="submit" variant={estimate?.ok || quiet ? "quiet" : "primary"} disabled={!valid || estimating}>{estimating ? "Estimating" : estimate?.ok ? "Estimate again" : "Estimate"}</Button>
        <span className="small aside">tokens, store size and build time, before building</span>
        <Help label="the estimate">What a corpus of {dataset.name} with {chosen ? `${chosen.name ?? chosen.recipe} version ${chosen.version}` : "the chosen vocabulary"} at {all ? "every token" : valid ? plural(Number(limit), "token") : "the chosen size"} would hold in tokens, take in the store, and how long its build would take. A sample of the dataset is tokenized to measure it; nothing is built or written.</Help>
      </p>
      {estimate && !estimate.ok && <Problems problems={estimate.problems} />}
      {estimate?.ok && (
        <div className="estimate">
          <p className="cue small estimate-head">Estimate, not built yet</p>
          <Forecast e={estimate.data} dataset={dataset} asked={all ? null : Number(limit)} onSize={(n) => { setLimit(String(n)); clear(); ask(String(n)); }} />
          <div className="row tight">
            <Button variant="primary" disabled={!estimate.data.fits || busy || starting} onClick={start}>
              Confirm and build{chosen ? ` with ${chosen.name ?? chosen.recipe} version ${chosen.version}` : ""}
            </Button>
            <span className="small aside">{busy ? "Another build of it waits or runs." : `${bytes(estimate.data.bytes)}; cannot be cancelled here.`}</span>
          </div>
          <EstimateFacts e={estimate.data} limited={!all} dataset={dataset} />
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
  const every = useCorpora([id]).get(id); // the corpora of every revision, for those of older revisions
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
  const built = corpora.filter((c) => c.state === "built").length;
  const build = (
    <NewBuild dataset={dataset} open={job} startSize={size} onDataset={load} quiet={built > 0}
      onStarted={(j) => { setMine(j); load(); window.scrollTo({ top: 0, behavior: "smooth" }); }} />
  );
  return (
    <>
      <PageHead title={`Corpora of ${dataset.name}`} meta={job ? <span className="cue">1 {job.state === "queued" ? "waiting its turn" : "being built"}</span> : undefined}
        chain={<>The token files of the dataset for training, one corpus per vocabulary and size, each at an address that stays the same whatever is built next. Its texts, with no tokenizer, are on <a href={href(["export", dataset.id])}>its export page</a>.</>} />
      <DatasetStages id={dataset.id} here="corpus" action="none" />
      {state === "no store" && <Problems problems={[listed.data.problem ?? "the store cannot be read here"]} />}
      {state === "checking an erasure" && <p>Checking erasure {listed.data.erasures?.join(", ")}: downloads wait.</p>}
      {mine && <JobLine key={mine.id} job={mine} onEnded={load} />}
      {job && job.id !== mine?.id && <JobLine key={job.id} job={job} onEnded={load} />}
      {corpora.length > 0 && (
        <Section title="Corpora built">
          <div className="table-scroll cards">
            <table className="ledger-table" aria-label="The corpora of the dataset">
              <thead><tr><th scope="col">vocabulary</th><th scope="col">size</th><th scope="col">state</th><th scope="col" className="number">texts</th><th scope="col" className="number">tokens</th><th scope="col">mix</th><th scope="col"><span className="visually-hidden">download</span></th></tr></thead>
              <tbody>
                {corpora.map((c) => (
                  <tr key={c.link}>
                    <td data-label="vocabulary"><a href={corpusHref(dataset.id, c, corpora)}>{vocabularyWords(c.vocabulary)}</a></td>
                    <td data-label="size">{sizeWords(c.max_tokens)}</td>
                    <td data-label="state" className={c.state === "stale" ? "not-allowed" : ""}>{c.state}{c.state === "stale" && c.corpus?.stale && <>: {c.corpus.stale.erasures.map((e, i) => <span key={e}>{i > 0 && ", "}<a href={href(["erasures"], { request: e })}>{e}</a></span>)}</>}</td>
                    <td data-label="texts" className="number ledger" title={c.corpus?.texts !== undefined ? takenWords(c.corpus.texts, c.corpus.conversations, c.corpus.tasks) : undefined}>{c.corpus?.texts?.toLocaleString("en") ?? ""}</td>
                    <td data-label="tokens" className="number ledger" title={c.corpus ? `${(c.corpus.total_tokens ?? 0).toLocaleString("en")} of plain text, ${(itemTokens(c.corpus) ?? 0).toLocaleString("en")} of conversations and tasks` : undefined}>{c.corpus ? ((c.corpus.total_tokens ?? 0) + (itemTokens(c.corpus) ?? 0)).toLocaleString("en") : ""}</td>
                    <td data-label="mix" className="small">{c.corpus ? mixLine(mixOf(dataset, c.corpus)) : ""}</td>
                    <td data-label="">{c.state === "built" && <ButtonLink variant="ink" href={corpusHref(dataset.id, c, corpora)}>Download</ButtonLink>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}
      <BuildPlace dataset={dataset} built={built > 0}>{build}</BuildPlace>
      <OlderCorpora dataset={dataset} corpora={every} />
    </>
  );
}

// The corpora built from the dataset's older revisions, folded: this page's
// list and the corpus pages serve only the current revision, and wpt train
// still takes one of them by its id. Each says its id, the revision it was
// built from, its vocabulary and size, what it holds, when it was built, and
// whether it is deleted (and why) or stale (and the erasures that made it so).
function OlderCorpora({ dataset, corpora }: { dataset: Dataset; corpora: BuiltCorpus[] | null | undefined }) {
  const { trained } = useRecipesAndVocabularies();
  const older = (corpora ?? []).filter((c) => c.revision < dataset.revision);
  if (older.length === 0) return null;
  const words = (sha256: string) => { const v = trained?.ok ? trained.data.find((t) => t.sha256 === sha256) : undefined; return v ? trainedWords(v) : <span className="ledger">{sha256.slice(0, 12)}</span>; };
  return (
    <details className="fold small">
      <summary className="cue">Of older revisions: {plural(older.length, "corpus", "corpora")}</summary>
      <div className="table-scroll cards">
        <table className="ledger-table" aria-label="The corpora of older revisions of the dataset">
          <thead><tr><th scope="col">corpus id</th><th scope="col" className="number">revision</th><th scope="col">vocabulary, size</th><th scope="col" className="number">texts</th><th scope="col" className="number">tokens</th><th scope="col">built (UTC)</th><th scope="col">state</th></tr></thead>
          <tbody>
            {older.map((c) => (
              <tr key={c.id}>
                <td className="ledger" data-label=""><Cut text={c.id} /></td>
                <td className="number ledger" data-label="revision">{c.revision}</td>
                <td data-label="vocabulary, size">{words(c.vocabulary)}, {sizeWords(c.max_tokens)}</td>
                <td className="number ledger" data-label="texts" title={takenWords(c.texts, c.conversations)}>{c.texts.toLocaleString("en")}</td>
                <td className="number ledger" data-label="tokens">{(c.total_tokens + (c.items_tokens ?? 0)).toLocaleString("en")}</td>
                <td className="ledger nowrap" data-label="built (UTC)">{c.built_at.replace("T", " ").slice(0, 16)}</td>
                <td data-label="state">{c.deleted ? <span className="aside">deleted: {c.deleted.why}</span>
                  : c.stale ? <span className="not-allowed">stale: {c.stale.erasures.map((e, i) => <span key={e}>{i > 0 && ", "}<a href={href(["erasures"], { request: e })}>{e}</a></span>)}</span> : "fresh"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

// Where the form of a new build sits: in its own block while nothing is
// built; folded once a corpus is built, so its download leads the page; and
// while the dataset has no vocabulary of its own yet, folded behind its next
// step, since a build then takes another recipe's vocabulary.
function BuildPlace({ dataset, built, children }: { dataset: Dataset; built: boolean; children: ReactNode }) {
  const at = usePipeline(dataset);
  if (!at) return null;
  const ownMissing = !dataset.vocabulary && at.own.length === 0;
  if (ownMissing) return (
    <Section title="Needs a vocabulary first">
      <p className="row tight">{at.next ? <ButtonLink variant="primary" href={at.next.href}>{at.next.label}</ButtonLink> : <span className="aside">Vocabulary: {at.vocabulary.word}</span>}</p>
      <details className="small">
        <summary className="cue">Build with another recipe's vocabulary</summary>
        {children}
      </details>
    </Section>
  );
  if (built) return <details className="fold"><summary className="cue">New build</summary>{children}</details>;
  return <Section title="New build" help="A build names a trained vocabulary and a size. It starts only after its estimate is shown and confirmed.">{children}</Section>;
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
        meta={<><span>{vocabularyWords(shown.data.vocabulary)}</span><span>{sizeWords(shown.data.max_tokens)}</span><span className="cue">{state}</span>{corpus?.built_at && state === "built" && <span>{corpus.built_at.replace("T", " ").slice(0, 16)} UTC</span>}</>}
        chain={<>The corpus of the <a href={href(["datasets", dataset.id])}>dataset {dataset.name}</a> as it is now. This address stays the same whatever is built next, so it can be shared.</>}
        actions={<><CopyLink /><ButtonLink variant="quiet" href={href(["download", dataset.id])}>Other corpora</ButtonLink></>} />
      <DatasetStages id={dataset.id} here="corpus" action="none" />
      {state === "no store" && <Problems problems={[shown.data.problem ?? "the store cannot be read here"]} />}
      {state === "checking an erasure" && <p>Checking erasure {shown.data.erasures?.join(", ")}: downloads wait.</p>}
      {state === "being built" && job && <JobLine key={job.id} job={job} onEnded={load} />}
      {state === "stale" && <p className="not-allowed">Stale{corpus?.stale ? <> ({corpus.stale.erasures.join(", ")})</> : null}: <a href={href(["download", dataset.id])}>build it again</a> to download.<Help label="a stale corpus">An admitted erasure reached a text of its latest corpus. Nothing starts a new build by itself.</Help></p>}
      {state === "not built" && <p>Not built{corpus?.deleted ? <> (deleted: {corpus.deleted.why})</> : null}. <a href={href(["download", dataset.id])}>Build one</a>.</p>}
      {state === "built" && corpus && <Built corpus={corpus} files={shown.data.files!} base={base} dataset={dataset} asked={shown.data.max_tokens} vocabulary={vocabulary} />}
    </>
  );
}

// A button that copies, for a terminal, one curl command per file of the
// corpus, each saving under the name the page gives it; its "?" names the
// training command that reads the corpus.
function CopyCommands({ files, base, corpus }: { files: File[]; base: string; corpus: string }) {
  const [copied, setCopied] = useState<boolean | null>(null);
  const quote = (t: string) => `'${t.replace(/'/g, "'\\''")}'`;
  const lines = files.map((f) => `curl -fL -o ${quote(f.download)} ${quote(new URL(`${base}/files/${part(f.name)}`, window.location.href).href)}`);
  const copy = async () => {
    try { await navigator.clipboard.writeText(`${lines.join("\n")}\n`); setCopied(true); } catch { setCopied(false); }
  };
  return (
    <span className="row tight">
      <Button variant="quiet" onClick={copy}>{copied === null ? "Copy the download commands" : copied ? "Commands copied" : "Copying refused: select them below"}</Button>
      <Help label="training on it">Training reads it from the store: <span className="ledger">uv run wpt train RECIPE {corpus.slice(0, 12)}</span>.</Help>
      {copied === false && <pre className="remote ledger small copy-by-hand" tabIndex={0}>{lines.join("\n")}</pre>}
    </span>
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

// The mix a built corpus got (mix.tsx): its conversations and tasks, a part
// of all its tokens, and each share, a part of its plain tokens, beside the part asked.
function mixOf(dataset: Dataset, corpus: Corpus | null): Mixed {
  return built(dataset.shares ?? null, corpus?.shares_tokens, corpus ? itemTokens(corpus) ?? 0 : 0, Boolean(corpus?.tasks), corpus?.total_tokens ?? 0);
}

// The shares a dataset asks for and those a built corpus got, each as a
// part of its plain text, and whether one is far off (UNEVEN, in points).
// When one is, it asks the estimate of the corpus's vocabulary and size for
// the size where every share comes even (even_size), the number the build
// form offers for that size.
function MixStat({ corpus, dataset, vocabulary, size }: { corpus: Corpus; dataset: Dataset; vocabulary: string; size: number | null }) {
  const m = mixOf(dataset, corpus);
  const uneven = off(m);
  const [even, setEven] = useState<number | null>(null);
  useEffect(() => {
    if (!uneven || size === null) return;
    get<Estimate>(`/api/datasets/${part(dataset.id)}/estimate?vocabulary=${part(vocabulary)}&max_tokens=${size}`)
      .then((a) => setEven(a.ok && a.data.even_size !== null && a.data.even_size > size ? a.data.even_size : null));
  }, [uneven, size, dataset.id, vocabulary]);
  if (!m.items && !m.shares.length) return null;
  return (
    <div className={`stat stat-mix${uneven ? " stat-off" : ""}`}>
      <MixBar m={m} />
      <span className="stat-label small">mix got
        {uneven && <Help label="uneven shares">Texts are never cut, and a share stops at its first text that does not fit in what is left of its part, so at a small size one long text can end a share early. A bigger size keeps closer to the shares.{even !== null && ` ${plural(even, "token")} is the smallest size where every share comes within ${Math.round(UNEVEN * 100)} percentage points of the part it asks, by the build estimate's forecast.`}</Help>}
      </span>
      {uneven && (
        <span className="row tight small">
          {even !== null && <a href={href(["download", dataset.id], { size: String(even) })}>Estimate {plural(even, "token")}</a>}
          {!dataset.upstream && <a href={href(["datasets", dataset.id], { edit: "1" })}>Change the shares</a>}
        </span>
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
  return (
    <>
      <div className="stats">
        <Stat value={inAll.toLocaleString("en")} label={asked !== null ? `tokens of ${asked.toLocaleString("en")} asked` : "tokens"} />
        <Stat value={(corpus.texts ?? 0).toLocaleString("en")} label={corpus.conversations || corpus.tasks ? takenWords(corpus.texts ?? 0, corpus.conversations, corpus.tasks) : "plain texts"} />
        <Stat value={bytes(total)} label="in all files" />
        <MixStat corpus={corpus} dataset={dataset} vocabulary={vocabulary} size={asked} />
      </div>
      <Section title="Files" help="The token files are what training reads; the tables say what the tokens came from. The token files and the tables are Parquet, the vocabulary is JSON, and the tasks are JSON lines."
        tools={<CopyCommands files={sorted} base={base} corpus={corpus.id} />}>
        <div className="table-scroll cards">
          <table className="ledger-table files-table" aria-label="The corpus's files">
            <thead><tr><th scope="col">what it is</th><th scope="col" className="number">size</th><th scope="col"><span className="visually-hidden">download</span></th><th scope="col">saved as</th></tr></thead>
            <tbody>
              {sorted.map((f) => {
                const r = role(f, dataset);
                return (
                  <tr key={f.name}>
                    <td className="wrap-cell" data-label="">{r.tokens ? <strong>{r.what}</strong> : r.what}{f.tokens !== null && r.tokens ? <span className="aside">, {plural(f.tokens, "token")}</span> : null}</td>
                    <td className="number ledger" data-label="size">{fileSize(f.bytes)}</td>
                    <td data-label=""><ButtonLink variant={r.tokens ? "ink" : "quiet"} href={`${base}/files/${part(f.name)}`} download={f.download}>Download</ButtonLink></td>
                    <td className="ledger small aside file-name" data-label="saved as" title={`${f.download}, stored as ${f.name}, sha256 ${f.sha256}`}>{f.download}</td>
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
function CorpusAt({ id, vocabulary, size }: { id: string; vocabulary: string; size: string }) {
  const { trained } = useRecipesAndVocabularies();
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
