// Tokenizers: the depot's tokenizer recipes (GET /api/tokenizers), each the
// dataset whose sample trains its vocabulary (this depot's, or another
// depot's named by the web address of its web app), the sample's bytes, the
// vocabulary's size and a note, with whether its version has a vocabulary,
// and each recipe a dataset names that is not made yet, with its make button;
// a new recipe (wpt depot tokenizer new), its dataset chosen in a list of
// this depot's or named by hand on another depot, its form opened and filled in by
// the address (?new=1&name=NAME&dataset=DATASET, as a dataset's page links
// to it), with a small sample and vocabulary for a quick test; once made,
// its page opens. #/tokenizers/NAME is one recipe (a name not made yet: the datasets that wait for it and its make button): its
// vocabularies, each with its file and the texts it trained on (folded, by
// title, a page at a time), the training of its version's vocabulary
// queued as a job once its estimate (GET /api/tokenizers/NAME/estimate: the
// sample's bytes and an approximate time) is shown and confirmed (?train=1
// opens the estimate at once), then followed to its end, its dataset's
// stages, its revisions folded, and its edit as a
// new revision, which raises its version when it changes what trains the
// vocabulary (wpt depot tokenizer edit). A vocabulary its dataset of this
// depot does not publish has a button that publishes it there (POST
// /api/datasets/ID, its row as it is but the vocabulary). A recipe of the
// upstream this depot follows has no edit.

import { Button, ButtonLink, Dialog, Select, Skeleton, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useRef, useState } from "react";
import { get, part, post, put, waiting, type Answer, type Job } from "../api";
import { EVERY, search } from "../filter";
import { Cut, Facts, Help, Joined, Outcome, PageHead, Pager, plural, Problems, optional, required, RequiredNote, Section, useDatasets, type Dataset } from "../parts";
import { href } from "../route";
import { pipelineChanged, publishVocabulary, usePipeline, usePipelineChange } from "../pipeline";
import { DatasetStages } from "../stages";
import { bytes, duration } from "./download";
import { JobLine, JobProgress, type Known } from "./jobs";
import type { Page } from "./index";

type Recipe = {
  id: string; name: string; version: number; revision: number; depot: string | null; dataset: string; dataset_name: string | null;
  sample_bytes: number; vocab_size: number; note: string | null; trained?: number | boolean; job: Job<unknown> | null;
};
type Vocabulary = { recipe: string; version: number; sha256: string; texts: number; bytes: number | null; made_at: string };
type Shown = { recipe: Recipe; revisions: Recipe[]; vocabularies: Vocabulary[]; upstream: boolean; trained: boolean };
// below this part of the sample asked, a vocabulary says why it trained on less
const SHORT_SAMPLE = 0.9;
type TrainingEstimate = { sample_bytes: number; version: number; seconds: number; basis: string };

const trainsOn = (r: Recipe) => (r.depot ? <><span className="ledger">{r.dataset}</span> <span className="aside">of</span> <Cut text={r.depot} className="ledger" /></>
  : <a href={href(["datasets", r.dataset])}>{r.dataset_name ?? r.dataset}</a>);

// The sample and the vocabulary a new recipe is filled in with: small, for a quick test.
const SAMPLE = 2_000_000;
// A size typed with or without a unit ("2 MB", "1.5GB", "500000"), in bytes; null when it is none.
const FACTORS: Record<string, number> = { "": 1, b: 1, byte: 1, bytes: 1, kb: 1e3, mb: 1e6, gb: 1e9, tb: 1e12 };
export function parseBytes(typed: string): number | null {
  const found = /^\s*(\d+(?:\.\d+)?)\s*([a-z]*)\s*$/i.exec(typed.replace(/,/g, ""));
  const factor = found ? FACTORS[found[2].toLowerCase()] : undefined;
  const n = found && factor ? Math.round(Number(found[1]) * factor) : NaN;
  return n >= 1 ? n : null;
}
const TOKENS = 2048;

function Tokenizers({ query }: { query: URLSearchParams }) {
  const [rows, setRows] = useState<Answer<Recipe[]> | null>(null);
  const [making, setMaking] = useState(query.get("new") === "1");
  const [answer, setAnswer] = useState<Answer<{ id: string }> | null>(null);
  const load = useCallback(() => { get<Recipe[]>("/api/tokenizers").then(setRows); }, []);
  useEffect(load, [load]);
  const { datasets } = useDatasets();
  // the recipes a dataset names that are not made yet, each with the first dataset that waits for it
  const missing = rows?.ok && datasets?.ok ? [...new Map(datasets.data.filter((d) => !rows.data.some((r) => r.name === d.tokenizer)).map((d) => [d.tokenizer, d] as const)).entries()] : [];
  return (
    <>
      <PageHead title="Tokenizers"
        chain={<>A tokenizer recipe says how a vocabulary trains: on a sample of a dataset, up to a number of tokens. Each version's vocabulary is kept in the depot once trained, for good, and a corpus build names one of them.</>}
        actions={<Button variant="quiet" onClick={() => { setAnswer(null); setMaking(true); }}>New recipe</Button>} />
      {!making && answer && !answer.ok && <Outcome answer={answer} done={() => ""} />}
      {!rows && <Skeleton lines={4} />}
      {rows && !rows.ok && <Problems problems={rows.problems} />}
      {missing.map(([name, d], i) => (
        <p key={`missing-${name}`} className="wanted row tight">
          <span><strong className="ledger">{name}</strong> <span className="aside">wanted by</span> <a href={href(["datasets", d.id])}>{d.name}</a></span>
          <ButtonLink variant={i === 0 ? "primary" : "quiet"} href={href(["tokenizers"], { new: "1", name, dataset: d.name })}>Make the recipe</ButtonLink>
        </p>
      ))}
      {rows?.ok && (rows.data.length === 0 ? (missing.length === 0 && <p>No tokenizer recipe yet.</p>) : (
        <div className="table-scroll">
          <table className="ledger-table" aria-label="The tokenizer recipes">
            <thead><tr><th scope="col">recipe</th><th scope="col" className="number">version</th><th scope="col">trains on</th><th scope="col" className="number">sample</th><th scope="col" className="number">vocabulary size</th><th scope="col">vocabulary</th></tr></thead>
            <tbody>
              {rows.data.map((r) => (
                <tr key={r.id}>
                  <td className="nowrap"><a href={href(["tokenizers", r.name])}>{r.name}</a></td>
                  <td className="number ledger">{r.version}</td>
                  <td>{trainsOn(r)}</td>
                  <td className="number ledger" title={plural(r.sample_bytes, "byte")}>{bytes(r.sample_bytes)}</td>
                  <td className="number ledger">{r.vocab_size.toLocaleString("en")}</td>
                  <td>{r.trained ? "trained" : r.job ? <JobProgress job={r.job} /> : <a href={href(["tokenizers", r.name], { train: "1" })}>train it</a>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <Dialog open={making} onClose={() => setMaking(false)} title="New tokenizer recipe">
        {making && <RecipeForm name={query.get("name") ?? undefined} dataset={query.get("dataset") ?? undefined} onDone={(a, name) => {
          setAnswer(a as Answer<{ id: string }>);
          if (a.ok) { setMaking(false); window.location.hash = href(["tokenizers", name], { made: "1" }); } // its page, with "Train its vocabulary"
        }} />}
      </Dialog>
    </>
  );
}

// A new recipe (POST /api/tokenizers), or, given one, its edit (PUT /api/tokenizers/NAME).
function RecipeForm({ recipe, name: givenName, dataset: givenDataset, onDone }: { recipe?: Recipe; name?: string; dataset?: string; onDone: (a: Answer<unknown>, name: string) => void }) {
  const { datasets } = useDatasets();
  const [name, setName] = useState(recipe?.name ?? givenName ?? "");
  const [depot, setDepot] = useState(recipe?.depot ?? "");
  const [elsewhere, setElsewhere] = useState(Boolean(recipe?.depot)); // a dataset of another depot, named by hand
  const [dataset, setDataset] = useState(recipe ? (recipe.depot ? recipe.dataset : recipe.dataset_name ?? recipe.dataset) : givenDataset ?? "");
  // the sample's size as typed, in bytes or with a unit ("2 MB"); an edit starts from the recipe's exact bytes unless the short form says the same
  const [sample, setSample] = useState(recipe ? (parseBytes(bytes(recipe.sample_bytes)) === recipe.sample_bytes ? bytes(recipe.sample_bytes) : String(recipe.sample_bytes)) : bytes(SAMPLE));
  const sampleBytes = parseBytes(sample);
  const [size, setSize] = useState(String(recipe?.vocab_size ?? TOKENS));
  const [note, setNote] = useState(recipe?.note ?? "");
  const [problems, setProblems] = useState<string[]>([]);
  const send = async () => {
    const body = { depot: depot.trim() || null, dataset: dataset.trim(), sample_bytes: sampleBytes, vocab_size: Number(size), note: note.trim() || null };
    const a = recipe ? await put(`/api/tokenizers/${part(recipe.name)}`, body) : await post("/api/tokenizers", { ...body, name: name.trim() });
    setProblems(a.ok ? [] : a.problems);
    onDone(a, recipe?.name ?? name.trim());
  };
  return (
    <form onSubmit={(e) => { e.preventDefault(); send(); }}>
      <RequiredNote />
      {recipe && <p className="form-note small">A new dataset, sample or size makes version {recipe.version + 1}.<Help label="versions">Its vocabulary trains anew; a change of the note alone keeps version {recipe.version}.</Help></p>}
      <div className="fields">
        {!recipe && <TextInput label={required("Name")} hint="Small letters, digits, dot, dash, underscore. Kept for good." required value={name} onChange={(e) => setName(e.target.value)} />}
        {!elsewhere && datasets?.ok && datasets.data.length > 0 ? (
          <Select label={required("Dataset")} required value={datasets.data.some((d) => d.name === dataset) ? dataset : ""} onChange={(e) => {
            if (e.target.value === "\u0000other") { setElsewhere(true); setDataset(""); } else setDataset(e.target.value);
          }}>
            <option value="" disabled>choose a dataset</option>
            {datasets.data.map((d) => <option key={d.id} value={d.name}>{d.name}</option>)}
            <option value={"\u0000other"}>a dataset of another depot</option>
          </Select>
        ) : (
          <>
            <TextInput label={elsewhere ? required("Other depot's address") : optional("Other depot's address")} hint="Its web app's address." value={depot} placeholder="https://..." onChange={(e) => setDepot(e.target.value)} />
            <TextInput label={required("Dataset")} required value={dataset} onChange={(e) => setDataset(e.target.value)} placeholder="its name" />
            {datasets?.ok && datasets.data.length > 0 && <Button variant="quiet" onClick={() => { setElsewhere(false); setDepot(""); setDataset(""); }}>Choose a dataset of this depot instead</Button>}
          </>
        )}
        <TextInput label={required("Sample size")} required value={sample} placeholder="2 MB" onChange={(e) => setSample(e.target.value)}
          hint={sampleBytes ? <>{plural(sampleBytes, "byte")} of plain text at most.</> : <span className="not-allowed">A size such as 2 MB or 500000.</span>} />
        <TextInput label={required("Vocabulary tokens")} hint={<>At least 513.<Help label="the sizes">The sample takes at most this many bytes of plain text, split between the dataset's shares; a dataset smaller than that gives all it has, and conversations are never in it. The vocabulary holds this many tokens: 513 are always there (one for each of the 256 possible bytes, one that ends a text, and 256 kept for later use), and the rest are learned from the sample. The sizes filled in are for a quick test; a vocabulary for real training takes a bigger sample and more tokens.</Help></>} required type="number" min="513" step="1" value={size} onChange={(e) => setSize(e.target.value)} />
        <TextInput label={optional("Note")} value={note} onChange={(e) => setNote(e.target.value)} />
      </div>
      <Problems problems={problems} />
      <div className="sticky-submit">
        <Button type="submit" variant="primary" disabled={(!recipe && !name.trim()) || !dataset.trim() || (elsewhere && !depot.trim()) || !sampleBytes || !(Number(size) >= 513)}>
          {recipe ? "Save the recipe" : "Make the recipe"}
        </Button>
      </div>
    </form>
  );
}

const PAGE = 50; // texts of a vocabulary shown at once

// The texts a vocabulary trained on, a page at a time, each by its title (POST /api/texts/search of the page's ids).
function TrainedOn({ name, version, total }: { name: string; version: number; total: number }) {
  const [offset, setOffset] = useState(0);
  const [rows, setRows] = useState<Answer<{ position: number; depot: string; text: string }[]> | null>(null);
  const [titles, setTitles] = useState<Map<string, string>>(new Map());
  useEffect(() => {
    get<{ position: number; depot: string; text: string }[]>(`/api/tokenizers/${part(name)}/vocabularies/${version}/texts?offset=${offset}&limit=${PAGE}`).then((a) => {
      setRows(a);
      const here = a.ok ? a.data.filter((r) => !r.depot).map((r) => r.text) : [];
      if (here.length) search({ ...EVERY, ids: here }, {}, PAGE).then((found) => {
        if (found.ok) setTitles(new Map(found.data.rows.map((r) => [r.id, r.title ?? r.id])));
      });
    });
  }, [name, version, offset]);
  if (!rows) return <Skeleton lines={3} />;
  if (!rows.ok) return <Problems problems={rows.problems} />;
  return (
    <>
      <ol className="plain small" start={offset + 1}>
        {rows.data.map((r) => <li key={r.position}>{r.depot ? <span className="ledger">{r.depot} {r.text}</span> : <a href={href(["texts", r.text])}>{titles.get(r.text) ?? r.text}</a>}</li>)}
      </ol>
      <Pager offset={offset} limit={PAGE} total={total} onOffset={setOffset} label="The texts it trained on" />
    </>
  );
}

// The button that publishes a vocabulary on the recipe's dataset of this
// depot, or the words that say it does already, with the next step. onDone
// reads the dataset again after any answer, so after a refusal for a change
// made since the page read it, the next click sends the current revision.
function PublishOn({ vocabulary, dataset, onDone }: { vocabulary: Vocabulary; dataset: Dataset; onDone: () => void }) {
  const [answer, setAnswer] = useState<Answer<unknown> | null>(null);
  const at = usePipeline(dataset);
  const publish = async () => {
    const a = await publishVocabulary(dataset, vocabulary.sha256);
    setAnswer(a);
    onDone();
  };
  if (dataset.vocabulary === vocabulary.sha256) return (
    <p className="small row tight"><span>{answer?.ok ? <strong className="cue">Published on <a href={href(["datasets", dataset.id])}>{dataset.name}</a></strong> : <>Published on <a href={href(["datasets", dataset.id])}>{dataset.name}</a></>}{at && at.fresh > 0 && <> · <a href={href(["download", dataset.id])}>{plural(at.fresh, "corpus", "corpora")} built</a></>}</span>
      {at && at.fresh === 0 && <ButtonLink variant="primary" href={href(["download", dataset.id])}>Build a corpus</ButtonLink>}</p>
  );
  return (
    <div className="row tight publish-on">
      <Button variant="primary" onClick={publish}>Publish it on {dataset.name}</Button>
      {dataset.vocabulary && <span className="small aside">Replaces its current one.</span>}
      <Help label="publishing">{dataset.vocabulary ? `${dataset.name} publishes another vocabulary now; this one would take its place.` : `${dataset.name} publishes none yet: model training needs one, and a corpus build picks the published one first.`} It makes a new revision of the dataset; publishing another vocabulary undoes it.</Help>
      {answer && !answer.ok && <Problems problems={answer.problems} />}
    </div>
  );
}

function OneTokenizer({ name, query }: { name: string; query: URLSearchParams }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [editing, setEditing] = useState(false);
  const [answer, setAnswer] = useState<Answer<unknown> | null>(null);
  const [said, setSaid] = useState("");
  const [estimate, setEstimate] = useState<Answer<TrainingEstimate> | null>(null);
  const [started, setStarted] = useState<Job<unknown> | null>(null); // the training this page queued, followed to its end
  // the list first, so a recipe not made yet is known without a request that fails; only the latest reading
  // is shown, so an answer to a reading sent before a training ended never hides what it trained
  const readings = useRef(0);
  const load = useCallback(() => {
    const reading = ++readings.current;
    const latest = () => reading === readings.current;
    get<{ name: string }[]>("/api/tokenizers").then((list) => {
      if (!latest()) return;
      if (list.ok && !list.data.some((r) => r.name === name)) setShown({ ok: false, status: 404, problems: [`no tokenizer recipe ${name}`] } as Answer<Shown>);
      else get<Shown>(`/api/tokenizers/${part(name)}`).then((a) => latest() && setShown(a));
    });
    get<Dataset[]>("/api/datasets").then((a) => latest() && a.ok && setDatasets(a.data));
  }, [name]);
  useEffect(load, [load]);
  usePipelineChange(load); // a job that ends, here or in a part of the page, changes what the recipe holds
  const estimated = useCallback(async () => { setAnswer(null); setEstimate(await get<TrainingEstimate>(`/api/tokenizers/${part(name)}/estimate`)); }, [name]);
  const wantsTraining = query.get("train") === "1";
  useEffect(() => {  // ?train=1: the estimate opens at once, when there is something to train
    if (wantsTraining && shown?.ok && !shown.data.trained && !shown.data.recipe.job && !shown.data.upstream) estimated();
  }, [wantsTraining, shown?.ok]);
  if (shown && !shown.ok && shown.status === 404) return <NotMade name={name} datasets={datasets} />;
  if (!shown) return <Skeleton lines={6} label="Reading the recipe" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const { recipe, revisions, vocabularies, upstream, trained } = shown.data;
  const dataset = recipe.depot ? null : datasets.find((d) => d.id === recipe.dataset) ?? null;
  const train = async () => {
    const a = await post<Job<unknown>>(`/api/tokenizers/${part(name)}/train`, { confirmed: true });
    setEstimate(null);
    if (a.ok) { setStarted(a.data); setAnswer(null); } else { setSaid(""); setAnswer(a); }
    if (a.ok && !waiting(a.data.state)) pipelineChanged(); // ended already: its line follows nothing, so the stages read again here
    load();
  };
  const following = started ?? recipe.job;
  const known: Known = { published: new Map(datasets.filter((d) => d.vocabulary).map((d) => [d.vocabulary!, { id: d.id, name: d.name }])) };
  const current = vocabularies.find((v) => v.version === recipe.version);
  const publishedOn = current ? datasets.find((d) => d.vocabulary === current.sha256) : undefined;
  // the current version's sample came out short of the size asked
  const short = current?.bytes != null && current.bytes < recipe.sample_bytes * SHORT_SAMPLE;
  return (
    <>
      <PageHead title={recipe.name}
        meta={<><a href={href(["tokenizers"])}>tokenizers</a>{!(publishedOn && publishedOn.id === dataset?.id) && <span>{publishedOn ? `published on ${publishedOn.name}` : trained ? "trained, not published" : "not trained yet"}</span>}{upstream && <span>the upstream's</span>}</>}
        chain={<>Its vocabulary trains on a sample of {trainsOn(recipe)}, then is published on the dataset to build corpora and train models. A change of what trains the vocabulary makes a new version, which trains its own.</>}
        actions={<>
          {!trained && !following && !upstream && <Button variant="primary" onClick={estimated}>Train its vocabulary</Button>}
          {!upstream && <Button variant="quiet" onClick={() => { setAnswer(null); setEditing(true); }}>Edit the recipe</Button>}
        </>} />
      {dataset && <DatasetStages id={dataset.id} here="vocabulary" action="none" />}
      {query.get("made") === "1" && !trained && !following && !answer && <p className="done cue" role="status">Made {recipe.name}.</p>}
      {!editing && <Outcome answer={answer} done={() => said} />}
      {following && !(publishedOn && following.state === "done") && <JobLine key={following.id} job={following} onEnded={load} known={known} />}
      <Facts rows={[["version", `v${recipe.version}`], ["dataset", dataset ? null : trainsOn(recipe)],
        ["sample", `at most ${bytes(recipe.sample_bytes)}`], ["vocabulary", plural(recipe.vocab_size, "token")]]} />
      {recipe.note && <p className="small aside">{recipe.note}</p>}
      <Section title="Vocabularies" help={short ? <>Less than the {bytes(recipe.sample_bytes)} asked, and nothing went wrong: {(dataset?.shares?.length ?? 0) > 1 ? "each share takes whole texts, in the dataset's order, up to its part of the sample" : "the sample takes whole texts, in the dataset's order"}, and stops at its first text that does not fit, so a few long texts end it early; a dataset smaller than the sample gives all it has.</> : undefined}>
        {vocabularies.length === 0 ? <p className="small aside">None yet.</p> : vocabularies.map((v) => (
          <div key={v.version} className="vocabulary">
            <p className="page-meta"><span className="cue">v{v.version}{v.version === recipe.version ? "" : " (older)"}</span><span>{plural(v.texts, "plain text")}</span>
              {v.bytes !== null && <span>{bytes(v.bytes)}{v.version === recipe.version && v.bytes < recipe.sample_bytes * SHORT_SAMPLE && <> of {bytes(recipe.sample_bytes)}, whole texts</>}</span>}
              <span>{v.made_at.replace("T", " ").slice(0, 16)} UTC</span></p>
            {v.version === recipe.version && dataset && !upstream && <PublishOn vocabulary={v} dataset={dataset} onDone={() => { setStarted(null); load(); }} />}
            <p><ButtonLink variant="ink" href={`/api/tokenizers/${part(name)}/vocabularies/${v.version}/file`} title={`Its sha256, which a corpus records: ${v.sha256}`}>Download the vocabulary</ButtonLink></p>
            {v.texts > 0 && (
              <details className="vocabulary-texts">
                <summary className="small">Its {plural(v.texts, "plain text")}</summary>
                <TrainedOn name={name} version={v.version} total={v.texts} />
              </details>
            )}
          </div>
        ))}
      </Section>
      {revisions.length > 1 && (
        <details className="fold small">
          <summary className="cue">{plural(revisions.length, "revision")}</summary>
          <table className="ledger-table" aria-label="Its revisions">
            <thead><tr><th scope="col" className="number">revision</th><th scope="col" className="number">version</th><th scope="col">dataset</th><th scope="col" className="number">sample</th><th scope="col" className="number">tokens</th><th scope="col">note</th></tr></thead>
            <tbody>
              {revisions.map((r) => (
                <tr key={r.revision}>
                  <td className="number ledger">{r.revision}</td><td className="number ledger">{r.version}</td>
                  <td>{r.depot ? `${r.dataset} of ${r.depot}` : datasets.find((d) => d.id === r.dataset)?.name ?? r.dataset}</td>
                  <td className="number ledger">{bytes(r.sample_bytes)}</td><td className="number ledger">{r.vocab_size.toLocaleString("en")}</td><td>{r.note ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
      <Dialog open={estimate !== null} onClose={() => setEstimate(null)} title={`Train the vocabulary of ${recipe.name}?`}>
        {estimate && !estimate.ok && <Problems problems={estimate.problems} />}
        {estimate?.ok && (
          <>
            <dl className="facts">
              <dt className="aside">version</dt><dd>{estimate.data.version}, a vocabulary of {plural(recipe.vocab_size, "token")}</dd>
              <dt className="aside">sample</dt><dd>at most {bytes(estimate.data.sample_bytes)} of the plain texts of {trainsOn(recipe)}</dd>
              <dt className="aside">time</dt><dd>{estimate.data.seconds < 60 ? "" : "about "}{duration(estimate.data.seconds)}<Help label="the estimate">{estimate.data.basis}.</Help></dd>
            </dl>
            <p className="small aside">A job of the <a href={href(["jobs"])}>queue</a>; cannot be cancelled here.</p>
            <Button variant="primary" onClick={train}>Confirm and start the training</Button>
          </>
        )}
      </Dialog>
      <Dialog open={editing} onClose={() => setEditing(false)} title={`Edit ${recipe.name}`}>
        {editing && <RecipeForm recipe={recipe} onDone={(a) => { setSaid(!a.ok ? "" : (a.data as { version: number }).version === recipe.version ? `Saved: still version ${recipe.version}.` : `Saved as version ${(a.data as { version: number }).version}, not trained yet.`); setAnswer(a); if (a.ok) { setEditing(false); load(); } }} />}
      </Dialog>
    </>
  );
}

// A recipe a dataset names, not made yet: the datasets that wait for it, the
// stages of the first, and the button that makes it, filled in for that dataset.
function NotMade({ name, datasets }: { name: string; datasets: Dataset[] }) {
  const waiting = datasets.filter((d) => d.tokenizer === name);
  const first = waiting[0] ?? null;
  return (
    <>
      <PageHead title={name}
        meta={<><a href={href(["tokenizers"])}>tokenizers</a><span>no recipe yet</span>{waiting.length > 0 && <span>wanted by <Joined values={waiting.map((d) => d.id)} render={(id) => <a href={href(["datasets", id])}>{waiting.find((d) => d.id === id)!.name}</a>} /></span>}</>}
        chain={<>A tokenizer recipe a dataset names, not made yet: once made, its vocabulary trains on a sample of the dataset.</>}
      />
      {first && <DatasetStages id={first.id} here="vocabulary" action="none" />}
      <div className="empty-state">
        <Facts rows={[["dataset", first?.name ?? null], ["sample", `at most ${bytes(SAMPLE)}`], ["vocabulary", plural(TOKENS, "token")]]} />
        <ButtonLink variant="primary" href={href(["tokenizers"], { new: "1", name, dataset: first?.name ?? null })}>Make the recipe</ButtonLink>
      </div>
    </>
  );
}

export const tokenizers: Page = {
  path: "tokenizers", title: "Tokenizers", group: "data",
  Component: ({ rest, query }) => (rest[0] ? <OneTokenizer key={rest[0]} name={rest[0]} query={query} /> : <Tokenizers key={query.toString()} query={query} />),
};
