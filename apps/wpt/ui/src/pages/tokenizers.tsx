// Tokenizers: the depot's tokenizer recipes (GET /api/tokenizers), each the
// dataset whose sample trains its vocabulary (this depot's, or another
// depot's named by the web address of its web app), the sample's bytes, the
// vocabulary's size and a note, with whether its version has a vocabulary;
// a new recipe (wpt depot tokenizer new), its dataset chosen in a list of
// this depot's or named by hand on another depot, its form opened and filled in by
// the address (?new=1&name=NAME&dataset=DATASET, as a dataset's page links
// to it), with a small sample and vocabulary for a quick test; once made,
// its page opens. #/tokenizers/NAME is one recipe: its revisions, its
// vocabularies, each with its file and the texts it trained on (folded, by
// title, a page at a time), the training of its version's vocabulary
// queued as a job once its estimate (GET /api/tokenizers/NAME/estimate: the
// sample's bytes and an approximate time) is shown and confirmed (?train=1
// opens the estimate at once), then followed to its end, and its edit as a
// new revision, which raises its version when it changes what trains the
// vocabulary (wpt depot tokenizer edit). A vocabulary its dataset of this
// depot does not publish has a button that publishes it there (POST
// /api/datasets/ID, its row as it is but the vocabulary). A recipe of the
// upstream this depot follows has no edit.

import { Button, ButtonLink, Dialog, Select, Skeleton, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, part, post, put, type Answer, type Job } from "../api";
import { EVERY, search } from "../filter";
import { Cut, Outcome, PageHead, Pager, plural, Problems, optional, publishVocabulary, required, RequiredNote, Section, useDatasets, type Dataset } from "../parts";
import { href } from "../route";
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
const TOKENS = 2048;

function Tokenizers({ query }: { query: URLSearchParams }) {
  const [rows, setRows] = useState<Answer<Recipe[]> | null>(null);
  const [making, setMaking] = useState(query.get("new") === "1");
  const [answer, setAnswer] = useState<Answer<{ id: string }> | null>(null);
  const load = useCallback(() => { get<Recipe[]>("/api/tokenizers").then(setRows); }, []);
  useEffect(load, [load]);
  return (
    <>
      <PageHead title="Tokenizers"
        chain={<>A tokenizer recipe says how a vocabulary trains: on a sample of a dataset, up to a number of tokens. Each version's vocabulary is kept in the depot once trained, for good, and a corpus build names one of them.</>}
        actions={<Button variant="primary" onClick={() => { setAnswer(null); setMaking(true); }}>New recipe</Button>} />
      {!making && answer && !answer.ok && <Outcome answer={answer} done={() => ""} />}
      {!rows && <Skeleton lines={4} />}
      {rows && !rows.ok && <Problems problems={rows.problems} />}
      {rows?.ok && (rows.data.length === 0 ? <p>No tokenizer recipe yet.</p> : (
        <div className="table-scroll">
          <table className="ledger-table" aria-label="The tokenizer recipes">
            <thead><tr><th scope="col">recipe</th><th scope="col" className="number">version</th><th scope="col">trains on</th><th scope="col" className="number">sample</th><th scope="col" className="number">tokens</th><th scope="col">vocabulary</th></tr></thead>
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
  const [sample, setSample] = useState(String(recipe?.sample_bytes ?? SAMPLE));
  const [size, setSize] = useState(String(recipe?.vocab_size ?? TOKENS));
  const [note, setNote] = useState(recipe?.note ?? "");
  const [problems, setProblems] = useState<string[]>([]);
  const send = async () => {
    const body = { depot: depot.trim() || null, dataset: dataset.trim(), sample_bytes: Number(sample), vocab_size: Number(size), note: note.trim() || null };
    const a = recipe ? await put(`/api/tokenizers/${part(recipe.name)}`, body) : await post("/api/tokenizers", { ...body, name: name.trim() });
    setProblems(a.ok ? [] : a.problems);
    onDone(a, recipe?.name ?? name.trim());
  };
  return (
    <form onSubmit={(e) => { e.preventDefault(); send(); }}>
      <RequiredNote />
      {recipe && <p className="form-note small">A change of the dataset, the sample's bytes or the vocabulary's size makes version {recipe.version + 1}, whose vocabulary trains anew; a change of the note alone keeps version {recipe.version}.</p>}
      {!recipe && <p className="form-note small">A small sample and vocabulary are filled in, for a quick test; a vocabulary for real training takes more of both.</p>}
      <div className="fields">
        {!recipe && <TextInput label={required("Name")} required value={name} onChange={(e) => setName(e.target.value)} hint="Lowercase letters, digits, '.', '_' and '-'; it never changes." />}
        {!elsewhere && datasets?.ok && datasets.data.length > 0 ? (
          <Select label={required("Dataset")} required value={datasets.data.some((d) => d.name === dataset) ? dataset : ""} onChange={(e) => {
            if (e.target.value === "\u0000other") { setElsewhere(true); setDataset(""); } else setDataset(e.target.value);
          }} hint="The dataset whose texts the vocabulary learns from.">
            <option value="" disabled>choose a dataset</option>
            {datasets.data.map((d) => <option key={d.id} value={d.name}>{d.name}</option>)}
            <option value={"\u0000other"}>a dataset of another depot</option>
          </Select>
        ) : (
          <>
            <TextInput label={elsewhere ? required("Other depot's address") : optional("Other depot's address")} value={depot} placeholder="https://..." onChange={(e) => setDepot(e.target.value)}
              hint="The web address of another depot's web app; the vocabulary learns from texts it sends." />
            <TextInput label={required("Dataset")} required value={dataset} onChange={(e) => setDataset(e.target.value)}
              hint={depot.trim() ? "The other depot's dataset, by its name." : "One of this depot's datasets, by its name."} />
            {datasets?.ok && datasets.data.length > 0 && <Button variant="quiet" onClick={() => { setElsewhere(false); setDepot(""); setDataset(""); }}>Choose a dataset of this depot instead</Button>}
          </>
        )}
        <TextInput label={required("Sample bytes")} required type="number" min="1" step="1" value={sample} onChange={(e) => setSample(e.target.value)}
          hint={<>{Number(sample) >= 1 ? <strong>{bytes(Number(sample))}</strong> : null} of plain text at most, split between the dataset's shares; a dataset smaller than that gives all it has.</>} />
        <TextInput label={required("Vocabulary tokens")} required type="number" min="513" step="1" value={size} onChange={(e) => setSize(e.target.value)}
          hint="At most this many tokens. 513 of them are always there (one for each of the 256 possible bytes, one that ends a text, and 256 kept for later use); the rest are learned from the sample." />
        <TextInput label={optional("Note")} value={note} onChange={(e) => setNote(e.target.value)} />
      </div>
      <Problems problems={problems} />
      <Button type="submit" variant="primary" disabled={(!recipe && !name.trim()) || !dataset.trim() || (elsewhere && !depot.trim()) || !(Number(sample) >= 1) || !(Number(size) >= 513)}>
        {recipe ? "Save the recipe" : "Make the recipe"}
      </Button>
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
  const publish = async () => {
    const a = await publishVocabulary(dataset, vocabulary.sha256);
    setAnswer(a);
    onDone();
  };
  if (dataset.vocabulary === vocabulary.sha256) return (
    <p className="small">{answer?.ok ? <strong className="cue">Published on {dataset.name}.</strong> : <>{dataset.name} publishes it.</>} Model training takes it, and corpora are built with it; publishing another vocabulary later replaces it. <ButtonLink variant="primary" href={href(["download", dataset.id])}>Build a corpus</ButtonLink></p>
  );
  return (
    <div className="row tight publish-on">
      <Button variant="primary" onClick={publish}>Publish it on {dataset.name}</Button>
      <span className="small aside">{dataset.vocabulary ? `${dataset.name} publishes another vocabulary now; this one would take its place.` : `${dataset.name} publishes none yet: model training needs one, and a corpus build picks the published one first.`} It makes a new revision of the dataset; publishing another vocabulary undoes it.</span>
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
  const load = useCallback(() => {
    get<Shown>(`/api/tokenizers/${part(name)}`).then(setShown);
    get<Dataset[]>("/api/datasets").then((a) => a.ok && setDatasets(a.data));
  }, [name]);
  useEffect(load, [load]);
  const estimated = useCallback(async () => { setAnswer(null); setEstimate(await get<TrainingEstimate>(`/api/tokenizers/${part(name)}/estimate`)); }, [name]);
  const wantsTraining = query.get("train") === "1";
  useEffect(() => {  // ?train=1: the estimate opens at once, when there is something to train
    if (wantsTraining && shown?.ok && !shown.data.trained && !shown.data.recipe.job && !shown.data.upstream) estimated();
  }, [wantsTraining, shown?.ok]);
  if (!shown) return <Skeleton lines={6} label="Reading the recipe" />;
  if (!shown.ok) return shown.status === 404 ? (
    <>
      <PageHead title={name} chain={<>No <a href={href(["tokenizers"])}>tokenizer recipe</a> has this name yet. A dataset that names it waits for it to be made.</>}
        actions={<ButtonLink variant="primary" href={href(["tokenizers"], { new: "1", name, dataset: datasets.find((d) => d.tokenizer === name)?.name ?? null })}>Make the recipe {name}</ButtonLink>} />
    </>
  ) : <Problems problems={shown.problems} />;
  const { recipe, revisions, vocabularies, upstream, trained } = shown.data;
  const dataset = recipe.depot ? null : datasets.find((d) => d.id === recipe.dataset) ?? null;
  const train = async () => {
    const a = await post<Job<unknown>>(`/api/tokenizers/${part(name)}/train`, { confirmed: true });
    setEstimate(null);
    if (a.ok) { setStarted(a.data); setAnswer(null); } else { setSaid(""); setAnswer(a); }
    load();
  };
  const following = started ?? recipe.job;
  const known: Known = { published: new Map(datasets.filter((d) => d.vocabulary).map((d) => [d.vocabulary!, { id: d.id, name: d.name }])) };
  const current = vocabularies.find((v) => v.version === recipe.version);
  const publishedOn = current ? datasets.find((d) => d.vocabulary === current.sha256) : undefined;
  return (
    <>
      <PageHead title={recipe.name}
        meta={<><span title="A change of what trains the vocabulary makes a new version, which trains its own.">version {recipe.version}</span><span>trains on {trainsOn(recipe)}</span><span>{publishedOn ? `its vocabulary is published on ${publishedOn.name}` : trained ? "its vocabulary is trained, not published" : "not trained yet"}</span>{upstream && <span className="aside">the upstream's</span>}</>}
        chain={<>One of the <a href={href(["tokenizers"])}>tokenizer recipes</a>. Its vocabulary trains on a sample of {trainsOn(recipe)}, then is published on the dataset to build corpora and train models.</>}
        actions={<>
          {!trained && !following && !upstream && <Button variant="primary" onClick={estimated}>Train its vocabulary</Button>}
          {!upstream && <Button variant="quiet" onClick={() => { setAnswer(null); setEditing(true); }}>Edit it</Button>}
        </>} />
      {query.get("made") === "1" && !trained && !following && !answer && <p className="next-step">Made the recipe {recipe.name}. Next: train its vocabulary, after an estimate of its time.</p>}
      {!editing && <Outcome answer={answer} done={() => said} />}
      {following && <JobLine key={following.id} job={following} onEnded={load} known={known} />}
      <Section title="What trains its vocabulary">
        <dl className="facts">
          <dt className="aside">dataset</dt><dd>{trainsOn(recipe)}</dd>
          <dt className="aside">sample</dt><dd>{bytes(recipe.sample_bytes)} of plain text at most <span className="aside">({plural(recipe.sample_bytes, "byte")})</span></dd>
          <dt className="aside">vocabulary</dt><dd>{plural(recipe.vocab_size, "token")} at most</dd>
          {recipe.note && <><dt className="aside">note</dt><dd>{recipe.note}</dd></>}
        </dl>
      </Section>
      <Section title="Vocabularies, one per version">
        {vocabularies.length === 0 ? <p className="small">No vocabulary yet.</p> : vocabularies.map((v) => (
          <div key={v.version} className="vocabulary">
            <p><span className="cue">Version {v.version}{v.version === recipe.version ? "" : " (an older version)"}</span>: trained on {plural(v.texts, "text")}{v.bytes !== null && <>, {bytes(v.bytes)}</>}, {plural(recipe.vocab_size, "token")} at most; made {v.made_at.replace("T", " ").slice(0, 16)} UTC.</p>
            {v.bytes !== null && v.version === recipe.version && v.bytes < recipe.sample_bytes * SHORT_SAMPLE && <p className="small aside">Less than the {bytes(recipe.sample_bytes)} asked, and nothing went wrong: {(dataset?.shares?.length ?? 0) > 1 ? "each share takes whole texts, in the dataset's order, up to its part of the sample" : "the sample takes whole texts, in the dataset's order"}, and stops at its first text that does not fit, so a few long texts end it early; a dataset smaller than the sample gives all it has.</p>}
            {v.version === recipe.version && dataset && !upstream && <PublishOn vocabulary={v} dataset={dataset} onDone={load} />}
            <p className="small"><a href={`/api/tokenizers/${part(name)}/vocabularies/${v.version}/file`} title={`Its sha256, which a corpus records: ${v.sha256}`}>Download its file</a></p>
            {v.texts > 0 && (
              <details className="vocabulary-texts">
                <summary className="small">The {plural(v.texts, "text")} it trained on</summary>
                <TrainedOn name={name} version={v.version} total={v.texts} />
              </details>
            )}
          </div>
        ))}
      </Section>
      <Section title="Revisions">
        <p className="small aside">Every edit makes a revision; one that changes what trains the vocabulary makes a new version too.</p>
        <ol className="plain small">
          {revisions.map((r) => <li key={r.revision}>revision {r.revision}: version {r.version}, {r.depot ? `${r.dataset} of ${r.depot}` : datasets.find((d) => d.id === r.dataset)?.name ?? r.dataset}, {bytes(r.sample_bytes)}, {plural(r.vocab_size, "token")}{r.note ? `, ${r.note}` : ""}</li>)}
        </ol>
      </Section>
      <Dialog open={estimate !== null} onClose={() => setEstimate(null)} title={`Train the vocabulary of ${recipe.name}?`}>
        {estimate && !estimate.ok && <Problems problems={estimate.problems} />}
        {estimate?.ok && (
          <>
            <p>The vocabulary of version {estimate.data.version}, of at most {plural(recipe.vocab_size, "token")}, trains on a sample of at most {bytes(estimate.data.sample_bytes)} of {trainsOn(recipe)}. It runs as a job of the <a href={href(["jobs"])}>queue</a>, one job at a time, and cannot be cancelled here. Once trained, a vocabulary is kept for good; next, this page offers to publish it on the dataset.</p>
            <dl className="facts">
              <dt className="aside">time</dt><dd>{estimate.data.seconds < 60 ? "" : "about "}{duration(estimate.data.seconds)}</dd>
              <dt className="aside">basis</dt><dd className="small">{estimate.data.basis}.</dd>
            </dl>
            <Button variant="primary" onClick={train}>Confirm and start the training</Button>
          </>
        )}
      </Dialog>
      <Dialog open={editing} onClose={() => setEditing(false)} title={`Edit ${recipe.name}`}>
        {editing && <RecipeForm recipe={recipe} onDone={(a) => { setSaid(!a.ok ? "" : (a.data as { version: number }).version === recipe.version ? `Saved ${recipe.name}. It is still version ${recipe.version}, so its vocabulary stays as it is.` : `Saved ${recipe.name} as version ${(a.data as { version: number }).version}, whose vocabulary is not trained yet: train it with the button above.`); setAnswer(a); if (a.ok) { setEditing(false); load(); } }} />}
      </Dialog>
    </>
  );
}

export const tokenizers: Page = {
  path: "tokenizers", title: "Tokenizers", group: "data",
  Component: ({ rest, query }) => (rest[0] ? <OneTokenizer key={rest[0]} name={rest[0]} query={query} /> : <Tokenizers key={query.toString()} query={query} />),
};
