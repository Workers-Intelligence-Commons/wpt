// Tokenizers: the depot's tokenizer recipes (GET /api/tokenizers), each the
// dataset whose sample trains its vocabulary (this depot's, or another
// depot's named by the web address of its web app), the sample's bytes, the
// vocabulary's size and a note, with whether its version has a vocabulary;
// a new recipe (wpt depot tokenizer new). #/tokenizers/NAME is one recipe:
// its revisions, its vocabularies, each with its file and the texts it
// trained on, a page at a time, the training of its version's vocabulary
// queued as a job once its estimate (GET /api/tokenizers/NAME/estimate: the
// sample's bytes and an approximate time) is shown and confirmed, and its edit as a new revision, which raises its version
// when it changes what trains the vocabulary (wpt depot tokenizer edit). A
// recipe of the upstream this depot follows has no edit.

import { Button, Dialog, Skeleton, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, part, post, put, type Answer, type Job } from "../api";
import { Cut, Outcome, PageHead, Pager, plural, Problems, optional, required, RequiredNote, Section, useDatasets } from "../parts";
import { href } from "../route";
import { bytes, duration } from "./download";
import { JobProgress } from "./jobs";
import type { Page } from "./index";

type Recipe = {
  id: string; name: string; version: number; revision: number; depot: string | null; dataset: string; dataset_name: string | null;
  sample_bytes: number; vocab_size: number; note: string | null; trained?: number | boolean; job: Job<unknown> | null;
};
type Vocabulary = { recipe: string; version: number; sha256: string; texts: number; bytes: number | null; made_at: string };
type Shown = { recipe: Recipe; revisions: Recipe[]; vocabularies: Vocabulary[]; upstream: boolean; trained: boolean };
type TrainingEstimate = { sample_bytes: number; version: number; seconds: number; basis: string };

const trainsOn = (r: Recipe) => (r.depot ? <><span className="ledger">{r.dataset}</span> <span className="aside">of</span> <Cut text={r.depot} className="ledger" /></>
  : <a href={href(["datasets", r.dataset])}>{r.dataset_name ?? r.dataset}</a>);

function Tokenizers() {
  const [rows, setRows] = useState<Answer<Recipe[]> | null>(null);
  const [making, setMaking] = useState(false);
  const [answer, setAnswer] = useState<Answer<{ id: string }> | null>(null);
  const load = useCallback(() => { get<Recipe[]>("/api/tokenizers").then(setRows); }, []);
  useEffect(load, [load]);
  return (
    <>
      <PageHead title="Tokenizers"
        chain={<>A tokenizer recipe says how a vocabulary trains: on a sample of a dataset, up to a number of tokens. Each version's vocabulary is kept in the depot once trained, for good, and a corpus build names one of them.</>}
        actions={<Button variant="primary" onClick={() => { setAnswer(null); setMaking(true); }}>New recipe</Button>} />
      {!making && <Outcome answer={answer} done={(d) => `Tokenizer recipe ${d.id}: version 1.`} />}
      {!rows && <Skeleton lines={4} />}
      {rows && !rows.ok && <Problems problems={rows.problems} />}
      {rows?.ok && (rows.data.length === 0 ? <p>No tokenizer recipe yet.</p> : (
        <div className="table-scroll">
          <table className="ledger-table" aria-label="The tokenizer recipes">
            <thead><tr><th scope="col">recipe</th><th scope="col" className="number">version</th><th scope="col">trains on</th><th scope="col" className="number">sample bytes</th><th scope="col" className="number">tokens</th><th scope="col">vocabulary</th></tr></thead>
            <tbody>
              {rows.data.map((r) => (
                <tr key={r.id}>
                  <td className="nowrap"><a href={href(["tokenizers", r.name])}>{r.name}</a></td>
                  <td className="number ledger">{r.version}</td>
                  <td>{trainsOn(r)}</td>
                  <td className="number ledger">{r.sample_bytes.toLocaleString("en")}</td>
                  <td className="number ledger">{r.vocab_size.toLocaleString("en")}</td>
                  <td>{r.trained ? "trained" : r.job ? <JobProgress job={r.job} /> : <span className="aside">not trained</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <Dialog open={making} onClose={() => setMaking(false)} title="New tokenizer recipe">
        {making && <RecipeForm onDone={(a) => { setAnswer(a as Answer<{ id: string }>); if (a.ok) { setMaking(false); load(); } }} />}
      </Dialog>
    </>
  );
}

// A new recipe (POST /api/tokenizers), or, given one, its edit (PUT /api/tokenizers/NAME).
function RecipeForm({ recipe, onDone }: { recipe?: Recipe; onDone: (a: Answer<unknown>) => void }) {
  const { datasets } = useDatasets();
  const [name, setName] = useState(recipe?.name ?? "");
  const [depot, setDepot] = useState(recipe?.depot ?? "");
  const [dataset, setDataset] = useState(recipe ? (recipe.depot ? recipe.dataset : recipe.dataset_name ?? recipe.dataset) : "");
  const [sample, setSample] = useState(String(recipe?.sample_bytes ?? ""));
  const [size, setSize] = useState(String(recipe?.vocab_size ?? ""));
  const [note, setNote] = useState(recipe?.note ?? "");
  const [problems, setProblems] = useState<string[]>([]);
  const send = async () => {
    const body = { depot: depot.trim() || null, dataset: dataset.trim(), sample_bytes: Number(sample), vocab_size: Number(size), note: note.trim() || null };
    const a = recipe ? await put(`/api/tokenizers/${part(recipe.name)}`, body) : await post("/api/tokenizers", { ...body, name: name.trim() });
    setProblems(a.ok ? [] : a.problems);
    onDone(a);
  };
  return (
    <form onSubmit={(e) => { e.preventDefault(); send(); }}>
      <RequiredNote />
      {recipe && <p className="form-note small">A change of the dataset, the sample's bytes or the vocabulary's size makes version {recipe.version + 1}, whose vocabulary trains anew; a change of the note alone keeps version {recipe.version}.</p>}
      <div className="fields">
        {!recipe && <TextInput label={required("Name")} required value={name} onChange={(e) => setName(e.target.value)} hint="Lowercase letters, digits, '.', '_' and '-'; it never changes." />}
        <TextInput label={optional("Another depot")} value={depot} placeholder="this depot" onChange={(e) => setDepot(e.target.value)}
          hint="Empty: a dataset of this depot. Else the web address of another depot's web app (https://...), whose sample route gives the texts." />
        <TextInput label={required("Dataset")} required value={dataset} list="recipe-datasets" onChange={(e) => setDataset(e.target.value)}
          hint={depot.trim() ? "The other depot's dataset, by its name." : "One of this depot's datasets, by its name."} />
        <datalist id="recipe-datasets">{!depot.trim() && datasets?.ok && datasets.data.map((d) => <option key={d.id} value={d.name} />)}</datalist>
        <TextInput label={required("Sample bytes")} required type="number" min="1" step="1" value={sample} onChange={(e) => setSample(e.target.value)}
          hint="The UTF-8 bytes of plain text the vocabulary trains on at most: each share takes its part and stops at its first text that would pass it." />
        <TextInput label={required("Vocabulary tokens")} required type="number" min="513" step="1" value={size} onChange={(e) => setSize(e.target.value)}
          hint="At most this many tokens, 513 of them fixed (the 256 bytes, the end of a text, 256 reserved ids)." />
        <TextInput label={optional("Note")} value={note} onChange={(e) => setNote(e.target.value)} />
      </div>
      <Problems problems={problems} />
      <Button type="submit" variant="primary" disabled={(!recipe && !name.trim()) || !dataset.trim() || !(Number(sample) >= 1) || !(Number(size) >= 513)}>
        {recipe ? "Save the recipe" : "Make the recipe"}
      </Button>
    </form>
  );
}

const PAGE = 50; // texts of a vocabulary shown at once

// The texts a vocabulary trained on, a page at a time.
function TrainedOn({ name, version, total }: { name: string; version: number; total: number }) {
  const [offset, setOffset] = useState(0);
  const [rows, setRows] = useState<Answer<{ position: number; depot: string; text: string }[]> | null>(null);
  useEffect(() => {
    get<{ position: number; depot: string; text: string }[]>(`/api/tokenizers/${part(name)}/vocabularies/${version}/texts?offset=${offset}&limit=${PAGE}`).then(setRows);
  }, [name, version, offset]);
  if (!rows) return <Skeleton lines={3} />;
  if (!rows.ok) return <Problems problems={rows.problems} />;
  return (
    <>
      <ol className="plain small" start={offset + 1}>
        {rows.data.map((r) => <li key={r.position}>{r.depot ? <span className="ledger">{r.depot} {r.text}</span> : <a className="ledger" href={href(["texts", r.text])}>{r.text}</a>}</li>)}
      </ol>
      <Pager offset={offset} limit={PAGE} total={total} onOffset={setOffset} label="The texts it trained on" />
    </>
  );
}

function OneTokenizer({ name }: { name: string }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  const [editing, setEditing] = useState(false);
  const [answer, setAnswer] = useState<Answer<unknown> | null>(null);
  const [said, setSaid] = useState("");
  const [estimate, setEstimate] = useState<Answer<TrainingEstimate> | null>(null);
  const load = useCallback(() => { get<Shown>(`/api/tokenizers/${part(name)}`).then(setShown); }, [name]);
  useEffect(load, [load]);
  useEffect(() => {  // a training under way: read again until it ends
    if (!shown?.ok || !shown.data.recipe.job) return;
    const timer = window.setTimeout(load, 3000);
    return () => window.clearTimeout(timer);
  }, [shown, load]);
  if (!shown) return <Skeleton lines={6} label="Reading the recipe" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const { recipe, revisions, vocabularies, upstream, trained } = shown.data;
  const train = async () => {
    const a = await post<Job<unknown>>(`/api/tokenizers/${part(name)}/train`, { confirmed: true });
    setEstimate(null);
    setSaid(a.ok ? `Queued: ${a.data.title}.` : "");
    setAnswer(a);
    load();
  };
  const estimated = async () => { setAnswer(null); setEstimate(await get<TrainingEstimate>(`/api/tokenizers/${part(name)}/estimate`)); };
  return (
    <>
      <PageHead title={recipe.name}
        meta={<><span className="ledger">{recipe.id}</span><span>version {recipe.version}</span><span>revision {recipe.revision}</span>{upstream && <span className="aside">the upstream's</span>}</>}
        chain={<>One of the <a href={href(["tokenizers"])}>tokenizer recipes</a>. Its vocabulary trains on a sample of {trainsOn(recipe)}; a training runs as a job of the <a href={href(["jobs"])}>queue</a>.</>}
        actions={<>
          {!trained && !recipe.job && !upstream && <Button variant="primary" onClick={estimated}>Train its vocabulary</Button>}
          {!upstream && <Button variant="quiet" onClick={() => { setAnswer(null); setEditing(true); }}>Edit it</Button>}
        </>} />
      {!editing && <Outcome answer={answer} done={() => said} />}
      {recipe.job && <p className="small">Its vocabulary job: {recipe.job.state}, <JobProgress job={recipe.job} /></p>}
      <Section title="What trains its vocabulary">
        <dl className="facts">
          <dt className="aside">dataset</dt><dd>{trainsOn(recipe)}</dd>
          <dt className="aside">sample</dt><dd>{plural(recipe.sample_bytes, "byte")} at most</dd>
          <dt className="aside">vocabulary</dt><dd>{plural(recipe.vocab_size, "token")} at most</dd>
          {recipe.note && <><dt className="aside">note</dt><dd>{recipe.note}</dd></>}
        </dl>
      </Section>
      <Section title="Vocabularies, one per version">
        {vocabularies.length === 0 ? <p className="small">No vocabulary yet.</p> : vocabularies.map((v) => (
          <details key={v.version} className="vocabulary" open={v.version === recipe.version}>
            <summary className="small">version {v.version}: <span className="ledger">{v.sha256.slice(0, 16)}</span>, {plural(v.texts, "text")}{v.bytes !== null && <>, {plural(v.bytes, "byte")}</>}, made {v.made_at.replace("T", " ").slice(0, 16)} UTC</summary>
            <p className="small"><a href={`/api/tokenizers/${part(name)}/vocabularies/${v.version}/file`}>Its file</a> <span className="ledger aside">sha256 {v.sha256}</span></p>
            {v.texts > 0 && <TrainedOn name={name} version={v.version} total={v.texts} />}
          </details>
        ))}
      </Section>
      <Section title="Revisions">
        <ol className="plain small">
          {revisions.map((r) => <li key={r.revision}>revision {r.revision}: version {r.version}, {r.depot ? `${r.dataset} of ${r.depot}` : r.dataset}, {plural(r.sample_bytes, "byte")}, {plural(r.vocab_size, "token")}{r.note ? `, ${r.note}` : ""}</li>)}
        </ol>
      </Section>
      <Dialog open={estimate !== null} onClose={() => setEstimate(null)} title={`Train the vocabulary of ${recipe.name}?`}>
        {estimate && !estimate.ok && <Problems problems={estimate.problems} />}
        {estimate?.ok && (
          <>
            <p>The vocabulary of version {estimate.data.version}, of at most {plural(recipe.vocab_size, "token")}, trains on a sample of at most {bytes(estimate.data.sample_bytes)} of {trainsOn(recipe)}. It runs as a job of the <a href={href(["jobs"])}>queue</a>, one job at a time, and only the command line cancels it. Once trained, a vocabulary is kept for good.</p>
            <dl className="facts">
              <dt className="aside">time</dt><dd>about {duration(estimate.data.seconds)}</dd>
              <dt className="aside">basis</dt><dd className="small">{estimate.data.basis}.</dd>
            </dl>
            <Button variant="primary" onClick={train}>Confirm and start the training</Button>{" "}
            <Button variant="quiet" onClick={() => setEstimate(null)}>Cancel</Button>
          </>
        )}
      </Dialog>
      <Dialog open={editing} onClose={() => setEditing(false)} title={`Edit ${recipe.name}`}>
        {editing && <RecipeForm recipe={recipe} onDone={(a) => { setSaid(a.ok ? `${recipe.name}: version ${(a.data as { version: number }).version}.` : ""); setAnswer(a); if (a.ok) { setEditing(false); load(); } }} />}
      </Dialog>
    </>
  );
}

export const tokenizers: Page = {
  path: "tokenizers", title: "Tokenizers", group: "data",
  Component: ({ rest }) => (rest[0] ? <OneTokenizer key={rest[0]} name={rest[0]} /> : <Tokenizers />),
};
