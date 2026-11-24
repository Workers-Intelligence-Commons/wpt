// Tokenizers: the depot's tokenizer recipes (GET /api/tokenizers), each the
// snapshot and kinds whose sample trains its vocabulary, the sample's bytes,
// the vocabulary's size and a note, with whether its version has a
// vocabulary; a recipe written before snapshots names a dataset and trains
// nothing until an edit names a snapshot: "Use snapshot NAME" makes that
// edit in one click with the snapshot the upgrade made of that dataset
// (snapshot_offer), on its row and on its page. A new recipe (wpt depot tokenizer
// new) is made of clicks: its snapshot picked in a list (the latest first, or
// the one the address names: ?new=1&snapshot=ID), its kinds as chips from the
// snapshot's summary, the sample and the vocabulary sizes as preset buttons,
// its name filled in from the snapshot's; once made, its page opens.
// #/tokenizers/NAME is one recipe: where it stands as a strip of stages
// (its snapshot, the recipe, its vocabulary, and the build of a model recipe
// naming it), its vocabularies, each with its file, the
// texts it trained on (folded, by title, a page at a time) and a link that
// composes a model recipe with it on its snapshot's page; the training of its
// version's vocabulary queued as a job once its estimate (GET
// /api/tokenizers/NAME/estimate: the sample's bytes and an approximate time)
// is shown and confirmed (?train=1 opens the estimate at once), then followed
// to its end; its revisions folded; and its edit as a new revision, which
// raises its version when it changes what trains the vocabulary (wpt depot
// tokenizer edit). A recipe of the upstream this depot follows has no edit.

import { Button, ButtonLink, Dialog, Picker, Skeleton, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { get, part, post, put, waiting, type Answer, type Job } from "../api";
import { EVERY, search } from "../filter";
import { Help, Outcome, PageHead, Pager, plural, Problems, optional, PLAIN, required, RequiredNote, Section } from "../parts";
import { href } from "../route";
import { pipelineChanged, snapshotHref, Stage, usePipelineChange, useSnapshots, type Snapshot, type SnapshotShown, type Tokenizer as Recipe } from "../pipeline";
import { bytes, duration } from "../sizes";
import { JobLine, JobProgress } from "./jobs";
import type { Page } from "./index";

type Vocabulary = { recipe: string; version: number; sha256: string; texts: number; bytes: number | null; made_at: string };
type Shown = { recipe: Recipe; revisions: Recipe[]; vocabularies: Vocabulary[]; upstream: boolean; trained: boolean };
// below this part of the sample asked, a vocabulary says why it trained on less
const SHORT_SAMPLE = 0.9;
type TrainingEstimate = { sample_bytes: number; version: number; seconds: number; basis: string };

// The kinds a sample takes for now (wpt_depot's SAMPLE_KINDS).
const SAMPLE_KINDS = [PLAIN];
const kindsWords = (kinds: string[] | null | undefined) => (kinds?.length ? kinds.join(", ") : PLAIN);
// What a recipe's vocabulary trains on: its snapshot and kinds, or the dataset a revision written before snapshots names.
const trainsOn = (r: Recipe) => (r.sample
  ? <>{r.snapshot_name ? <a href={snapshotHref({ id: r.sample.snapshot })}>{r.snapshot_name}</a> : <span className="ledger">{r.sample.snapshot}</span>} <span className="aside">{kindsWords(r.sample.kinds)}</span></>
  : <span className="act">the dataset {r.dataset_name ?? r.dataset}, from before snapshots</span>);

// A recipe written before snapshots, given the snapshot made of its dataset in one click: an edit
// (PUT /api/tokenizers/NAME) naming that snapshot, the rest kept, which makes its next version.
function UseSnapshot({ recipe, primary = false, onDone }: { recipe: Recipe; primary?: boolean; onDone: (a: Answer<unknown>) => void }) {
  const [sending, setSending] = useState(false);
  const offer = recipe.snapshot_offer;
  if (!recipe.needs_snapshot || !offer) return null;
  const use = async () => {
    setSending(true);
    const a = await put(`/api/tokenizers/${part(recipe.name)}`, { snapshot: offer.id, kinds: [PLAIN], sample_bytes: recipe.sample_bytes, vocab_size: recipe.vocab_size, note: recipe.note });
    setSending(false);
    if (a.ok) pipelineChanged();
    onDone(a);
  };
  return <Button variant={primary ? "primary" : "quiet"} disabled={sending} onClick={use}
    title={`Its vocabulary then trains on the plain texts of the snapshot ${offer.name}, which the upgrade made from the dataset ${recipe.dataset_name ?? recipe.dataset} this recipe named: the texts of every library it read, in its order and bounds; this makes version ${recipe.version + 1}.`}>Use snapshot {offer.name}</Button>;
}

// The sample and the vocabulary a new recipe is filled in with: small, for a quick test; and the sizes offered as buttons.
const SAMPLE = 2_000_000;
const SAMPLES = [2e6, 2e7, 2e8, 1e9];
const VOCABULARIES = [2048, 8192, 32768, 65536];
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
  const [answer, setAnswer] = useState<Answer<unknown> | null>(null);
  const load = useCallback(() => { get<Recipe[]>("/api/tokenizers").then(setRows); }, []);
  useEffect(load, [load]);
  return (
    <>
      <PageHead title="Tokenizers"
        chain={<>A tokenizer recipe says how a vocabulary trains: on a sample of a <a href={href(["snapshots"])}>snapshot</a>'s texts, up to a number of tokens. Each version's vocabulary is kept in the depot once trained, for good, and a model recipe names one of them, and the <a href={href(["build"])}>Build</a> page builds its corpus.</>}
        actions={<Button variant={rows?.ok && rows.data.length === 0 ? "primary" : "quiet"} onClick={() => { setAnswer(null); setMaking(true); }}>New tokenizer recipe</Button>} />
      {!making && answer && <Outcome answer={answer} done={() => "Saved: it names the snapshot now, ready to train."} />}
      {!rows && <Skeleton lines={4} />}
      {rows && !rows.ok && <Problems problems={rows.problems} />}
      {rows?.ok && (rows.data.length === 0 ? <p className="empty-state">No tokenizer recipe yet.</p> : (
        <div className="table-scroll cards">
          <table className="ledger-table" aria-label="The tokenizer recipes">
            <thead><tr><th scope="col">tokenizer recipe</th><th scope="col" className="number">version</th><th scope="col">trains on</th><th scope="col" className="number">sample</th><th scope="col" className="number">vocabulary size</th><th scope="col">vocabulary</th></tr></thead>
            <tbody>
              {rows.data.map((r) => (
                <tr key={r.id}>
                  <td className="nowrap" data-label="tokenizer recipe"><a href={href(["tokenizers", r.name])}>{r.name}</a></td>
                  <td className="number ledger" data-label="version">{r.version}</td>
                  <td data-label="trains on">{trainsOn(r)}</td>
                  <td className="number ledger" data-label="sample" title={plural(r.sample_bytes, "byte")}>{bytes(r.sample_bytes)}</td>
                  <td className="number ledger" data-label="vocabulary size">{r.vocab_size.toLocaleString("en")}</td>
                  <td data-label="vocabulary">{r.trained ? "trained" : r.job ? <JobProgress job={r.job} /> : r.needs_snapshot ? (r.snapshot_offer ? <UseSnapshot recipe={r} onDone={(a) => { setAnswer(a); load(); }} /> : <span className="aside">needs a snapshot: edit it</span>) : <a href={href(["tokenizers", r.name], { train: "1" })}>train it</a>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <Dialog open={making} onClose={() => setMaking(false)} title="New tokenizer recipe">
        {making && <RecipeForm taken={rows?.ok ? rows.data.map((r) => r.name) : []} snapshot={query.get("snapshot") ?? undefined} onDone={(a, name) => {
          setAnswer(a);
          if (a.ok) { setMaking(false); window.location.hash = href(["tokenizers", name], { made: "1" }); } // its page, with "Train its vocabulary"
        }} />}
      </Dialog>
    </>
  );
}

// A free name for a new recipe of the snapshot: tok-NAME, then tok-NAME-2, ...
function freeName(snapshot: string, taken: string[]): string {
  const base = `tok-${snapshot}`.toLowerCase().replace(/[^a-z0-9._-]+/g, "-");
  if (!taken.includes(base)) return base;
  for (let n = 2; ; n += 1) if (!taken.includes(`${base}-${n}`)) return `${base}-${n}`;
}

// Size buttons, the one equal to value pressed, and a field for another size behind "Other".
function SizeChoice({ label, value, presets, words, onChange, field }: { label: ReactNode; value: number | null; presets: number[]; words: (n: number) => string; onChange: (n: number) => void; field: ReactNode }) {
  const [other, setOther] = useState(value !== null && !presets.includes(value));
  return (
    <fieldset className="chips-field">
      <legend className="cue">{label}</legend>
      <div className="chips">
        {presets.map((p) => <button key={p} type="button" className="chip-button" aria-pressed={!other && value === p} onClick={() => { setOther(false); onChange(p); }}>{words(p)}</button>)}
        <button type="button" className="chip-button" aria-pressed={other} onClick={() => setOther(true)}>Other</button>
      </div>
      {other && field}
    </fieldset>
  );
}

// A new recipe (POST /api/tokenizers), or, given one, its edit (PUT /api/tokenizers/NAME).
function RecipeForm({ recipe, taken = [], snapshot: givenSnapshot, onDone }: { recipe?: Recipe; taken?: string[]; snapshot?: string; onDone: (a: Answer<unknown>, name: string) => void }) {
  const { answer: listed } = useSnapshots();
  const snapshots: Snapshot[] = listed?.ok ? [...listed.data].reverse() : [];
  const [snapshot, setSnapshot] = useState<string | null>(recipe?.sample?.snapshot ?? givenSnapshot ?? null);
  const [summary, setSummary] = useState<SnapshotShown | null>(null);
  const [kinds, setKinds] = useState<string[]>(recipe?.sample?.kinds?.length ? recipe.sample.kinds : [PLAIN]);
  const [name, setName] = useState(recipe?.name ?? "");
  const [named, setNamed] = useState(false); // the name was typed: no longer filled in from the snapshot's
  // the sample's size as typed, in bytes or with a unit ("2 MB")
  const [sample, setSample] = useState(recipe ? (parseBytes(bytes(recipe.sample_bytes)) === recipe.sample_bytes ? bytes(recipe.sample_bytes) : String(recipe.sample_bytes)) : bytes(SAMPLE));
  const sampleBytes = parseBytes(sample);
  const [size, setSize] = useState(String(recipe?.vocab_size ?? TOKENS));
  const [note, setNote] = useState(recipe?.note ?? "");
  const [problems, setProblems] = useState<string[]>([]);
  // the latest snapshot when none is named
  useEffect(() => { if (!snapshot && snapshots.length) setSnapshot(snapshots[0].id); }, [snapshots.length]);
  const chosen = snapshots.find((s) => s.id === snapshot) ?? null;
  useEffect(() => { if (!recipe && !named && chosen) setName(freeName(chosen.name, taken)); }, [chosen?.id]);
  useEffect(() => { setSummary(null); if (snapshot) get<SnapshotShown>(`/api/snapshots/${part(snapshot)}`).then((a) => a.ok && setSummary(a.data)); }, [snapshot]);
  const send = async () => {
    const body = { snapshot, kinds, sample_bytes: sampleBytes, vocab_size: Number(size), note: note.trim() || null };
    const a = recipe ? await put(`/api/tokenizers/${part(recipe.name)}`, body) : await post("/api/tokenizers", { ...body, name: name.trim() });
    setProblems(a.ok ? [] : a.problems);
    onDone(a, recipe?.name ?? name.trim());
  };
  const plainHeld = summary?.summary.kinds[PLAIN]?.[2] ?? null;
  return (
    <form onSubmit={(e) => { e.preventDefault(); send(); }}>
      <RequiredNote />
      {recipe && <p className="form-note small">A new snapshot, kinds, sample or size makes version {recipe.version + 1}.<Help label="versions">Its vocabulary trains anew; a change of the note alone keeps version {recipe.version}.</Help></p>}
      <div className="fields">
        {!listed && <Skeleton lines={1} />}
        {listed && !listed.ok && <Problems problems={listed.problems} />}
        {listed?.ok && (snapshots.length === 0
          ? <p className="small">No snapshot yet: a vocabulary trains on one. <ButtonLink variant="quiet" href={href(["snapshots"], { new: "1" })}>Make a snapshot</ButtonLink></p>
          : <div className="wic-field"><span className="cue">{required("Snapshot")}</span>
              <Picker label="Snapshot" options={snapshots.map((s) => ({ value: s.id, label: s.name, slug: `change ${s.at_change}`, group: snapshots.some((o) => o.owner !== s.owner) ? s.owner : undefined }))} value={snapshot} onChange={setSnapshot} /></div>)}
        {summary && (
          <fieldset className="chips-field">
            <legend className="cue">Kinds<Help label="kinds">A sample takes plain texts only for now; the other kinds of the snapshot are shown for what it holds.</Help></legend>
            <div className="chips">
              {Object.entries(summary.summary.kinds).map(([k, held]) => (
                <button key={k} type="button" className="chip-button" disabled={!SAMPLE_KINDS.includes(k)} aria-pressed={kinds.includes(k)}
                  onClick={() => setKinds((now) => (now.includes(k) ? (now.length > 1 ? now.filter((x) => x !== k) : now) : [...now, k]))}>{k} <span className="aside">{bytes(held[2])}</span></button>
              ))}
            </div>
          </fieldset>
        )}
        <SizeChoice label={<>{required("Sample size")}: {sampleBytes ? bytes(sampleBytes) : "…"}</>} value={sampleBytes} presets={SAMPLES} words={bytes} onChange={(n) => setSample(bytes(n))}
          field={<TextInput label="Other sample size" value={sample} placeholder="2 MB" onChange={(e) => setSample(e.target.value)}
            hint={sampleBytes ? <>{plural(sampleBytes, "byte")} at most.</> : <span className="not-allowed">A size such as 2 MB or 500000.</span>} />} />
        {plainHeld !== null && sampleBytes !== null && sampleBytes > plainHeld && <p className="small aside">The snapshot holds {bytes(plainHeld)} of plain text: the sample takes it all.</p>}
        <SizeChoice label={<>{required("Vocabulary tokens")}: {Number(size) >= 513 ? Number(size).toLocaleString("en") : "…"}<Help label="the sizes">The vocabulary holds this many tokens: 513 are always there (one for each of the 256 possible bytes, one that ends a text, and 256 kept for later use), and the rest are learned from the sample. The smallest sizes are for a quick test.</Help></>}
          value={Number(size) || null} presets={VOCABULARIES} words={(n) => n.toLocaleString("en")} onChange={(n) => setSize(String(n))}
          field={<TextInput label="Other vocabulary size" hint="At least 513." type="number" min="513" step="1" value={size} onChange={(e) => setSize(e.target.value)} />} />
        {!recipe && <TextInput label={required("Name")} hint="Filled in from the snapshot's. Small letters, digits, dot, dash, underscore. Kept for good." required value={name} onChange={(e) => { setNamed(true); setName(e.target.value); }} />}
        <TextInput label={optional("Note")} value={note} onChange={(e) => setNote(e.target.value)} />
      </div>
      <Problems problems={problems} />
      <div className="sticky-submit">
        <Button type="submit" variant="primary" disabled={(!recipe && !name.trim()) || !snapshot || kinds.length === 0 || !sampleBytes || !(Number(size) >= 513)}>
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

function OneTokenizer({ name, query }: { name: string; query: URLSearchParams }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
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
  }, [name]);
  useEffect(load, [load]);
  usePipelineChange(load); // a job that ends, here or in a part of the page, changes what the recipe holds
  const { answer: snapshotsRead } = useSnapshots();
  const estimated = useCallback(async () => { setAnswer(null); setEstimate(await get<TrainingEstimate>(`/api/tokenizers/${part(name)}/estimate`)); }, [name]);
  const wantsTraining = query.get("train") === "1";
  useEffect(() => {  // ?train=1: the estimate opens at once, when there is something to train
    if (wantsTraining && shown?.ok && !shown.data.trained && !shown.data.recipe.job && !shown.data.upstream && !shown.data.recipe.needs_snapshot) estimated();
  }, [wantsTraining, shown?.ok]);
  if (!shown) return <Skeleton lines={6} label="Reading the recipe" />;
  if (!shown.ok) return <><PageHead title={name} meta={<a href={href(["tokenizers"])}>tokenizers</a>} /><Problems problems={shown.problems} /></>;
  const { recipe, revisions, vocabularies, upstream, trained } = shown.data;
  const train = async () => {
    const a = await post<Job<unknown>>(`/api/tokenizers/${part(name)}/train`, { confirmed: true });
    setEstimate(null);
    if (a.ok) { setStarted(a.data); setAnswer(null); } else { setSaid(""); setAnswer(a); }
    if (a.ok && !waiting(a.data.state)) pipelineChanged(); // ended already: its line follows nothing, so the stages read again here
    load();
  };
  const following = started ?? recipe.job;
  const current = vocabularies.find((v) => v.version === recipe.version);
  // where a model recipe with one of its vocabularies is composed: the snapshot its sample read
  const composeOn = (sha256: string) => (recipe.sample ? snapshotHref({ id: recipe.sample.snapshot }, { compose: "1", vocabulary: sha256 }) : href(["snapshots"]));
  // the current version's sample came out short of the size asked
  const short = current?.bytes != null && current.bytes < recipe.sample_bytes * SHORT_SAMPLE;
  // the snapshot its sample reads
  const sampled = recipe.sample && snapshotsRead?.ok ? snapshotsRead.data.find((one) => one.id === recipe.sample!.snapshot) ?? null : null;
  return (
    <>
      <PageHead title={recipe.name}
        meta={<><a href={href(["tokenizers"])}>tokenizers</a><span>tokenizer recipe</span><span>{trained ? "trained" : "not trained yet"}</span>{upstream && <span>the upstream's</span>}</>}
        chain={<>A tokenizer recipe: its vocabulary trains on a sample of {trainsOn(recipe)}; a model recipe names the vocabulary by its sha256. A change of what trains the vocabulary makes a new version, which trains its own.</>}
        actions={<>
          {!trained && !following && !upstream && !recipe.needs_snapshot && <Button variant="primary" onClick={estimated}>Train its vocabulary</Button>}
          {!upstream && <UseSnapshot recipe={recipe} primary onDone={(a) => { setSaid(a.ok ? "Saved: it names the snapshot now, ready to train." : ""); setAnswer(a); if (a.ok) load(); }} />}
          {trained && current && <ButtonLink variant="primary" href={composeOn(current.sha256)}>New recipe with this vocabulary</ButtonLink>}
          {!upstream && <Button variant="quiet" onClick={() => { setAnswer(null); setEditing(true); }}>Edit the tokenizer recipe</Button>}
        </>} />
      {query.get("made") === "1" && !trained && !following && !answer && <p className="done cue" role="status">Made {recipe.name}.</p>}
      {!editing && <Outcome answer={answer} done={() => said} />}
      {following && <JobLine key={following.id} job={following} onEnded={load} />}
      <ol className="stages stages-4" aria-label={`Where ${recipe.name} stands`}>
        {recipe.sample
          ? <Stage name="Snapshot" to={snapshotHref({ id: recipe.sample.snapshot })} state="done" mark={false} value={recipe.snapshot_name ?? recipe.sample.snapshot}
              label={<>{sampled ? `${plural(sampled.texts, "text")}, ` : ""}{kindsWords(recipe.sample.kinds)}</>} />
          : <Stage name="Snapshot" to={href(["snapshots"])} state="problem" word="none" value="none" empty label={<>the dataset {recipe.dataset_name ?? recipe.dataset}, from before snapshots</>}
              line={recipe.snapshot_offer ? undefined : "Edit it to name a snapshot."} />}
        <Stage name="Tokenizer recipe" state="done" mark={false} current value={`v${recipe.version}`} label={`a sample of at most ${bytes(recipe.sample_bytes)}`} />
        <Stage name="Vocabulary" state={trained ? "done" : following ? "waiting" : recipe.needs_snapshot ? "waiting" : "next"} word={trained ? undefined : following ? "training" : recipe.needs_snapshot ? "waiting" : "to train"}
          value={recipe.vocab_size.toLocaleString("en")} label="tokens" />
        <Stage name="Build" to={href(["build"])} state={trained ? "next" : "waiting"} word={trained ? "a recipe file" : "waiting"}
          value="recipe" empty label="a model recipe naming its vocabulary, handed over on the Build page" />
      </ol>
      {recipe.note && <p className="small aside">{recipe.note}</p>}
      <Section title="Vocabularies" help={short ? <>Less than the {bytes(recipe.sample_bytes)} asked, and nothing went wrong: the sample takes the texts that fit it, in the snapshot's order, a text longer than about a million characters as pieces of that length, and closes once nearly full or after a thousand texts in a row that do not fit; a snapshot smaller than the sample gives all it has.</> : undefined}>
        {vocabularies.length === 0 ? <p className="small aside">None yet.</p> : vocabularies.map((v) => (
          <div key={v.version} className="vocabulary">
            <p className="page-meta"><span className="cue">v{v.version}{v.version === recipe.version ? "" : " (older)"}</span><span>{plural(v.texts, "plain text")}</span>
              {v.bytes !== null && <span>{bytes(v.bytes)}{v.version === recipe.version && v.bytes < recipe.sample_bytes * SHORT_SAMPLE && <> of {bytes(recipe.sample_bytes)}, whole texts</>}</span>}
              <span>{v.made_at.replace("T", " ").slice(0, 16)} UTC</span></p>
            <p className="small aside">sha256 <span className="ledger">{v.sha256}</span></p>
            <p className="row tight"><ButtonLink variant="ink" href={`/api/tokenizers/${part(name)}/vocabularies/${v.version}/file`} title={`Its sha256, which a recipe names: ${v.sha256}`}>Download the vocabulary</ButtonLink>
              <ButtonLink variant="quiet" href={composeOn(v.sha256)}>New recipe with this vocabulary</ButtonLink></p>
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
            <thead><tr><th scope="col" className="number">revision</th><th scope="col" className="number">version</th><th scope="col">trains on</th><th scope="col" className="number">sample</th><th scope="col" className="number">tokens</th><th scope="col">note</th></tr></thead>
            <tbody>
              {revisions.map((r) => (
                <tr key={r.revision}>
                  <td className="number ledger">{r.revision}</td><td className="number ledger">{r.version}</td>
                  <td>{r.sample ? `${r.sample.snapshot} ${kindsWords(r.sample.kinds)}` : `dataset ${r.dataset}`}</td>
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

export const tokenizers: Page = {
  path: "tokenizers", title: "Tokenizers", group: "data",
  Component: ({ rest, query }) => (rest[0] ? <OneTokenizer key={rest[0]} name={rest[0]} query={query} /> : <Tokenizers key={query.toString()} query={query} />),
};
