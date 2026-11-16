// Datasets: what to train on (wpt depot dataset new), and written out as a
// remote (export); a remote is read back in (wpt depot dataset import) by
// the commands. A new dataset names a tokenizer recipe of the depot (one of
// the Tokenizers page's, or the name a recipe made next will take, filled in
// from the dataset's name), its libraries of this depot ticked (the one the
// address names, ?new=1&library=ID, ticked already), its shares (each tag
// offered with how many texts hold it, and one no ticked library's text
// holds marked) and its optional max_tokens; while it is filled in, it says
// what the ticked libraries hold, measured by the server (POST
// /api/libraries/measure): texts and characters, characters per share, the
// characters the mix keeps and the share that limits it, the texts no share
// takes, and the conversations and tasks every dataset takes; a share's tag
// is checked once typing pauses. Once made, the dataset's page opens;
// ?edit=1 on a dataset's page opens its edit form. A dataset that
// reads a library of another depot is made and changed by the commands
// only. #/datasets/ID is one dataset's view: its libraries with the moment
// of their depot each is read at (the time of that change, from POST
// /api/changes/times, or its number when the time is not known), its shares, its limit, the corpora
// built from it when the store can be read (each with the revision of the
// dataset it was built from, its vocabulary and size linking to its download
// page, stale or deleted), what its libraries hold now, its next step as
// the main button (make its tokenizer recipe, train its vocabulary, publish
// it, build a corpus), the links to its download page (download.tsx),
// where a build starts, and to its export page (export.tsx), its remote
// (its description for another machine, folded away),
// the vocabulary it publishes (the corpus of that vocabulary at a recipe's
// total_tokens is the one training takes), set or cleared among the trained
// vocabularies, and the edit of its name, libraries, shares and limit, each
// change a new revision (wpt depot dataset edit). A dataset that belongs to
// the upstream this depot follows says so and has no edit: only the
// upstream changes it.

import { Button, ButtonLink, Dialog, Select, Skeleton, TextArea, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { get, part, post, type Answer } from "../api";
import { characters, Cut, local, optional, Outcome, PageHead, plural, Problems, publishVocabulary, required, RequiredNote, requiredGroup, Section, useLibraries, useMeasure, type Dataset, type DatasetRef, type Measure } from "../parts";
import { href } from "../route";
import { TickList } from "../ticks";
import type { Page } from "./index";

type Other = { depot: string; library: string };
type Share = { tag: string; share: string };

function Datasets({ query }: { query: URLSearchParams }) {
  const { libraries } = useLibraries();
  const [rows, setRows] = useState<Answer<Dataset[]> | null>(null);
  const [answer, setAnswer] = useState<Answer<any> | null>(null);
  const [said, setSaid] = useState<(d: any) => string>(() => () => "");
  const [making, setMaking] = useState(query.get("new") === "1");
  const [remote, setRemote] = useState<{ name: string; text: string } | null>(null); // the dialog of an exported remote
  const load = useCallback(() => { get<Dataset[]>("/api/datasets").then(setRows); }, []);
  useEffect(load, [load]);
  const show = (a: Answer<any>, words: (d: any) => string) => { setSaid(() => words); setAnswer(a); load(); };

  const exported = async (dataset: string) => {
    const a = await get<unknown>(`/api/datasets/${part(dataset)}/remote`);
    if (a.ok) { setAnswer(null); setRemote({ name: dataset, text: JSON.stringify(a.data, null, 2) }); } else show(a, () => "");
  };

  return (
    <>
      <PageHead title="Datasets"
        chain={<>A dataset reads <a href={href(["libraries"])}>libraries</a>, each as it stood when the dataset was made or last edited, and builds corpora of their texts for training, with the vocabulary of its <a href={href(["tokenizers"])}>tokenizer recipe</a>.</>}
        actions={<Button variant="primary" onClick={() => setMaking(true)}>New dataset</Button>} />
      <Outcome answer={answer} done={said} />
      {rows && !rows.ok && <Problems problems={rows.problems} />}
      {rows?.ok && (rows.data.length === 0 ? <p>No dataset yet.</p> : (
        <div className="table-scroll cards">
          <table className="ledger-table" aria-label="The datasets">
            <thead><tr><th scope="col">dataset</th><th scope="col">tokenizer recipe</th><th scope="col">libraries</th><th scope="col">shares, in order</th><th scope="col">vocabulary</th><th scope="col"><span className="visually-hidden">actions</span></th></tr></thead>
            <tbody>
              {rows.data.map((d) => (
                <tr key={d.id}>
                  <td className="nowrap" data-label=""><a href={href(["datasets", d.id])}><strong>{d.name}</strong></a>{d.upstream ? <span className="aside"> the upstream's</span> : null}</td>
                  <td data-label="tokenizer recipe"><a href={href(["tokenizers", d.tokenizer])}><Cut text={d.tokenizer} /></a></td>
                  <td data-label="libraries">
                    <span className="cut">
                      {d.libraries.map((r, i) => (
                        <span key={`${r.depot}-${r.library}`}>
                          {i > 0 && ", "}
                          {!local(d, r) ? <span className="ledger">{r.depot} {r.library}</span> : <a href={href(["libraries", r.library])}>{libraries.find((l) => l.id === r.library)?.name ?? r.library}</a>}
                        </span>
                      ))}
                    </span>
                  </td>
                  <td data-label="shares"><Cut text={d.shares?.length ? d.shares.map((s) => `${s.tag} ${Math.round(s.share * 1000) / 10} %`).join(", ") : "every text"} /></td>
                  <td data-label="vocabulary">{d.vocabulary ? "published" : <span className="aside">none published</span>}</td>
                  <td className="nowrap" data-label="">
                    <a href={href(["download", d.id])}>Corpora</a>{" "}
                    <a href={href(["export", d.id])}>Texts</a>{" "}
                    <Button variant="quiet" onClick={() => exported(d.name)} title="The description of the dataset that the command wpt depot dataset import reads on another machine">For another machine</Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}

      <Dialog open={making} onClose={() => setMaking(false)} title="New dataset">
        {making && <DatasetForm library={query.get("library")} onDone={(a) => {
          if (!a.ok) return;
          setMaking(false);
          window.location.hash = href(["datasets", (a.data as { id: string }).id], { made: "1" }); // its page, with its next step
        }} />}
      </Dialog>
      <Dialog open={remote !== null} onClose={() => setRemote(null)} title={`${remote?.name ?? ""}, described for another machine`}>
        {remote && <Remote text={remote.text} />}
      </Dialog>
    </>
  );
}

// The form of a dataset: a new one (POST /api/datasets), or, given the
// dataset, its edit as a new revision (POST /api/datasets/ID), its tokenizer fixed.
function DatasetForm({ dataset, library, onDone }: { dataset?: Dataset; library?: string | null; onDone: (a: Answer<any>, name: string) => void }) {
  const { libraries } = useLibraries();
  const [recipes, setRecipes] = useState<Answer<{ name: string }[]> | null>(null);
  const [tags, setTags] = useState<{ tag: string; texts: number }[]>([]);
  const [name, setName] = useState(dataset?.name ?? "");
  const [tokenizer, setTokenizer] = useState(dataset?.tokenizer ?? "");
  const typedTokenizer = useRef(false); // until the recipe's name is typed, it follows the dataset's
  const [mine, setMine] = useState<string[]>(dataset ? dataset.libraries.filter((r) => local(dataset, r)).map((r) => r.library) : library ? [library] : []);
  const [others] = useState<Other[]>(dataset ? dataset.libraries.filter((r) => !local(dataset, r)).map((r) => ({ depot: r.depot ?? "", library: r.library })) : []);
  const [shares, setShares] = useState<Share[]>(dataset?.shares?.map((s) => ({ tag: s.tag, share: String(s.share) })) ?? []);
  const [limit, setLimit] = useState(dataset?.max_tokens ? String(dataset.max_tokens) : "");
  const [problems, setProblems] = useState<string[]>([]);
  useEffect(() => { if (!dataset) get<{ name: string }[]>("/api/tokenizers").then(setRecipes); }, [dataset]);
  useEffect(() => { get<{ tag: string; texts: number }[]>("/api/tags").then((a) => a.ok && setTags(a.data)); }, []);
  const named = (n: string) => { setName(n); if (!dataset && !typedTokenizer.current) setTokenizer(recipeName(n)); };
  // a new dataset of the library the address names takes that library's name until a name is typed
  const typedName = useRef(false);
  const from = !dataset && library ? libraries.find((l) => l.id === library)?.name : undefined;
  useEffect(() => { if (from && !typedName.current) named(from); }, [from]);
  const known = recipes?.ok ? recipes.data.some((r) => r.name === tokenizer.trim()) : true;
  const send = async () => {
    const body = {
      name,
      libraries: mine.map((library) => ({ library })),
      shares: shares.length ? shares.map((s) => ({ tag: s.tag, share: Number(s.share) })) : null,
      max_tokens: limit.trim() ? Number(limit) : null,
      vocabulary: dataset?.vocabulary ?? null,
    };
    const a = dataset ? await post(`/api/datasets/${part(dataset.id)}`, { ...body, revision: dataset.revision }) : await post("/api/datasets", { ...body, tokenizer });
    setProblems(a.ok ? [] : a.problems);
    onDone(a, name);
  };
  const move = (i: number, by: number) => {
    const next = [...shares];
    [next[i], next[i + by]] = [next[i + by], next[i]];
    setShares(next);
  };
  const count = mine.length + others.length;
  // the share tags Holds checks: as typed, once typing paused, so a tag half typed is never called missing
  const typedTags = shares.filter((s) => s.tag.trim() && Number(s.share) >= 0 && s.share.trim() !== "").map((s) => ({ tag: s.tag.trim(), share: Number(s.share) }));
  const [settled, setSettled] = useState(typedTags);
  const typedKey = JSON.stringify(typedTags);
  useEffect(() => { const timer = window.setTimeout(() => setSettled(JSON.parse(typedKey)), 800); return () => window.clearTimeout(timer); }, [typedKey]);
  const measured = useMeasure(mine, settled.length ? settled : null); // read once, for the summary and the line by the button
  return (
    <form onSubmit={(e) => { e.preventDefault(); send(); }}>
      <RequiredNote />
      {dataset && <p className="form-note small">Its id <span className="ledger">{dataset.id}</span> and its tokenizer recipe <span className="ledger">{dataset.tokenizer}</span> stay. A library it reads already keeps the change it is read at; a library added is read at its depot's change now. A corpus built before keeps its own copy of the dataset as it was.</p>}
      <div className="fields">
        <TextInput label={required("Name")} required value={name} onChange={(e) => { typedName.current = true; named(e.target.value); }} />
        {!dataset && (
          <>
            <TextInput label={required("Tokenizer recipe")} required value={tokenizer} list="tokenizer-recipes" onChange={(e) => { typedTokenizer.current = true; setTokenizer(e.target.value); }}
              hint={tokenizer.trim() && !known
                ? <>No recipe has this name yet: the dataset's page offers to make it next, filled in for this dataset.</>
                : <>An existing recipe, or a new name; the dataset's page then offers to make it.</>} />
            <datalist id="tokenizer-recipes">{recipes?.ok && recipes.data.map((r) => <option key={r.name} value={r.name} />)}</datalist>
          </>
        )}
        <TextInput label={optional("Most tokens")} type="number" min="1" step="1" value={limit} placeholder="no limit"
          onChange={(e) => setLimit(e.target.value)}
          hint={<SizeSplit limit={limit} shares={settled} found={measured} />} />
      </div>
      {recipes && !recipes.ok && <Problems problems={recipes.problems} />}
      <fieldset className="form-part">
        <legend className="wic-label">{requiredGroup("Libraries")}</legend>
        <p className="wic-field-hint">At least one. Each is read as it stands now; a later edit of the dataset reads it again.</p>
      <TickList label="Of this depot" chosen={mine} onChange={setMine}
        options={libraries.map((l) => ({ value: l.id, words: `${l.id} ${l.name}`, label: <>{l.name}{l.locked_at ? <span className="aside"> locked</span> : null}</> }))} />
      {others.length > 0 && <p className="form-note small">It reads libraries of another depot ({others.map((o) => `${o.depot} ${o.library}`).join(", ")}), so only the command line changes it.</p>}
      <details className="small aside">
        <summary>From the command line</summary>
        A library of another depot is named with <span className="ledger">wpt depot dataset new</span> or <span className="ledger">wpt depot dataset edit</span>.
      </details>
      </fieldset>
      <fieldset className="form-part">
        <legend className="wic-label">{optional("Shares, in order")}</legend>
        <p className="wic-field-hint">A plain text counts for the first share whose tag it has; a plain text no share takes is left out. No share: every text. Conversations and tasks are always in, whatever the shares, and a size gives them the share of the bytes they hold in the corpus of every token.</p>
        <datalist id="share-tags">{tags.map((t) => <option key={t.tag} value={t.tag}>{plural(t.texts, "text")}</option>)}</datalist>
        {shares.map((s, i) => (
          <div key={i} className="entry-row">
            <TextInput label={required("Tag")} required value={s.tag} list="share-tags" onChange={(e) => setShares(shares.map((o, j) => (j === i ? { ...o, tag: e.target.value } : o)))} />
            <TextInput label={required("Share (0.3 is 30 %)")} required type="number" step="any" min="0" max="1" value={s.share} placeholder="0.3"
              onChange={(e) => setShares(shares.map((o, j) => (j === i ? { ...o, share: e.target.value } : o)))} />
            <Button variant="quiet" disabled={i === 0} onClick={() => move(i, -1)}>Up</Button>
            <Button variant="quiet" disabled={i === shares.length - 1} onClick={() => move(i, 1)}>Down</Button>
            <Button variant="quiet" onClick={() => setShares(shares.filter((_, j) => j !== i))}>Remove</Button>
          </div>
        ))}
        <Button variant="quiet" onClick={() => setShares([...shares, { tag: "", share: "" }])}>Add a share</Button>
      </fieldset>
      <HoldsView libraries={mine} shares={settled} found={measured} />
      <Problems problems={problems} />
      <div className="sticky-submit">
      <HoldsLine libraries={mine} found={measured} />
      <Button type="submit" variant="primary" disabled={!name.trim() || (!dataset && !tokenizer.trim()) || count === 0 || others.length > 0 || shares.some((s) => !s.tag.trim() || s.share.trim() === "") || (limit.trim() !== "" && !(Number(limit) >= 1))}>
        {dataset ? "Save the dataset" : "Make the dataset"}
      </Button>
      </div>
    </form>
  );
}

// The name a dataset's tokenizer recipe is given until one is typed: the
// dataset's name in the letters a recipe's name takes, then "-tok".
export const recipeName = (dataset: string) => {
  const base = dataset.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  return base ? `${base}-tok` : "";
};

// What libraries hold now (useMeasure, the server's measure): their texts
// and characters, the plain texts each share takes (a plain text counts
// for the first share whose tag it has), with its characters, and the
// conversations and tasks, which a dataset always takes. With shares, it
// says what the corpus of every token keeps of each share: the share that
// holds the least for its proportion keeps all its texts and limits the
// others, so a few short texts of one share can hold a whole mix down, and
// what conversations and tasks take of any size. A share that takes no
// text is marked. Nothing shows until the measure is read, so no number
// shows before it is true.
function Holds({ libraries, shares }: { libraries: string[]; shares: { tag: string; share: number }[] }) {
  const found = useMeasure(libraries, shares.length ? shares : null);
  return <HoldsView libraries={libraries} shares={shares} found={found} />;
}

// The line beside the dataset form's button, which stays in view while the
// form scrolls: what the mix keeps, so the click is made knowing it.
function HoldsLine({ libraries, found }: { libraries: string[]; found: Measure | string | null }) {
  if (!libraries.length || found === null || typeof found === "string") return null;
  const limiting = found.shares.find((s) => s.limits);
  return (
    <span className="small" role="status">
      {plural(found.texts, "text")}, {characters(found.characters)}
      {found.mix && <>; the corpus of every token keeps {characters(found.mix.kept_characters)} of plain text{limiting && <>, held down by {limiting.tag}</>}</>}
    </span>
  );
}

function HoldsView({ libraries, shares, found }: { libraries: string[]; shares: { tag: string; share: number }[]; found: Measure | string | null }) {
  if (!libraries.length) return null;
  if (found === null) return <div className="holds small"><p className="aside" role="status">Measuring what its libraries hold…</p></div>;
  if (typeof found === "string") return <div className="holds small"><Problems problems={[found]} /></div>;
  const missing = found.shares.filter((s) => s.texts === 0).map((s) => s.tag);
  const limiting = found.shares.find((s) => s.limits);
  const whole = shares.reduce((t, s) => t + s.share, 0) || 1;
  return (
    <div className="holds small" aria-live="polite">
      <p>
        <span className="cue">Its libraries hold {plural(found.texts, "text")} now</span>, <span title={plural(found.characters, "character")}>{characters(found.characters)}</span>:
        {" "}{plural(found.plain.texts, "plain text")} ({characters(found.plain.characters)}) and {plural(found.items.texts, "conversation or task", "conversations and tasks")} ({characters(found.items.characters)}), always in{found.mix ? `, at ${percent(found.mix.items_share)} of any size` : ", at their share of any size"}
        {found.other.texts > 0 && <>; {plural(found.other.texts, "item")} of other kinds, which no corpus takes</>}.
      </p>
      {found.shares.length > 0 && (
        <ul className="plain">
          {found.shares.map((s) => (
            <li key={s.tag}><span>
              <strong>{s.tag}</strong> <span className="ledger">{percent(s.share! / whole)}</span>: {plural(s.texts, "plain text")}, {characters(s.characters)}
              {s.texts > 0 && s.kept_characters !== undefined && (s.limits
                ? <>; <strong>it limits the mix</strong>: the corpus of every token keeps all of it, and each other share only its proportion of it</>
                : <>; the corpus of every token keeps {s.kept_characters >= s.characters ? "all of it" : <>{characters(s.kept_characters)} of it</>}</>)}
            </span></li>
          ))}
          {found.unshared && found.unshared.texts > 0 && <li className="aside"><span>No share takes {plural(found.unshared.texts, "plain text")} ({characters(found.unshared.characters)}): they are left out.</span></li>}
        </ul>
      )}
      {found.mix && limiting && (
        <p>So the corpus of every token holds {characters(found.mix.kept_characters)} of plain text in this mix: a few short texts of {limiting.tag} hold the other shares down. Add texts tagged {limiting.tag}, lower its share, or accept the smaller corpus.</p>
      )}
      {missing.length > 0 && <p className="not-allowed"><strong>No plain text of these libraries counts for the share {missing.join(", ")}:</strong> none has the tag, or each one that has it counts for an earlier share. That share would take nothing; check its spelling and the order.</p>}
    </div>
  );
}

const percent = (n: number) => `${Math.round(n * 1000) / 10}\u00a0%`; // a no-break space: the number and % never part

// The hint of the dataset form's size: how a corpus of the size typed splits
// between the conversations and tasks (their part of the bytes of the
// corpus of every token, from the measure) and each share (its proportion
// of the rest), in tokens, each a part the build fills up to, never past.
function SizeSplit({ limit, shares, found }: { limit: string; shares: { tag: string; share: number }[]; found: Measure | string | null }) {
  const size = Number(limit);
  if (!limit.trim() || !(size >= 1)) return <>The size a corpus build fills in for this dataset. Empty: every token.</>;
  if (found === null || typeof found === "string") return <>The size a corpus build fills in for this dataset.</>;
  const itemShare = found.mix ? found.mix.items_share : found.items.bytes / ((found.items.bytes + found.plain.bytes) || 1);
  const items = Math.floor(size * itemShare);
  const whole = shares.reduce((t, s) => t + s.share, 0) || 1;
  const about = (n: number) => `about ${Math.round(n).toLocaleString("en")}`;
  const parts = [...(items > 0 ? [`conversations and tasks ${about(items)}`] : []),
    ...(shares.length ? shares.map((s) => `${s.tag} ${about((size - items) * s.share / whole)}`) : [`plain text ${about(size - items)}`])];
  return <>The size a corpus build fills in. At {size.toLocaleString("en")} tokens: {parts.join(", ")}. A build's estimate says what really comes out, since a share stops at its first text that does not fit.</>;
}

type Trained = { sha256: string; recipe: string; name: string | null; version: number };
const trainedWords = (v: Trained) => `${v.name ?? v.recipe} version ${v.version}`;

// The vocabulary a dataset publishes, set or cleared among the trained ones
// (publishVocabulary). The page reads the dataset again after any answer, so
// after a refusal for a change made since it was read, the next one sends the
// current revision.
// The one trained last is chosen while it publishes none.
function Published({ dataset, trained, onDone }: { dataset: Dataset; trained: Answer<Trained[]> | null; onDone: (a: Answer<any>, words: string) => void }) {
  const [chosen, setChosen] = useState(dataset.vocabulary ?? "");
  const mineFirst = trained?.ok ? [...trained.data].sort((a, b) => Number(b.name === dataset.tokenizer) - Number(a.name === dataset.tokenizer)) : [];
  useEffect(() => setChosen(dataset.vocabulary ?? mineFirst[0]?.sha256 ?? ""), [dataset.vocabulary, trained]);
  const send = async () => {
    const a = await publishVocabulary(dataset, chosen || null);
    const picked = mineFirst.find((v) => v.sha256 === chosen);
    onDone(a, chosen ? `${dataset.name} now publishes ${picked ? trainedWords(picked) : "the vocabulary chosen"}: model training takes it, and corpora can be built with it.` : `${dataset.name} publishes no vocabulary now.`);
  };
  if (!trained) return <Skeleton lines={2} label="Reading the vocabularies" />;
  if (!trained.ok) return <Problems problems={trained.problems} />;
  const current = trained.data.find((v) => v.sha256 === dataset.vocabulary);
  return (
    <>
      <p className="small">{dataset.vocabulary
        ? <>It publishes {current ? trainedWords(current) : "a vocabulary the depot no longer holds"}. Model training takes the corpus of that vocabulary, <a href={href(["download", dataset.id])}>built here</a>.</>
        : <>It publishes no vocabulary yet. Model training needs one; a corpus build can use any trained vocabulary.</>}</p>
      {!dataset.upstream && (trained.data.length === 0
        ? <p className="small aside">No vocabulary is trained yet: the next step, above, leads there.</p>
        : (
          <form className="entry-row" onSubmit={(e) => { e.preventDefault(); send(); }}>
            <Select id="published-vocabulary" label="Vocabulary it publishes" value={chosen} onChange={(e) => setChosen(e.target.value)}>
              <option value="">none</option>
              {mineFirst.map((v) => <option key={v.sha256} value={v.sha256}>{trainedWords(v)}</option>)}
            </Select>
            <Button type="submit" variant={dataset.vocabulary ? "quiet" : "primary"} disabled={chosen === (dataset.vocabulary ?? "")}>{chosen ? "Publish it" : "Publish none"}</Button>
            {chosen === (dataset.vocabulary ?? "") && <span className="small aside">{chosen ? "Already published: choose another to replace it." : "None is published."}</span>}
          </form>
        ))}
    </>
  );
}

type Corpus = { id: string; revision: number; vocabulary: string; max_tokens: number | null; built_at: string; texts: number; conversations: number; total_tokens: number; stale: { erasures: string[] } | null; deleted: { at: string; why: string; whole: boolean } | null };
type Shown = {
  dataset: Dataset; upstream: boolean; remote: unknown; corpora: Corpus[] | null; store: string | null;
  libraries: (DatasetRef & { local: boolean; name: string | null; locked_at: string | null })[];
};
type Recipe = { name: string; version: number; trained?: number | boolean; job: unknown };

// The step that comes next for a dataset, as the main button of its page,
// with a sentence that says why: make its tokenizer recipe (filled in on the
// Tokenizers page), train its vocabulary (the recipe's page, its estimate
// open), publish it (the section below), then build a corpus.
function nextStep(dataset: Dataset, recipes: Answer<Recipe[]> | null, trained: Answer<Trained[]> | null): { button: ReactNode; words: ReactNode } | null {
  if (!recipes?.ok || !trained?.ok) return null;
  const recipe = recipes.data.find((r) => r.name === dataset.tokenizer);
  const vocabularies = trained.data.filter((v) => v.name === dataset.tokenizer);
  const toPublish = () => { const el = document.getElementById("published-vocabulary"); el?.scrollIntoView({ block: "center" }); el?.focus(); };
  if (!recipe) return { button: <ButtonLink variant="primary" href={href(["tokenizers"], { new: "1", name: dataset.tokenizer, dataset: dataset.name })}>Make its tokenizer recipe</ButtonLink>,
    words: <>Next: make its tokenizer recipe, {dataset.tokenizer}, which trains the vocabulary its corpora are tokenized with.</> };
  if (!recipe.trained && vocabularies.length === 0) return { button: <ButtonLink variant="primary" href={href(["tokenizers", recipe.name], { train: "1" })}>Train its vocabulary</ButtonLink>,
    words: recipe.job ? <>Its vocabulary is being trained: <a href={href(["tokenizers", recipe.name])}>follow it</a>.</> : <>Next: train the vocabulary of {recipe.name}, after an estimate of its time.</> };
  if (!dataset.vocabulary) return { button: <Button variant="primary" onClick={toPublish}>Publish its vocabulary</Button>,
    words: <>Next: publish the trained vocabulary, below, so model training can take it.</> };
  return { button: <ButtonLink variant="primary" href={href(["download", dataset.id])}>Build a corpus</ButtonLink>,
    words: <>Its vocabulary is published: build a corpus of tokens, or download one already built.</> };
}

// The UTC time of each of this depot's changes asked for (POST
// /api/changes/times), "YYYY-MM-DD HH:MM"; null until read, and empty when
// the reading fails, so the change numbers show instead.
function useChangeTimes(changes: number[]): Map<number, string> | null {
  const [found, setFound] = useState<Map<number, string> | null>(null);
  const key = [...new Set(changes)].sort((a, b) => a - b).join(",");
  useEffect(() => {
    if (!key) return;
    post<{ change: number; made_at: string | null }[]>("/api/changes/times", { changes: key.split(",").map(Number) })
      .then((a) => setFound(new Map(a.ok ? a.data.filter((r) => r.made_at).map((r) => [r.change, r.made_at!.replace("T", " ").slice(0, 16)]) : [])));
  }, [key]);
  return found;
}

// One dataset: what it holds, its next step, what it reads, the corpora built from it, its remote, and its edit.
// edit: the address asked for its edit (?edit=1, from a corpus page whose shares came out uneven), opened at once
function OneDataset({ id, made, edit = false }: { id: string; made: boolean; edit?: boolean }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  const [recipes, setRecipes] = useState<Answer<Recipe[]> | null>(null);
  const [trained, setTrained] = useState<Answer<Trained[]> | null>(null);
  const [editing, setEditing] = useState(edit);
  useEffect(() => { if (edit) setEditing(true); }, [edit]); // the address can ask for it while the page is open
  const [answer, setAnswer] = useState<Answer<any> | null>(null);
  const [said, setSaid] = useState("");
  const load = useCallback(() => {
    get<Shown>(`/api/datasets/${part(id)}`).then(setShown);
    get<Recipe[]>("/api/tokenizers").then(setRecipes);
    get<Trained[]>("/api/vocabularies").then(setTrained);
  }, [id]);
  useEffect(load, [load]);
  const changeTimes = useChangeTimes(shown?.ok ? shown.data.libraries.filter((r) => !r.depot).map((r) => r.change) : []);
  if (!shown) return <Skeleton lines={6} label="Reading the dataset" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const { dataset, libraries, corpora } = shown.data;
  const next = shown.data.upstream ? null : nextStep(dataset, recipes, trained);
  const mine = libraries.filter((r) => r.local).map((r) => r.library);
  const times = changeTimes;
  return (
    <>
      <PageHead title={dataset.name}
        meta={<><span>tokenizer recipe <a href={href(["tokenizers", dataset.tokenizer])}>{dataset.tokenizer}</a></span><span>{plural(libraries.length, "library", "libraries")}</span><span>{dataset.max_tokens ? `a new corpus starts at ${plural(dataset.max_tokens, "token")}` : "no default corpus size"}</span>{shown.data.upstream ? <span className="aside">the upstream's</span> : null}<span className="aside" title="Each edit of a dataset makes a new revision; a corpus records the revision it was built from.">{dataset.revision === 1 ? "never edited" : `edited ${dataset.revision - 1 === 1 ? "once" : `${dataset.revision - 1} times`}`}</span></>}
        note={shown.data.upstream ? "It belongs to the upstream this depot follows: only the upstream edits it, and a pull brings its edits here. Its corpus can still be built here." : undefined}
        chain={<>One of the <a href={href(["datasets"])}>datasets</a>. It reads libraries, and gives their texts as files on <a href={href(["export", dataset.id])}>its export page</a>, with no tokenizer, and token files for training on <a href={href(["download", dataset.id])}>its corpora page</a>.</>}
        actions={<>
          {next?.button}
          <ButtonLink variant="quiet" href={href(["download", dataset.id])}>Corpora</ButtonLink>
          <ButtonLink variant="quiet" href={href(["export", dataset.id])}>Export its texts</ButtonLink>
          {!shown.data.upstream && <Button variant="quiet" onClick={() => { setAnswer(null); setEditing(true); }}>Edit it</Button>}
        </>} />
      {made && !answer && <p className="done cue" role="status">Made the dataset {dataset.name}.</p>}
      {next && <p className="next-step">{next.words}</p>}
      {!editing && <Outcome answer={answer} done={() => said} />}
      <Holds libraries={mine} shares={dataset.shares ?? []} />
      <Section title="The vocabulary it publishes">
        <Published dataset={dataset} trained={trained} onDone={(a, words) => { setSaid(words); setAnswer(a); load(); }} />
      </Section>
      <Section title="Libraries it reads">
        <p className="small aside">Each library is read as it stood when the dataset was made or last edited; a text put in or taken out after that changes nothing here until the dataset is edited again.</p>
        <div className="table-scroll cards">
          <table className="ledger-table" aria-label="The libraries it reads">
            <thead><tr><th scope="col">library</th><th scope="col">now</th><th scope="col">depot</th><th scope="col">read as it stood on (UTC)</th></tr></thead>
            <tbody>
              {libraries.map((r) => (
                <tr key={`${r.depot}-${r.library}`}>
                  <td className="wide-cell" data-label="">{r.local ? <a href={href(["libraries", r.library])}>{r.name}</a> : <span className="ledger">{r.library}</span>}</td>
                  <td data-label="now">{r.local ? (r.locked_at ? "locked" : "open") : <span className="aside">unknown</span>}</td>
                  <td className="ledger" data-label="depot">{r.depot ? <Cut text={r.depot} /> : "this depot"}</td>
                  <td className="ledger nowrap" data-label="read as it stood on" title={`change ${r.change} of its depot`}>{!r.depot && times?.get(r.change) ? times.get(r.change) : `change ${r.change}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <Section title="Shares, in order">
        {dataset.shares?.length ? (
          <ol className="plain small">
            {dataset.shares.map((s) => <li key={s.tag}><a href={href(["tags", s.tag])}>{s.tag}</a> <span className="ledger">{Math.round(s.share * 1000) / 10} %</span></li>)}
          </ol>
        ) : <p className="small">No share: every text of its libraries.</p>}
        <p className="small aside">Conversations and tasks are always in, and keep the same part of every corpus size as in the corpus of every token; the shares split the rest.</p>
      </Section>
      <Section title="Corpora built from it">
        {corpora === null ? (
          <p className="small aside">The store cannot be read here, so only the dataset's definition is shown. {shown.data.store}</p>
        ) : corpora.length === 0 ? <p className="small">No corpus is built from it yet.</p> : (
          <div className="table-scroll cards">
            <table className="ledger-table" aria-label="The corpora built from it">
              <thead><tr><th scope="col">vocabulary, size</th><th scope="col" className="number">texts</th><th scope="col" className="number">tokens of plain text</th><th scope="col" className="number">conversations</th><th scope="col">built (UTC)</th><th scope="col">state</th></tr></thead>
              <tbody>
                {corpora.map((c) => {
                  const v = trained?.ok ? trained.data.find((t) => t.sha256 === c.vocabulary) : undefined;
                  return (
                    <tr key={c.id}>
                      <td className="nowrap" data-label=""><a href={href(["download", dataset.id, c.vocabulary, c.max_tokens === null ? "all" : String(c.max_tokens)])}>{v ? trainedWords(v) : <span className="ledger">{c.vocabulary.slice(0, 12)}</span>}, {c.max_tokens === null ? "every token" : plural(c.max_tokens, "token")}</a></td>
                      <td className="number ledger" data-label="texts">{c.texts.toLocaleString("en")}</td>
                      <td className="number ledger" data-label="tokens of plain text">{c.total_tokens.toLocaleString("en")}</td>
                      <td className="number ledger" data-label="conversations">{c.conversations.toLocaleString("en")}</td>
                      <td className="ledger nowrap" data-label="built (UTC)">{c.built_at.replace("T", " ").slice(0, 16)}{c.revision < dataset.revision && <span className="aside"> of an older revision</span>}</td>
                      <td data-label="">{c.deleted ? <span className="aside">deleted: {c.deleted.why}</span>
                        : c.stale ? <span className="not-allowed">stale: {c.stale.erasures.map((e, i) => <span key={e}>{i > 0 && ", "}<a href={href(["erasures"], { request: e })}>{e}</a></span>)}</span> : <span className="aside">fresh</span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Section>
      <details className="raw small">
        <summary>Its description for another machine</summary>
        <p className="aside">The command <span className="ledger">wpt depot dataset import</span> reads it on another machine to make the same dataset there.</p>
        <Button variant="quiet" onClick={() => navigator.clipboard?.writeText(JSON.stringify(shown.data.remote, null, 2))}>Copy it</Button>
        <pre className="remote ledger small">{JSON.stringify(shown.data.remote, null, 2)}</pre>
      </details>
      <Dialog open={editing} onClose={() => setEditing(false)} title={`Edit ${dataset.name}`}>
        {editing && <DatasetForm dataset={dataset} onDone={(a, name) => { setSaid(`Saved ${name}. A corpus built before keeps the dataset as it was.`); setAnswer(a); if (a.ok) { setEditing(false); load(); } }} />}
      </Dialog>
    </>
  );
}

// An exported remote, to copy to another machine.
function Remote({ text }: { text: string }) {
  return (
    <>
      <TextArea label="Its description" className="code" rows={14} value={text} readOnly
        hint="Copy it to another machine, then import it there with wpt depot dataset import." />
      <Button variant="ink" onClick={() => navigator.clipboard?.writeText(text)}>Copy it</Button>
    </>
  );
}

export const datasets: Page = {
  path: "datasets", title: "Datasets", group: "data",
  Component: ({ rest, query }) => (rest[0] ? <OneDataset key={rest[0]} id={rest[0]} made={query.get("made") === "1"} edit={query.get("edit") === "1"} /> : <Datasets key={query.toString()} query={query} />),
};
