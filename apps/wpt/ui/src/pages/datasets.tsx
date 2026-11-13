// Datasets: what to train on (wpt depot dataset new), and written out as a
// remote (export); a remote is read back in (wpt depot dataset import) and a
// corpus built into the store (wpt depot build) by the commands. A new dataset's tokenizer recipe is chosen among the
// repository's, its libraries of this depot ticked; a dataset that reads a
// library of another depot is made and changed by the commands only. #/datasets/ID is one dataset's view: its libraries with
// the change number of their depot each is read at, its shares, the corpora
// built from it when the store can be read (each with the revision of the
// dataset it was built from), its remote, and the edit of its name,
// libraries and shares as a new revision (wpt depot dataset edit). A
// dataset that belongs to the upstream this depot follows says so and has no
// edit: only the upstream changes it.

import { Button, Dialog, Select, Skeleton, TextArea, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, part, post, type Answer } from "../api";
import { Cut, local, optional, Outcome, PageHead, plural, Problems, required, RequiredNote, requiredGroup, Section, useLibraries, type Dataset, type DatasetRef } from "../parts";
import { href } from "../route";
import { TickList } from "../ticks";
import type { Page } from "./index";

type Other = { depot: string; library: string };
type Share = { tag: string; share: string };

function Datasets() {
  const { libraries } = useLibraries();
  const [rows, setRows] = useState<Answer<Dataset[]> | null>(null);
  const [answer, setAnswer] = useState<Answer<any> | null>(null);
  const [said, setSaid] = useState<(d: any) => string>(() => () => "");
  const [making, setMaking] = useState(false);
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
        chain={<>A dataset reads <a href={href(["libraries"])}>libraries</a>, each as it stood at a change of its depot, and builds a corpus of their texts for training with <span className="ledger">wpt depot build</span>. A remote carries it to another machine, where <span className="ledger">wpt depot dataset import</span> reads it.</>}
        actions={<Button variant="primary" onClick={() => setMaking(true)}>New dataset</Button>} />
      <Outcome answer={answer} done={said} />
      {rows && !rows.ok && <Problems problems={rows.problems} />}
      {rows?.ok && (rows.data.length === 0 ? <p>No dataset yet.</p> : (
        <div className="table-scroll">
          <table className="ledger-table" aria-label="The datasets">
            <thead><tr><th scope="col">dataset</th><th scope="col">id</th><th scope="col">tokenizer</th><th scope="col">libraries, each at a change of its depot</th><th scope="col">shares, in order</th><th scope="col">actions</th></tr></thead>
            <tbody>
              {rows.data.map((d) => (
                <tr key={d.id}>
                  <td className="nowrap"><a href={href(["datasets", d.id])}>{d.name}</a>{d.upstream && <span className="aside"> the upstream's</span>}</td>
                  <td className="ledger nowrap">{d.id}</td>
                  <td className="ledger"><Cut text={d.tokenizer} /></td>
                  <td>
                    <span className="cut" title={d.libraries.map((r) => `${r.depot ?? ""} ${r.library} at change ${r.change}`).join("; ")}>
                      {d.libraries.map((r, i) => (
                        <span key={`${r.depot}-${r.library}`}>
                          {i > 0 && ", "}
                          {!local(d, r) ? <span className="ledger">{r.depot} {r.library}</span> : <a href={href(["libraries", r.library])}>{libraries.find((l) => l.id === r.library)?.name ?? r.library}</a>}
                          <span className="aside"> at {r.change}</span>
                        </span>
                      ))}
                    </span>
                  </td>
                  <td><Cut text={d.shares?.length ? d.shares.map((s) => `${s.tag} ${Math.round(s.share * 1000) / 10} %`).join(", ") : "every text"} /></td>
                  <td className="nowrap">
                    <Button variant="quiet" onClick={() => exported(d.name)}>Export</Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}

      <Dialog open={making} onClose={() => setMaking(false)} title="New dataset">
        {making && <DatasetForm onDone={(a, name) => { show(a, (d) => `Dataset ${d.id}: ${name}.`); if (a.ok) setMaking(false); }} />}
      </Dialog>
      <Dialog open={remote !== null} onClose={() => setRemote(null)} title={`The remote of ${remote?.name ?? ""}`}>
        {remote && <Remote text={remote.text} />}
      </Dialog>
    </>
  );
}

// The form of a dataset: a new one (POST /api/datasets), or, given the
// dataset, its edit as a new revision (POST /api/datasets/ID), its tokenizer fixed.
function DatasetForm({ dataset, onDone }: { dataset?: Dataset; onDone: (a: Answer<any>, name: string) => void }) {
  const { libraries } = useLibraries();
  const [recipes, setRecipes] = useState<Answer<string[]> | null>(null);
  const [name, setName] = useState(dataset?.name ?? "");
  const [tokenizer, setTokenizer] = useState(dataset?.tokenizer ?? "");
  const [mine, setMine] = useState<string[]>(dataset ? dataset.libraries.filter((r) => local(dataset, r)).map((r) => r.library) : []);
  const [others] = useState<Other[]>(dataset ? dataset.libraries.filter((r) => !local(dataset, r)).map((r) => ({ depot: r.depot ?? "", library: r.library })) : []);
  const [shares, setShares] = useState<Share[]>(dataset?.shares?.map((s) => ({ tag: s.tag, share: String(s.share) })) ?? []);
  const [problems, setProblems] = useState<string[]>([]);
  useEffect(() => { if (!dataset) get<string[]>("/api/tokenizers").then(setRecipes); }, [dataset]);
  const send = async () => {
    const body = {
      name,
      libraries: mine.map((library) => ({ library })),
      shares: shares.length ? shares.map((s) => ({ tag: s.tag, share: Number(s.share) })) : null,
    };
    const a = dataset ? await post(`/api/datasets/${part(dataset.id)}`, body) : await post("/api/datasets", { ...body, tokenizer });
    setProblems(a.ok ? [] : a.problems);
    onDone(a, name);
  };
  const move = (i: number, by: number) => {
    const next = [...shares];
    [next[i], next[i + by]] = [next[i + by], next[i]];
    setShares(next);
  };
  const count = mine.length + others.length;
  return (
    <form onSubmit={(e) => { e.preventDefault(); send(); }}>
      <RequiredNote />
      {dataset && <p className="form-note small">Its id <span className="ledger">{dataset.id}</span> and its tokenizer recipe <span className="ledger">{dataset.tokenizer}</span> stay. A library it reads already keeps the change it is read at; a library added is read at its depot's change now. A corpus built before keeps its own copy of the dataset as it was.</p>}
      <div className="fields">
        <TextInput label={required("Name")} required value={name} onChange={(e) => setName(e.target.value)} />
        {!dataset && (
          <Select label={required("Tokenizer recipe")} required value={tokenizer} onChange={(e) => setTokenizer(e.target.value)} hint="A recipe of data/recipes/tokenizers.">
            <option value="">{!recipes ? "Reading the recipes" : recipes.ok ? "Choose a recipe" : "No recipe could be read"}</option>
            {recipes?.ok && recipes.data.map((r) => <option key={r} value={r}>{r}</option>)}
          </Select>
        )}
      </div>
      {recipes && !recipes.ok && <Problems problems={recipes.problems} />}
      <fieldset className="form-part">
        <legend className="wic-label">{requiredGroup("Libraries")}</legend>
        <p className="wic-field-hint">At least one, of this depot. A library of another depot is named with <span className="ledger">wpt depot dataset new</span> or <span className="ledger">edit</span>.</p>
      <TickList label="Of this depot" chosen={mine} onChange={setMine}
        options={libraries.map((l) => ({ value: l.id, words: `${l.id} ${l.name}`, label: <>{l.name} <span className="ledger aside">{l.id}</span>{l.locked_at ? <span className="aside"> locked</span> : null}</>, count: l.texts }))} />
      {others.length > 0 && <p className="form-note small">It reads libraries of another depot ({others.map((o) => `${o.depot} ${o.library}`).join(", ")}): only <span className="ledger">wpt depot dataset edit</span> changes it.</p>}
      </fieldset>
      <fieldset className="form-part">
        <legend className="wic-label">{optional("Shares, in order")}</legend>
        <p className="wic-field-hint">A text counts for the first share whose tag it has; a text no share takes is left out. No share: every text.</p>
        {shares.map((s, i) => (
          <div key={i} className="entry-row">
            <TextInput label={required("Tag")} required value={s.tag} onChange={(e) => setShares(shares.map((o, j) => (j === i ? { ...o, tag: e.target.value } : o)))} />
            <TextInput label={required("Share (0.3 is 30 %)")} required type="number" step="any" min="0" max="1" value={s.share} placeholder="0.3"
              onChange={(e) => setShares(shares.map((o, j) => (j === i ? { ...o, share: e.target.value } : o)))} />
            <Button variant="quiet" disabled={i === 0} onClick={() => move(i, -1)}>Up</Button>
            <Button variant="quiet" disabled={i === shares.length - 1} onClick={() => move(i, 1)}>Down</Button>
            <Button variant="quiet" onClick={() => setShares(shares.filter((_, j) => j !== i))}>Remove</Button>
          </div>
        ))}
        <Button variant="quiet" onClick={() => setShares([...shares, { tag: "", share: "" }])}>Add a share</Button>
      </fieldset>
      <Problems problems={problems} />
      <Button type="submit" variant="primary" disabled={!name.trim() || (!dataset && !tokenizer) || count === 0 || others.length > 0 || shares.some((s) => !s.tag.trim() || s.share.trim() === "")}>
        {dataset ? "Save the dataset" : "Make the dataset"}
      </Button>
    </form>
  );
}

type Corpus = { id: string; revision: number; built_at: string; texts: number; conversations: number; total_tokens: number; stale: { erasures: string[] } | null };
type Shown = {
  dataset: Dataset; upstream: boolean; remote: unknown; corpora: Corpus[] | null; store: string | null;
  libraries: (DatasetRef & { local: boolean; name: string | null; locked_at: string | null })[];
};

// One dataset: what it reads, the corpora built from it, its remote, and its edit.
function OneDataset({ id }: { id: string }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  const [editing, setEditing] = useState(false);
  const [answer, setAnswer] = useState<Answer<any> | null>(null);
  const [said, setSaid] = useState("");
  const load = useCallback(() => { get<Shown>(`/api/datasets/${part(id)}`).then(setShown); }, [id]);
  useEffect(load, [load]);
  if (!shown) return <Skeleton lines={6} label="Reading the dataset" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const { dataset, libraries, corpora } = shown.data;
  return (
    <>
      <PageHead title={dataset.name}
        meta={<><span className="ledger">{dataset.id}</span><span>revision {dataset.revision}</span><span>tokenizer <span className="ledger">{dataset.tokenizer}</span></span><span>{plural(libraries.length, "library", "libraries")}</span>{shown.data.upstream && <span className="aside">the upstream's</span>}</>}
        note={shown.data.upstream ? "It belongs to the upstream this depot follows: only the upstream edits it, and a pull brings its edits here. Its corpus can still be built here." : undefined}
        chain={<>One of the <a href={href(["datasets"])}>datasets</a>. It reads <a href={href(["libraries"])}>libraries</a>, each as it stood at a change of its depot, and builds a corpus of their texts for training with <span className="ledger">wpt depot build {dataset.name}</span>.</>}
        actions={!shown.data.upstream && <Button variant="primary" onClick={() => { setAnswer(null); setEditing(true); }}>Edit it</Button>} />
      {!editing && <Outcome answer={answer} done={() => said} />}
      <Section title="Libraries, each at a change of its depot">
        <div className="table-scroll">
          <table className="ledger-table" aria-label="The libraries it reads">
            <thead><tr><th scope="col">library</th><th scope="col">id</th><th scope="col">depot</th><th scope="col" className="number">change</th><th scope="col">now</th></tr></thead>
            <tbody>
              {libraries.map((r) => (
                <tr key={`${r.depot}-${r.library}`}>
                  <td className="wide-cell">{r.local ? <a href={href(["libraries", r.library])}>{r.name}</a> : <span className="aside">another depot's</span>}</td>
                  <td className="ledger nowrap">{r.library}</td>
                  <td className="ledger">{r.depot ? <Cut text={r.depot} /> : "this depot"}</td>
                  <td className="number ledger">{r.change}</td>
                  <td>{r.local ? (r.locked_at ? "locked" : "open") : <span className="aside">unknown</span>}</td>
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
      </Section>
      <Section title="Corpora built from it">
        {corpora === null ? (
          <p className="small aside">The store cannot be read here, so only the dataset's definition is shown. {shown.data.store}</p>
        ) : corpora.length === 0 ? <p className="small">No corpus of the store is built from it.</p> : (
          <div className="table-scroll">
            <table className="ledger-table" aria-label="The corpora built from it">
              <thead><tr><th scope="col">corpus</th><th scope="col" className="number">of revision</th><th scope="col">built (UTC)</th><th scope="col" className="number">texts</th><th scope="col" className="number">tokens of plain text</th><th scope="col">stale</th></tr></thead>
              <tbody>
                {corpora.map((c) => (
                  <tr key={c.id}>
                    <td className="ledger nowrap" title={c.id}>{c.id.slice(0, 12)}</td>
                    <td className="number ledger">{c.revision}{c.revision < dataset.revision && <span className="aside"> older</span>}</td>
                    <td className="ledger nowrap">{c.built_at.replace("T", " ").slice(0, 16)}</td>
                    <td className="number ledger">{c.texts.toLocaleString("en")}</td>
                    <td className="number ledger">{c.total_tokens.toLocaleString("en")}</td>
                    <td>{c.stale ? <span className="not-allowed">stale: {c.stale.erasures.map((e, i) => <span key={e}>{i > 0 && ", "}<a href={href(["erasures"], { request: e })}>{e}</a></span>)}</span> : <span className="aside">no</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
      <Section title="Its remote" tools={<Button variant="quiet" onClick={() => navigator.clipboard?.writeText(JSON.stringify(shown.data.remote, null, 2))}>Copy the remote</Button>}>
        <pre className="remote ledger small">{JSON.stringify(shown.data.remote, null, 2)}</pre>
      </Section>
      <Dialog open={editing} onClose={() => setEditing(false)} title={`Edit ${dataset.name}`}>
        {editing && <DatasetForm dataset={dataset} onDone={(a, name) => { setSaid(`Dataset ${dataset.id}: ${name}, changed.`); setAnswer(a); if (a.ok) { setEditing(false); load(); } }} />}
      </Dialog>
    </>
  );
}

// An exported remote, to copy to another machine.
function Remote({ text }: { text: string }) {
  return (
    <>
      <TextArea label="Remote" className="code" rows={14} value={text} readOnly
        hint="Copy it to another machine, then import it there with wpt depot dataset import." />
      <Button variant="ink" onClick={() => navigator.clipboard?.writeText(text)}>Copy the remote</Button>
    </>
  );
}

export const datasets: Page = {
  path: "datasets", title: "Datasets", group: "data",
  Component: ({ rest }) => (rest[0] ? <OneDataset key={rest[0]} id={rest[0]} /> : <Datasets />),
};
