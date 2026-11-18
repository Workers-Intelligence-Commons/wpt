// Libraries: named collections of admitted texts. #/libraries lists them,
// each with how many texts it holds (the depot's counts, GET /api/counts) and
// the characters of its plain texts and of its items (POST /api/libraries/measure),
// whether it is locked and the datasets that read it, and makes a new one;
// the Texts page (texts/texts.tsx) opens each one, so
// #/libraries/ID opens that library in it. This file keeps
// what the Texts page's head of a library uses: the forms that make a
// library (wpt depot library new), empty or with the admitted texts of the
// whole depot or of a filter saved in this browser, put in as one write over
// that filter, rename it (rename) and fork it (fork),
// and the list of the revisions a locked library pinned, marking a text
// whose current revision is newer. #/libraries/ID/problems/DECISION is one
// reported problem's view: the problem, who reported it, its texts.

import { Button, Dialog, Select, Skeleton, TextInput } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, part, post, type Answer } from "../api";
import { WriteProgress, leftOut, useWrite, type Left } from "../decide-forms";
import { EVERY, sent, type Selection, type TextFilter } from "../filter";
import { characters, Facts, figures, Help, LibraryLink, PageHead, PlacePager, plural, Problems, readers, required, RequiredNote, Section, StatusBadge, useDatasets, useLibraries, useMeasure, type Library } from "../parts";
import { ITEMS } from "../mix";
import { href } from "../route";
import { useSavedFilters } from "../saved-filters";
import { MatchedTexts } from "../table";
import { when, type DecisionShown } from "./decisions";
import type { Page } from "./index";

// onMade: called once the library is made and, when it starts with texts, once they are in it (or the write failed,
// which the form says, the library staying, empty).
export function NewLibrary({ onMade }: { onMade: (a: Answer<{ id: string }>) => void }) {
  const [name, setName] = useState("");
  const [answer, setAnswer] = useState<Answer<{ id: string }> | null>(null);
  const [start, setStart] = useState("admitted"); // "admitted", "empty", or "saved:NAME"
  const [admitted, setAdmitted] = useState<number | null>(null);
  const [problems, setProblems] = useState<string[] | null>(null);
  const [making, setMaking] = useState(false);
  const saved = useSavedFilters();
  const { send, job, busy } = useWrite();
  useEffect(() => { get<{ statuses: Record<string, number> }>("/api/counts").then((a) => a.ok && setAdmitted(a.data.statuses.admitted ?? 0)); }, []);
  const filter: TextFilter | null = start === "admitted" ? { ...EVERY, statuses: ["admitted"] } : start.startsWith("saved:") ? saved.find((f) => `saved:${f.name}` === start)?.filter ?? null : null;
  const make = async () => {
    setProblems(null);
    setMaking(true);
    const a = await post<{ id: string }>("/api/libraries", { name });
    setAnswer(a);
    if (a.ok && filter) {
      const put = await send<{ joined: number; left_out: number; left: Left }>(`/api/libraries/${encodeURIComponent(a.data.id)}/texts`, { selection: sent({ filter }) });
      if (!put.ok) { setMaking(false); setProblems([`The library ${name.trim()} is made, but its texts were not put in it:`, ...put.problems]); return; }
    }
    setMaking(false);
    onMade(a);
  };
  return (
    <form onSubmit={(e) => { e.preventDefault(); if (!making) make(); }}>
      <RequiredNote />
      <TextInput label={required("Name")} required value={name} placeholder="first model, October 2026" onChange={(e) => setName(e.target.value)} />
      <Select label={required("It starts with")} hint="Only admitted texts join a library. Texts can be put in or taken out later on the Texts page." value={start} onChange={(e) => setStart(e.target.value)}>
        <option value="admitted">every admitted text of the depot{admitted !== null ? ` (${admitted.toLocaleString("en")})` : ""}</option>
        {saved.map((f) => <option key={f.name} value={`saved:${f.name}`}>the admitted texts of the saved filter {f.name}</option>)}
        <option value="empty">nothing: choose its texts on the Texts page</option>
      </Select>
      {answer && !answer.ok && <Problems problems={answer.problems} />}
      {problems && <Problems problems={problems} />}
      <WriteProgress job={job} />
      <Button type="submit" variant="primary" disabled={!name.trim() || making || busy}>
        {start === "admitted" && admitted !== null ? `Make it with ${plural(admitted, "text")}` : start === "empty" ? "Make it empty" : "Make it with those texts"}
      </Button>
    </form>
  );
}

// next: the place after which the next page starts, null on the last page
type Pins = { total: number; newer: number; next: string | null; rows: { id: string; title: string | null; revision: number; current: number }[] };

// The revisions a locked library pinned, a page at a time, each text marked when its current revision is newer.
// A page starts after the place of the last row of the one before (places: those of the pages read on the way
// here), so a far page reads as little as the first.
export function Pinned({ id }: { id: string }) {
  const [places, setPlaces] = useState<string[]>([]);
  const [limit, setLimit] = useState(25);
  const [pins, setPins] = useState<Answer<Pins> | null>(null);
  const after = places.at(-1);
  useEffect(() => { post<Pins>(`/api/libraries/${encodeURIComponent(id)}/pins`, { limit, ...(after ? { after } : {}) }).then(setPins); }, [id, after, limit]);
  const next = pins?.ok ? pins.data.next : null;
  return (
    <>
      {!pins && <Skeleton lines={3} label="Reading the revisions it pinned" />}
      {pins && !pins.ok && <Problems problems={pins.problems} />}
      {pins?.ok && (pins.data.total === 0 ? <p>It pinned no text.</p> : (
        <>
          <p className="small">{plural(pins.data.total, "text")}{pins.data.newer > 0 ? <>, {pins.data.newer.toLocaleString("en")} with a newer revision</> : null}.<Help label="pinned revisions">Each text is read at the revision the library pinned; a newer revision is not read.</Help></p>
          <div className="table-scroll">
            <table className="ledger-table" aria-label="The revisions it pinned">
              <thead><tr><th scope="col">text</th><th scope="col">id</th><th scope="col" className="number">pinned revision</th><th scope="col" className="number">current revision</th></tr></thead>
              <tbody>
                {pins.data.rows.map((r) => (
                  <tr key={r.id}>
                    <td className="wide-cell"><a href={href(["texts", r.id], { library: id })} className="cut" title={r.title ?? r.id}>{r.title ?? "untitled"}</a></td>
                    <td className="ledger nowrap">{r.id}</td>
                    <td className="number ledger">{r.revision}</td>
                    <td className="number ledger">{r.current}{r.current > r.revision && <span className="aside"> newer</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <PlacePager onFirst={() => setPlaces([])} onPrevious={places.length ? () => setPlaces(places.slice(0, -1)) : null}
            onNext={next ? () => setPlaces([...places, next]) : null} limit={limit} onLimit={(n) => { setLimit(n); setPlaces([]); }}
            label="Pages of the revisions it pinned" />
        </>
      ))}
    </>
  );
}

// A new name for the library, open or locked: its name is not what it holds.
export function RenameForm({ library, onDone }: { library: Library; onDone: (a: Answer<unknown>, words: string) => void }) {
  const [name, setName] = useState(library.name);
  const rename = async () => onDone(await post(`/api/libraries/${encodeURIComponent(library.id)}/name`, { name: name.trim() }), `It is named ${name.trim()} now.`);
  return (
    <form className="action-form" onSubmit={(e) => { e.preventDefault(); rename(); }}>
      <RequiredNote />
      <TextInput label={required("Name")} required value={name} onChange={(e) => setName(e.target.value)}
        hint={library.locked_at ? "Locked libraries can be renamed too." : "Its id and texts stay."} />
      <Button type="submit" variant="primary" disabled={!name.trim() || name.trim() === library.name}>Rename it</Button>
    </form>
  );
}

// One problem reported on a library: the problem, who reported it and when (its decision), and its texts.
function OneProblem({ library, decision }: { library: string; decision: string }) {
  const { libraries } = useLibraries();
  const [found, setFound] = useState<Answer<DecisionShown> | null>(null);
  useEffect(() => { get<DecisionShown>(`/api/decisions/${part(decision)}`).then(setFound); }, [decision]);
  if (!found) return <Skeleton lines={4} label="Reading the problem" />;
  if (!found.ok) return <Problems problems={found.problems} />;
  const { decision: d, problem } = found.data;
  if (!problem || problem.library !== library) return <Problems problems={[`the decision ${decision} records no problem of the library ${library}`]} />;
  const held = libraries.find((l) => l.id === library);
  return (
    <>
      <PageHead title={problem.problem}
        meta={<><span>reported on <LibraryLink library={held ?? { id: library }} /></span><span>{problem.texts.length ? plural(problem.texts.length, "text") : "the whole library"}</span></>}
        chain={<>A problem reported on one of the <a href={href(["texts"])}>libraries</a>. It changes no text and no membership; the <a href={href(["decisions", d.id])}>decision {d.id}</a> records who reported it.</>} />
      <Facts rows={[
        ["reported by", d.name],
        ["email", <span className="ledger">{d.email}</span>],
        ["when (UTC)", <span className="ledger">{when(d.made_at)}</span>],
        ["decision", <span className="row tight"><a className="ledger" href={href(["decisions", d.id])}>{d.id}</a><StatusBadge status={d.status} /></span>],
      ]} />
      {problem.texts.length > 0 && (
        <Section title="Its texts">
          <MatchedTexts filter={{ ...EVERY, ids: problem.texts }} label="The texts it names" empty="The depot no longer holds any of its texts." />
        </Section>
      )}
    </>
  );
}

// A fork without the texts of a selection, or with every text (without null).
export function ForkForm({ library, without, onDone }: { library: Library; without: Selection | null; onDone: (a: Answer<unknown>, words: string) => void }) {
  const [name, setName] = useState("");
  const { send, job, busy } = useWrite();
  const fork = async () => {
    const a = await send<{ id: string; left: Left }>(`/api/libraries/${encodeURIComponent(library.id)}/forks`, { name, ...(without ? { without: sent(without) } : {}) });
    onDone(a, a.ok ? `Library ${a.data.id}: ${name}, open.${leftOut(a.data.left)}` : "");
  };
  return (
    <div className="action-form">
      <RequiredNote />
      <TextInput label={required("Name of the fork")} required value={name} onChange={(e) => setName(e.target.value)} />
      <Button variant="primary" disabled={!name.trim() || busy} onClick={fork}>
        {without ? "Fork without the ticked texts" : "Fork with every text"}
      </Button>
      <WriteProgress job={job} />
    </div>
  );
}

const short = (n: number) => characters(n).replace(/ characters?$/, "");

// One library's row: its texts, the characters of its plain texts and of its
// conversations and tasks (useMeasure, the depot's counts), its state and the
// datasets that read it.
function LibraryRow({ library: l, texts, reading }: { library: Library; texts: number | null; reading: { id: string; name: string }[] }) {
  const found = figures(useMeasure([l.id]));
  return (
    <tr>
      <td className="wrap-cell" data-label=""><a href={href(["texts"], { library: l.id })}>{l.name}</a></td>
      <td className="number figure" data-label="texts">{texts !== null ? texts.toLocaleString("en") : ""}</td>
      <td className="number figure" data-label="plain chars" title={found ? plural(found.plain.characters, "character") : undefined}>{found ? short(found.plain.characters) : ""}</td>
      <td className="number figure" data-label={`${ITEMS} chars`} title={found ? plural(found.items.characters, "character") : undefined}>{found ? short(found.items.characters) : ""}</td>
      <td data-label="">{l.locked_at ? `locked ${l.locked_at.slice(0, 10)}` : "open"}{l.upstream ? <span className="aside">, the upstream's</span> : null}</td>
      <td className="wrap-cell" data-label="read by"><span className="row-links">{reading.length ? reading.map((d, i) => <span key={d.id}>{i > 0 && ", "}<a href={href(["datasets", d.id])}>{d.name}</a></span>)
        : <a href={href(["datasets"], { new: "1", library: l.id })}>make a dataset from it</a>}</span></td>
    </tr>
  );
}

// Every library, with its size, its state and the datasets that read it; ?new=1 opens the form of a new one.
function Libraries({ query }: { query: URLSearchParams }) {
  const { libraries, loaded, problems } = useLibraries();
  const { datasets } = useDatasets();
  const [counts, setCounts] = useState<Record<string, number> | null>(null);
  const [making, setMaking] = useState(query.get("new") === "1"); // ?new=1: the form opens at once
  useEffect(() => { get<{ libraries: Record<string, number> }>("/api/counts").then((a) => a.ok && setCounts(a.data.libraries)); }, []);
  return (
    <>
      <PageHead title="Libraries"
        chain={<>Named collections of admitted texts, which <a href={href(["datasets"])}>datasets</a> read. A library opens on the <a href={href(["texts"])}>Texts page</a>, where texts are put in it.</>}
        actions={<Button variant="quiet" onClick={() => setMaking(true)}>New library</Button>} />
      {!loaded && <Skeleton lines={4} label="Reading the libraries" />}
      <Problems problems={problems} />
      {loaded && (libraries.length === 0 ? <p className="aside">No library yet.</p> : (
        <div className="table-scroll cards">
          <table className="ledger-table" aria-label="The libraries">
            <thead><tr><th scope="col">library</th><th scope="col" className="number">texts</th><th scope="col" className="number" title="Characters of plain text">plain chars</th><th scope="col" className="number" title={`Characters of ${ITEMS}`}>{ITEMS} chars</th><th scope="col">state</th><th scope="col">read by</th></tr></thead>
            <tbody>
              {libraries.map((l) => <LibraryRow key={l.id} library={l} texts={counts ? counts[l.id] ?? 0 : null} reading={readers(datasets?.ok ? datasets.data : [], l.id)} />)}
            </tbody>
          </table>
        </div>
      ))}
      <Dialog open={making} onClose={() => setMaking(false)} title="New library">
        {making && <NewLibrary onMade={(a) => { if (a.ok) { setMaking(false); window.location.hash = href(["texts"], { library: a.data.id }); } }} />}
      </Dialog>
    </>
  );
}

// An address of a library that the Texts page now shows: replaced by its own, so the back button skips it.
function Moved({ to }: { to: string }) {
  useEffect(() => { window.location.replace(to); }, [to]);
  return <Skeleton lines={2} label="Opening the Texts page" />;
}

export const libraries: Page = {
  path: "libraries", title: "Libraries", group: "data",
  Component: ({ rest, query }) => (rest[0] && rest[1] === "problems" && rest[2] ? <OneProblem key={rest[2]} library={rest[0]} decision={rest[2]} />
    : rest[0] ? <Moved to={href(["texts"], { library: rest[0] })} /> : <Libraries key={query.toString()} query={query} />),
};
