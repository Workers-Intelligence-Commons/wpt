// Libraries: named collections of admitted texts. The workspace
// (workspace/workspace.tsx) lists them and opens each one, so #/libraries
// leads there, and #/libraries/ID opens that library in it. This file keeps
// what the workspace's head of a library uses: the forms that make a
// library (wpt depot library new), rename it (rename) and fork it (fork),
// and the list of the revisions a locked library pinned, marking a text
// whose current revision is newer. #/libraries/ID/problems/DECISION is one
// reported problem's view: the problem, who reported it, its texts.

import { Button, Skeleton, TextInput } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, part, post, type Answer } from "../api";
import { EVERY } from "../filter";
import { Facts, LibraryLink, PageHead, Pager, plural, Problems, required, RequiredNote, Section, StatusBadge, useLibraries, type Library } from "../parts";
import { href } from "../route";
import { MatchedTexts } from "../table";
import { when, type DecisionShown } from "./decisions";
import type { Page } from "./index";

export function NewLibrary({ onMade }: { onMade: (a: Answer<{ id: string }>) => void }) {
  const [name, setName] = useState("");
  const [answer, setAnswer] = useState<Answer<{ id: string }> | null>(null);
  const make = async () => {
    const a = await post<{ id: string }>("/api/libraries", { name });
    setAnswer(a);
    onMade(a);
  };
  return (
    <form onSubmit={(e) => { e.preventDefault(); make(); }}>
      <RequiredNote />
      <TextInput label={required("Name")} required value={name} placeholder="first model, October 2026" onChange={(e) => setName(e.target.value)}
        hint="It opens empty; admitted texts join it from the workspace." />
      {answer && !answer.ok && <Problems problems={answer.problems} />}
      <Button type="submit" variant="primary" disabled={!name.trim()}>Make the library</Button>
    </form>
  );
}

type Pins = { total: number; newer: number; rows: { id: string; title: string | null; revision: number; current: number }[] };

// The revisions a locked library pinned, a page at a time, each text marked when its current revision is newer.
export function Pinned({ id }: { id: string }) {
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(25);
  const [pins, setPins] = useState<Answer<Pins> | null>(null);
  useEffect(() => { post<Pins>(`/api/libraries/${encodeURIComponent(id)}/pins`, { offset, limit }).then(setPins); }, [id, offset, limit]);
  return (
    <>
      {!pins && <Skeleton lines={3} label="Reading the revisions it pinned" />}
      {pins && !pins.ok && <Problems problems={pins.problems} />}
      {pins?.ok && (pins.data.total === 0 ? <p>It pinned no text.</p> : (
        <>
          <p className="small">{plural(pins.data.total, "text")}, each at the revision it reads; {pins.data.newer > 0 ? <>{plural(pins.data.newer, "text")} {pins.data.newer === 1 ? "has" : "have"} a newer revision now, which it does not read.</> : "none has a newer revision."}</p>
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
          <Pager offset={offset} limit={limit} total={pins.data.total} onOffset={setOffset} onLimit={(n) => { setLimit(n); setOffset(0); }} label="Pages of the revisions it pinned" />
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
        hint={library.locked_at ? "Locked, it still takes a new name: the name is not what it holds." : "Its id and its texts stay."} />
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

// A fork without exactly the chosen texts, or with every text (without null).
export function ForkForm({ library, without, onDone }: { library: Library; without: string[] | null; onDone: (a: Answer<unknown>, words: string) => void }) {
  const [name, setName] = useState("");
  const fork = async () => {
    const a = await post<{ id: string }>(`/api/libraries/${encodeURIComponent(library.id)}/forks`, { name, ...(without?.length ? { without } : {}) });
    onDone(a, a.ok ? `Library ${a.data.id}: ${name}, open.` : "");
  };
  return (
    <div className="action-form">
      <RequiredNote />
      <TextInput label={required("Name of the fork")} required value={name} onChange={(e) => setName(e.target.value)} />
      <Button variant="primary" disabled={!name.trim() || (without !== null && without.length === 0)} onClick={fork}>
        {without ? `Fork without ${plural(without.length, "text")}` : "Fork with every text"}
      </Button>
    </div>
  );
}

// An address of the libraries that the workspace now shows: replaced by the workspace's own, so the back button skips it.
function Moved({ to }: { to: string }) {
  useEffect(() => { window.location.replace(to); }, [to]);
  return <Skeleton lines={2} label="Opening the workspace" />;
}

export const libraries: Page = {
  path: "libraries", title: "Libraries", group: null,
  Component: ({ rest }) => (rest[0] && rest[1] === "problems" && rest[2] ? <OneProblem key={rest[2]} library={rest[0]} decision={rest[2]} />
    : <Moved to={rest[0] ? href(["texts"], { library: rest[0] }) : href(["texts"])} />),
};
