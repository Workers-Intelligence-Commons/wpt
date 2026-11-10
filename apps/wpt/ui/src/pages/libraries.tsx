// Libraries: named collections of admitted texts, made, locked and forked
// (wpt depot library new, lock, fork), and a problem with one reported (wpt
// depot library report). #/libraries/ID is one library's page: its texts,
// filtered, sorted and ticked, and a report or a fork without either the
// ticked texts or those the filter matches, after their review, and its
// rename (wpt depot library rename). #/libraries/ID/problems/DECISION is one
// reported problem's view: the problem, who reported it, its texts. Texts
// join a library from the Texts page. Locking a library pins the revision of
// each text it holds: it reads those for good and takes no new text, while
// its texts still take decisions and edits as new revisions; its page lists
// the revisions it pinned and marks a text whose current revision is newer.
// Only an erasure changes what it reads; a problem can still be reported on
// it, and it can still be renamed. A library that belongs to the upstream
// this depot follows says so, and only the upstream changes it: its page
// keeps only the fork, which makes a library of this depot's own.

import { Button, ButtonLink, Dialog, Skeleton, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, part, post, type Answer } from "../api";
import { EVERY, FilterBar, useSearch, type TextFilter } from "../filter";
import { Facts, LibraryLink, Outcome, PageHead, Pager, plural, Problems, readers, required, RequiredNote, Section, StatusBadge, useDatasets, useLibraries, type Library } from "../parts";
import { ReportForm } from "../decide-forms";
import { Review } from "../review";
import { href } from "../route";
import { MatchedTexts, TextTable } from "../table";
import { when, type DecisionShown } from "./decisions";
import type { Page } from "./index";

function Libraries() {
  const { libraries, loaded, problems, reload } = useLibraries();
  const { datasets } = useDatasets();
  const [making, setMaking] = useState(false);
  const [answer, setAnswer] = useState<Answer<{ id: string }> | null>(null);
  const found = datasets?.ok ? datasets.data : [];
  return (
    <>
      <PageHead title="Libraries"
        chain={<>A library holds admitted <a href={href(["texts"])}>texts</a>; <a href={href(["datasets"])}>datasets</a> read libraries. Locking one keeps it as it is for good.</>}
        actions={<Button variant="primary" onClick={() => { setAnswer(null); setMaking(true); }}>New library</Button>} />
      {!making && <Outcome answer={answer} done={(d) => <>Library <a href={href(["libraries", d.id])}>{d.id}</a>: open.</>} />}
      <Problems problems={problems} />
      {!loaded ? <Skeleton lines={3} label="Reading the libraries" /> : libraries.length === 0 ? <p>No library yet.</p> : (
        <div className="table-scroll">
          <table className="ledger-table" aria-label="The libraries">
            <thead><tr><th scope="col">library</th><th scope="col">id</th><th scope="col">state</th><th scope="col" className="number">texts</th><th scope="col">read by</th><th scope="col">forked from</th><th scope="col" className="number">problems</th></tr></thead>
            <tbody>
              {libraries.map((l) => {
                const reading = readers(found, l.id);
                const parent = libraries.find((p) => p.id === l.forked_from);
                return (
                  <tr key={l.id}>
                    <td className="wide-cell"><a href={href(["libraries", l.id])} className="cut" title={l.name}>{l.name}</a>{l.upstream && <span className="aside"> the upstream's</span>}</td>
                    <td className="ledger nowrap">{l.id}</td>
                    <td>{l.locked_at ? "locked" : "open"}</td>
                    <td className="number ledger"><a href={href(["texts"], { library: l.id })}>{l.texts.toLocaleString("en")}</a></td>
                    <td>{reading.length ? <span className="cut" title={reading.map((d) => d.name).join(", ")}>{reading.map((d, i) => <span key={d.id}>{i > 0 && ", "}<a href={href(["datasets", d.id])}>{d.name}</a></span>)}</span> : <span className="aside">none</span>}</td>
                    <td>{parent ? <LibraryLink library={parent} /> : l.forked_from ? <span className="ledger">{l.forked_from}</span> : ""}</td>
                    <td className="number ledger">{l.problems?.length ?? 0}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <Dialog open={making} onClose={() => setMaking(false)} title="New library">
        {making && <NewLibrary onMade={(a) => { setAnswer(a); if (a.ok) { setMaking(false); reload(); } }} />}
      </Dialog>
    </>
  );
}

function NewLibrary({ onMade }: { onMade: (a: Answer<{ id: string }>) => void }) {
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
        hint="It opens empty; admitted texts join it from the texts page." />
      {answer && !answer.ok && <Problems problems={answer.problems} />}
      <Button type="submit" variant="primary" disabled={!name.trim()}>Make the library</Button>
    </form>
  );
}

type Act = "report" | "fork";
type Which = "whole" | "ticked" | "filter"; // the library as a whole (or every text, for a fork), the ticked texts, the filter's

function OneLibrary({ id }: { id: string }) {
  const { libraries, loaded, problems, reload } = useLibraries();
  const { datasets } = useDatasets();
  const [filter, setFilter] = useState<TextFilter>({ ...EVERY, libraries: [id] });
  const search = useSearch(filter);
  const [ticked, setTicked] = useState<Set<string>>(new Set());
  const [acting, setActing] = useState<{ act: Act; which: Which } | null>(null);
  const [locking, setLocking] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [done, setDone] = useState<Answer<unknown> | null>(null);
  const [said, setSaid] = useState("");
  const apply = useCallback((f: TextFilter) => { setTicked(new Set()); setFilter({ ...f, libraries: [id] }); }, [id]);
  const library = libraries.find((l) => l.id === id);
  if (problems.length) return <Problems problems={problems} />;
  if (!loaded) return <Skeleton lines={4} label="Reading the library" />;
  if (!library) return <Problems problems={[`no library ${id}`]} />;
  const parent = libraries.find((l) => l.id === library.forked_from);
  const tick = (ids: string[], on: boolean) => {
    const next = new Set(ticked);
    ids.forEach((t) => (on ? next.add(t) : next.delete(t)));
    setTicked(next);
  };
  const finished = (a: Answer<unknown>, words: string) => {
    setDone(a);
    setSaid(words);
    if (a.ok) { setActing(null); setLocking(false); setRenaming(false); setTicked(new Set()); reload(); search.reload(); }
  };
  const lock = async () => finished(await post(`/api/libraries/${encodeURIComponent(id)}/lock`, {}), `${library.name} is locked.`);
  const filtered = JSON.stringify(filter) !== JSON.stringify({ ...EVERY, libraries: [id] });
  const answer = search.answer;
  const reach: TextFilter | null = acting === null || acting.which === "whole" ? null : acting.which === "ticked" ? { ...EVERY, ids: [...ticked] } : filter;
  const reading = readers(datasets?.ok ? datasets.data : [], id);
  const act = (a: Act, which: Which) => () => { setDone(null); setActing({ act: a, which }); };
  const open = !library.locked_at;
  const theirs = library.upstream;
  return (
    <>
      <PageHead title={library.name}
        meta={<>
          <span>{open ? "open" : `locked ${library.locked_at!.replace("T", " ").slice(0, 16)} UTC`}</span>
          <span className="ledger">{library.id}</span>
          {theirs && <span className="aside">the upstream's</span>}
          <a href={href(["texts"], { library: library.id })}>{plural(library.texts, "text")}</a>
          {parent && <span>forked from <LibraryLink library={parent} /></span>}
        </>}
        chain={<>One of the <a href={href(["libraries"])}>libraries</a>. Admitted <a href={href(["texts"])}>texts</a> join it{open ? "" : " while it is open"}; {reading.length
          ? <>read by {reading.map((d, i) => <span key={d.id}>{i > 0 && ", "}<a href={href(["datasets", d.id])}>{d.name}</a></span>)}.</>
          : <>no <a href={href(["datasets"])}>dataset</a> reads it yet.</>}</>}
        actions={<>
          {open && !theirs && <ButtonLink variant={ticked.size > 0 ? "quiet" : "primary"} href={href(["texts"], { status: "admitted", where: `NOT list_contains(libraries, '${id.replace(/'/g, "''")}')` })}>Put texts in it</ButtonLink>}
          {!theirs && <Button variant={open || ticked.size > 0 ? "quiet" : "primary"} onClick={act("report", "whole")}>Report a problem</Button>}
          <Button variant={theirs && ticked.size === 0 ? "primary" : "quiet"} onClick={act("fork", "whole")}>Fork it</Button>
          {open && !theirs && <Button variant="quiet" onClick={() => { setDone(null); setLocking(true); }}>Lock it</Button>}
          {!theirs && <Button variant="quiet" onClick={() => { setDone(null); setRenaming(true); }}>Rename it</Button>}
        </>}
        note={theirs ? <>It belongs to the upstream this depot follows: only the upstream names it, locks it, puts texts in it or takes a problem on it, and a pull brings what it did here. A fork of it is a library of this depot's own.</>
          : open ? undefined : "Locked: it reads the revision it pinned of each of its texts, for good, and no text joins or leaves it. Its texts still take decisions and edits, as new revisions it does not read; only an erasure changes what it reads."} />
      {!acting && !locking && !renaming && <Outcome answer={done} done={() => said} />}
      {(library.problems ?? []).length > 0 && (
        <details className="problems section">
          <summary className="cue">{plural(library.problems!.length, "reported problem")}</summary>
          <ul className="plain small">
            {library.problems!.map((p) => (
              <li key={p.decision}>
                <a href={href(["libraries", library.id, "problems", p.decision])} className="cut" title={p.problem}>{p.problem}</a> <a className="ledger aside" href={href(["decisions", p.decision])}>{p.decision}</a>
                {p.texts.length > 0 ? <> <a href={href(["texts"], { where: `list_contains(${JSON.stringify(p.texts).replace(/"/g, "'")}, id)` })}>{plural(p.texts.length, "text")}</a></> : <span className="aside"> the whole library</span>}
              </li>
            ))}
          </ul>
        </details>
      )}

      {!open && <Pinned id={id} />}
      <Section title={open ? "Its texts" : "Its texts as they are now"}>
        <FilterBar value={filter} onApply={apply} libraries={libraries} fixedLibrary={id} />
        {!answer && <Skeleton lines={6} label="Reading its texts" />}
        {answer && !answer.ok && <Problems problems={answer.problems} />}
        {answer?.ok && (answer.data.total === 0 ? <p>No text matches.</p> : (
          <>
            <div className="table-tools small">
              <span>{plural(answer.data.total, "text")} {filtered ? "match the filter" : "in it"}</span>
              {filtered && !theirs && <Button variant="quiet" onClick={act("report", "filter")}>Report a problem on them</Button>}
              {filtered && <Button variant="quiet" onClick={act("fork", "filter")}>Fork without them</Button>}
            </div>
            <TextTable rows={answer.data.rows} libraries={libraries} ticked={ticked} onTick={tick} busy={search.pending} order={search.order} onOrder={search.setOrder} label="The library's texts" otherLibraries={id} />
            <Pager offset={search.offset} limit={search.limit} total={answer.data.total} onOffset={search.setOffset} onLimit={search.setLimit} label="Pages of the library's texts" />
          </>
        ))}
      </Section>
      {ticked.size > 0 && (
        <div className="selection-bar" role="region" aria-label="The ticked texts">
          <strong className="cue">{plural(ticked.size, "text")} ticked</strong>
          {!theirs && <Button variant="primary" onClick={() => { setDone(null); setActing({ act: "report", which: "ticked" }); }}>Report a problem on them</Button>}
          <Button variant={theirs ? "primary" : "quiet"} onClick={() => { setDone(null); setActing({ act: "fork", which: "ticked" }); }}>Fork without them</Button>
          <Button variant="quiet" onClick={() => setTicked(new Set())}>Untick all</Button>
        </div>
      )}

      <Dialog open={acting !== null} onClose={() => setActing(null)}
        title={acting?.act === "report" ? (acting.which === "whole" ? `Report a problem with ${library.name}` : `Report a problem with texts of ${library.name}`)
          : acting?.which === "whole" ? `Fork ${library.name}` : `Fork ${library.name} without these texts`}>
        {acting && (
          <div className={reach ? "wide" : undefined}>
            {reach ? (
              <Review filter={reach} libraries={libraries}>
                {(chosen) => (acting.act === "report"
                  ? <ReportForm library={library} chosen={chosen} onDone={finished} />
                  : <ForkForm library={library} without={chosen} onDone={finished} />)}
              </Review>
            ) : acting.act === "report" ? <ReportForm library={library} chosen={null} onDone={finished} /> : <ForkForm library={library} without={null} onDone={finished} />}
            {done && !done.ok && <Problems problems={done.problems} />}
          </div>
        )}
      </Dialog>
      <Dialog open={renaming} onClose={() => setRenaming(false)} title={`Rename ${library.name}`}>
        {renaming && <RenameForm library={library} onDone={finished} />}
        {renaming && done && !done.ok && <Problems problems={done.problems} />}
      </Dialog>
      <Dialog open={locking} onClose={() => setLocking(false)} title={`Lock ${library.name} for good`}>
        <p>Locking {library.name} (<span className="ledger">{library.id}</span>), which holds {plural(library.texts, "text")}, cannot be undone: it pins the current revision of each, reads those for good, and no text joins or leaves it again. A later decision or edit of one of its texts makes a new revision it does not read.</p>
        <Button variant="primary" onClick={lock}>Lock it for good</Button>
        {done && !done.ok && <Problems problems={done.problems} />}
      </Dialog>
    </>
  );
}

type Pins = { total: number; newer: number; rows: { id: string; title: string | null; revision: number; current: number }[] };

// The revisions a locked library pinned, a page at a time, each text marked when its current revision is newer.
function Pinned({ id }: { id: string }) {
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(25);
  const [pins, setPins] = useState<Answer<Pins> | null>(null);
  useEffect(() => { post<Pins>(`/api/libraries/${encodeURIComponent(id)}/pins`, { offset, limit }).then(setPins); }, [id, offset, limit]);
  return (
    <Section title="The revisions it pinned">
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
                    <td className="wide-cell"><a href={href(["texts", r.id])} className="cut" title={r.title ?? r.id}>{r.title ?? "untitled"}</a></td>
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
    </Section>
  );
}

// A new name for the library, open or locked: its name is not what it holds.
function RenameForm({ library, onDone }: { library: Library; onDone: (a: Answer<unknown>, words: string) => void }) {
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
        chain={<>A problem reported on one of the <a href={href(["libraries"])}>libraries</a>. It changes no text and no membership; the <a href={href(["decisions", d.id])}>decision {d.id}</a> records who reported it.</>} />
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
function ForkForm({ library, without, onDone }: { library: Library; without: string[] | null; onDone: (a: Answer<unknown>, words: string) => void }) {
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

export const libraries: Page = {
  path: "libraries", title: "Libraries", group: "data",
  Component: ({ rest }) => (rest[0] && rest[1] === "problems" && rest[2] ? <OneProblem key={rest[2]} library={rest[0]} decision={rest[2]} />
    : rest[0] ? <OneLibrary key={rest[0]} id={rest[0]} /> : <Libraries />),
};
