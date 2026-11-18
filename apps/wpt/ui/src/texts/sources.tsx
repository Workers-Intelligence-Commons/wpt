// The Open menu of the Texts page: everything that can be opened, each a
// filter of the texts (filter.tsx) that a link puts into the address: every
// text, each status, the libraries (open, then locked), the tags, the
// licenses, and the filters saved in this browser (saved-filters.ts), with
// the form, opened by a button, that saves the open filter under a name, and a way to forget one.
// A status, a tag or a license narrows the open filter: it takes the place
// of that part of the filter only, the search and the other parts kept, and
// pressed again while it applies, it drops that part. Every text, a library
// and a saved filter open on their own.
// Each count is what its link opens: every text and each library count the
// depot's texts (GET /api/counts, read at once and again after every
// change); a status, a tag or a license counts the open filter with that
// value in place of its own part (counts.tsx), which shows once counted,
// and the depot's count stands for it while the open filter is every text.
// The entry
// whose filter is the open one is marked as the current one. A library just
// made opens at once. The boxes that find a tag or a license empty when
// what is open changes.

import { Button, Dialog, TextInput } from "@workers-intelligence-commons/components";
import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { get, type Answer } from "../api";
import { settledCounts, shownCounts, useCounts, type Counting, type Counts } from "../counts";
import { EVERY, toQuery, type TextFilter } from "../filter";
import { NewLibrary } from "../pages/libraries";
import { Problems, required, STATUSES, StatusBadge, type Library, type Status } from "../parts";
import { href } from "../route";
import { forgetFilter, saveFilter, useSavedFilters } from "../saved-filters";


// the depot's counts: every current text, and how many have each status, tag and license
type DepotCounts = { texts: number; statuses: Record<Status, number>; tags: Record<string, number>; licenses: Record<string, number>; libraries: Record<string, number> };
type Part = "statuses" | "tags" | "licenses";
// the most used first, then by name
const ranked = (found: Record<string, number>) => Object.entries(found).sort(([a, x], [b, y]) => y - x || a.localeCompare(b)).map(([key, n]) => ({ key, n }));

const same = (a: TextFilter, b: TextFilter) => JSON.stringify({ ...a, ids: null }) === JSON.stringify({ ...b, ids: null });
const every = (f: TextFilter) => same(f, EVERY);

// How many texts one value of a part counts in a count of texts.
function countOf(c: Counts, part: Part, value: string): number {
  if (part === "statuses") return c.statuses[value as Status] ?? 0;
  if (part === "tags") return c.tags.find((t) => t.tag === value)?.texts ?? 0;
  return c.licenses.find((l) => l.license === value)?.texts ?? 0;
}

// version: raised after any change, so the counts are read again. into: the
// library texts are being chosen for, which a status, tag or license link
// keeps in the address, so narrowing the texts stays inside that task.
export function Sources({ open, libraries, version, onLibraryMade, into = null }: {
  open: TextFilter; libraries: Library[]; version: number; onLibraryMade: () => void; into?: string | null;
}) {
  const [read, setRead] = useState<{ answer: Answer<DepotCounts>; version: number } | null>(null);
  useEffect(() => { get<DepotCounts>("/api/counts").then((answer) => setRead({ answer, version })); }, [version]);
  const answer = read?.answer ?? null;
  const counts = answer?.ok ? answer.data : null;
  const depotStale = read !== null && read.version !== version; // the depot's counts of before the change, read again
  // what a status, tag or license link opens: the open filter without that part, then narrowed to the value
  const base = (part: Part): TextFilter => (open[part].length ? { ...open, ids: null, [part]: [] } : { ...open, ids: null });
  const countable = (f: TextFilter) => (every(f) ? null : f);
  const counted: Record<Part, Counting | null> = {
    statuses: useCounts(countable(base("statuses")), version),
    tags: useCounts(countable(base("tags")), version),
    licenses: useCounts(countable(base("licenses")), version),
  };
  // undefined while it is counted
  // whether a part's counts shown are those of before the change, while they are read again
  const staleFor = (part: Part) => every(base(part)) ? depotStale
    : Boolean(counted[part] && !counted[part]!.done && !(part === "statuses" && counted[part]!.settled));
  const countFor = (part: Part, value: string): number | undefined => {
    if (every(base(part))) return counts ? part === "statuses" ? counts.statuses[value as Status] : counts[part][value] ?? 0 : undefined;
    const c = counted[part];
    const shown = part === "statuses" ? settledCounts(c) : shownCounts(c); // a status's count is final once settled
    return shown ? countOf(shown, part, value) : undefined;
  };
  const saved = useSavedFilters();
  const [making, setMaking] = useState(false);
  const openKey = JSON.stringify({ ...open, ids: null }); // a find box empties when what is open changes
  const entry = (filter: TextFilter, label: ReactNode, count?: number, title?: string) => (
    <a className="source wic-focus" href={href(["texts"], toQuery(filter))} aria-current={same(filter, open) ? "true" : undefined} title={title}>
      <span className="source-label">{label}</span>
      {count !== undefined && <span className={`aside source-count${depotStale ? " recounting" : ""}`}>{count.toLocaleString("en")}</span>}
    </a>
  );
  // the open filter with one of its parts set to a single value, or without it when that value alone applies already
  const narrowed = (part: "statuses" | "tags" | "licenses", value: string): TextFilter => {
    const alone = open[part].length === 1 && open[part][0] === value;
    return { ...open, ids: null, [part]: alone ? [] : [value] };
  };
  const part = (key: Part, value: string, label: ReactNode) => {
    const alone = open[key].length === 1 && open[key][0] === value;
    const count = countFor(key, value);
    return (
      <a className="source wic-focus" href={href(["texts"], { ...toQuery(narrowed(key, value)), into })} aria-current={alone ? "true" : undefined}
        title={alone ? "Applies now; press again to drop it from the filter"
          : `Narrows the open filter, which keeps its search and its other parts${count === undefined ? "; its count shows once counted" : `: ${count.toLocaleString("en")} texts${staleFor(key) ? ", counted before the change, counting again" : ""}`}`}>
        <span className="source-label">{label}</span>
        {count !== undefined && <span className={`aside source-count${staleFor(key) ? " recounting" : ""}`}>{count.toLocaleString("en")}</span>}
      </a>
    );
  };
  // a tag or license list, the most counted first: within the open filter once counted, else in the depot
  const rankedFor = (key: "tags" | "licenses") => {
    if (!counts) return [];
    const rows = ranked(counts[key]);
    if (!rows.every((row) => countFor(key, row.key) !== undefined)) return rows;
    // under a filter, a value no open text has is left out, unless the filter names it
    return rows.map((row) => ({ key: row.key, n: countFor(key, row.key)! })).filter((row) => row.n > 0 || open[key].includes(row.key))
      .sort((a, b) => b.n - a.n || a.key.localeCompare(b.key));
  };
  const own = libraries.filter((l) => !l.locked_at);
  const locked = libraries.filter((l) => l.locked_at);
  return (
    <nav className="sources small" aria-label="What to open">
      <ul className="plain source-list">
        <li>{entry(EVERY, "Every text", counts?.texts)}</li>
        {answer && !answer.ok && <li><Problems problems={answer.problems} /></li>}
      </ul>
      <SourceGroup title="Statuses" tools={staleFor("statuses") ? <span className="aside">counting again</span> : undefined}>
        {STATUSES.map((s) => <li key={s} className={countFor("statuses", s) === 0 ? "none-held" : undefined}>{part("statuses", s, <StatusBadge status={s} />)}</li>)}
      </SourceGroup>
      <SourceGroup title="Libraries" tools={<Button variant="quiet" onClick={() => setMaking(true)}>New library</Button>}>
        {libraries.length === 0 && <li className="aside">No library yet.</li>}
        {[...own, ...locked].map((l) => (
          <li key={l.id}>{entry({ ...EVERY, libraries: [l.id] }, <>{l.name}{l.locked_at && <span className="aside"> locked</span>}{l.upstream ? <span className="aside"> upstream</span> : null}</>, counts ? counts.libraries[l.id] ?? 0 : undefined, `${l.name} (${l.id})`)}</li>
        ))}
      </SourceGroup>
      <Findable key={`tags ${openKey}`} title="Tags" loaded={Boolean(counts)} rows={rankedFor("tags")} recounting={staleFor("tags")}
        entry={(key) => part("tags", key, key)} />
      <Findable key={`licenses ${openKey}`} title="Licenses" loaded={Boolean(counts)} rows={rankedFor("licenses")} recounting={staleFor("licenses")}
        entry={(key) => part("licenses", key, <span className="ledger">{key}</span>)} />
      <SourceGroup title="Saved filters">
        {saved.length === 0 && <li className="aside">None saved in this browser.</li>}
        {saved.map((s) => (
          <li key={s.name} className="saved">
            {entry(s.filter, s.name)}
            <button type="button" className="forget wic-focus" aria-label={`Forget the saved filter ${s.name}`} title="Forget it" onClick={() => forgetFilter(s.name)}>✕</button>
          </li>
        ))}
        <li><SaveFilter open={open} /></li>
      </SourceGroup>
      <Dialog open={making} onClose={() => setMaking(false)} title="New library">
        {making && <NewLibrary onMade={(a) => {
          if (!a.ok) return;
          setMaking(false);
          onLibraryMade();
          window.location.hash = href(["texts"], { library: a.data.id }); // the new library opens, with "Put texts in it"
        }} />}
      </Dialog>
    </nav>
  );
}

function SourceGroup({ title, tools, children, className }: { title: string; tools?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`source-group ${className ?? ""}`} aria-label={title}>
      <div className="source-head"><h2 className="cue source-title">{title}</h2>{tools}</div>
      <ul className="plain source-list">{children}</ul>
    </section>
  );
}

// A long list (tags, licenses), every entry, the most used first, scrolled in place, with a box that keeps the entries
// holding the words typed. recounting: its counts are those of before a change, read again. The list is one stop of
// the Tab key: the arrow keys, Home and End move between its entries, and Tab leaves it.
const FIND = 8; // entries above which a list has a box to find them

function Findable({ title, loaded, rows, recounting, entry }: { title: string; loaded: boolean; rows: { key: string; n: number }[]; recounting: boolean; entry: (key: string, n: number) => ReactNode }) {
  const [typed, setTyped] = useState("");
  const words = typed.toLowerCase().split(/\s+/).filter(Boolean);
  const found = rows.filter((r) => words.every((w) => r.key.toLowerCase().includes(w)));
  const list = useRef<HTMLUListElement>(null);
  const [current, setCurrent] = useState(0); // the entry the Tab key reaches
  const links = () => [...(list.current?.querySelectorAll<HTMLAnchorElement>("a.source") ?? [])];
  useEffect(() => { links().forEach((a, i) => { a.tabIndex = i === Math.min(current, found.length - 1) ? 0 : -1; }); });
  const move = (e: KeyboardEvent<HTMLUListElement>) => {
    const all = links();
    const at = all.indexOf(document.activeElement as HTMLAnchorElement);
    if (at < 0) return;
    const to = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: all.length - 1 }[e.key];
    if (to === undefined || to < 0 || to >= all.length) return;
    e.preventDefault();
    all[to].focus();
  };
  return (
    <SourceGroup title={title} className="source-long" tools={recounting ? <span className="aside">counting again</span> : undefined}>
      {rows.length > FIND && (
        <li><input className="wic-input wic-focus source-find" type="search" value={typed} placeholder={`Find ${title.toLowerCase()}`}
          aria-label={`Find ${title.toLowerCase()}`} onChange={(e) => setTyped(e.target.value)} /></li>
      )}
      {!loaded && <li className="aside">counting</li>}
      {loaded && rows.length === 0 && <li className="aside">None.</li>}
      {loaded && rows.length > 0 && found.length === 0 && <li className="aside">None holds “{typed.trim()}”.</li>}
      {found.length > 0 && (
        <li className="source-scroll-item">
          <ul ref={list} className="plain source-list source-scroll" aria-label={`${title}, ${found.length.toLocaleString("en")}`} onKeyDown={move}
            onFocus={(e) => { const at = links().indexOf(e.target as Element as HTMLAnchorElement); if (at >= 0) setCurrent(at); }}>
            {found.map((r) => <li key={r.key}>{entry(r.key, r.n)}</li>)}
          </ul>
        </li>
      )}
    </SourceGroup>
  );
}

// Keep the open filter in this browser under a name.
function SaveFilter({ open }: { open: TextFilter }) {
  const [name, setName] = useState("");
  const [kept, setKept] = useState<boolean | null>(null);
  const [shown, setShown] = useState(false);
  const save = () => { const ok = saveFilter(name.trim(), open); setKept(ok); setName(""); if (ok) setShown(false); };
  if (!shown) return (
    <>
      <Button variant="quiet" onClick={() => { setKept(null); setShown(true); }}>Save the open filter</Button>
      {kept === false && <Problems problems={["This browser refuses to keep it (its storage is off or full)."]} />}
    </>
  );
  return (
    <form className="save-filter" onSubmit={(e) => { e.preventDefault(); if (name.trim()) save(); }}>
      <TextInput label={required("Save the open filter as")} required autoFocus value={name} onChange={(e) => { setName(e.target.value); setKept(null); }}
        hint="Kept in this browser only; the same name replaces it." />
      <div className="row tight">
        <Button type="submit" variant="quiet" disabled={!name.trim()}>Save it</Button>
        <Button variant="quiet" onClick={() => setShown(false)}>Cancel</Button>
      </div>
      {kept === false && <Problems problems={["This browser refuses to keep it (its storage is off or full)."]} />}
    </form>
  );
}
