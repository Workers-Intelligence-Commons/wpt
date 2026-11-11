// The workspace's left column: everything that can be opened, each a
// filter of the texts (filter.tsx) that a link puts into the address: every
// text, each status, the libraries (open, then locked), the tags, the
// licenses, and the filters saved in this browser (saved-filters.ts), with
// the form, opened by a button, that saves the open filter under a name, and a way to forget one.
// The counts come from one search over every text. The entry whose filter is
// the open one is marked as the current one.

import { Button, Dialog, TextInput } from "@workers-intelligence-commons/components";
import { useEffect, useState, type ReactNode } from "react";
import { type Answer } from "../api";
import { EVERY, search, toQuery, type Found, type TextFilter } from "../filter";
import { NewLibrary } from "../pages/libraries";
import { Outcome, plural, Problems, required, STATUSES, StatusBadge, type Library } from "../parts";
import { href } from "../route";
import { forgetFilter, saveFilter, useSavedFilters } from "../saved-filters";

const SHOWN = 30; // tags and licenses listed before typing finds the others

const same = (a: TextFilter, b: TextFilter) => JSON.stringify({ ...a, ids: null }) === JSON.stringify({ ...b, ids: null });

// version: raised after any change, so the counts are read again.
export function Sources({ open, libraries, version, onLibraryMade }: {
  open: TextFilter; libraries: Library[]; version: number; onLibraryMade: () => void;
}) {
  const [every, setEvery] = useState<Answer<Found> | null>(null);
  useEffect(() => { search(EVERY, 0, 1).then(setEvery); }, [version]);
  const saved = useSavedFilters();
  const [making, setMaking] = useState(false);
  const [made, setMade] = useState<Answer<{ id: string }> | null>(null);
  const counts = every?.ok ? every.data : null;
  const entry = (filter: TextFilter, label: ReactNode, count?: number, title?: string) => (
    <a className="source wic-focus" href={href(["texts"], toQuery(filter))} aria-current={same(filter, open) ? "true" : undefined} title={title}>
      <span className="source-label">{label}</span>
      {count !== undefined && <span className="aside source-count">{count.toLocaleString("en")}</span>}
    </a>
  );
  const own = libraries.filter((l) => !l.locked_at);
  const locked = libraries.filter((l) => l.locked_at);
  return (
    <nav className="sources small" aria-label="What to open">
      {every && !every.ok && <Problems problems={every.problems} />}
      <ul className="plain source-list">
        <li>{entry(EVERY, "Every text", counts?.total)}</li>
      </ul>
      <SourceGroup title="Statuses">
        {STATUSES.map((s) => <li key={s}>{entry({ ...EVERY, statuses: [s] }, <StatusBadge status={s} />, counts?.statuses[s])}</li>)}
      </SourceGroup>
      <SourceGroup title="Libraries" tools={<Button variant="quiet" onClick={() => { setMade(null); setMaking(true); }}>New library</Button>}>
        {counts && libraries.length === 0 && <li className="aside">No library yet.</li>}
        {[...own, ...locked].map((l) => (
          <li key={l.id}>{entry({ ...EVERY, libraries: [l.id] }, <>{l.name}{l.locked_at && <span className="aside"> locked</span>}{l.upstream && <span className="aside"> upstream</span>}</>, l.texts, `${l.name} (${l.id})`)}</li>
        ))}
      </SourceGroup>
      {!making && <Outcome answer={made} done={(d) => <>Library <a href={href(["texts"], { library: d.id })}>{d.id}</a>: open.</>} />}
      <Findable title="Tags" loaded={counts !== null} rows={(counts?.tags ?? []).map((t) => ({ key: t.tag, n: t.texts }))}
        entry={(key, n) => entry({ ...EVERY, tags: [key] }, key, n)} />
      <Findable title="Licenses" loaded={counts !== null} rows={(counts?.licenses ?? []).map((l) => ({ key: l.license, n: l.texts }))}
        entry={(key, n) => entry({ ...EVERY, licenses: [key] }, <span className="ledger">{key}</span>, n)} />
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
        {making && <NewLibrary onMade={(a) => { setMade(a); if (a.ok) { setMaking(false); onLibraryMade(); } }} />}
      </Dialog>
    </nav>
  );
}

function SourceGroup({ title, tools, children }: { title: string; tools?: ReactNode; children: ReactNode }) {
  return (
    <section className="source-group" aria-label={title}>
      <div className="source-head"><h2 className="cue source-title">{title}</h2>{tools}</div>
      <ul className="plain source-list">{children}</ul>
    </section>
  );
}

// A long list (tags, licenses), the most used first, with a box that keeps the entries holding the words typed.
function Findable({ title, loaded, rows, entry }: { title: string; loaded: boolean; rows: { key: string; n: number }[]; entry: (key: string, n: number) => ReactNode }) {
  const [typed, setTyped] = useState("");
  const words = typed.toLowerCase().split(/\s+/).filter(Boolean);
  const found = rows.filter((r) => words.every((w) => r.key.toLowerCase().includes(w)));
  return (
    <SourceGroup title={title}>
      {rows.length > SHOWN && (
        <li><input className="wic-input wic-focus source-find" type="search" value={typed} placeholder={`Find ${title.toLowerCase()}`}
          aria-label={`Find ${title.toLowerCase()}`} onChange={(e) => setTyped(e.target.value)} /></li>
      )}
      {loaded && rows.length === 0 && <li className="aside">None.</li>}
      {found.slice(0, SHOWN).map((r) => <li key={r.key}>{entry(r.key, r.n)}</li>)}
      {found.length > SHOWN && <li className="aside">{plural(found.length - SHOWN, "more")}: type to find them.</li>}
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
