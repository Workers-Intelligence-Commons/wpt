// The workspace: the depot's texts and libraries in three columns. On the
// left, everything that can be opened (sources.tsx); in the middle, the
// playlist of what is open (playlist.tsx), or, in the blob mode, the whole
// of it as one continuous text (blob.tsx) in place of the playlist and the
// editor; on the right, the current text in its editor with its record and
// actions (text-panel.tsx). What is open is a filter of the texts
// (filter.tsx), kept in the address with the current text and the mode:
// #/texts/ID?library=L&view=blob. The filter bar refines it. When it is one
// library, its head stands above the columns (library-head.tsx); ticked
// texts are acted on through the bar at the bottom (bulk.tsx).
//
// The keys of the playlist work anywhere but in a field, the editor or a
// dialog: J or the down arrow (in the playlist) opens the next text, K or
// the up arrow the previous one, X ticks the current text, E puts the
// cursor in the editor, and A or R admits or refuses the current text and
// opens the next one. A and R send a public decision under the declaration
// this browser keeps (declarant.ts), so the legend beside the playlist
// shows the confirmation they make and the name they record, and the line
// over the columns says what each press did; without a name and email, the
// key asks for them and decides nothing. Leaving a text whose editor holds
// unsaved changes, by a key or a link, asks first.

import { Button, Dialog, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { useAbout } from "../about";
import { post, type Answer } from "../api";
import type { Deciding } from "../blocked";
import { type EditorHandle } from "../code-editor";
import { declared, readDeclarant, saveDeclarant, useDeclarant } from "../declarant";
import { BY_ID, EVERY, FilterBar, fromQuery, search, toQuery, useSearch, type TextFilter } from "../filter";
import { plural, Problems, required, RequiredNote, useLibraries } from "../parts";
import { href } from "../route";
import { useSavedFilters } from "../saved-filters";
import { Blob } from "./blob";
import { Bulk } from "./bulk";
import { LibraryHead } from "./library-head";
import { Playlist } from "./playlist";
import { Sources } from "./sources";
import { TextPanel } from "./text-panel";

const LIMIT = 100; // texts of a playlist page

// Which text to open once the playlist's page is read again: its first, its
// last, or the one after a text just decided (index: where that text stood,
// where the next one now stands when the decision took it out of the list).
type Pick = { kind: "first" } | { kind: "last" } | { kind: "after"; id: string; index: number };
type Said = { status: Deciding; id: string; title: string; answer: Answer<{ decision: string }> } | { words: ReactNode };

export function Workspace({ current, query }: { current: string | null; query: URLSearchParams }) {
  const queryText = query.toString();
  const filter = useMemo(() => fromQuery(new URLSearchParams(queryText)), [queryText]);
  const blob = query.get("view") === "blob";
  const { libraries, reload: reloadLibraries } = useLibraries();
  const playlist = useSearch(filter, LIMIT);
  const { answer, offset, limit } = playlist;
  const rows = answer?.ok ? answer.data.rows : [];
  const total = answer?.ok ? answer.data.total : 0;
  const [ticked, setTicked] = useState<Set<string>>(new Set());
  // a new filter, however it was reached (a link, the filter bar, the back button), starts with nothing ticked
  const filterKey = JSON.stringify({ ...filter, ids: null });
  const [tickedFor, setTickedFor] = useState(filterKey);
  if (tickedFor !== filterKey) { setTickedFor(filterKey); setTicked(new Set()); }
  const [tickingAll, setTickingAll] = useState(false);
  const [version, setVersion] = useState(0); // raised after every change, so each column reads again
  const [said, setSaid] = useState<Said | null>(null);
  const [asking, setAsking] = useState<Deciding | null>(null);
  const [deciding, setDeciding] = useState(false);
  const dirty = useRef(false);
  const editor = useRef<EditorHandle | null>(null);
  const pick = useRef<Pick | null>(null);
  const who = useDeclarant();
  const { about } = useAbout();
  const saved = useSavedFilters();
  const library = filter.libraries.length === 1 ? libraries.find((l) => l.id === filter.libraries[0]) ?? null : null;

  const address = useCallback((id: string | null, q: Record<string, string | string[] | null>) => href(id ? ["texts", id] : ["texts"], q), []);
  const into = query.get("into"); // the library "Put texts in it" came from: the one the ticked texts go into first
  const here = useMemo(() => ({ ...toQuery(filter), view: blob ? "blob" : null, into }), [filter, blob, into]);
  const link = useCallback((id: string) => address(id, here), [address, here]);
  const leave = () => {
    if (!dirty.current) return true;
    const ok = window.confirm("The editor holds changes that are not saved. Leave them?");
    if (ok) dirty.current = false;
    return ok;
  };
  const go = (id: string) => { window.location.hash = link(id); };
  const changed = useCallback(() => { setVersion((v) => v + 1); playlist.reload(); reloadLibraries(); }, [playlist.reload, reloadLibraries]);
  const apply = useCallback((f: TextFilter) => {
    window.location.hash = address(current, { ...toQuery(f), view: blob ? "blob" : null });
  }, [address, current, blob]);
  const onDirty = useCallback((d: boolean) => { dirty.current = d; }, []);
  const shownId = useRef<string | null>(null); // the text the right column shows now: A, R and T act only on it
  const onShown = useCallback((id: string | null) => { shownId.current = id; }, []);
  const handle = useCallback((h: EditorHandle | null) => { editor.current = h; }, []);

  // once the page asked for is read, open the text a move or a decision chose
  useEffect(() => {
    const wanted = pick.current;
    if (!wanted || playlist.pending || !answer?.ok) return;
    pick.current = null;
    const found = answer.data.rows;
    if (wanted.kind === "first") { if (found[0]) go(found[0].id); return; }
    if (wanted.kind === "last") { if (found.length) go(found[found.length - 1].id); return; }
    const at = found.findIndex((r) => r.id === wanted.id);
    const next = at >= 0 ? found[at + 1] : found[wanted.index];
    if (next) go(next.id);
    else if (offset + limit < answer.data.total) { pick.current = { kind: "first" }; playlist.setOffset(offset + limit); }
  });

  // nothing open: the first text of the playlist opens, as if J was pressed
  useEffect(() => {
    if (!current && !blob && !playlist.pending && answer?.ok && answer.data.rows[0])
      window.location.replace(link(answer.data.rows[0].id));
  }, [current, blob, playlist.pending, answer, link]);

  const move = (step: 1 | -1) => {
    if (!leave()) return;
    const at = rows.findIndex((r) => r.id === current);
    if (at === -1 && rows.length) return go(rows[0].id);
    if (step === 1) {
      if (at < rows.length - 1) go(rows[at + 1].id);
      else if (offset + limit < total) { pick.current = { kind: "first" }; playlist.setOffset(offset + limit); }
    } else if (at > 0) go(rows[at - 1].id);
    else if (offset > 0) { pick.current = { kind: "last" }; playlist.setOffset(Math.max(0, offset - limit)); }
  };

  const decide = async (status: Deciding) => {
    if (!current || deciding || shownId.current !== current) return;
    if (!declared(who)) { setAsking(status); return; }
    if (!about) { setSaid({ words: <Problems problems={["The server's confirmation is not read yet, so no decision can be sent."]} /> }); return; }
    if (!leave()) return;
    const at = rows.findIndex((r) => r.id === current);
    const row = rows[at];
    const title = row ? row.title ?? (row.kind ? `an item of kind ${row.kind}` : "untitled text") : current;
    setDeciding(true);
    const a = await post<{ decision: string }>("/api/decisions", { status, texts: [current], declaration: { name: who.name.trim(), email: who.email.trim(), accepted: true } });
    setDeciding(false);
    setSaid({ status, id: current, title, answer: a });
    if (!a.ok) return;
    pick.current = { kind: "after", id: current, index: Math.max(at, 0) };
    changed();
  };

  // T in an open library of this depot: the current text leaves it (it stays admitted), then the next opens
  const takeOut = async () => {
    if (!current || deciding || shownId.current !== current || !library || library.locked_at || library.upstream) return;
    if (!leave()) return;
    const at = rows.findIndex((r) => r.id === current);
    const row = rows[at];
    const title = row ? row.title ?? (row.kind ? `an item of kind ${row.kind}` : "untitled text") : current;
    setDeciding(true);
    const a = await post<{ taken_out: string[]; not_held: string[] }>(`/api/libraries/${encodeURIComponent(library.id)}/taken-out`, { texts: [current] });
    setDeciding(false);
    if (!a.ok) { setSaid({ words: <Problems problems={[`Not taken out: ${title} (${current}).`, ...a.problems]} /> }); return; }
    setSaid({ words: <>Taken out of {library.name}: {title} <span className="ledger">({current})</span>; it stays admitted.</> });
    pick.current = { kind: "after", id: current, index: Math.max(at, 0) };
    changed();
  };

  const tick = (ids: string[], on: boolean) => {
    const next = new Set(ticked);
    ids.forEach((id) => (on ? next.add(id) : next.delete(id)));
    setTicked(next);
  };
  const tickAll = async () => {
    setTickingAll(true);
    const a = await search(filter, 0, 1, BY_ID, true);
    setTickingAll(false);
    if (a.ok) setTicked(new Set(a.data.ids));
  };

  // the keys, read again at every render so they act on what is shown now
  useEffect(() => {
    if (blob) return;
    const press = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
      const target = e.target instanceof Element ? e.target : null;
      if (target?.closest("input, textarea, select, [contenteditable], .cm-editor") || document.querySelector("dialog[open]")) return;
      const inList = Boolean(target?.closest(".playlist"));
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      if (key === "j" || (key === "ArrowDown" && inList)) move(1);
      else if (key === "k" || (key === "ArrowUp" && inList)) move(-1);
      else if (key === "a") decide("admitted");
      else if (key === "r") decide("refused");
      else if (key === "t") takeOut();
      else if (key === "x" && current) tick([current], !ticked.has(current));
      else if (key === "e") editor.current?.focus();
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", press);
    return () => window.removeEventListener("keydown", press);
  });

  // a link followed while the editor holds unsaved changes asks first
  const guard = (e: MouseEvent) => {
    const a = e.target instanceof Element ? e.target.closest("a[href^='#']") : null;
    if (a && dirty.current && !leave()) { e.preventDefault(); e.stopPropagation(); }
  };

  const openFromBlob = (id: string, place: number) => {
    playlist.setOffset(Math.floor(place / limit) * limit);
    window.location.hash = address(id, toQuery(filter));
  };
  const named = saved.find((s) => JSON.stringify({ ...s.filter, ids: null }) === JSON.stringify({ ...filter, ids: null }))?.name;
  const counts = answer?.ok ? answer.data : null;

  return (
    <div className="workspace-frame" onClickCapture={guard}>
      <FilterBar value={filter} onApply={apply} libraries={libraries} />
      <div className="open-bar">
        {library ? <LibraryHead library={library} libraries={libraries} onChanged={changed} /> : (
          <section className="open-head" aria-label="What is open">
            <h1 className="title open-title">{named ?? describe(filter)}</h1>
            {counts && <span className="small aside">
              {plural(counts.total, "text")}: {counts.statuses["under review"].toLocaleString("en")} under review, {counts.statuses.admitted.toLocaleString("en")} admitted, {counts.statuses.refused.toLocaleString("en")} refused{counts.locked > 0 && <>; {counts.locked.toLocaleString("en")} in a locked library</>}
            </span>}
          </section>
        )}
        <div className="view-switch small" role="group" aria-label="How the open texts are shown">
          <a className="wic-focus" href={address(current, { ...toQuery(filter), view: null })} aria-current={!blob ? "true" : undefined}>Playlist and editor</a>
          <a className="wic-focus" href={address(current, { ...toQuery(filter), view: "blob" })} aria-current={blob ? "true" : undefined}>Blob: every text in one</a>
        </div>
      </div>
      <div className={`workspace${blob ? " blob-mode" : ""}`}>
        <Sources open={filter} libraries={libraries} version={version} onLibraryMade={changed} />
        {blob ? <Blob filter={filter} order={playlist.order} onOpen={openFromBlob} /> : <>
          <div className="playlist-column">
            {!answer && <p className="aside">Reading the texts</p>}
            {answer && !answer.ok && <Problems problems={answer.problems} />}
            {answer?.ok && (
              <Playlist found={answer.data} busy={playlist.pending} current={current} link={link} ticked={ticked} onTick={tick}
                onTickAll={tickAll} tickingAll={tickingAll} order={playlist.order} onOrder={playlist.setOrder}
                offset={offset} limit={limit} onOffset={playlist.setOffset} />
            )}
          </div>
          <div className="editor-column">
            {current ? <TextPanel id={current} version={version} onChanged={changed} onDirty={onDirty} handle={handle} onShown={onShown} />
              : <p className="aside">No text is open: choose one in the playlist, or press J for the first.</p>}
          </div>
        </>}
      </div>
      <Bulk ticked={ticked} libraries={libraries} library={library} into={into} onUntick={() => setTicked(new Set())}
        onDone={(words) => { setSaid({ words }); setTicked(new Set()); changed(); }} />
      {said && <SaidLine said={said} />}
      {!blob && <Legend takes={Boolean(library && !library.locked_at && !library.upstream)} who={declared(who) ? who : null} confirmation={about?.confirmation ?? null} requirements={about?.requirements ?? null} />}
      <Dialog open={asking !== null} onClose={() => setAsking(null)} title="Who decides from this browser?">
        {asking && <AskDeclarant status={asking} onSaved={() => { setAsking(null); setSaid({ words: <>Saved. Press {asking === "admitted" ? "A" : "R"} again to {asking === "admitted" ? "admit" : "refuse"} the current text.</> }); }} />}
      </Dialog>
    </div>
  );
}

// What is open, in words, when it is no saved filter and no single library.
function describe(f: TextFilter): string {
  const plain = JSON.stringify({ ...f, ids: null });
  if (plain === JSON.stringify(EVERY)) return "Every text";
  const only = (part: Partial<TextFilter>) => plain === JSON.stringify({ ...EVERY, ...part });
  if (f.statuses.length === 1 && only({ statuses: f.statuses })) return `Texts ${f.statuses[0]}`;
  if (f.tags.length === 1 && only({ tags: f.tags })) return `The tag ${f.tags[0]}`;
  if (f.licenses.length === 1 && only({ licenses: f.licenses })) return `The license ${f.licenses[0]}`;
  return "Texts the filter matches";
}

// What the latest key press, or the latest action on ticked texts, did.
function SaidLine({ said }: { said: Said }) {
  if ("words" in said) return <div className="said-line cue" role="status">{said.words}</div>;
  const { answer, status, id, title } = said;
  if (!answer.ok) return <div className="said-line" role="status"><Problems problems={[`${status === "admitted" ? "Not admitted" : "Not refused"}: ${title} (${id}).`, ...answer.problems]} /></div>;
  return (
    <p className="said-line done cue" role="status">
      {status === "admitted" ? "Admitted" : "Refused"}: {title} <span className="ledger">({id})</span>, by the public decision <a className="ledger" href={href(["decisions", answer.data.decision])}>{answer.data.decision}</a>.
    </p>
  );
}

// The keys of the playlist, in one line under the columns, and what A and R confirm and record.
function Legend({ takes, who, confirmation, requirements }: { takes: boolean; who: { name: string; email: string } | null; confirmation: string | null; requirements: string | null }) {
  const keys: [string, string][] = [
    ["J K", "next, previous"], ["A", "admit, then next"], ["R", "refuse, then next"], ["X", "tick"], ["E", "edit"], ["Ctrl F", "find and replace"], ["Ctrl S", "save"],
    ...(takes ? [["T", "take out of this library, then next"] as [string, string]] : []),
  ];
  return (
    <section className="key-legend small" aria-label="Keys">
      <span className="keys">
        {keys.map(([key, what]) => <span key={key} className="key"><kbd>{key}</kbd> {what}</span>)}
      </span>
      {who ? (
        <span className="aside" title={confirmation ? `Each press of A or R is a public decision and confirms: “${confirmation}”` : undefined}>
          A and R decide in public as {who.name}, <span className="ledger">{who.email}</span> (<a href={href(["settings"])}>change</a>){requirements && <>; each press confirms the <a href={requirements} target="_blank" rel="noreferrer">requirements</a></>}.
        </span>
      ) : <span className="act"><strong>No name and email set:</strong> A and R ask for them first.</span>}
    </section>
  );
}

// A key that decides without a name and email set asks for them; saving them decides nothing.
function AskDeclarant({ status, onSaved }: { status: Deciding; onSaved: () => void }) {
  const [who, setWho] = useState(readDeclarant);
  const [kept, setKept] = useState<boolean | null>(null);
  const { about } = useAbout();
  const save = () => {
    const ok = saveDeclarant({ name: who.name.trim(), email: who.email.trim() });
    setKept(ok);
    if (ok) onSaved();
  };
  return (
    <form className="action-form" onSubmit={(e) => { e.preventDefault(); save(); }}>
      <p>{status === "admitted" ? "Admitting" : "Refusing"} a text is a public decision that records a legal name and an email. Both stay in this browser, and nobody verifies them. Nothing is decided now: press the key again once they are saved.</p>
      {about && <p className="small">Each press then confirms: <q>{about.confirmation}</q> <a href={about.requirements} target="_blank" rel="noreferrer">The requirements</a>.</p>}
      <RequiredNote />
      <TextInput label={required("Legal name")} required value={who.name} autoComplete="name" onChange={(e) => setWho({ ...who, name: e.target.value })} />
      <TextInput label={required("Email")} required type="email" value={who.email} autoComplete="email" onChange={(e) => setWho({ ...who, email: e.target.value })} />
      <Button type="submit" variant="primary" disabled={!who.name.trim() || !who.email.trim()}>Save them</Button>
      {kept === false && <Problems problems={["This browser refuses to keep them (its storage is off or full), so no key can decide."]} />}
    </form>
  );
}
