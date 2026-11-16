// The workspace: the depot's texts and libraries in three columns. On the
// left, everything that can be opened (sources.tsx); in the middle, the
// playlist of what is open (playlist.tsx), or, in the blob mode, the whole
// of it as one continuous text (blob.tsx) in place of the playlist and the
// editor; on the right, the current text in its editor with its record and
// actions (text-panel.tsx). What is open is a filter of the texts
// (filter.tsx), kept in the address with the current text and the mode:
// #/texts/ID?library=L&view=blob. The filter bar refines it. When it is one
// library, its head stands above the columns (library-head.tsx); ticked
// texts are acted on through the bar at the bottom (bulk.tsx). The ticked
// texts are those ticked one by one, or, once "Tick all" is pressed, every
// text the filter matches less those unticked since, at the change number
// the playlist's page was read at; either way the bar sends them as a
// selection (filter.tsx), never as the list of every id. Opened from a
// library's "Put texts in it" (into=L), a banner says which library the
// texts are chosen for (the admitted texts it does not hold); a link that
// names texts by id may say what they are in words (what=...), which the
// filter bar shows in place of their number. Clearing the filter opens the
// first page and its first text, so the list reads from its start.
//
// The keys of the playlist work anywhere but in a field, the editor or a
// dialog: J or the down arrow (in the playlist) opens the next text, K or
// the up arrow the previous one, X ticks the current text, E puts the
// cursor in the editor (on a conversation, it opens its form), A admits the
// current text and opens the next one, R asks first, in one line with an
// optional reason, then refuses it on Enter and opens the next one (Escape
// refuses nothing), and U puts an admitted or refused text back under
// review the same way, undoing its admission or refusal. Each sends a public decision under the declaration this
// browser keeps (declarant.ts), so the legend beside the playlist shows the
// confirmation they make and the name they record, and the line under the
// columns says what each press did; without a name and email, the key asks
// for them and decides nothing. Leaving a text whose editor holds unsaved
// changes, by a key or a link, asks first. On a phone, a bar at the bottom
// of the screen does what J, K, A, R and U do, with that line just above it.
// Its Admit, Refuse and Back to review all ask first, in a line above the bar that holds
// the confirmation and the name the decision records, since a tap beside
// another button can land on them and the legend is out of view there.

import { Button, Dialog, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { useAbout } from "../about";
import { write, type Answer } from "../api";
import type { Deciding } from "../blocked";
import { type EditorHandle } from "../code-editor";
import { declared, readDeclarant, saveDeclarant, useDeclarant } from "../declarant";
import { CountingNote, matched, shownCounts, useCounts } from "../counts";
import { type Decided } from "../decide-forms";
import { EVERY, FilterBar, fromQuery, named as namedIds, sent, toQuery, useSearch, type Selection, type TextFilter } from "../filter";
import { plural, Problems, required, RequiredNote, textName, useLibraries, type Library } from "../parts";
import { href } from "../route";
import { useSavedFilters } from "../saved-filters";
import { Blob } from "./blob";
import { Bulk } from "./bulk";
import { LibraryFrom } from "./library-from";
import { LibraryHead, outsideOf } from "./library-head";
import { Playlist } from "./playlist";
import { Sources } from "./sources";
import { TextPanel } from "./text-panel";

const LIMIT = 100; // texts of a playlist page
const SAID_SECONDS = 8; // seconds a line of what was done stays

// Whether texts show in a monospace type (the default) or in the serif of
// prose, kept in this browser.
const MONO = "wpt-depot-monospace";
function readMono(): boolean {
  try { return window.localStorage.getItem(MONO) !== "no"; } catch { return true; }
}
function keepMono(on: boolean) {
  try { window.localStorage.setItem(MONO, on ? "yes" : "no"); } catch { /* not kept: the default comes back next time */ }
}

// Which text to open once the playlist's page is read again: its first, its
// last, or the one after a text just decided (index: where that text stood,
// where the next one now stands when the decision took it out of the list).
type Pick = { kind: "first" } | { kind: "last" } | { kind: "after"; id: string; index: number };
type Said = { status: Deciding; id: string; title: string; answer: Answer<{ decision: string }> } | { words: ReactNode };
// What is ticked: the ids ticked one by one (all false), or every text the
// filter matches at change less the ids unticked (all true).
type Ticked = { all: boolean; ids: Set<string>; change: number | null };
const NONE: Ticked = { all: false, ids: new Set(), change: null };

export function Workspace({ current, query }: { current: string | null; query: URLSearchParams }) {
  const queryText = query.toString();
  const filter = useMemo(() => fromQuery(new URLSearchParams(queryText)), [queryText]);
  const blob = query.get("view") === "blob";
  const { libraries, reload: reloadLibraries } = useLibraries();
  const playlist = useSearch(filter, LIMIT);
  const { answer } = playlist;
  const rows = answer?.ok ? answer.data.rows : [];
  const [ticked, setTicked] = useState<Ticked>(NONE);
  // a new filter, however it was reached (a link, the filter bar, the back button), starts with nothing ticked
  const filterKey = JSON.stringify(filter);
  const [tickedFor, setTickedFor] = useState(filterKey);
  if (tickedFor !== filterKey) { setTickedFor(filterKey); setTicked(NONE); }
  const [version, setVersion] = useState(0); // raised after every change, so each column reads again
  const [said, setSaid] = useState<Said | null>(null);
  const [asking, setAsking] = useState<Deciding | null>(null);
  const [confirming, setConfirming] = useState<{ id: string; title: string; status: Deciding } | null>(null); // R's or the phone's question, until confirmed or Escape
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
  const intoLibrary = into ? libraries.find((l) => l.id === into) ?? null : null;
  const outsideKey = (f: TextFilter, library: string) => JSON.stringify(f.outside) === JSON.stringify(outsideOf(library).outside);
  const choosing = intoLibrary && outsideKey(filter, intoLibrary.id) ? intoLibrary : null; // its filter still open
  // what a link's named texts are, in words (#/texts?id=...&what=...), shown in place of their number while they stay named
  const what = query.get("what");
  const idsLabel = what && filter.ids ? what : undefined;
  const here = useMemo(() => ({ ...toQuery(filter), view: blob ? "blob" : null, into, what }), [filter, blob, into, what]);
  const link = useCallback((id: string) => address(id, here), [address, here]);
  const leave = () => {
    if (!dirty.current) return true;
    const ok = window.confirm("The editor holds changes that are not saved. Leave them?");
    if (ok) dirty.current = false;
    return ok;
  };
  const go = (id: string) => { window.location.hash = link(id); };
  const changed = useCallback(() => { setVersion((v) => v + 1); playlist.reload(); reloadLibraries(); }, [playlist.reload, reloadLibraries]);
  // a filter cleared to every text goes back to the playlist, since the blob would read every text as one page
  const apply = useCallback((f: TextFilter) => {
    const every = JSON.stringify(f) === JSON.stringify(EVERY);
    window.location.hash = address(every ? null : current, { ...toQuery(f), view: blob && !every ? "blob" : null, into: into && outsideKey(f, into) ? into : null,
      what: what && JSON.stringify(f.ids) === JSON.stringify(filter.ids) ? what : null });
  }, [address, current, blob, into, what, filter.ids]);
  const onDirty = useCallback((d: boolean) => { dirty.current = d; }, []);
  const shownId = useRef<string | null>(null); // the text the right column shows now: A, R and T act only on it
  const onShown = useCallback((id: string | null) => { shownId.current = id; }, []);
  const handle = useCallback((h: EditorHandle | null) => { editor.current = h; }, []);

  // once the page asked for is read, open the text a move or a decision chose
  useEffect(() => {
    const wanted = pick.current;
    if (!wanted || playlist.pending || !answer?.ok) return;
    pick.current = null;
    placedAt.current = current; // the text left behind is not looked for again
    const found = answer.data.rows;
    // no text left to open (a write took every text out of the filter): the editor closes rather than show one it left out
    if (!found.length && (wanted.kind === "first" || wanted.kind === "last")) { window.location.replace(address(null, here)); return; }
    if (wanted.kind === "first") { go(found[0].id); return; }
    if (wanted.kind === "last") { go(found[found.length - 1].id); return; }
    const at = found.findIndex((r) => r.id === wanted.id);
    const next = at >= 0 ? found[at + 1] : found[wanted.index];
    if (next) go(next.id);
    else if (playlist.next) { pick.current = { kind: "first" }; playlist.next(); }
  });

  // nothing open: the first text of the playlist opens, as if J was pressed.
  // A text opened that the page read does not hold (from a link, the
  // address, or a filter just changed): in the order of ids, the playlist
  // reads the page that starts with it, so it is marked and J goes on from
  // it. When even that page does not hold it, a new filter opens its first
  // text, so the editor never shows a text the filter left out; the same
  // filter keeps it open (moving page by page never checks).
  const checkedFilter = useRef(filterKey);
  // after a write over many texts, the open text is checked again as for a new filter: it may no longer match;
  // the check waits for the page read after the write, since the page read before it still holds the text
  const readBefore = useRef<typeof answer | null>(null);
  const recheck = () => { checkedFilter.current = ""; placedAt.current = null; readBefore.current = answer; };
  const placedAt = useRef<string | null>(null); // the text last checked against the page
  const placing = useRef<string | null>(null); // the text and filter whose page was asked for
  useEffect(() => {
    if (blob || playlist.pending || !answer?.ok) return;
    if (readBefore.current === answer) return;
    readBefore.current = null;
    const found = answer.data.rows;
    if (!current) { if (found[0]) window.location.replace(link(found[0].id)); return; }
    if (pick.current) return;
    const newFilter = checkedFilter.current !== filterKey;
    if (!newFilter && placedAt.current === current) return;
    const held = found.some((r) => r.id === current);
    const key = `${current} ${filterKey}`;
    if (!held && placing.current !== key) {
      placing.current = key;
      playlist.open(current);
      return;
    }
    checkedFilter.current = filterKey;
    placedAt.current = current;
    if (!held && newFilter) {
      if (answer.data.previous) { pick.current = { kind: "first" }; playlist.first(); }
      else window.location.replace(found[0] ? link(found[0].id) : address(null, here));
    }
  });

  // the line of a key press fades after a while, and at once when the filter
  // changes; a refusal, and what an action on many texts did, stay until then
  useEffect(() => {
    if (!said || "words" in said || !said.answer.ok) return;
    const timer = window.setTimeout(() => setSaid(null), SAID_SECONDS * 1000);
    return () => window.clearTimeout(timer);
  }, [said]);
  const [saidFor, setSaidFor] = useState(filterKey);
  if (saidFor !== filterKey) { setSaidFor(filterKey); if (said) setSaid(null); }
  if (confirming && confirming.id !== current) setConfirming(null); // the question is about the text shown only

  // the keys go back to the playlist, on its current text
  const toList = useCallback(() => {
    const row = document.querySelector<HTMLElement>(".playlist-row.current .playlist-link") ?? document.querySelector<HTMLElement>(".playlist-link");
    row?.focus({ preventScroll: true });
  }, []);
  const [mono, setMono] = useState(readMono);

  const move = (step: 1 | -1) => {
    if (!leave()) return;
    const at = rows.findIndex((r) => r.id === current);
    if (at === -1 && rows.length) return go(rows[0].id);
    if (step === 1) {
      if (at < rows.length - 1) go(rows[at + 1].id);
      else if (playlist.next) { pick.current = { kind: "first" }; playlist.next(); }
    } else if (at > 0) go(rows[at - 1].id);
    else if (playlist.previous) { pick.current = { kind: "last" }; playlist.previous(); }
  };

  const titleOf = (id: string) => {
    const row = rows.find((r) => r.id === id);
    return row ? textName(row) : id;
  };
  // R, U, or the phone's Admit, Refuse or Back to review: the question first, which decides nothing
  const ask = (status: Deciding) => {
    if (!current || deciding || shownId.current !== current) return;
    if (same(status)) return;
    if (!declared(who)) { setAsking(status); return; }
    setConfirming({ id: current, title: titleOf(current), status });
  };
  // a decision that would leave the current text's status as it is says so and records nothing
  const same = (status: Deciding) => {
    if (!current || rows.find((r) => r.id === current)?.status !== status) return false;
    setSaid({ words: status === "under review"
      ? <>Under review already: {titleOf(current)}. There is no admission or refusal to undo.</>
      : <>{status === "admitted" ? "Admitted" : "Refused"} already: {titleOf(current)}. Nothing is recorded, since another decision would change nothing.</> });
    return true;
  };
  const decide = async (status: Deciding, reason?: string) => {
    if (!current || deciding || shownId.current !== current) return;
    if (same(status)) return;
    if (!declared(who)) { setAsking(status); return; }
    if (!about) { setSaid({ words: <Problems problems={["The server's confirmation is not read yet, so no decision can be sent."]} /> }); return; }
    if (!leave()) return;
    const at = rows.findIndex((r) => r.id === current);
    const title = titleOf(current);
    setDeciding(true);
    const a = await write<Decided>("/api/decisions", { status, selection: sent(namedIds([current])), ...(reason?.trim() ? { reason: reason.trim() } : {}),
      declaration: { name: who.name.trim(), email: who.email.trim(), accepted: true } });
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
    const title = row ? textName(row) : current;
    setDeciding(true);
    const a = await write<{ taken_out: number; not_held: number }>(`/api/libraries/${encodeURIComponent(library.id)}/taken-out`, { selection: sent(namedIds([current])) });
    setDeciding(false);
    if (!a.ok) { setSaid({ words: <Problems problems={[`Not taken out: ${title}.`, ...a.problems]} /> }); return; }
    if (!a.data.taken_out) { setSaid({ words: <>{library.name} did not hold {title}.</> }); return; }
    setSaid({ words: <>Taken out of {library.name}: {title}; it stays admitted.</> });
    pick.current = { kind: "after", id: current, index: Math.max(at, 0) };
    changed();
  };

  const currentStatus = rows.find((r) => r.id === current)?.status ?? null;
  const isTicked = (id: string) => ticked.all !== ticked.ids.has(id);
  const tick = (ids: string[], on: boolean) => {
    const next = new Set(ticked.ids);
    ids.forEach((id) => (on !== ticked.all ? next.add(id) : next.delete(id)));
    const change = ticked.change ?? (answer?.ok ? answer.data.change : null);
    setTicked(!ticked.all && next.size === 0 ? NONE : { ...ticked, ids: next, change });
  };
  // every text the filter matches at the change the playlist's page was read at; nothing is read
  const tickAll = () => { if (answer?.ok) setTicked({ all: true, ids: new Set(), change: answer.data.change }); };

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
      else if (key === "r") ask("refused");
      else if (key === "u") ask("under review");
      else if (key === "t") takeOut();
      else if (key === "x" && current) tick([current], !isTicked(current));
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

  // the playlist's page starts at the text, so it holds it
  const openFromBlob = (id: string) => {
    playlist.open(id);
    window.location.hash = address(id, toQuery(filter));
  };
  const savedName = saved.find((s) => JSON.stringify({ ...EVERY, ...s.filter }) === JSON.stringify(filter))?.name;
  const counts = useCounts(filter, version);
  const admittedHere = address(null, toQuery({ ...filter, statuses: ["admitted"] }));
  const total = counts?.done && !counts.problems ? matched(counts) : null; // how many texts are open, once counted
  const shown = shownCounts(counts); // the counts to show: those done, or the last ones while counted again
  const shownTotal = shown ? matched(shown) : null;
  const selection: Selection | null = ticked.all
    ? { filter, change: ticked.change ?? undefined, unticked: [...ticked.ids] }
    : ticked.ids.size ? { ...namedIds([...ticked.ids]), change: ticked.change ?? undefined } : null;
  const tickedCount = ticked.all ? (total === null ? null : Math.max(0, total - ticked.ids.size)) : ticked.ids.size;

  return (
    <div className={`workspace-frame${mono ? " mono" : ""}`} onClickCapture={guard}>
      <FilterBar value={filter} onApply={apply} libraries={libraries} onLeave={toList}
        idsLabel={idsLabel} within={counts?.done && !counts.problems ? counts : null} />
      {choosing && total === 0 && (
        <p className="next-step into-banner" role="note">
          Every admitted text is in <strong>{choosing.name}</strong>. <a href={href(["texts"], { library: choosing.id })}>Open it</a>, or <a href={href(["datasets"], { new: "1", library: choosing.id })}>make a dataset from it</a>.
        </p>
      )}
      {choosing && total !== 0 && (
        <p className="next-step into-banner" role="note">
          Choosing texts to put in <strong>{choosing.name}</strong>: the admitted texts it does not hold yet{total !== null ? <> ({total.toLocaleString("en")})</> : null}. Tick them, or tick them all, then press <q>Put in {choosing.name}</q> at the bottom.{" "}
          <a href={href(["texts"], { library: choosing.id })}>Back to {choosing.name}</a>
        </p>
      )}
      <div className="open-bar">
        {library ? <LibraryHead library={library} libraries={libraries} onChanged={changed} version={version} /> : (
          <section className="open-head" aria-label="What is open">
            <h1 className="title open-title">{choosing ? `Admitted texts not in ${choosing.name}` : savedName ?? describe(filter, libraries, idsLabel ?? null)}</h1>
            {(() => {
              const c = counts && !counts.done && counts.before ? counts.before : counts && counts.counted > 0 ? counts : null;
              return c && <span className={`small aside${c === counts ? "" : " recounting"}`}>
                {counts!.done || c !== counts ? "" : "so far "}{plural(matched(c), "text")}: {c.statuses["under review"].toLocaleString("en")} under review, {c.statuses.admitted.toLocaleString("en")} admitted, {c.statuses.refused.toLocaleString("en")} refused{c.locked > 0 && <>; {c.locked.toLocaleString("en")} in a locked library</>}
              </span>;
            })()}
            <span className="small"><CountingNote counting={counts} /></span>
            {!choosing && (
              <LibraryFrom filter={filter} change={answer?.ok ? answer.data.change : null} admitted={shown ? shown.statuses.admitted : null}
                onDone={(words) => { setSaid({ words }); changed(); }} />
            )}
          </section>
        )}
        <div className="view-switch small" role="group" aria-label="How the open texts are shown">
          <a className="wic-focus" href={address(current, { ...toQuery(filter), view: null })} aria-current={!blob ? "true" : undefined}>Playlist and editor</a>
          <a className="wic-focus" href={address(current, { ...toQuery(filter), view: "blob" })} aria-current={blob ? "true" : undefined}>Blob: every text in one</a>
          <label className="tick"><input type="checkbox" className="wic-check-box wic-focus" checked={mono} onChange={(e) => { setMono(e.target.checked); keepMono(e.target.checked); }} /> <span>Monospace type</span></label>
        </div>
      </div>
      <div className={`workspace${blob ? " blob-mode" : ""}`}>
        <Sources open={filter} libraries={libraries} version={version} onLibraryMade={changed} />
        {blob ? <Blob filter={filter} order={playlist.order} onOpen={openFromBlob} /> : <>
          <div className="playlist-column">
            {!answer && <p className="aside">Reading the texts</p>}
            {answer && !answer.ok && <Problems problems={answer.problems} />}
            {answer?.ok && (
              <Playlist found={answer.data} busy={playlist.pending} current={current} link={link} isTicked={isTicked} everyTicked={ticked.all && ticked.ids.size === 0}
                onTick={tick} onTickAll={tickAll} total={shownTotal} sorted={about?.sorted ?? 0} order={playlist.order} onOrder={playlist.setOrder}
                limit={playlist.limit} onFirst={playlist.first} onPrevious={playlist.previous} onNext={playlist.next} counting={counts}
                empty={library && JSON.stringify(filter) === JSON.stringify({ ...EVERY, libraries: [library.id] })
                  ? <>{library.name} is empty. {!library.locked_at && !library.upstream ? <a href={href(["texts"], { ...toQuery(outsideOf(library.id)), into: library.id })}>Put texts in it</a> : null}</>
                  : choosing ? <>Every admitted text is in {choosing.name} already. <a href={href(["texts"], { library: choosing.id })}>Back to {choosing.name}</a></> : undefined} />
            )}
          </div>
          <div className="editor-column">
            {current ? <TextPanel id={current} version={version} onChanged={changed} onDirty={onDirty} handle={handle} onShown={onShown} find={filter.inText ? filter.search : undefined} />
              : <p className="aside">No text is open: choose one in the playlist, or press J for the first.</p>}
          </div>
        </>}
      </div>
      <Bulk selection={selection} count={tickedCount} libraries={libraries} library={library} into={choosing?.id ?? into} admittedHere={admittedHere} onUntick={() => setTicked(NONE)}
        onDone={(words) => { setSaid({ words }); setTicked(NONE); recheck(); changed(); }} />
      <div className="dock">
        {confirming && <DecideAsk title={confirming.title} status={confirming.status} busy={deciding} who={who} confirmation={about?.confirmation ?? null}
          onDecide={(reason) => { setConfirming(null); decide(confirming.status, reason); toList(); }}
          onCancel={() => { setConfirming(null); toList(); }} />}
        {said && <SaidLine said={said} />}
        {!blob && current && (
          <div className="phone-bar" role="group" aria-label="Move and decide">
            <Button variant="quiet" onClick={() => move(-1)} aria-keyshortcuts="K">Previous</Button>
            <Button variant="primary" onClick={() => ask("admitted")} disabled={deciding}>Admit…</Button>
            <Button variant="quiet" onClick={() => ask("refused")} disabled={deciding} aria-keyshortcuts="R">Refuse…</Button>
            {currentStatus && currentStatus !== "under review" && <Button variant="quiet" onClick={() => ask("under review")} disabled={deciding} aria-keyshortcuts="U" aria-label="Back to review" title="Back to review: undo its admission or refusal">To review…</Button>}
            <Button variant="quiet" onClick={() => move(1)} aria-keyshortcuts="J">Next</Button>
          </div>
        )}
      </div>
      {!blob && <Legend takes={Boolean(library && !library.locked_at && !library.upstream)} who={declared(who) ? who : null} confirmation={about?.confirmation ?? null} requirements={about?.requirements ?? null} />}
      <Dialog open={asking !== null} onClose={() => setAsking(null)} title="Who decides from this browser?">
        {asking && <AskDeclarant status={asking} onSaved={() => { setAsking(null); setSaid({ words: <>Saved. Press {KEYS[asking]} again to {VERBS[asking]} the current text.</> }); }} />}
      </Dialog>
    </div>
  );
}

// What is open, in words, when it is no saved filter and no single library:
// every part of the filter, the texts named by what a link called them.
function describe(f: TextFilter, libraries: Library[], idsName: string | null): string {
  const or = (list: string[]) => (list.length < 2 ? list.join("") : `${list.slice(0, -1).join(", ")} or ${list[list.length - 1]}`);
  const and = (list: string[]) => (list.length < 2 ? list.join("") : `${list.slice(0, -1).join(", ")} and ${list[list.length - 1]}`);
  const status = f.statuses.length === 1 ? f.statuses[0] : null;
  const head = status === "admitted" ? "Admitted texts" : status === "refused" ? "Refused texts" : status === "under review" ? "Texts under review"
    : f.statuses.length ? `Texts ${or(f.statuses)}` : "Texts";
  const words = f.search.trim();
  const parts = [
    f.tags.length ? `tagged ${and(f.tags)}` : null,
    f.licenses.length ? `under ${or(f.licenses)}` : null,
    f.ids ? (idsName ?? `among ${f.ids.length} named`) : null,
    f.libraries.length ? `in ${or(f.libraries.map((id) => libraries.find((l) => l.id === id)?.name ?? id))}` : null,
    f.outside.length ? `not in ${or(f.outside.map((id) => libraries.find((l) => l.id === id)?.name ?? id))}` : null,
    f.locked === true ? "in a locked library" : f.locked === false ? "in no locked library" : null,
    f.decision ? `on the decision ${f.decision}` : null,
    words ? `holding “${words}”${f.wholeWords ? " as whole words" : ""} in ${f.inText ? "their title or text" : "their title"}` : null,
    f.where.trim() ? "matching an SQL condition" : null,
  ].filter((p): p is string => p !== null);
  if (!parts.length && head === "Texts") return "Every text";
  return [head, ...parts].join(", ");
}

// The key that decides each way, the decision in words, and what it did.
const KEYS: Record<Deciding, string> = { admitted: "A", refused: "R", "under review": "U" };
const VERBS: Record<Deciding, string> = { admitted: "admit", refused: "refuse", "under review": "put back under review" };
const DONE: Record<Deciding, string> = { admitted: "Admitted", refused: "Refused", "under review": "Put back under review" };
const NOT_DONE: Record<Deciding, string> = { admitted: "Not admitted", refused: "Not refused", "under review": "Not put back under review" };

// The question before a decision, in one line above the phone's bar: what
// it does, the confirmation it makes and the name it records, then a button
// that decides. A refusal and a putting back under review take an optional
// public reason, and Enter decides; Escape decides nothing.
function DecideAsk({ title, status, busy, who, confirmation, onDecide, onCancel }: {
  title: string; status: Deciding; busy: boolean; who: { name: string; email: string }; confirmation: string | null;
  onDecide: (reason: string) => void; onCancel: () => void;
}) {
  const [reason, setReason] = useState("");
  const refusing = status === "refused";
  const again = status === "under review";
  const reasoned = refusing || again; // a reason is asked for, and Enter decides
  const head = refusing ? "Refuse" : again ? "Put back under review" : "Admit";
  const escape = (e: { key: string; preventDefault: () => void }) => { if (e.key === "Escape") { e.preventDefault(); onCancel(); } };
  return (
    <form className={`decide-ask said-line${refusing ? " refusing" : ""}`} role="group" aria-label={`${head} ${title}`}
      onSubmit={(e) => { e.preventDefault(); onDecide(reason); }} onKeyDown={escape}>
      <span>{head} <strong>{title}</strong>? A public decision{refusing ? ": it leaves every open library" : again ? " that undoes its admission or refusal: the decisions before it no longer count, and it leaves every open library until admitted again" : ""}, recorded as {who.name}, <span className="ledger">{who.email}</span>.
        {confirmation && <span className="small"> It confirms: <q>{confirmation}</q></span>}</span>
      {reasoned && <input className="wic-input wic-focus decide-reason" autoFocus value={reason} placeholder="Reason, optional and public" aria-label="Reason (optional, public)"
        onChange={(e) => setReason(e.target.value)} />}
      <Button type="submit" variant="primary" disabled={busy} autoFocus={!reasoned}>{refusing ? "Refuse it" : again ? "Put it back" : "Admit it"}{reasoned && <span className="key-hint">(Enter)</span>}</Button>
      <Button variant="quiet" onClick={onCancel}>Keep it<span className="key-hint">(Esc)</span></Button>
    </form>
  );
}

// What the latest key press, or the latest action on ticked texts, did.
function SaidLine({ said }: { said: Said }) {
  if ("words" in said) return <div className="said-line cue" role="status">{said.words}</div>;
  const { answer, status, id, title } = said;
  if (!answer.ok) return <div className="said-line" role="status"><Problems problems={[`${NOT_DONE[status]}: ${title}.`, ...answer.problems]} /></div>;
  return (
    <p className="said-line done cue" role="status" title={`${id}, decision ${answer.data.decision}`}>
      {DONE[status]}: {title}. <a href={href(["decisions", answer.data.decision])}>See the decision</a>
    </p>
  );
}

// The keys of the playlist, in one line under the columns, and what A, R and U confirm and record.
function Legend({ takes, who, confirmation, requirements }: { takes: boolean; who: { name: string; email: string } | null; confirmation: string | null; requirements: string | null }) {
  const keys: [string, string][] = [
    ["J K", "next, previous"], ["A", "admit, then next"], ["R", "refuse (asks a reason), then next"], ["U", "back to review (asks a reason), then next"], ["X", "tick"], ["E", "edit"], ["Ctrl F", "find and replace"], ["Ctrl S", "save"],
    ...(takes ? [["T", "take out of this library, then next"] as [string, string]] : []),
  ];
  return (
    <section className="key-legend small" aria-label="Keys">
      <span className="keys key-hint">
        {keys.map(([key, what]) => <span key={key} className="key"><kbd>{key}</kbd> {what}</span>)}
      </span>
      {who ? (
        <span className="aside" title={confirmation ? `Each press of A, R or U is a public decision and confirms: “${confirmation}”` : undefined}>
          A, R and U decide in public as {who.name}, <span className="ledger">{who.email}</span> (<a href={href(["settings"])}>change</a>){requirements && <>; each press confirms the <a href={requirements} target="_blank" rel="noreferrer">requirements</a></>}.
        </span>
      ) : <span className="act"><strong>No name and email set:</strong> A, R and U ask for them first.</span>}
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
      <p>{status === "admitted" ? "Admitting" : status === "refused" ? "Refusing" : "Putting back under review"} a text is a public decision that records a legal name and an email. Both stay in this browser, and nobody verifies them. Nothing is decided now: press the key again once they are saved.</p>
      {about && <p className="small">Each press then confirms: <q>{about.confirmation}</q> <a href={about.requirements} target="_blank" rel="noreferrer">The requirements</a>.</p>}
      <RequiredNote />
      <TextInput label={required("Legal name")} required value={who.name} autoComplete="name" onChange={(e) => setWho({ ...who, name: e.target.value })} />
      <TextInput label={required("Email")} required type="email" value={who.email} autoComplete="email" onChange={(e) => setWho({ ...who, email: e.target.value })} />
      <Button type="submit" variant="primary" disabled={!who.name.trim() || !who.email.trim()}>Save them</Button>
      {kept === false && <Problems problems={["This browser refuses to keep them (its storage is off or full), so no key can decide."]} />}
    </form>
  );
}
