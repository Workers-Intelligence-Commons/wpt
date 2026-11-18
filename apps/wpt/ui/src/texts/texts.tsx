// The Texts page: the depot's texts one after another, to read, score,
// tag, edit and decide on with as few moves of the pointer as can be. On
// top, the quick filters (under review, not scored, the newest first,
// every text), the filter bar (filter.tsx) and the "Open" menu, which
// holds everything that can be opened with its count (sources.tsx); then
// the open text, its content always in view and always editable
// (text-panel.tsx), and beside it the review panel (panel.tsx), whose
// buttons stay in place from one text to the next. The list of the open
// texts (playlist.tsx) hides at the left edge and slides out while the
// pointer rests on that edge, or while the keyboard is in it; on a touch
// screen, the "Texts" button opens it. In the blob mode, the whole of what
// is open reads as one continuous text (blob.tsx) in place of the open
// text. What is open is a filter of the texts (filter.tsx), kept in the
// address with the current text and the mode:
// #/texts/ID?library=L&view=blob. When it is one library, its head stands
// above the text (library-head.tsx); ticked texts are acted on through the
// bar at the bottom (bulk.tsx). The ticked texts are those ticked one by
// one, or, once the box of every text is ticked, every text the filter
// matches less those unticked since, at the change number the list's page
// was read at; either way the bar sends them as a selection (filter.tsx),
// never as the list of every id. Opened from a library's "Put texts in
// it" (into=L), a banner says which library the texts are chosen for (the
// admitted texts it does not hold); a link that names texts by id may say
// what they are in words (what=...), which the filter bar shows in place of
// their number. Clearing the filter opens the first page and its first
// text, so the list reads from its start. The next two texts are read
// ahead, so Next shows them at once.
//
// The keys work anywhere but in a field, the editor or a dialog (Escape
// leaves the editor): J or the down arrow (in the list) opens the next
// text, K or the up arrow the previous one, X ticks the current text, E
// puts the cursor in the editor (on a conversation, it opens its form), A
// admits the current text and opens the next one (asking first when the
// text was refused, since that reverses a decision), R asks first, in one
// line with an optional reason, then refuses it on Enter and opens the next
// one (Escape refuses nothing), U puts an admitted or refused text back
// under review the same way, and T takes it out of the open library. The
// panel's buttons do the same: Admit & next, Refuse, ‹ and Next. Each
// decision is public, under the declaration this browser keeps
// (declarant.ts), which the panel names; the line under the text says what
// each press did; without a name and email, a decision asks for them and
// decides nothing. Leaving a text whose editor holds unsaved changes, by a
// key or a link, asks first.

import { Button, Dialog, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { useAbout } from "../about";
import { write, type Answer } from "../api";
import type { Deciding } from "../blocked";
import { type EditorHandle } from "../code-editor";
import { declared, readDeclarant, saveDeclarant, useDeclarant } from "../declarant";
import { CountingNote, matched, settledCounts, useCounts } from "../counts";
import { type Decided } from "../decide-forms";
import { EVERY, FilterBar, fromQuery, named as namedIds, sent, toQuery, useSearch, type Selection, type TextFilter } from "../filter";
import { Help, plural, Problems, required, RequiredNote, textName, useLibraries, type Library } from "../parts";
import { href } from "../route";
import { useSavedFilters } from "../saved-filters";
import { Blob } from "./blob";
import { Bulk } from "./bulk";
import { LibraryFrom } from "./library-from";
import { LibraryHead, outsideOf } from "./library-head";
import { Playlist } from "./playlist";
import { Sources } from "./sources";
import { prefetch, TextPanel } from "./text-panel";

const LIMIT = 100; // texts of a playlist page
const SAID_SECONDS = 8; // seconds a line of what was done stays
const NOT_SCORED = "scores IS NULL"; // the quick filter of the texts with no score

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

export function Texts({ current, query }: { current: string | null; query: URLSearchParams }) {
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
  // the open filter narrows the texts it offers further than the admitted texts it does not hold
  const base = choosing ? outsideOf(choosing.id) : null;
  const narrowed = base !== null && (Object.keys(EVERY) as (keyof TextFilter)[]).some((k) => JSON.stringify(filter[k]) !== JSON.stringify(base[k]));
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

  // the keys go back to the page: the field that held them lets go, and the list stays where it was
  const toList = useCallback(() => {
    const at = document.activeElement;
    if (at instanceof HTMLElement && at !== document.body) at.blur();
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
      : status === "admitted" ? <>Admitted already: {titleOf(current)}. Nothing is recorded; on to the next text.</>
      : <>Refused already: {titleOf(current)}. Nothing is recorded, since another decision would change nothing.</> });
    return true;
  };
  const decide = async (status: Deciding, reason?: string) => {
    if (!current || deciding || shownId.current !== current) return;
    if (same(status)) { if (status === "admitted") move(1); return; } // Admit & next on an admitted text only moves on
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
      // Escape leaves the editor (but its find panel, which Escape closes), so the keys work again
      if (e.key === "Escape" && target?.closest(".cm-editor") && !target.closest(".cm-panel")) { (target as HTMLElement).blur(); e.preventDefault(); return; }
      if (target?.closest("input, textarea, select, [contenteditable], .cm-editor, .number-pad") || document.querySelector("dialog[open]")) return;
      const inList = Boolean(target?.closest(".playlist"));
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      if (key === "j" || (key === "ArrowDown" && inList)) move(1);
      else if (key === "k" || (key === "ArrowUp" && inList)) move(-1);
      else if (key === "a") { if (currentStatus === "refused") ask("admitted"); else decide("admitted"); } // reversing a refusal asks first
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
  const total = counts && (counts.done || counts.settled) && !counts.problems ? matched(counts) : null; // how many texts are open, once counted
  const shown = settledCounts(counts); // the counts to show: those whose statuses are final, or the last ones while counted again
  const shownTotal = shown ? matched(shown) : null;
  const selection: Selection | null = ticked.all
    ? { filter, change: ticked.change ?? undefined, unticked: [...ticked.ids] }
    : ticked.ids.size ? { ...namedIds([...ticked.ids]), change: ticked.change ?? undefined } : null;
  const tickedCount = ticked.all ? (total === null ? null : Math.max(0, total - ticked.ids.size)) : ticked.ids.size;

  // the moves and decisions, in the panel under the tags (panel.tsx): ‹, Next, Refuse and Admit & next, then back to review
  const actions = (
    <div className="panel-buttons" role="group" aria-label="Move and decide">
      <Button variant="quiet" className="move-button" onClick={() => move(-1)} aria-keyshortcuts="K" title="The previous text (K)">‹<span className="key-hint">K</span></Button>
      <Button variant="primary" className="next-button" onClick={() => move(1)} aria-keyshortcuts="J" title="The next text (J)">Next ›<span className="key-hint">J</span></Button>
      <Button variant="quiet" className="refuse-button" onClick={() => ask("refused")} disabled={deciding} aria-keyshortcuts="R" title="Refuse it, with an optional reason, then the next text (R)">Refuse<span className="key-hint">R</span></Button>
      <Button variant="ink" className="admit-button" onClick={() => { if (currentStatus === "refused") ask("admitted"); else decide("admitted"); }} disabled={deciding} aria-keyshortcuts="A"
        title="Admit it, then the next text (A)">Admit &amp; next<span className="key-hint">A</span></Button>
      {currentStatus && currentStatus !== "under review" && <Button variant="quiet" className="review-button" onClick={() => ask("under review")} disabled={deciding} aria-keyshortcuts="U"
        title="Back to review: undo its admission or refusal (U)">Back to review<span className="key-hint">U</span></Button>}
      {library && !library.locked_at && !library.upstream && <Button variant="quiet" className="review-button" onClick={takeOut} disabled={deciding} aria-keyshortcuts="T"
        title={`Take it out of ${library.name}, then the next text (T)`}>Take out<span className="key-hint">T</span></Button>}
    </div>
  );
  // the quick filters: each replaces what it is about and keeps the rest
  const quick: { label: string; on: boolean; title: string; go: () => void }[] = [
    { label: "Under review", on: filter.statuses.length === 1 && filter.statuses[0] === "under review", title: "Only the texts under review",
      go: () => apply({ ...filter, statuses: filter.statuses.length === 1 && filter.statuses[0] === "under review" ? [] : ["under review"] }) },
    { label: "Not scored", on: filter.where.trim() === NOT_SCORED, title: "Only the texts with no score yet (an SQL condition, slower on a large depot)",
      go: () => apply({ ...filter, where: filter.where.trim() === NOT_SCORED ? "" : NOT_SCORED }) },
    { label: "Newest first", on: playlist.order.by === "id" && playlist.order.descending, title: "The texts added last first",
      go: () => playlist.setOrder(playlist.order.by === "id" && playlist.order.descending ? { by: "id", descending: false } : { by: "id", descending: true }) },
    { label: "All", on: JSON.stringify(filter) === JSON.stringify(EVERY), title: "Every text of the depot", go: () => apply(EVERY) },
  ];
  // the list of the open texts: at the left edge, out while the pointer rests there or the keyboard is in it, or after the Texts button
  const [listOut, setListOut] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const later = (out: boolean, ms: number) => { window.clearTimeout(timer.current); timer.current = window.setTimeout(() => setListOut(out), ms); };
  useEffect(() => () => window.clearTimeout(timer.current), []);
  // out after the Texts button (a phone, no pointer to rest): choosing a text, or a tap outside it, puts it back
  const byButton = useRef(false);
  useEffect(() => { if (byButton.current) { byButton.current = false; setListOut(false); } }, [current]);
  useEffect(() => {
    if (!listOut || !byButton.current) return;
    const tap = (e: Event) => {
      if (e.target instanceof Element && e.target.closest(".list-drawer, .list-button")) return;
      byButton.current = false; setListOut(false);
    };
    window.addEventListener("pointerdown", tap);
    return () => window.removeEventListener("pointerdown", tap);
  }, [listOut]);
  // the next two texts, read ahead so Next shows them at once
  useEffect(() => {
    const at = rows.findIndex((r) => r.id === current);
    if (at >= 0) prefetch(rows.slice(at + 1, at + 3).map((r) => r.id), version);
  }, [current, rows, version]);
  const [opening, setOpening] = useState(false); // the Open menu, closed by Escape or a click outside it
  useEffect(() => {
    if (!opening) return;
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") setOpening(false); };
    const click = (e: Event) => { if (!(e.target instanceof Element && e.target.closest(".open-menu"))) setOpening(false); };
    window.addEventListener("keydown", key);
    window.addEventListener("mousedown", click);
    return () => { window.removeEventListener("keydown", key); window.removeEventListener("mousedown", click); };
  }, [opening]);
  return (
    <div className={`texts-frame${mono ? " mono" : ""}`} onClickCapture={guard}>
      <div className="texts-top">
        <div className="quick-filters" role="group" aria-label="Quick filters">
          <Button variant="quiet" className="list-button" aria-expanded={listOut} onClick={() => { byButton.current = !listOut; setListOut(!listOut); }} title="The list of the open texts">Texts</Button>
          {quick.map((q) => <Button key={q.label} variant="quiet" className={`quick${q.on ? " on" : ""}`} aria-pressed={q.on} title={q.title} onClick={q.go}>{q.label}</Button>)}
          <div className="open-menu">
            <Button variant="quiet" aria-expanded={opening} onClick={() => setOpening(!opening)} title="Every text, each status, library, tag and license with its count, and the saved filters">Open ▾</Button>
            {opening && (
              <div className="open-pop" onClick={(e) => { if (e.target instanceof Element && e.target.closest("a")) setOpening(false); }}>
                <Sources open={filter} libraries={libraries} version={version} onLibraryMade={changed} into={choosing?.id ?? null} />
              </div>
            )}
          </div>
        </div>
        <FilterBar value={filter} onApply={apply} libraries={libraries} onLeave={toList}
          idsLabel={idsLabel} within={counts?.done && !counts.problems ? counts : null} />
        <div className="view-switch small" role="group" aria-label="How the open texts are shown">
          <a className="wic-focus" href={address(current, { ...toQuery(filter), view: null })} aria-current={!blob ? "true" : undefined} title="One text at a time, with the review panel">Text</a>
          <a className="wic-focus" href={address(current, { ...toQuery(filter), view: "blob" })} aria-current={blob ? "true" : undefined} title="Every open text one after another">Blob</a>
          <label className="tick" title="Show texts in a monospace type"><input type="checkbox" className="wic-check-box wic-focus" checked={mono} onChange={(e) => { setMono(e.target.checked); keepMono(e.target.checked); }} /> <span>Mono</span></label>
        </div>
      </div>
      {choosing && (
        <p className="next-step into-banner row tight" role="note">
          <span>{total === 0
            ? <>Nothing left to put in <strong>{choosing.name}</strong>{narrowed ? " under this filter" : ""}.</>
            : <>Putting texts in <strong>{choosing.name}</strong>: tick, then <q>Put in</q>.</>}</span>
          <a href={href(["texts"], { library: choosing.id })}>{total === 0 ? `Open ${choosing.name}` : "Back"}</a>
          {total === 0 && <a href={href(["datasets"], { new: "1", library: choosing.id })}>Make a dataset from it</a>}
        </p>
      )}
      <div className="open-bar small">
        {library ? <LibraryHead library={library} libraries={libraries} onChanged={changed} version={version} /> : (
          <section className="open-head" aria-label="What is open">
            <h1 className="open-title">{(choosing ? null : savedName) ?? describe(filter, libraries, idsLabel ?? null)}</h1>
            {(() => {
              const final = counts && (counts.done || counts.settled);
              const c = counts && !final && counts.before ? counts.before : counts && (final || counts.counted > 0) ? counts : null;
              return c && <span className={`aside${c === counts ? "" : " recounting"}`}>
                {final || c !== counts ? "" : "so far "}{plural(matched(c), "text")}: {c.statuses["under review"].toLocaleString("en")} under review, {c.statuses.admitted.toLocaleString("en")} admitted, {c.statuses.refused.toLocaleString("en")} refused{c.locked > 0 && <>; {c.locked.toLocaleString("en")} in a locked library</>}
              </span>;
            })()}
            <span><CountingNote counting={counts} /></span>
            {!choosing && (
              <LibraryFrom filter={filter} change={answer?.ok ? answer.data.change : null} admitted={shown ? shown.statuses.admitted : null}
                onDone={(words) => { setSaid({ words }); changed(); }} />
            )}
          </section>
        )}
        {!blob && current && <span className="aside ledger position">{(() => { const at = rows.findIndex((r) => r.id === current); return at >= 0 ? `${at + 1} of ${rows.length} on this page` : ""; })()}</span>}
      </div>
      <div className={`texts-body${blob ? " blob-mode" : ""}${listOut ? " list-out" : ""}`}>
        {!blob && <div className="list-rail" aria-hidden="true" onMouseEnter={() => later(true, 250)} onMouseLeave={() => { if (listOut) later(false, 300); else window.clearTimeout(timer.current); }} />}
        {!blob && (
          <div className="list-drawer" onMouseEnter={() => window.clearTimeout(timer.current)} onMouseLeave={() => later(false, 300)}
            onFocus={() => { window.clearTimeout(timer.current); setListOut(true); }}
            onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) later(false, 300); }}>
            {!answer && <p className="aside">Reading the texts</p>}
            {answer && !answer.ok && <Problems problems={answer.problems} />}
            {answer?.ok && (
              <Playlist found={answer.data} busy={playlist.pending} current={current} link={link} isTicked={isTicked} everyTicked={ticked.all && ticked.ids.size === 0}
                onTick={tick} onTickAll={tickAll} onUntickAll={() => setTicked(NONE)} total={shownTotal} order={playlist.order} onOrder={playlist.setOrder}
                limit={playlist.limit} onFirst={playlist.first} onPrevious={playlist.previous} onNext={playlist.next} counting={counts}
                empty={library && JSON.stringify(filter) === JSON.stringify({ ...EVERY, libraries: [library.id] })
                  ? <>{library.name} is empty. {!library.locked_at && !library.upstream ? <a href={href(["texts"], { ...toQuery(outsideOf(library.id)), into: library.id })}>Put texts in it</a> : null}</>
                  : choosing ? <>None left.</> : undefined} />
            )}
          </div>
        )}
        {blob ? <Blob filter={filter} order={playlist.order} onOpen={openFromBlob} />
          : current ? <TextPanel id={current} actions={actions} version={version} onChanged={changed} onSaved={(words) => { setSaid({ words }); playlist.reload(); }}
              onDirty={onDirty} handle={handle} onShown={onShown} find={filter.inText ? filter.search : undefined} />
            : <p className="aside text-column">No text is open: choose one in the list at the left edge, or press J for the first.</p>}
      </div>
      <Bulk selection={selection} count={tickedCount} libraries={libraries} library={library} into={choosing?.id ?? into} admittedHere={admittedHere} onUntick={() => setTicked(NONE)}
        onDone={(words) => { setSaid({ words }); setTicked(NONE); recheck(); changed(); }} />
      <div className="dock">
        {confirming && <DecideAsk title={confirming.title} status={confirming.status} busy={deciding} who={who} confirmation={about?.confirmation ?? null} requirements={about?.requirements ?? null}
          onDecide={(reason) => { setConfirming(null); decide(confirming.status, reason); toList(); }}
          onCancel={() => { setConfirming(null); toList(); }} />}
        {said && <SaidLine said={said} />}
      </div>
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

// The question before a decision, in one line above the phone's bar (and
// before A reverses a refusal): what it does, the name and email it records,
// the confirmation it makes and the link to the requirements, then a button
// that decides; what else the decision does is behind its "?". A refusal and a putting back under review take an optional
// public reason, and Enter decides; Escape decides nothing.
function DecideAsk({ title, status, busy, who, confirmation, requirements, onDecide, onCancel }: {
  title: string; status: Deciding; busy: boolean; who: { name: string; email: string }; confirmation: string | null; requirements: string | null;
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
      <span>{head} <strong>{title}</strong>? Public, as {who.name}, <span className="ledger">{who.email}</span>.
        {confirmation && <> It confirms: <q>{confirmation}</q></>}{requirements && <> <a href={requirements} target="_blank" rel="noreferrer">The requirements</a>.</>}
        {(refusing || again) && <Help label="this decision">{refusing ? "It leaves every open library." : "It undoes its admission or refusal: the decisions before it no longer count, and it leaves every open library until admitted again."}</Help>}</span>
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
