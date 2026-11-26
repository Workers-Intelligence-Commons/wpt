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
// #/texts/ID?kind=conversation&view=blob, and the order when it is not the
// order of ids (order=-length: the longest first; "Biggest first" sets it,
// so the blob reads the texts that take the most room first, from the index
// of lengths whatever the filter). That address is kept in this browser, and
// the Texts link of the top bar leads back to it from any page (place.ts):
// a new filter or search replaces it. Ticked texts are acted on through
// the bar at the bottom (bulk.tsx). The ticked texts are those ticked one by
// one, or, once the box of every text is ticked, every text the filter
// matches less those unticked since, at the change number the list's page
// was read at; either way the bar sends them as a selection (filter.tsx),
// never as the list of every id. A link that names texts by id may say
// what they are in words (what=...), which the filter bar shows in place of
// their number. A filter changed on the page, cleared included, opens the
// first page and its first text, read once, so the list reads from its
// start; until it comes, the place of the text says it is being read, with
// the seconds it takes. The head counts the open texts by status, at once
// from the depot's counts when they hold the filter; by tag and license too
// once the Tags or Licenses menu opens (counts.tsx). The next two texts are
// read ahead, so Next shows them at once.
//
// The keys work anywhere but in a field, the editor or a dialog (Escape
// leaves the editor): J or the down arrow (in the list) opens the next
// text, K or the up arrow the previous one, X ticks the current text, E
// puts the cursor in the editor (on a conversation, it opens its form), A
// admits the current text and opens the next one (asking first when the
// text was refused, since that reverses a decision), R refuses it at once
// and opens the next one, then offers reasons for that refusal in a line at
// the bottom (ReasonLine), which names the text refused by its title and id:
// chips, the usual ones and those kept in this
// browser (reasons.ts), picked by a click or the keys 1 to 9, and a field
// for another; the reason picked becomes that refusal's public reason (POST
// /api/decisions/ID/reason), and none is needed. Undo on that line, or Z,
// puts the text just refused back under review at once, a public decision
// too, and opens it again, so a stray R costs one click. U asks first, in one line
// with an optional reason, then puts an admitted or refused text back under
// review on Enter (Escape does nothing). The panel's buttons do the same:
// Admit & next, Refuse, ‹ and Next. Each decision is public, under the
// declaration this browser keeps (declarant.ts), which the panel names; the
// line under the text says what each press did; without a name and email, a
// decision asks for them and decides nothing. An edit of the text is saved
// before the page leaves it, by a key, a button or a link, and before a
// decision on it, so Admit & next admits the version just saved; a save
// refused keeps the text open and says why.

import { Button, Dialog, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { useAbout } from "../about";
import { post, write, type Answer } from "../api";
import type { Deciding } from "../blocked";
import { type EditorHandle } from "../code-editor";
import { declared, readDeclarant, saveDeclarant, useDeclarant } from "../declarant";
import { CountingNote, matched, settledCounts, useCounts, BY_TAG_AND_LICENSE, STATUSES_ONLY } from "../counts";
import { type Decided } from "../decide-forms";
import { EVERY, FilterBar, fromQuery, goneParts, named as namedIds, sent, toQuery, useSearch, withoutGone, type Order, type Selection, type TextFilter } from "../filter";
import { Help, PageHead, plural, Problems, required, RequiredNote, textName } from "../parts";
import { href } from "../route";
import { useSavedFilters } from "../saved-filters";
import { Blob, textsReader } from "./blob";
import { Bulk } from "./bulk";
import { Playlist } from "./playlist";
import { keepTextsPlace } from "./place";
import { Sources } from "./sources";
import { forgetReason, keepReason, USUAL, useOwnReasons } from "./reasons";
import { decideOne } from "./refuse";
import { prefetch, TextPanel, type Saver } from "./text-panel";

const LIMIT = 100; // texts of a playlist page
const SAID_SECONDS = 8; // seconds a line of what was done stays
const NOT_SCORED = "scores IS NULL"; // the quick filter of the texts with no score

// Whether texts show in a monospace type (the default) or in the serif of
// prose, kept in this browser.
const MONO = "depot-monospace";
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

// An address written before snapshots that limited the texts to libraries
// (filter.tsx, GONE_PARTS) is not opened: without that limit it would show,
// and let a bulk action reach, more texts than it was written for. The page
// says so and offers the same address without those parts as a choice.
export function Texts({ current, query }: { current: string | null; query: URLSearchParams }) {
  const gone = goneParts(query);
  if (gone.length) {
    const kept = withoutGone(query);
    return (
      <>
        <PageHead title="Texts" />
        <Problems problems={[`This address limits the texts to libraries (${gone.join(", ")}), which no longer exist since snapshots replaced them. It is not opened: without that limit it would show more texts than it was written for, and a bulk action would reach them all.`]} />
        <p className="small">A hand-picked set of texts is a tag now. <a href={href(current ? ["texts", current] : ["texts"], Object.fromEntries([...new Set(kept.keys())].map((k) => [k, kept.getAll(k)])))}>Open the same filter without the library limit</a>, knowing it matches every text its other parts match.</p>
      </>
    );
  }
  return <TextsPage current={current} query={query} />;
}

function TextsPage({ current, query }: { current: string | null; query: URLSearchParams }) {
  const queryText = query.toString();
  const filter = useMemo(() => fromQuery(new URLSearchParams(queryText)), [queryText]);
  const blob = query.get("view") === "blob";
  const orderParam = query.get("order");
  const playlist = useSearch(filter, LIMIT, orderOf(orderParam) ?? undefined);
  // the order the address names, taken whenever the address changes it
  const { setOrder } = playlist;
  useEffect(() => { const o = orderOf(orderParam); if (o) setOrder(o); }, [orderParam, setOrder]);
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
  const [confirming, setConfirming] = useState<{ id: string; title: string; status: Deciding } | null>(null); // U's or A's question, until confirmed or Escape
  const [refusal, setRefusal] = useState<Refusal | null>(null); // the refusal made last, which takes a reason afterwards
  const [deciding, setDeciding] = useState(false);
  const saver = useRef<Saver | null>(null);
  const editor = useRef<EditorHandle | null>(null);
  const pick = useRef<Pick | null>(null);
  const who = useDeclarant();
  const { about } = useAbout();
  const saved = useSavedFilters();

  const address = useCallback((id: string | null, q: Record<string, string | string[] | null>) => href(id ? ["texts", id] : ["texts"], q), []);
  // the place reached, kept in this browser for the Texts link of the top bar (place.ts)
  useEffect(() => { keepTextsPlace(window.location.hash); }, [current, queryText]);
  // what a link's named texts are, in words (#/texts?id=...&what=...), shown in place of their number while they stay named
  const what = query.get("what");
  const idsLabel = what && filter.ids ? what : undefined;
  const here = useMemo(() => ({ ...toQuery(filter), view: blob ? "blob" : null, what, order: orderParam }), [filter, blob, what, orderParam]);
  const link = useCallback((id: string) => address(id, here), [address, here]);
  // before the page leaves the text or decides on it: its edit saved, or the text kept open with why
  const settle = async (): Promise<boolean> => {
    const s = saver.current;
    if (!s || !s.dirty()) return true;
    setSaid({ words: <>Saving the edit…</> });
    const r = await s.save();
    setSaid({ words: r.words });
    return r.ok;
  };
  const go = (id: string) => { window.location.hash = link(id); };
  const changed = useCallback(() => { setVersion((v) => v + 1); playlist.reload(); }, [playlist.reload]);
  // a filter changed on the page opens the first text of its first page, read once; a filter cleared to every text
  // goes back to the playlist, unless the texts are read the longest first, which reads every text a page at a time
  // from the index of lengths
  const apply = useCallback((f: TextFilter) => {
    const every = JSON.stringify(f) === JSON.stringify(EVERY);
    window.location.hash = address(null, { ...toQuery(f), view: blob && (!every || orderParam) ? "blob" : null,
      what: what && JSON.stringify(f.ids) === JSON.stringify(filter.ids) ? what : null, order: orderParam });
  }, [address, blob, what, filter.ids, orderParam]);
  const onSaver = useCallback((s: Saver | null) => { saver.current = s; }, []);
  const shownId = useRef<string | null>(null); // the text the right column shows now: A, R and U act only on it
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
    if (!found.length && wanted.kind === "first" && readsOn()) { pick.current = wanted; playlist.next?.(); return; }
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
    if (!current) {
      if (found[0]) window.location.replace(link(found[0].id));
      else if (readsOn()) playlist.next?.(); // a page cut at the reading bound with no match: the next one reads on by itself
      return;
    }
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
  if (saidFor !== filterKey) { setSaidFor(filterKey); if (said) setSaid(null); if (refusal) setRefusal(null); }
  if (confirming && confirming.id !== current) setConfirming(null); // the question is about the text shown only

  // the keys go back to the page: the field that held them lets go, and the list stays where it was
  const toList = useCallback(() => {
    const at = document.activeElement;
    if (at instanceof HTMLElement && at !== document.body) at.blur();
  }, []);
  const [mono, setMono] = useState(readMono);

  const move = async (step: 1 | -1) => {
    if (!(await settle())) return;
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
  // U, or A on a refused text: the question first, which decides nothing
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
    const edited = Boolean(saver.current?.dirty()); // saving puts it under review: the decision then changes its status
    // back to review: the decision goes first, with its reason, and the edit is saved after it, since a saved edit
    // puts the text under review already and the decision would then be refused, its reason lost
    const back = status === "under review";
    if ((back || !edited) && same(status)) { if (status === "admitted") move(1); return; } // Admit & next on an admitted text only moves on
    if (!declared(who)) { setAsking(status); return; }
    if (!about) { setSaid({ words: <Problems problems={["The server's confirmation is not read yet, so no decision can be sent."]} /> }); return; }
    const at = rows.findIndex((r) => r.id === current);
    const title = titleOf(current);
    const id = current;
    setDeciding(true);
    setRefusal(null);
    if (!back && !(await settle())) { setDeciding(false); return; }
    setSaid({ words: <>{status === "admitted" ? "Admitting" : status === "refused" ? "Refusing" : "Putting back under review"} {title}…</> });
    const a = await write<Decided>("/api/decisions", { status, selection: sent(namedIds([id])), ...(reason?.trim() ? { reason: reason.trim() } : {}),
      declaration: { name: who.name.trim(), email: who.email.trim(), accepted: true } });
    // the edit, saved once the text is back under review; one that cannot be saved keeps the text open, said
    if (back && a.ok && edited && saver.current) {
      const r = await saver.current.save();
      if (!r.ok) { setDeciding(false); setSaid({ words: <>Put back under review as decision {a.data.decision}. {r.words}</> }); changed(); return; }
    }
    setDeciding(false);
    if (status === "refused" && a.ok) { setSaid(null); setRefusal({ decision: a.data.decision, id, title, reason: null, sending: null, problem: null }); }
    else setSaid({ status, id, title, answer: a });
    if (!a.ok) return;
    pick.current = { kind: "after", id, index: Math.max(at, 0) };
    changed();
  };
  // a reason picked or typed for the refusal made last becomes its public reason; another one replaces it
  const giveReason = async (reason: string) => {
    const r = refusal;
    if (!r || !reason.trim() || r.sending) return;
    setRefusal({ ...r, sending: reason.trim(), problem: null });
    const a = await post<{ reason: string }>(`/api/decisions/${encodeURIComponent(r.decision)}/reason`,
      { reason: reason.trim(), declaration: { name: who.name.trim(), email: who.email.trim(), accepted: true } });
    setRefusal((now) => now && now.decision === r.decision
      ? { ...now, sending: null, reason: a.ok ? a.data.reason : now.reason, problem: a.ok ? null : a.problems.join(" ") } : now);
  };
  // Undo on the refusal line, or Z: the text just refused back under review at once, and open again
  const undoRefusal = async () => {
    const r = refusal;
    if (!r || deciding) return;
    if (!(await settle())) return;
    setDeciding(true);
    setRefusal(null);
    setSaid({ words: <>Undoing the refusal of {r.title}…</> });
    const a = await decideOne(r.id, "under review");
    setDeciding(false);
    setSaid({ status: "under review", id: r.id, title: r.title, answer: a });
    if (!a.ok) return;
    changed();
    go(r.id);
  };
  const ownReasons = useOwnReasons();
  const chips = [...USUAL, ...ownReasons];

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
      else if (key === "r") decide("refused");
      else if (refusal && /^[1-9]$/.test(key) && chips[Number(key) - 1]) giveReason(chips[Number(key) - 1]);
      else if (refusal && key === "z") undoRefusal();
      else if (key === "u") ask("under review");
      else if (key === "x" && current) tick([current], !isTicked(current));
      else if (key === "e") editor.current?.focus();
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", press);
    return () => window.removeEventListener("keydown", press);
  });

  // a link of the app followed while the editor holds an edit: the edit is saved first, then the link followed
  const guard = (e: MouseEvent) => {
    const a = e.target instanceof Element ? e.target.closest<HTMLAnchorElement>("a[href^='#']") : null;
    if (!a || !saver.current?.dirty() || e.ctrlKey || e.metaKey || e.shiftKey) return;
    e.preventDefault();
    e.stopPropagation();
    const to = a.getAttribute("href")!;
    settle().then((ok) => { if (ok) window.location.hash = to.slice(1); });
  };

  // the playlist's page starts at the text, so it holds it
  const openFromBlob = (id: string) => {
    playlist.open(id);
    window.location.hash = address(id, { ...toQuery(filter), order: orderParam });
  };
  const biggest = playlist.order.by === "length" && playlist.order.descending;
  const savedName = saved.find((s) => !s.gone.length && JSON.stringify({ ...EVERY, ...s.filter }) === JSON.stringify(filter))?.name;
  // the count shows the statuses; by tag and license too once the Tags or Licenses menu opened for this filter
  const [withinFor, setWithinFor] = useState<string | null>(null);
  const countWithin = useCallback(() => setWithinFor(filterKey), [filterKey]);
  const counts = useCounts(filter, version, withinFor === filterKey ? BY_TAG_AND_LICENSE : STATUSES_ONLY);
  const within = counts?.done && !counts.problems && BY_TAG_AND_LICENSE.every((p) => counts.known.includes(p)) ? counts : null;
  const admittedHere = address(null, toQuery({ ...filter, statuses: ["admitted"] }));
  const total = counts && (counts.done || counts.settled) && !counts.problems ? matched(counts) : null; // how many texts are open, once counted
  // an empty page cut at the server's reading bound reads on by itself, until a text matches, the filter's
  // texts end, or the count says none matches
  const noneMatch = Boolean(counts?.done && !counts.problems && matched(counts) === 0);
  const readsOn = () => Boolean(answer?.ok && answer.data.rows.length === 0 && answer.data.partial && playlist.next && !noneMatch);
  // the first texts of a filter on their way: one line in place of the text says so, the count included
  const readingFirst = !blob && !current && (!answer || playlist.pending || readsOn()) && !noneMatch;
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
      <Button variant="quiet" className="refuse-button" onClick={() => decide("refused")} disabled={deciding} aria-keyshortcuts="R" title="Refuse it at once, then the next text; a reason can be picked after (R)">Refuse<span className="key-hint">R</span></Button>
      <Button variant="ink" className="admit-button" onClick={() => { if (currentStatus === "refused") ask("admitted"); else decide("admitted"); }} disabled={deciding} aria-keyshortcuts="A"
        title="Admit it, then the next text (A)">Admit &amp; next<span className="key-hint">A</span></Button>
      {currentStatus && currentStatus !== "under review" && <Button variant="quiet" className="review-button" onClick={() => ask("under review")} disabled={deciding} aria-keyshortcuts="U"
        title="Back to review: undo its admission or refusal (U)">Back to review<span className="key-hint">U</span></Button>}
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
    { label: "Biggest first", on: biggest, title: "The open texts read as one strip, the longest first, each with its size in the margin",
      go: () => { if (biggest) playlist.setOrder({ by: "id", descending: false }); window.location.hash = address(biggest ? current : null, { ...here, view: biggest ? null : "blob", order: biggest ? null : "-length" }); } },
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
        <div className="texts-top-row">
        <div className="quick-filters" role="group" aria-label="Quick filters">
          <Button variant="quiet" className="list-button" aria-expanded={listOut} onClick={() => { byButton.current = !listOut; setListOut(!listOut); }} title="The list of the open texts"><span aria-hidden="true">☰</span> List</Button>
          {quick.map((q) => <Button key={q.label} variant="quiet" className={`quick${q.on ? " on" : ""}`} aria-pressed={q.on} title={q.title} onClick={q.go}>{q.label}</Button>)}
          <div className="open-menu">
            <Button variant="quiet" aria-expanded={opening} onClick={() => setOpening(!opening)} title="Every text, each status, kind, tag and license with its count, and the saved filters">Open ▾</Button>
            {opening && (
              <div className="open-pop" onClick={(e) => { if (e.target instanceof Element && e.target.closest("a")) setOpening(false); }}>
                <Sources open={filter} version={version} />
              </div>
            )}
          </div>
        </div>
        <div className="view-switch small" role="group" aria-label="How the open texts are shown">
          <a className="wic-focus" href={address(current, { ...toQuery(filter), view: null, order: orderParam })} aria-current={!blob ? "true" : undefined} title="One text at a time, with the review panel">Text</a>
          <a className="wic-focus" href={address(current, { ...toQuery(filter), view: "blob", order: orderParam })} aria-current={blob ? "true" : undefined} title="Every open text one after another">Blob</a>
          <a className="wic-focus" href={href(["sizes"], { status: filter.statuses.length === 1 ? filter.statuses[0] : "admitted" })} title="Which tags and texts weigh the most">Size map</a>
          <label className="tick" title="Show texts in a monospace type"><input type="checkbox" className="wic-check-box wic-focus" checked={mono} onChange={(e) => { setMono(e.target.checked); keepMono(e.target.checked); }} /> <span>Mono</span></label>
        </div>
        </div>
        <FilterBar value={filter} onApply={apply} onLeave={toList}
          idsLabel={idsLabel} within={within} onCounts={countWithin}
          countsNote={within ? "Counted within the open texts." : counts && !counts.done && !counts.problems && withinFor === filterKey
            ? <>Counts of the whole depot for now; <CountingNote counting={counts} /></> : "Counts of the whole depot."} />
      </div>
      <div className="open-bar small">
        <section className="open-head" aria-label="What is open">
          <h1 className="open-title">{savedName ?? describe(filter, idsLabel ?? null)}</h1>
          {(() => {
            const final = counts && (counts.done || counts.settled);
            const c = counts && !final && counts.before ? counts.before : counts && (final || counts.counted > 0) ? counts : null;
            return c && <span className={`aside${c === counts ? "" : " recounting"}`}>
              {final || c !== counts ? "" : "so far "}{plural(matched(c), "text")}: {c.statuses["under review"].toLocaleString("en")} under review, {c.statuses.admitted.toLocaleString("en")} admitted, {c.statuses.refused.toLocaleString("en")} refused
            </span>;
          })()}
          {!counts?.done && !readingFirst && <span><CountingNote counting={counts} /></span>}
        </section>
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
                limit={playlist.limit} onFirst={playlist.first} onPrevious={playlist.previous} onNext={playlist.next} counting={counts} />
            )}
          </div>
        )}
        {blob ? <Blob readKey={JSON.stringify([filter, playlist.order])} read={textsReader(filter, playlist.order)} onOpen={openFromBlob}
            short={playlist.order.by === "length"} refused={filter.statuses.length === 1 && filter.statuses[0] === "refused"} />
          : current ? <TextPanel id={current} actions={actions} version={version} onChanged={changed} onSaved={(words) => { setSaid({ words }); playlist.reload(); }}
              saver={onSaver} handle={handle} onShown={onShown} find={filter.inText ? filter.search : undefined} />
            : readingFirst ? <Reading key={filterKey} words={`${answer?.ok && answer.data.partial && rows.length === 0
                ? "No match among the texts read so far; reading on" : "Reading the first texts of this filter"}${counts && !counts.done && !counts.problems ? " and counting them" : ""}`} />
            : answer?.ok && rows.length === 0 && (!playlist.next || noneMatch) ? <p className="aside text-column">No text matches this filter.</p>
            : <p className="aside text-column">No text is open: choose one in the list at the left edge, or press J for the first.</p>}
      </div>
      <Bulk selection={selection} count={tickedCount} admittedHere={admittedHere} onUntick={() => setTicked(NONE)}
        onDone={(words) => { setSaid({ words }); setTicked(NONE); recheck(); changed(); }} />
      <div className="dock">
        {confirming && <DecideAsk title={confirming.title} status={confirming.status} busy={deciding} who={who} confirmation={about?.confirmation ?? null} requirements={about?.requirements ?? null}
          onDecide={(reason) => { setConfirming(null); decide(confirming.status, reason); toList(); }}
          onCancel={() => { setConfirming(null); toList(); }} />}
        {refusal && <ReasonLine refusal={refusal} chips={chips} own={ownReasons} onPick={giveReason} onUndo={undoRefusal} busy={deciding}
          onClose={() => { setRefusal(null); toList(); }} onLeave={toList} />}
        {said && <SaidLine said={said} />}
      </div>
      <Dialog open={asking !== null} onClose={() => setAsking(null)} title="Who decides from this browser?">
        {asking && <AskDeclarant status={asking} onSaved={() => { setAsking(null); setSaid({ words: <>Saved. Press {KEYS[asking]} again to {VERBS[asking]} the current text.</> }); }} />}
      </Dialog>
    </div>
  );
}

// A read on its way, in place of the text: shown at once, with the seconds it has taken once past one.
function Reading({ words }: { words: string }) {
  const [since] = useState(() => Date.now());
  const [now, setNow] = useState(since);
  useEffect(() => { const timer = window.setInterval(() => setNow(Date.now()), 500); return () => window.clearInterval(timer); }, []);
  const seconds = Math.floor((now - since) / 1000);
  return <p className="text-column reading small" role="status" aria-live="polite"><span className="running-dot" aria-hidden="true" /> {words}…{seconds >= 1 && ` ${seconds} s`}</p>;
}

// The order an address names: -length the longest first, length the shortest first; none for any other.
function orderOf(param: string | null): Order | null {
  if (param === "-length" || param === "length") return { by: "length", descending: param === "-length" };
  return null;
}

// What is open, in words, when it is no saved filter: every part of the
// filter, the texts named by what a link called them.
function describe(f: TextFilter, idsName: string | null): string {
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
    f.kinds.length ? `of kind ${or(f.kinds)}` : null,
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

// The question before U puts a text back under review, or A reverses a
// refusal, in one line at the bottom: what it does, the name and email it
// records, the confirmation it makes and the link to the requirements, then
// a button that decides; what else the decision does is behind its "?". A
// putting back under review takes an optional public reason, and Enter
// decides; Escape decides nothing.
function DecideAsk({ title, status, busy, who, confirmation, requirements, onDecide, onCancel }: {
  title: string; status: Deciding; busy: boolean; who: { name: string; email: string }; confirmation: string | null; requirements: string | null;
  onDecide: (reason: string) => void; onCancel: () => void;
}) {
  const [reason, setReason] = useState("");
  const again = status === "under review";
  const reasoned = again; // a reason is asked for, and Enter decides
  const head = again ? "Put back under review" : "Admit";
  const escape = (e: { key: string; preventDefault: () => void }) => { if (e.key === "Escape") { e.preventDefault(); onCancel(); } };
  return (
    <form className="decide-ask said-line" role="group" aria-label={`${head} ${title}`}
      onSubmit={(e) => { e.preventDefault(); onDecide(reason); }} onKeyDown={escape}>
      <span>{head} <strong>{title}</strong>? Public, as {who.name}, <span className="ledger">{who.email}</span>.
        {confirmation && <> It confirms: <q>{confirmation}</q></>}{requirements && <> <a href={requirements} target="_blank" rel="noreferrer">The requirements</a>.</>}
        {again && <Help label="this decision">It undoes its admission or refusal: the decisions before it no longer count. A snapshot made before keeps the revision it holds.</Help>}</span>
      {reasoned && <input className="wic-input wic-focus decide-reason" autoFocus value={reason} placeholder="Reason, optional and public" aria-label="Reason (optional, public)"
        onChange={(e) => setReason(e.target.value)} />}
      <Button type="submit" variant="primary" disabled={busy} autoFocus={!reasoned}>{again ? "Put it back" : "Admit it"}{reasoned && <span className="key-hint">(Enter)</span>}</Button>
      <Button variant="quiet" onClick={onCancel}>Keep it<span className="key-hint">(Esc)</span></Button>
    </form>
  );
}

// The refusal made last, its reason as the depot holds it, the reason on its way, and why the last one was refused.
type Refusal = { decision: string; id: string; title: string; reason: string | null; sending: string | null; problem: string | null };

// After a refusal, in one line at the bottom: what was refused, with its
// decision, then the reasons to attach to it, each a chip (the keys 1 to 9
// pick the first nine), a field for another reason (Enter attaches it), and
// "+ chip", which attaches the reason typed and keeps it as a chip of this
// browser; a chip of
// this browser's has a cross that forgets it. The chip of the reason the
// refusal holds is marked, and the one on its way outlined.
function ReasonLine({ refusal, chips, own, onPick, onUndo, busy, onClose, onLeave }: {
  refusal: Refusal; chips: string[]; own: string[]; onPick: (reason: string) => void; onUndo: () => void; busy: boolean; onClose: () => void; onLeave: () => void;
}) {
  const [typed, setTyped] = useState("");
  const [kept, setKept] = useState<boolean | null>(null);
  const words = typed.trim();
  return (
    <div className="said-line reason-line done cue" role="group" aria-label={`The reason of the refusal of ${refusal.title}`}>
      <Button variant="quiet" className="undo-button" onClick={onUndo} disabled={busy} aria-keyshortcuts="Z"
        title="Put it back under review at once (a public decision too) and open it again (Z)">Undo<span className="key-hint">Z</span></Button>
      <span>Refused: <a href={href(["texts", refusal.id])} title={`Open ${refusal.id} again`}><strong>{refusal.title}</strong></a>{" "}
        <span className="ledger aside">{refusal.id}</span> (<a href={href(["decisions", refusal.decision])} title={`Decision ${refusal.decision}`}>its decision</a>).{" "}
        {refusal.problem ? <span className="act">Reason not attached: {refusal.problem}</span>
          : refusal.sending ? <>Attaching “{refusal.sending}”…</>
          : refusal.reason ? <>Its public reason: “{refusal.reason}”.</> : <>A reason, if you want one, public:</>}</span>
      <span className="reason-chips">
        {chips.map((c, i) => (
          <span key={c} className="preset">
            <button type="button" className={`choice wic-focus${refusal.reason === c ? " on" : ""}${refusal.sending === c ? " waiting" : ""}`}
              aria-pressed={refusal.reason === c} disabled={refusal.sending !== null} onClick={() => onPick(c)} title={i < 9 ? `Key ${i + 1}` : undefined}>
              {i < 9 && <span className="key-hint">{i + 1}</span>}{c}</button>
            {own.includes(c) && <button type="button" className="forget wic-focus" aria-label={`Forget the reason ${c}`} title="Forget this reason" onClick={() => forgetReason(c)}>✕</button>}
          </span>
        ))}
      </span>
      <form className="reason-typed" onSubmit={(e) => { e.preventDefault(); if (words) { onPick(words); onLeave(); } }}>
        <input className="wic-input wic-focus" value={typed} placeholder="Another reason (Enter)" aria-label="Another reason, public"
          onChange={(e) => { setTyped(e.target.value); setKept(null); }} onKeyDown={(e) => { if (e.key === "Escape") { e.preventDefault(); onLeave(); } }} />
        {words && !chips.includes(words) && <button type="button" className="choice wic-focus" title="Attach it, and keep it as a chip in this browser"
          onClick={() => { const ok = keepReason(words); setKept(ok); onPick(words); if (ok) setTyped(""); }}>+ chip</button>}
        {kept === false && <span className="act">This browser keeps nothing.</span>}
      </form>
      <Button variant="quiet" onClick={onClose} aria-label="Close the reasons, keeping the refusal" title="Close this line; the refusal stays">✕</Button>
    </div>
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
