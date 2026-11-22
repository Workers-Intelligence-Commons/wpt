// A filter of the depot's texts, the address query it is kept in, the
// paged search of the texts it matches through POST /api/texts/search, and
// the compact bar that edits it: the menus and the plain search apply each
// change after a short pause, the SQL condition only on Enter. The server
// turns every part into its condition (wpt_depot.filters); the only SQL
// the page sends is the condition the person types.
// A page of texts comes with no count (counts.tsx counts them apart): it
// starts after the place of a row or ends before one, so the pages are
// read one after another, forward or back, never by their number.

import { Button } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useRef, useState } from "react";
import { get, post, type Answer } from "./api";
import { STATUSES, type Status } from "./parts";
import { TickMenu } from "./ticks";

export type TextFilter = {
  ids: string[] | null; // only these texts; null: any
  tags: string[]; // the text has every one
  licenses: string[]; // the text names one of them
  statuses: Status[]; // the text has one of them
  kinds: string[]; // the text is of one of these kinds (plain, conversation, task...); empty: any
  decision: string | null; // a decision the text is on; null: any
  search: string; // words the title (with inText, the text too) holds, each one, or the exact id; empty: any
  where: string; // an SQL condition over the texts table, typed by the person; empty: none
  inText: boolean; // the search's words are looked for in the text itself too
  wholeWords: boolean; // each word of the search matches only as a whole word ("rivers" never in "drivers")
};

export const EVERY: TextFilter = { ids: null, tags: [], licenses: [], statuses: [], kinds: [], decision: null, search: "", where: "", inText: false, wholeWords: false };

export type Row = {
  id: string; kind: string | null; title: string | null; authors: string[] | null; tags: string[]; licenses: string[];
  scores: { name: string; value: number }[] | null; status: Status; length: number;
};
// where a page starts or ends: a row's values of the order, as the server gives them
export type Place = (string | number)[];
// change: the depot's change number the page was read at, which a selection of its texts names
// partial: the page stopped at the server's reading bound, so it can hold few rows or none and next reads on;
// sortable: the texts it reads from are few enough for every order, not only id and status
export type Found = { rows: Row[]; next: Place | null; previous: Place | null; partial: boolean; sortable: boolean; change: number };
export type Order = { by: "id" | "title" | "status" | "kind" | "tags" | "licenses" | "length"; descending: boolean };
export const BY_ID: Order = { by: "id", descending: false };
// a page: the first (none), the one after a place, the one before it, or the one that starts at a text's place
export type At = { after?: Place; before?: Place; at?: string };

// What the server reads: every part of the filter, the plain search's words
// as typed (the server splits them at spaces), and the SQL condition the
// person typed, or none when blank.
export const body = (f: TextFilter) => {
  const { search: words, inText, wholeWords, where, ...rest } = f;
  return { ...rest, where: where.trim() === "" ? null : where, search: words.trim() === "" ? null : words, in_text: inText, whole_words: wholeWords };
};

// The texts a write takes (a decision, a put...): those the filter matches
// when it runs, less the ids unticked and every text changed after the change
// number its page was read at, which the write counts apart. Never a list of
// ids the browser read: a text's own page names that one text in the filter.
export type Selection = { filter: TextFilter; change?: number; unticked?: string[] };
export const named = (ids: string[]): Selection => ({ filter: { ...EVERY, ids } });
export const sent = (s: Selection) => ({
  filter: body(s.filter), ...(s.change !== undefined ? { change: s.change } : {}), ...(s.unticked?.length ? { unticked: s.unticked } : {}),
});

// The page at of the texts a filter matches.
export const search = (filter: TextFilter, at: At = {}, limit = 50, order: Order = BY_ID) =>
  post<Found>("/api/texts/search", { filter: body(filter), ...at, limit, order: order.by, descending: order.descending });

// The parts an address written before snapshots may carry, which limited the
// texts to libraries; libraries no longer exist, so such an address is not
// opened (the Texts page says why): without that limit it would match more
// texts than the one it was written for.
export const GONE_PARTS = ["library", "outside", "locked"] as const;
export const goneParts = (query: URLSearchParams): string[] => GONE_PARTS.filter((part) => query.has(part));

// The same address less those parts, opened only once the person chooses it.
export function withoutGone(query: URLSearchParams): URLSearchParams {
  const kept = new URLSearchParams(query);
  for (const part of GONE_PARTS) kept.delete(part);
  return kept;
}

// The filter a page's query names (#/texts?tag=en&status=admitted&kind=conversation&id=a&decision=d&q=w&words=whole&where=...).
// The Texts page refuses an address with a part of GONE_PARTS before it reads one.
export function fromQuery(query: URLSearchParams): TextFilter {
  const ids = query.getAll("id");
  return {
    ids: ids.length ? ids : null, tags: query.getAll("tag"), licenses: query.getAll("license"),
    statuses: query.getAll("status").filter((s): s is Status => STATUSES.includes(s as Status)), kinds: query.getAll("kind"),
    decision: query.get("decision"),
    search: query.get("q") ?? "", where: query.get("where") ?? "", inText: query.get("in") === "text", wholeWords: query.get("words") === "whole",
  };
}

// The SQL condition goes into the address as typed, so the filter read back
// from it equals the draft and never rewrites what is being typed; body()
// alone decides that a blank condition means none.
export function toQuery(f: TextFilter): Record<string, string | string[] | null> {
  return { id: f.ids ?? [], tag: f.tags, license: f.licenses, status: f.statuses, kind: f.kinds, decision: f.decision, q: f.search || null, where: f.where || null,
    in: f.inText ? "text" : null, words: f.wholeWords ? "whole" : null };
}

// The texts a filter matches, a page at a time, in the order asked; back to
// the first page when the filter, the order or the page size changes. Only
// the answer to the latest request is kept: answer is null until the
// filter's first answer comes, and pending is true while a newer page or a
// reload is asked. next and previous move by one page when the answer says
// there is one; open starts a page at a text's place in the order.
export function useSearch(filter: TextFilter, initialLimit = 50) {
  const [at, setAt] = useState<At>({});
  const [limit, setLimitState] = useState(initialLimit);
  const [order, setOrderState] = useState<Order>(BY_ID);
  const [got, setGot] = useState<{ key: string; asked: number; answer: Answer<Found> } | null>(null);
  const [asked, setAsked] = useState(0);
  const filterKey = JSON.stringify(filter);
  const [shownFilter, setShownFilter] = useState(filterKey);
  if (filterKey !== shownFilter) {
    setShownFilter(filterKey);
    setAt({});
  }
  const key = JSON.stringify([filterKey, at, limit, order]);
  const reload = useCallback(() => setAsked((n) => n + 1), []);
  useEffect(() => {
    let live = true;
    const [f, a, l, by] = JSON.parse(key);
    search(JSON.parse(f), a, l, by).then((answer) => live && setGot({ key, asked, answer }));
    return () => { live = false; };
  }, [key, asked]);
  const answer = got && JSON.parse(got.key)[0] === filterKey ? got.answer : null;
  const pending = got === null || got.key !== key || got.asked !== asked;
  const found = answer?.ok ? answer.data : null;
  const next = found?.next ? () => setAt({ after: found.next! }) : null;
  const previous = found?.previous ? () => setAt({ before: found.previous! }) : null;
  const first = () => setAt({});
  const open = useCallback((id: string) => setAt({ at: id }), []);
  const setLimit = (n: number) => { setLimitState(n); setAt({}); };
  const setOrder = (o: Order) => { setOrderState(o); setAt({}); };
  return { answer, pending, next, previous, first, open, limit, setLimit, order, setOrder, reload };
}

type Tag = { tag: string; texts: number };
type License = { id: string; name: string; texts: number };

const PAUSE = 400; // milliseconds after the last change before a filter applies

// The filter's fields in one compact bar, each named inside its control
// ("Status: any"; Kinds lists only the kinds the depot has texts of, with
// their counts from GET /api/counts, and shows once there are two). A menu, a chip or
// the plain search applies after a short pause, with no button. The SQL
// condition applies only on Enter, so a key pressed in it never runs a
// half-typed condition; Enter, or Escape (which puts back the condition that
// applies), hands the keys back to the playlist (leave). The Tags and
// Licenses menus count the texts of the depot, or, given within, the open
// texts. The parts a link sets and no menu edits (the texts named, the
// decision they are on) show as chips, each removed by its cross. The SQL
// condition and the two ways of searching (in the texts too, whole words)
// show after "More filters", or at once while one of them is set. On a
// phone, only the search shows, and a button opens the rest.
export function FilterBar({ value, onApply, onLeave, idsLabel, within }: {
  value: TextFilter; onApply: (f: TextFilter) => void; onLeave?: () => void;
  idsLabel?: string; // what the texts named are, in words ("the 2 texts added just now")
  // the counts of the open texts once done: the Tags and Licenses menus then count within them, not over the depot
  within?: { tags: { tag: string; texts: number }[]; licenses: { license: string; texts: number }[] } | null;
}) {
  const [draft, setDraft] = useState(value);
  const [typedWhere, setTypedWhere] = useState(value.where);
  const [tags, setTags] = useState<Tag[]>([]);
  const [licenses, setLicenses] = useState<License[]>([]);
  const [kinds, setKinds] = useState<Record<string, number>>({});
  const applied = useRef(JSON.stringify(value));
  const valueKey = JSON.stringify(value);
  useEffect(() => {
    if (valueKey !== applied.current) { applied.current = valueKey; const v = JSON.parse(valueKey); setDraft(v); setTypedWhere(v.where); }
  }, [valueKey]);
  useEffect(() => {
    get<Tag[]>("/api/tags").then((a) => a.ok && setTags(a.data));
    get<License[]>("/api/licenses").then((a) => a.ok && setLicenses(a.data));
    get<{ kinds?: Record<string, number> }>("/api/counts").then((a) => a.ok && setKinds(a.data.kinds ?? {}));
  }, []);
  useEffect(() => {
    const key = JSON.stringify(draft);
    if (key === applied.current) return;
    const timer = window.setTimeout(() => { applied.current = key; onApply(draft); }, PAUSE);
    return () => window.clearTimeout(timer);
  }, [draft, onApply]);
  const change = (part: Partial<TextFilter>) => setDraft({ ...draft, ...part });
  const tagsWithin = within ? new Map(within.tags.map((t) => [t.tag, t.texts])) : null;
  const licensesWithin = within ? new Map(within.licenses.map((l) => [l.license, l.texts])) : null;
  const tagCount = (t: Tag) => (tagsWithin ? tagsWithin.get(t.tag) ?? 0 : t.texts);
  const licenseCount = (l: License) => (licensesWithin ? licensesWithin.get(l.id) ?? 0 : l.texts);
  const leave = (input: HTMLInputElement) => { input.blur(); onLeave?.(); };
  const cleared = EVERY;
  const pendingWhere = typedWhere !== draft.where;
  const [shown, setShown] = useState(false); // on a phone, the menus show only once asked for
  // the filters used less often (the SQL condition, the ways of searching) show once asked for, or while one is set
  const [more, setMore] = useState(false);
  const extra = more || draft.where !== "" || typedWhere !== "";
  const narrowed = JSON.stringify({ ...draft, search: "" }) !== JSON.stringify({ ...cleared, search: "" });
  const kindsHeld = Object.entries(kinds).filter(([k, n]) => n > 0 || draft.kinds.includes(k));
  const chips: [string, Partial<TextFilter>][] = [
    ...(draft.ids ? [[idsLabel ?? `${draft.ids.length === 1 ? "1 text" : `${draft.ids.length} texts`} named`, { ids: null }] as [string, Partial<TextFilter>]] : []),
    ...(draft.decision ? [[`on the decision ${draft.decision}`, { decision: null }] as [string, Partial<TextFilter>]] : []),
  ];
  return (
    <div className={`filter-bar${shown ? " shown" : ""}`} role="search" aria-label="Filter the texts">
      <Button variant="quiet" className="filter-toggle" aria-expanded={shown} onClick={() => setShown(!shown)}>{shown ? "Hide the filters" : narrowed ? "Filters (on)" : "Filters"}</Button>
      <input className="wic-input wic-focus find" type="search" value={draft.search} aria-label={draft.inText ? "Search titles, ids and the texts themselves" : "Search titles and ids"}
        placeholder={draft.inText ? "Search titles and texts" : "Search titles"}
        title={`Words ${draft.inText ? "the title or the text itself" : "the title of a text"} holds, each one, in any case, or its exact id. Enter or Escape goes back to the list, where J and K move.`}
        onChange={(e) => change({ search: e.target.value })}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === "Escape") { e.preventDefault(); leave(e.currentTarget); } }} />
      {(extra || draft.inText || draft.wholeWords) && <label className="tick small in-text" title="Look for the words inside each text as well as in its title. It reads every text the other filters leave, so it is slower on a big depot.">
        <input type="checkbox" className="wic-check-box wic-focus" checked={draft.inText} onChange={(e) => change({ inText: e.target.checked })} />
        <span>in the texts too</span>
      </label>}
      {(extra || draft.inText || draft.wholeWords) && <label className="tick small in-text" title="Each word matches only as a whole word: no letter, digit or underscore right before or after it, so “rivers” finds “rivers” but not “drivers”.">
        <input type="checkbox" className="wic-check-box wic-focus" checked={draft.wholeWords} onChange={(e) => change({ wholeWords: e.target.checked })} />
        <span>whole words</span>
      </label>}
      <TickMenu label="Status" options={STATUSES.map((s) => ({ value: s, label: s }))} chosen={draft.statuses}
        onChange={(statuses) => change({ statuses: statuses as Status[] })} />
      <TickMenu label="Tags" search options={tags.map((t) => ({ value: t.tag, label: t.tag, count: tagCount(t) })).sort((a, b) => b.count - a.count || a.value.localeCompare(b.value))} chosen={draft.tags}
        onChange={(tags) => change({ tags })} />
      <TickMenu label="Licenses" search options={licenses.map((l) => ({ value: l.id, label: l.id, words: `${l.id} ${l.name}`, count: licenseCount(l) })).sort((a, b) => b.count - a.count || a.value.localeCompare(b.value))}
        chosen={draft.licenses} onChange={(licenses) => change({ licenses })} />
      {(kindsHeld.length > 1 || draft.kinds.length > 0) && (
        <TickMenu label="Kinds" options={kindsHeld.map(([k, n]) => ({ value: k, label: k, count: n })).sort((a, b) => b.count - a.count || a.value.localeCompare(b.value))}
          chosen={draft.kinds} onChange={(kinds) => change({ kinds })} />
      )}
      {(JSON.stringify(draft) !== JSON.stringify(cleared) || typedWhere !== "") && (
        <button type="button" className="forget wic-focus clear-filter" aria-label="Clear the filter" title="Clear the filter: every text again" onClick={() => { setDraft(cleared); setTypedWhere(""); }}>✕</button>
      )}
      {chips.map(([label, removed]) => (
        <span key={label} className="chip small">{label}
          <button type="button" className="forget wic-focus" aria-label={`Remove the filter: ${label}`} onClick={() => change(removed)}>✕</button>
        </span>
      ))}
      {!extra && <Button variant="quiet" aria-expanded={false} onClick={() => setMore(true)}>More filters</Button>}
      {extra && <input className={`wic-input wic-focus sql${pendingWhere ? " pending" : ""}`} type="text" value={typedWhere} aria-label="SQL condition, applied on Enter"
        placeholder="SQL condition, then Enter"
        title="An SQL condition over the columns of the texts view, each text as it is now (id, revision, change, text, kind, content, licenses, notice, title, authors, tags, address, decisions, scores, status), such as title LIKE '%fox%'. It applies on Enter; Escape puts back the one that applies. Either way the keys go back to the list."
        onChange={(e) => setTypedWhere(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") { e.preventDefault(); change({ where: typedWhere }); leave(e.currentTarget); }
          else if (e.key === "Escape") { e.preventDefault(); setTypedWhere(draft.where); leave(e.currentTarget); }
        }} />}
      {pendingWhere && <span className="small aside sql-note">Enter applies it</span>}
    </div>
  );
}
