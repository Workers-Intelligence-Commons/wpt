// A filter of the depot's texts, the address query it is kept in, the
// paged search of the texts it matches through POST /api/texts/search, and
// the compact bar that edits it and applies each change after a short pause.

import { Button } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useRef, useState } from "react";
import { get, post, type Answer } from "./api";
import { STATUSES, type Library, type Status } from "./parts";
import { ChoiceMenu, TickMenu } from "./ticks";

export type TextFilter = {
  ids: string[] | null; // only these texts; null: any
  tags: string[]; // the text has every one
  licenses: string[]; // the text names one of them
  statuses: Status[]; // the text has one of them
  libraries: string[]; // one of them holds the text
  locked: boolean | null; // whether a locked library holds the text
  where: string; // an SQL condition over the texts table; empty: none
};

export const EVERY: TextFilter = { ids: null, tags: [], licenses: [], statuses: [], libraries: [], locked: null, where: "" };

export type Row = {
  id: string; kind: string | null; title: string | null; authors: string[] | null; tags: string[]; licenses: string[];
  libraries: string[]; status: Status; locked: boolean; length: number;
};
export type Found = {
  where: string; total: number; statuses: Record<Status, number>; locked: number;
  tags: { tag: string; texts: number }[]; licenses: { license: string; texts: number }[]; rows: Row[]; ids?: string[];
};
export type Order = { by: "id" | "title" | "status" | "tags" | "licenses" | "libraries" | "length"; descending: boolean };
export const BY_ID: Order = { by: "id", descending: false };

const body = (f: TextFilter) => ({ ...f, where: f.where.trim() === "" ? null : f.where });

// The texts a filter matches from offset; everyId: with the id of every one.
export const search = (filter: TextFilter, offset = 0, limit = 50, order: Order = BY_ID, everyId = false) =>
  post<Found>("/api/texts/search", { filter: body(filter), offset, limit, order: order.by, descending: order.descending, every_id: everyId });

// The filter a page's query names (#/texts?tag=en&status=admitted&library=x&locked=no&where=...).
export function fromQuery(query: URLSearchParams): TextFilter {
  const locked = query.get("locked");
  return {
    ids: null, tags: query.getAll("tag"), licenses: query.getAll("license"),
    statuses: query.getAll("status").filter((s): s is Status => STATUSES.includes(s as Status)), libraries: query.getAll("library"),
    locked: locked === "yes" ? true : locked === "no" ? false : null, where: query.get("where") ?? "",
  };
}

// The SQL condition goes into the address as typed, so the filter read back
// from it equals the draft and never rewrites what is being typed; body()
// alone decides that a blank condition means none.
export function toQuery(f: TextFilter): Record<string, string | string[] | null> {
  return { tag: f.tags, license: f.licenses, status: f.statuses, library: f.libraries,
    locked: f.locked === null ? null : f.locked ? "yes" : "no", where: f.where || null };
}

// The texts a filter matches, a page at a time, in the order asked; back to
// the first page when the filter, the order or the page size changes. Only
// the answer to the latest request is kept: answer is null until the
// filter's first answer comes, and pending is true while a newer page or a
// reload is asked.
export function useSearch(filter: TextFilter, initialLimit = 50) {
  const [offset, setOffset] = useState(0);
  const [limit, setLimitState] = useState(initialLimit);
  const [order, setOrderState] = useState<Order>(BY_ID);
  const [got, setGot] = useState<{ key: string; asked: number; answer: Answer<Found> } | null>(null);
  const [asked, setAsked] = useState(0);
  const filterKey = JSON.stringify(filter);
  const [shownFilter, setShownFilter] = useState(filterKey);
  if (filterKey !== shownFilter) {
    setShownFilter(filterKey);
    setOffset(0);
  }
  const key = JSON.stringify([filterKey, offset, limit, order]);
  const reload = useCallback(() => setAsked((n) => n + 1), []);
  useEffect(() => {
    let live = true;
    const [f, o, l, by] = JSON.parse(key);
    search(JSON.parse(f), o, l, by).then((answer) => live && setGot({ key, asked, answer }));
    return () => { live = false; };
  }, [key, asked]);
  const answer = got && JSON.parse(got.key)[0] === filterKey ? got.answer : null;
  const pending = got === null || got.key !== key || got.asked !== asked;
  const setLimit = (n: number) => { setLimitState(n); setOffset(0); };
  const setOrder = (o: Order) => { setOrderState(o); setOffset(0); };
  return { answer, pending, offset, setOffset, limit, setLimit, order, setOrder, reload };
}

type Tag = { tag: string; texts: number };
type License = { id: string; name: string; texts: number };

const PAUSE = 400; // milliseconds after the last change before a filter applies

// The filter's fields in one compact bar, each named inside its control
// ("Status: any"). Each change applies after a
// short pause, with no button. Leaves out the libraries when fixedLibrary
// is given (the filter of one library's texts), and the locked choice too
// when that library is locked, since every one of its texts is then in one.
export function FilterBar({ value, onApply, libraries, fixedLibrary }: { value: TextFilter; onApply: (f: TextFilter) => void; libraries: Library[]; fixedLibrary?: string }) {
  const [draft, setDraft] = useState(value);
  const [tags, setTags] = useState<Tag[]>([]);
  const [licenses, setLicenses] = useState<License[]>([]);
  const applied = useRef(JSON.stringify(value));
  const valueKey = JSON.stringify(value);
  useEffect(() => {
    if (valueKey !== applied.current) { applied.current = valueKey; setDraft(JSON.parse(valueKey)); }
  }, [valueKey]);
  useEffect(() => {
    get<Tag[]>("/api/tags").then((a) => a.ok && setTags(a.data));
    get<License[]>("/api/licenses").then((a) => a.ok && setLicenses(a.data));
  }, []);
  useEffect(() => {
    const key = JSON.stringify(draft);
    if (key === applied.current) return;
    const timer = window.setTimeout(() => { applied.current = key; onApply(draft); }, PAUSE);
    return () => window.clearTimeout(timer);
  }, [draft, onApply]);
  const change = (part: Partial<TextFilter>) => setDraft({ ...draft, ...part });
  const cleared = { ...EVERY, libraries: fixedLibrary ? [fixedLibrary] : [] };
  const lockedFixed = fixedLibrary !== undefined && Boolean(libraries.find((l) => l.id === fixedLibrary)?.locked_at);
  return (
    <div className="filter-bar" role="search" aria-label="Filter the texts">
      <TickMenu label="Status" options={STATUSES.map((s) => ({ value: s, label: s }))} chosen={draft.statuses}
        onChange={(statuses) => change({ statuses: statuses as Status[] })} />
      <TickMenu label="Tags" search options={tags.map((t) => ({ value: t.tag, label: t.tag, count: t.texts }))} chosen={draft.tags}
        onChange={(tags) => change({ tags })} />
      <TickMenu label="Licenses" search options={licenses.map((l) => ({ value: l.id, label: l.id, words: `${l.id} ${l.name}`, count: l.texts }))}
        chosen={draft.licenses} onChange={(licenses) => change({ licenses })} />
      {fixedLibrary === undefined && (
        <TickMenu label="Libraries" options={libraries.map((l) => ({ value: l.id, label: `${l.name}${l.locked_at ? " (locked)" : ""}`, words: `${l.id} ${l.name}`, count: l.texts }))}
          chosen={draft.libraries} onChange={(libraries) => change({ libraries })} />
      )}
      {!lockedFixed && (
        <ChoiceMenu label="In a locked library" chosen={draft.locked === null ? "either" : draft.locked ? "yes" : "no"}
          options={[{ value: "either", label: "either" }, { value: "no", label: "no" }, { value: "yes", label: "yes" }]}
          onChange={(v) => change({ locked: v === "either" ? null : v === "yes" })} />
      )}
      <input className="wic-input wic-focus sql" type="text" value={draft.where} aria-label="SQL condition"
        placeholder="SQL condition"
        title="An SQL condition over the columns of the texts view, each text as it is now (id, revision, change, text, kind, content, licenses, notice, title, authors, tags, address, decisions, libraries), such as title LIKE '%fox%'."
        onChange={(e) => change({ where: e.target.value })} />
      {JSON.stringify(draft) !== JSON.stringify(cleared) && (
        <Button variant="quiet" onClick={() => setDraft(cleared)}>Clear the filter</Button>
      )}
    </div>
  );
}
