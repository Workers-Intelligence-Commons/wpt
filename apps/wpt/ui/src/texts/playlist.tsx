// The list of the open texts, at the left edge of the Texts page: in order, a
// page at a time (the depot can hold a hundred million, so only one page is
// ever read): each a link that opens it in the editor, with its status, its
// scores (a letter of each name and its value, q5 e1), its length and a tick
// for the actions on ticked texts. The current text is
// marked. Above it, one box in the column of the rows' boxes ticks the page
// shown (every open text when there is one page), half ticked while only some
// rows are; once the page is ticked, a button beside it ticks every open text
// (naming how many once their count is done). The bar of ticked texts unticks them all. Its head chooses
// the order (the orders of POST /api/texts/search: id and status at any size,
// the others when the page says its listing is small enough to sort); its
// foot is the pager, which moves a page after another and shows the count
// as it grows. A page cut at the server's reading bound (partial) can hold
// few rows or none, and the next page reads on. The keys that move and decide are the Texts page's
// (texts.tsx).

import { Button } from "@workers-intelligence-commons/components";
import { useEffect, useRef, type ReactNode } from "react";
import { MatchedCount, type Counting } from "../counts";
import { type Found, type Order } from "../filter";
import { PlacePager, plural, textName, type Status } from "../parts";
import { shortValue } from "./choices";

const ORDERS: { by: Order["by"]; label: string }[] = [
  { by: "id", label: "id" }, { by: "title", label: "title" }, { by: "status", label: "status" }, { by: "length", label: "length" },
  { by: "tags", label: "tags" }, { by: "licenses", label: "licenses" }, { by: "kind", label: "kind" },
];

// A row's status as one mark, coloured as the status badges are.
const MARKS: Record<Status, string> = { "under review": "○", admitted: "■", refused: "✕" };

// A length in characters, short: 950, 12k, 3.4M.
const short = (n: number) => n < 1000 ? `${n}` : n < 1e6 ? `${Math.round(n / 1000)}k` : `${(n / 1e6).toFixed(1)}M`;

// INDEXED: the orders the depot serves at any size.
const INDEXED: Order["by"][] = ["id", "status"];

// isTicked: whether a text is ticked; everyTicked: every open text is, none
// unticked; total: how many texts are open, once counted.
export function Playlist({ found, busy, current, link, isTicked, everyTicked, onTick, onTickAll, onUntickAll, total, order, onOrder, limit, onFirst, onPrevious, onNext, counting, empty }: {
  found: Found; busy: boolean; current: string | null; link: (id: string) => string; isTicked: (id: string) => boolean; everyTicked: boolean;
  onTick: (ids: string[], on: boolean) => void; onTickAll: () => void; onUntickAll: () => void; total: number | null;
  order: Order; onOrder: (o: Order) => void; limit: number;
  onFirst: () => void; onPrevious: (() => void) | null; onNext: (() => void) | null; counting: Counting | null;
  empty?: ReactNode; // what an empty first page says in place of "No text matches."
}) {
  const list = useRef<HTMLOListElement>(null);
  // the current row stays in view as the keys move through the list
  // and the keyboard's focus moves with it when it was on a row, so one row alone looks current
  useEffect(() => {
    const row = list.current?.querySelector<HTMLElement>("[aria-current='true']");
    row?.scrollIntoView({ block: "nearest" });
    if (row && document.activeElement?.closest(".playlist-row") && !row.contains(document.activeElement)) row.querySelector<HTMLElement>(".playlist-link")?.focus({ preventScroll: true });
  }, [current, found]);
  // the box at the head of the rows' boxes ticks the page shown (on a single page, every open text); once the page
  // is ticked, a link beside it ticks every open text, as a mail list does; half ticked while only some rows are
  const pageIds = found.rows.map((r) => r.id);
  const onPage = found.rows.filter((r) => isTicked(r.id)).length;
  const onePage = !found.next && !found.previous;
  const pageTicked = found.rows.length > 0 && (everyTicked || onPage === found.rows.length);
  const someTicked = !pageTicked && onPage > 0;
  const allBox = useRef<HTMLInputElement>(null);
  useEffect(() => { if (allBox.current) allBox.current.indeterminate = someTicked; }, [someTicked]);
  const every = total === null ? "every open text" : `all ${total.toLocaleString("en")}`;
  const boxWords = everyTicked ? `${every[0].toUpperCase()}${every.slice(1)} ticked` : onPage > 0 ? `${onPage.toLocaleString("en")} on this page ticked`
    : onePage ? `Tick ${every}` : `Tick this page (${found.rows.length})`;
  const orders = ORDERS.filter((o) => INDEXED.includes(o.by) || o.by === order.by || found.sortable);
  return (
    <section className="playlist" aria-label="The open texts, in order">
      {found.rows.length > 0 && (
        <div className="playlist-ticks small">
          <label className="tick">
            <input ref={allBox} type="checkbox" className="wic-check-box wic-focus" checked={pageTicked}
              onChange={(e) => (e.target.checked ? onTick(pageIds, true) : everyTicked ? onUntickAll() : onTick(pageIds, false))} />
            <span className="tick-label">{boxWords}</span>
          </label>
          {!onePage && !everyTicked && pageTicked && <Button variant="quiet" onClick={onTickAll}>Tick {every}</Button>}
        </div>
      )}
      <div className="playlist-head small">
        <label className="row tight">
          <span className="aside">order</span>
          <select className="wic-input wic-select wic-focus" value={order.by} onChange={(e) => onOrder({ by: e.target.value as Order["by"], descending: order.descending })}>
            {orders.map((o) => <option key={o.by} value={o.by}>{o.label}</option>)}
          </select>
        </label>
        <Button variant="quiet" onClick={() => onOrder({ ...order, descending: !order.descending })} aria-label={order.descending ? "Descending; make it ascending" : "Ascending; make it descending"}>
          {order.descending ? "↓ descending" : "↑ ascending"}
        </Button>
        <span className="aside playlist-unit" title="Lengths in characters">chars</span>
      </div>
      {found.rows.length === 0 ? <p>{found.partial ? "No match among the texts read so far: the next page reads on." : found.previous ? "No text matches from here on." : empty ?? "No text matches."}</p> : (
        <ol ref={list} className="playlist-rows" aria-busy={busy}>
          {found.rows.map((r) => (
            <li key={r.id} className={`playlist-row${r.id === current ? " current" : ""}${isTicked(r.id) ? " ticked" : ""}`} aria-current={r.id === current ? "true" : undefined}>
              <input type="checkbox" className="wic-check-box wic-focus" aria-label={`Tick ${textName(r)}`} checked={isTicked(r.id)} onChange={(e) => onTick([r.id], e.target.checked)} />
              <a href={link(r.id)} className="playlist-link" title={`${r.title ?? r.id} (${r.id}), ${r.status}, ${plural(r.length, "character")}${r.scores?.length ? `; ${r.scores.map((one) => `${one.name} ${shortValue(one.value)}`).join(", ")}` : ""}`}>
                <span className={`status-mark ${r.status.replace(" ", "-")}`} aria-label={r.status}>{MARKS[r.status]}</span>
                <span className="playlist-title">{textName(r)}</span>
                {r.scores && r.scores.length > 0 && <span className="playlist-scores aside" aria-label={r.scores.map((one) => `${one.name} ${shortValue(one.value)}`).join(", ")}>
                  {r.scores.map((one) => `${one.name.slice(0, 1)}${shortValue(one.value)}`).join(" ")}</span>}
                <span className="playlist-length aside" aria-label={plural(r.length, "character")}>{short(r.length)}</span>
              </a>
            </li>
          ))}
        </ol>
      )}
      <PlacePager onFirst={onFirst} onPrevious={onPrevious} onNext={onNext} limit={limit} label="Pages of the open texts"
        counted={<MatchedCount counting={counting} />} />
    </section>
  );
}
