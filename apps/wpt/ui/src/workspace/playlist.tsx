// The workspace's middle column: the texts of what is open, in order, a
// page at a time (the depot can hold a hundred million, so only one page is
// ever read): each a link that opens it in the editor, with its status, its
// length and a tick for the actions on ticked texts. The current text is
// marked. Its head chooses the order (the orders of POST /api/texts/search:
// id and status at any size, the others once the count is done and finds at
// most the depot's limit, which no index serves) and ticks every text open,
// naming how many once their count is done; its
// foot is the pager, which moves a page after another and shows the count
// as it grows. The keys that move and decide are the workspace's
// (workspace.tsx), and so is their legend.

import { Button } from "@workers-intelligence-commons/components";
import { useEffect, useRef, type ReactNode } from "react";
import { MatchedCount, type Counting } from "../counts";
import { type Found, type Order } from "../filter";
import { PlacePager, plural, textName, type Status } from "../parts";

const ORDERS: { by: Order["by"]; label: string }[] = [
  { by: "id", label: "id" }, { by: "title", label: "title" }, { by: "status", label: "status" }, { by: "length", label: "length" },
  { by: "tags", label: "tags" }, { by: "licenses", label: "licenses" }, { by: "libraries", label: "libraries" },
];

// A row's status as one mark, coloured as the status badges are.
const MARKS: Record<Status, string> = { "under review": "○", admitted: "■", refused: "✕" };

// A length in characters, short: 950, 12k, 3.4M.
const short = (n: number) => n < 1000 ? `${n}` : n < 1e6 ? `${Math.round(n / 1000)}k` : `${(n / 1e6).toFixed(1)}M`;

// INDEXED: the orders the depot serves at any size.
const INDEXED: Order["by"][] = ["id", "status"];

// isTicked: whether a text is ticked; everyTicked: every open text is, none
// unticked; total: how many texts are open, once counted; sorted: the most
// matches the other orders are given for.
export function Playlist({ found, busy, current, link, isTicked, everyTicked, onTick, onTickAll, total, sorted, order, onOrder, limit, onFirst, onPrevious, onNext, counting, empty }: {
  found: Found; busy: boolean; current: string | null; link: (id: string) => string; isTicked: (id: string) => boolean; everyTicked: boolean;
  onTick: (ids: string[], on: boolean) => void; onTickAll: () => void; total: number | null; sorted: number;
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
  const allTicked = found.rows.length > 0 && found.rows.every((r) => isTicked(r.id));
  const orders = ORDERS.filter((o) => INDEXED.includes(o.by) || o.by === order.by || (total !== null && total <= sorted));
  return (
    <section className="playlist" aria-label="The open texts, in order">
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
        <label className="tick">
          <input type="checkbox" className="wic-check-box wic-focus" checked={allTicked} onChange={(e) => onTick(found.rows.map((r) => r.id), e.target.checked)} />
          <span className="tick-label">this page</span>
        </label>
        {found.rows.length > 0 && !everyTicked && (total !== null || found.next || found.previous) && (total === null || total > 1) && (
          <Button variant="quiet" onClick={onTickAll}>{total ? `Tick all ${total.toLocaleString("en")}` : "Tick all"}</Button>
        )}
        <span className="aside playlist-unit">lengths in characters</span>
      </div>
      {found.rows.length === 0 ? <p>{found.previous ? "No text matches from here on." : empty ?? "No text matches."}</p> : (
        <ol ref={list} className="playlist-rows" aria-busy={busy}>
          {found.rows.map((r) => (
            <li key={r.id} className={`playlist-row${r.id === current ? " current" : ""}${isTicked(r.id) ? " ticked" : ""}`} aria-current={r.id === current ? "true" : undefined}>
              <input type="checkbox" className="wic-check-box wic-focus" aria-label={`Tick ${textName(r)}`} checked={isTicked(r.id)} onChange={(e) => onTick([r.id], e.target.checked)} />
              <a href={link(r.id)} className="playlist-link" title={`${r.title ?? r.id} (${r.id}), ${r.status}, ${plural(r.length, "character")}`}>
                <span className={`status-mark ${r.status.replace(" ", "-")}`} aria-label={r.status}>{MARKS[r.status]}</span>
                <span className="playlist-title">{textName(r)}</span>
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
