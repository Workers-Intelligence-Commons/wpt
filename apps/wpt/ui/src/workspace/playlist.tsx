// The workspace's middle column: the texts of what is open, in order, a
// page at a time (the depot can hold a hundred million, so only one page is
// ever read): each a link that opens it in the editor, with its status, its
// length and a tick for the actions on ticked texts. The current text is
// marked. Its head chooses the order (the orders of POST /api/texts/search)
// and ticks every text open; its foot is the pager. The keys that move and
// decide are the workspace's (workspace.tsx), and so is their legend.

import { Button } from "@workers-intelligence-commons/components";
import { useEffect, useRef } from "react";
import { type Found, type Order } from "../filter";
import { Pager, plural, type Status } from "../parts";

const ORDERS: { by: Order["by"]; label: string }[] = [
  { by: "id", label: "id" }, { by: "title", label: "title" }, { by: "status", label: "status" }, { by: "length", label: "length" },
  { by: "tags", label: "tags" }, { by: "licenses", label: "licenses" }, { by: "libraries", label: "libraries" },
];

// A row's status as one mark, coloured as the status badges are.
const MARKS: Record<Status, string> = { "under review": "○", admitted: "■", refused: "✕" };

// A length in characters, short: 950, 12k, 3.4M.
const short = (n: number) => n < 1000 ? `${n}` : n < 1e6 ? `${Math.round(n / 1000)}k` : `${(n / 1e6).toFixed(1)}M`;

export function Playlist({ found, busy, current, link, ticked, onTick, onTickAll, tickingAll, order, onOrder, offset, limit, onOffset }: {
  found: Found; busy: boolean; current: string | null; link: (id: string) => string; ticked: Set<string>;
  onTick: (ids: string[], on: boolean) => void; onTickAll: () => void; tickingAll: boolean;
  order: Order; onOrder: (o: Order) => void; offset: number; limit: number; onOffset: (o: number) => void;
}) {
  const list = useRef<HTMLOListElement>(null);
  // the current row stays in view as the keys move through the list
  useEffect(() => { list.current?.querySelector("[aria-current='true']")?.scrollIntoView({ block: "nearest" }); }, [current, found]);
  const allTicked = found.rows.length > 0 && found.rows.every((r) => ticked.has(r.id));
  return (
    <section className="playlist" aria-label="The open texts, in order">
      <div className="playlist-head small">
        <label className="row tight">
          <span className="aside">order</span>
          <select className="wic-input wic-select wic-focus" value={order.by} onChange={(e) => onOrder({ by: e.target.value as Order["by"], descending: order.descending })}>
            {ORDERS.map((o) => <option key={o.by} value={o.by}>{o.label}</option>)}
          </select>
        </label>
        <Button variant="quiet" onClick={() => onOrder({ ...order, descending: !order.descending })} aria-label={order.descending ? "Descending; make it ascending" : "Ascending; make it descending"}>
          {order.descending ? "↓ descending" : "↑ ascending"}
        </Button>
        <label className="tick">
          <input type="checkbox" className="wic-check-box wic-focus" checked={allTicked} onChange={(e) => onTick(found.rows.map((r) => r.id), e.target.checked)} />
          <span className="tick-label">this page</span>
        </label>
        {found.total > found.rows.length && ticked.size < found.total && (
          <Button variant="quiet" disabled={tickingAll} onClick={onTickAll}>{tickingAll ? "Ticking" : `Tick all ${found.total.toLocaleString("en")}`}</Button>
        )}
      </div>
      {found.total === 0 ? <p>No text matches.</p> : (
        <ol ref={list} className="playlist-rows" aria-busy={busy} start={offset + 1}>
          {found.rows.map((r) => (
            <li key={r.id} className={`playlist-row${r.id === current ? " current" : ""}${ticked.has(r.id) ? " ticked" : ""}`} aria-current={r.id === current ? "true" : undefined}>
              <input type="checkbox" className="wic-check-box wic-focus" aria-label={`Tick ${r.id}`} checked={ticked.has(r.id)} onChange={(e) => onTick([r.id], e.target.checked)} />
              <a href={link(r.id)} className="playlist-link" title={`${r.title ?? r.id} (${r.id}), ${r.status}, ${plural(r.length, "character")}`}>
                <span className={`status-mark ${r.status.replace(" ", "-")}`} aria-label={r.status}>{MARKS[r.status]}</span>
                <span className="playlist-title">{r.title ?? (r.kind ? `an item of kind ${r.kind}` : "untitled text")}</span>
                <span className="playlist-length aside">{short(r.length)}</span>
              </a>
            </li>
          ))}
        </ol>
      )}
      <Pager offset={offset} limit={limit} total={found.total} onOffset={onOffset} label="Pages of the open texts" />
    </section>
  );
}
