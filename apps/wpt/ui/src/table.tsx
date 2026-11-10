// The texts of one page as a table of one-line rows: a long value is cut
// and shown whole on hover. A column header that names an order sorts by
// it, and a second click turns the order around; a box ticks each row when
// ticked is given. A library's own page names that library in otherLibraries,
// and its libraries column then shows only the other libraries of each text,
// or is left out when no text of the page is in another one. MatchedTexts
// is such a table, paged, of the texts a filter matches, for a view that
// shows the texts of one thing (a license, a tag, a decision, a problem).

import { Skeleton } from "@workers-intelligence-commons/components";
import { type ReactNode } from "react";
import { useSearch, type Order, type Row, type TextFilter } from "./filter";
import { LibraryLink, Pager, Problems, StatusBadge, useLibraries, type Library } from "./parts";
import { href } from "./route";

const name = (r: Row) => r.title ?? (r.kind ? `an item of kind ${r.kind}` : "untitled text");

type Column = { by: Order["by"]; label: string; className?: string; cell: (r: Row, byId: Map<string, Library>) => ReactNode };

const COLUMNS: Column[] = [
  { by: "id", label: "id", className: "ledger nowrap", cell: (r) => r.id },
  { by: "title", label: "title", className: "wide-cell", cell: (r) => <a href={href(["texts", r.id])} title={name(r)} className="cut">{name(r)}</a> },
  { by: "status", label: "status", className: "nowrap", cell: (r) => <StatusBadge status={r.status} /> },
  { by: "tags", label: "tags", cell: (r) => <Links values={r.tags} part="tags" /> },
  { by: "licenses", label: "licenses", className: "ledger", cell: (r) => <Links values={r.licenses} part="licenses" /> },
  {
    by: "libraries", label: "libraries", cell: (r, byId) => (
      <span className="cut" title={r.libraries.map((id) => byId.get(id)?.name ?? id).join(", ")}>
        {r.libraries.map((id, i) => <span key={id}>{i > 0 && ", "}<LibraryLink library={byId.get(id) ?? { id }} /></span>)}
      </span>
    ),
  },
  { by: "length", label: "characters", className: "number ledger", cell: (r) => r.length.toLocaleString("en") },
];

// Values on one line, each linked to its view (#/tags/TAG, #/licenses/ID), cut when too long.
function Links({ values, part }: { values: string[]; part: string }) {
  return (
    <span className="cut" title={values.join(", ")}>
      {values.map((v, i) => <span key={v}>{i > 0 && ", "}<a href={href([part, v])}>{v}</a></span>)}
    </span>
  );
}

// The texts a filter matches, a page at a time, sortable; empty: what to say when none does.
export function MatchedTexts({ filter, label, empty }: { filter: TextFilter; label: string; empty: string }) {
  const { libraries } = useLibraries();
  const { answer, pending, offset, setOffset, limit, setLimit, order, setOrder } = useSearch(filter, 25);
  if (!answer) return <Skeleton lines={4} label={`Reading ${label.toLowerCase()}`} />;
  if (!answer.ok) return <Problems problems={answer.problems} />;
  if (answer.data.total === 0) return <p className="small">{empty}</p>;
  return (
    <>
      <TextTable rows={answer.data.rows} libraries={libraries} busy={pending} order={order} onOrder={setOrder} label={label} />
      <Pager offset={offset} limit={limit} total={answer.data.total} onOffset={setOffset} onLimit={setLimit} label={`Pages of ${label.toLowerCase()}`} />
    </>
  );
}

export function TextTable({ rows, libraries, order, onOrder, ticked, onTick, busy = false, label, otherLibraries }: {
  rows: Row[]; libraries: Library[]; order?: Order; onOrder?: (o: Order) => void; ticked?: Set<string>;
  onTick?: (ids: string[], on: boolean) => void; busy?: boolean; label: string; otherLibraries?: string;
}) {
  const shown = (r: Row) => (otherLibraries ? { ...r, libraries: r.libraries.filter((id) => id !== otherLibraries) } : r);
  const columns = otherLibraries && rows.every((r) => shown(r).libraries.length === 0) ? COLUMNS.filter((c) => c.by !== "libraries") : COLUMNS;
  const heading = (c: Column) => (otherLibraries && c.by === "libraries" ? "other libraries" : c.label);
  const byId = new Map(libraries.map((l) => [l.id, l]));
  const allTicked = rows.length > 0 && rows.every((r) => ticked?.has(r.id));
  const sort = (by: Order["by"]) => onOrder?.({ by, descending: order?.by === by ? !order.descending : false });
  return (
    <div className="table-scroll">
      <table className="ledger-table" aria-busy={busy} aria-label={label}>
        <thead>
          <tr>
            {ticked && onTick && (
              <th scope="col" className="tick-cell">
                <input type="checkbox" className="wic-check-box wic-focus" aria-label="Tick every row of this page" checked={allTicked}
                  onChange={(e) => onTick(rows.map((r) => r.id), e.target.checked)} />
              </th>
            )}
            {columns.map((c) => (
              <th key={c.label} scope="col" className={c.className?.includes("number") ? "number" : undefined}
                aria-sort={order?.by === c.by ? (order.descending ? "descending" : "ascending") : undefined}>
                {onOrder ? (
                  <button type="button" className="sort wic-focus" onClick={() => sort(c.by)}>
                    {heading(c)}<span aria-hidden="true">{order?.by === c.by ? (order.descending ? " ↓" : " ↑") : ""}</span>
                  </button>
                ) : heading(c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className={ticked?.has(r.id) ? "ticked" : undefined}>
              {ticked && onTick && (
                <td className="tick-cell">
                  <input type="checkbox" className="wic-check-box wic-focus" aria-label={`Tick ${r.id}`} checked={ticked.has(r.id)}
                    onChange={(e) => onTick([r.id], e.target.checked)} />
                </td>
              )}
              {columns.map((c) => <td key={c.label} className={c.className}>{c.cell(shown(r), byId)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
