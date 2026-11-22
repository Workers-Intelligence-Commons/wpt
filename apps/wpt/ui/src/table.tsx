// The texts of one page as a table of one-line rows, each named by its
// title, its id in the title's hover text: a long value is cut and shown
// whole on hover. A column header that names an order sorts by
// it, and a second click turns the order around; a box ticks each row when
// ticked is given. A kind column links each kind to the texts of that kind.
// MatchedTexts is such a table, paged, of the texts a filter matches, for a
// view that shows the texts of one thing (a license, a tag, a decision).

import { Skeleton } from "@workers-intelligence-commons/components";
import { type ReactNode } from "react";
import { MatchedCount, useCounts } from "./counts";
import { useSearch, type Order, type Row, type TextFilter } from "./filter";
import { PLAIN, PlacePager, Problems, StatusBadge, textName } from "./parts";
import { href } from "./route";

const name = textName;

type Column = { by: Order["by"]; label: string; className?: string; cell: (r: Row) => ReactNode };

const COLUMNS: Column[] = [
  { by: "title", label: "title", className: "wide-cell", cell: (r) => <a href={href(["texts", r.id])} title={`${name(r)} (${r.id})`} className="cut">{name(r)}</a> },
  { by: "kind", label: "kind", className: "nowrap", cell: (r) => { const kind = r.kind ?? PLAIN; return <a href={href(["texts"], { kind })} title={`Every text of kind ${kind}`}>{kind}</a>; } },
  { by: "status", label: "status", className: "nowrap", cell: (r) => <StatusBadge status={r.status} /> },
  { by: "tags", label: "tags", cell: (r) => <Links values={r.tags} part="tags" /> },
  { by: "licenses", label: "licenses", className: "ledger", cell: (r) => <Links values={r.licenses} part="licenses" /> },
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

// The texts a filter matches, a page at a time, sortable, with their count
// as it grows; empty: what to say when none does.
export function MatchedTexts({ filter, label, empty }: { filter: TextFilter; label: string; empty: string }) {
  const { answer, pending, next, previous, first, limit, setLimit, order, setOrder } = useSearch(filter, 25);
  const counting = useCounts(filter);
  if (!answer) return <Skeleton lines={4} label={`Reading ${label.toLowerCase()}`} />;
  if (!answer.ok) return <Problems problems={answer.problems} />;
  if (answer.data.rows.length === 0 && !previous) return <p className="small">{empty}</p>;
  return (
    <>
      <TextTable rows={answer.data.rows} busy={pending} order={order} onOrder={setOrder} label={label} />
      <PlacePager onFirst={first} onPrevious={previous} onNext={next} limit={limit} onLimit={setLimit} label={`Pages of ${label.toLowerCase()}`}
        counted={<MatchedCount counting={counting} />} />
    </>
  );
}

export function TextTable({ rows, order, onOrder, ticked, onTick, busy = false, label }: {
  rows: Row[]; order?: Order; onOrder?: (o: Order) => void; ticked?: Set<string>;
  onTick?: (ids: string[], on: boolean) => void; busy?: boolean; label: string;
}) {
  const columns = COLUMNS;
  const heading = (c: Column) => c.label;
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
                  <input type="checkbox" className="wic-check-box wic-focus" aria-label={`Tick ${name(r)}`} checked={ticked.has(r.id)}
                    onChange={(e) => onTick([r.id], e.target.checked)} />
                </td>
              )}
              {columns.map((c) => <td key={c.label} className={c.className}>{c.cell(r)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
