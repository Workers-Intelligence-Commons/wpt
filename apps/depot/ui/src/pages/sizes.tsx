// #/sizes: the size map, which tags and which texts weigh the most, in
// characters (a size in tokens needs a vocabulary). Of the depot's texts of
// one status now (?status=admitted, the default; GET /api/counts/STATUS,
// read from the depot's counts at once), or of a snapshot
// (?snapshot=ID; its summary, GET /api/snapshots/ID). Each part, tag, kind
// or license, is a bar as long as its share of the largest, with its texts
// and characters; a click on a tag or a kind opens those texts in the Texts
// page as one strip, the longest first (#/texts?...&view=blob&order=-length):
// their state now, the snapshot's own texts being read whole on its strip.
// Beside them, the biggest texts: the first page of the texts of that
// status by length (POST /api/texts/search, from the index of lengths), or
// of the snapshot's strip without passages, its order made first by a job
// that the page follows; each with a bar as long as its share of the biggest
// and a Refuse button (refuse.tsx: at once, then Undo); a click on one opens
// it in a pane to edit, admit or refuse it (strip.tsx). The title names what
// the map is of, a status now or a snapshot; the statuses are one switch; the
// navigation marks Snapshots for a snapshot's map, Texts for the others.

import { Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState, type CSSProperties } from "react";
import { get, part, type Answer } from "../api";
import { EVERY, search, toQuery, type Found } from "../filter";
import { PageHead, plural, Problems, Section, STATUSES, Switch, textName, type Status } from "../parts";
import { snapshotHref, type Sized, type SnapshotShown } from "../pipeline";
import { href } from "../route";
import { bytes } from "../sizes";
import { characters } from "../texts/blob";
import { RefuseRow } from "../texts/refuse";
import type { Page } from "./index";
import { OrderProgress, TextPane, useSnapshotOrder } from "./strip";

type Held = { kinds: Record<string, Sized>; tags: Record<string, Sized>; licenses: Record<string, Sized> };
type Biggest = { id: string; title: string | null; kind: string | null; length: number; status?: string | null };
const BIGGEST = 30; // the biggest texts listed
const PARTS = 40; // the parts of each kind of part listed, the largest first

// The parts of one kind (tags, kinds, licenses) as bars of their characters, the largest first; open: the address
// of a part's texts as a strip, or none.
function Bars({ parts, label, open }: { parts: Record<string, Sized>; label: string; open?: (name: string) => string }) {
  const rows = Object.entries(parts).sort((a, b) => b[1][1] - a[1][1] || a[0].localeCompare(b[0]));
  const most = rows[0]?.[1][1] || 1;
  if (!rows.length) return <p className="small aside">None.</p>;
  return (
    <>
      <ul className="plain size-bars small" aria-label={label}>
        {rows.slice(0, PARTS).map(([name, [texts, chars, size]]) => (
          <li key={name} title={`${name}: ${plural(texts, "text")}, ${chars.toLocaleString("en")} characters, ${bytes(size)}`}>
            {open ? <a href={open(name)} className="cut" title={`Read the texts of ${name} as one strip, the longest first`}>{name}</a> : <span className="cut">{name}</span>}
            <span className="number ledger">{characters(chars)}</span>
            <span className="number ledger aside">{plural(texts, "text")}</span>
            <span className="size-bar" aria-hidden="true" style={{ "--share": `${Math.max(0.5, (chars / most) * 100)}%` } as CSSProperties} />
          </li>
        ))}
      </ul>
      {rows.length > PARTS && <p className="small aside">and {plural(rows.length - PARTS, "more")}, smaller.</p>}
    </>
  );
}

// refused: whether every text listed is refused (no Refuse button then)
function BiggestList({ rows, onOpen, more, refused = false }: { rows: Biggest[]; onOpen: (id: string) => void; more: string; refused?: boolean }) {
  if (!rows.length) return <p className="small aside">No text.</p>;
  const most = rows[0].length || 1;
  return (
    <>
      <ol className="plain biggest small">
        {rows.map((r) => (
          <li key={r.id} title={`${r.id}: ${r.length.toLocaleString("en")} characters`}>
            <button type="button" className="link-button cut" onClick={() => onOpen(r.id)} title={`Open ${r.id} to edit, admit or refuse it`}>{textName(r)}</button>
            <span className="number ledger">{characters(r.length)}</span>
            <span className="size-bar" aria-hidden="true" style={{ "--share": `${Math.max(0.5, (r.length / most) * 100)}%` } as CSSProperties} />
            {refused ? <span /> : <RefuseRow id={r.id} title={textName(r)} refused={r.status === "refused"} />}
            {r.status && r.status !== "admitted" && <span className="small blob-status">{r.status} now</span>}
          </li>
        ))}
      </ol>
      <p className="small"><a href={more}>Read them all as one strip, the longest first</a></p>
    </>
  );
}

function SizeMap({ query }: { query: URLSearchParams }) {
  const snapshot = query.get("snapshot");
  const status = (STATUSES.includes(query.get("status") as Status) ? query.get("status") : "admitted") as Status;
  const [held, setHeld] = useState<Answer<Held> | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [biggest, setBiggest] = useState<Answer<Found> | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const { answer: order, again } = useSnapshotOrder(snapshot, true, BIGGEST, false); // the biggest texts, once the order is made
  useEffect(() => {
    setHeld(null); setBiggest(null); setName(null);
    if (snapshot) {
      get<SnapshotShown>(`/api/snapshots/${part(snapshot)}`).then((a) => {
        setHeld(a.ok ? { ok: true, data: a.data.summary } : a);
        if (a.ok) setName(a.data.name);
      });
    } else {
      get<Held>(`/api/counts/${part(status)}`).then(setHeld);
      search({ ...EVERY, statuses: [status] }, {}, BIGGEST, { by: "length", descending: true }).then(setBiggest);
    }
  }, [snapshot, status]);
  // a part's texts now, as one strip, the longest first
  const strip = (q: Record<string, string | string[] | null>) => href(["texts"], { ...toQuery({ ...EVERY, statuses: [status] }), ...q, view: "blob", order: "-length" });
  const allTexts = snapshot ? href(["snapshots", snapshot, "strip"]) : strip({});
  const sum = held?.ok ? Object.values(held.data.kinds).reduce((t, s) => [t[0] + s[0], t[1] + s[1], t[2] + s[2]], [0, 0, 0]) : null;
  return (
    <>
      <PageHead title={snapshot ? <>Size map of the snapshot {name ?? snapshot}</> : <>Size map of the texts {status} now</>}
        meta={<>{snapshot ? <><a href={snapshotHref({ id: snapshot })}>{name ?? snapshot}</a><a href={href(["sizes"], { status: "admitted" })}>the depot's texts now</a></>
          : <Switch label="Status" options={STATUSES.map((s) => ({ label: s, href: href(["sizes"], { status: s }), on: s === status, title: `The texts ${s} now` }))} />}
          {sum && <span>{plural(sum[0], "text")}, {characters(sum[1])} characters, {bytes(sum[2])}</span>}</>}
        chain={<>Which tags and which texts take the most room, in characters, {snapshot ? "in the snapshot as it holds them" : `among the ${status} texts now`}. A tag or a kind opens its texts {snapshot ? "as they are now " : ""}as one strip, the longest first; a text opens beside the map, to edit, admit or refuse it.</>} />
      {!held && <Skeleton lines={6} label="Reading the sizes" />}
      {held && !held.ok && <Problems problems={held.problems} />}
      {held?.ok && (
        <div className="board size-map">
          <Section title="Biggest texts" help="Sorted by length in characters.">
            {snapshot ? (order?.ok && order.data.ready
              ? <BiggestList rows={order.data.rows as Biggest[]} onOpen={setOpen} more={allTexts} />
              : <OrderProgress answer={order} onAgain={again} />)
              : biggest === null ? <Skeleton lines={6} label="Reading the biggest texts" />
              : !biggest.ok ? <Problems problems={biggest.problems} />
              : <BiggestList rows={biggest.data.rows} onOpen={setOpen} more={allTexts} refused={status === "refused"} />}
          </Section>
          <Section title="By tag" help="A text with several tags counts once for each.">
            <Bars parts={held.data.tags} label="characters by tag" open={(t) => strip({ tag: t })} />
          </Section>
          <Section title="By kind">
            <Bars parts={held.data.kinds} label="characters by kind" open={(k) => strip({ kind: k })} />
          </Section>
          <Section title="By license" help="A text under several licenses counts once for each.">
            <Bars parts={held.data.licenses} label="characters by license" open={(l) => strip({ license: l })} />
          </Section>
        </div>
      )}
      {open && <TextPane id={open} onClose={() => setOpen(null)} />}
    </>
  );
}

export const sizes: Page = { path: "sizes", title: "Size map", group: null, under: (query) => (query.get("snapshot") ? "snapshots" : "texts"),
  Component: ({ query }) => <SizeMap query={query} /> };
