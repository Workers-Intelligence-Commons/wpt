// #/snapshots/ID/strip: a whole snapshot read as one strip (texts/blob.tsx),
// its longest texts first, or its shortest with ?order=length (one switch in
// the head shows which and changes it), three lines of each by default, each at its
// revision as of the snapshot, with its size in characters and, when it is
// no longer admitted, its status now in the margin; a click on a text opens
// it as it is now in a pane beside the strip (TextPane: the Texts page of
// that one text, with its editor and its decisions), so the strip stays
// where it was read to, and Close or Escape puts the pane away. The
// order by length is made once for each snapshot, by a job of the server's
// read lane (POST /api/snapshots/ID/strip queues it on the first read and
// answers its state): until it is made, the page says the request was
// taken and how many of the snapshot's texts the job has read, reading the
// state again every POLL milliseconds; the strip shows as soon as it ends.
// A job that ended without the order says why, with "Try again", which
// queues a new one (again: true); the page never queues one by itself.

import { Button, Skeleton } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useMemo, useState } from "react";
import { get, part, post, type Answer, type Job } from "../api";
import { PageHead, plural, Problems, Switch } from "../parts";
import { snapshotHref, type SnapshotShown } from "../pipeline";
import { href } from "../route";
import { Blob, CHARACTERS, EACH, FIRST, TEXTS, type StripPage, type StripReader } from "../texts/blob";

const POLL = 500; // milliseconds between two readings of the order's job

export type StripAnswer = ({ ready: true } & StripPage<number> & { total: number })
  | { ready: false; job: Job<unknown> | null; texts: number };

// A page of the snapshot's strip: rows (with passages unless passages is false) after a position.
export const readStrip = (id: string, after: number | null, descending: boolean, limit = TEXTS, passages = true, again = false) =>
  post<StripAnswer>(`/api/snapshots/${part(id)}/strip`, { ...(after !== null ? { after } : {}), limit, characters: CHARACTERS, descending, passages, each: EACH, ...(again ? { again } : {}) });

// The order of a snapshot by length, made: null while it is read, its first page (of limit texts, the longest first
// when descending, with their passages or not) once ready, or its refusal; while its job runs, the job's last state;
// once a job ended without it, that job, until again() queues a new one.
export function useSnapshotOrder(id: string | null, descending = true, limit = FIRST, passages = true) {
  const [answer, setAnswer] = useState<Answer<StripAnswer> | null>(null);
  const [tries, setTries] = useState(0); // raised by again()
  const again = useCallback(() => setTries((n) => n + 1), []);
  useEffect(() => {
    setAnswer(null);
    if (!id) return;
    let live = true, timer: number | undefined, first = tries > 0;
    const look = async () => {
      const a = await readStrip(id, null, descending, limit, passages, first);
      first = false;
      if (!live) return;
      setAnswer(a);
      if (a.ok && !a.data.ready && (!a.data.job || a.data.job.state === "queued" || a.data.job.state === "running"))
        timer = window.setTimeout(look, POLL);
    };
    look();
    return () => { live = false; window.clearTimeout(timer); };
  }, [id, descending, limit, passages, tries]);
  return { answer, again };
}

// What the order's job says while it runs: queued, or how many texts it read of how many; once it ended without the
// order, why, with "Try again" (onAgain).
export function OrderProgress({ answer, onAgain }: { answer: Answer<StripAnswer> | null; onAgain: () => void }) {
  if (!answer) return <p className="small aside" role="status">Asking the server for the snapshot's order by size…</p>;
  if (!answer.ok) return <Problems problems={answer.problems} />;
  if (answer.data.ready) return null;
  const { job, texts } = answer.data;
  if (!job) return <p className="small aside" role="status">Starting the order by size…</p>;
  if (job.state === "refused" || job.state === "failed" || job.state === "cancelled")
    return (
      <div className="order-progress" role="alert">
        <Problems problems={[`Ordering the snapshot by size ${job.state}: ${job.problem ?? "no reason given"}`]} />
        <p className="small"><Button variant="quiet" onClick={onAgain}>Try again</Button> <a href={href(["jobs"])}>The jobs</a></p>
      </div>
    );
  const read = typeof job.progress?.texts === "number" ? job.progress.texts : 0;
  const of = typeof job.progress?.of === "number" ? job.progress.of : texts;
  const share = of ? Math.min(1, read / of) : 0;
  return (
    <div className="order-progress" role="status">
      <p className="small">{job.state === "queued" ? "Queued: the snapshot is ordered by size once, by a job of the read lane, which waits for the job before it."
        : <>Ordering the snapshot by size, once for good: {read.toLocaleString("en")} of {plural(of, "text")} read ({Math.floor(share * 100)}%).</>}</p>
      <progress max={1} value={share} aria-label="texts read" />
      <p className="small aside">Every other page works meanwhile. <a href={href(["jobs"])}>The jobs</a></p>
    </div>
  );
}

export function SnapshotStrip({ id, query }: { id: string; query: URLSearchParams }) {
  const descending = query.get("order") !== "length";
  const [shown, setShown] = useState<Answer<SnapshotShown> | null>(null);
  useEffect(() => { get<SnapshotShown>(`/api/snapshots/${part(id)}`).then(setShown); }, [id]);
  const { answer: order, again } = useSnapshotOrder(id, descending); // its first page, once the order is made
  const [open, setOpen] = useState<string | null>(null); // the text shown in the pane
  const ready = order?.ok && order.data.ready ? order.data : null;
  const read: StripReader<number> = useMemo(() => async (after) => {
    const a = await readStrip(id, after, descending);
    if (!a.ok) return a;
    if (!a.data.ready) return { ok: false, status: 409, problems: ["the snapshot's order by size is being made again"] };
    return { ok: true, data: { rows: a.data.rows, next: a.data.next } };
  }, [id, descending]);
  const name = shown?.ok ? shown.data.name : id;
  return (
    <div className="strip-page">
      <PageHead title={<>{name}, read whole</>}
        meta={<><a href={snapshotHref({ id })}>{name}</a>{shown?.ok && <span>{plural(shown.data.texts, "text")}</span>}
          <Switch label="Order" options={[
            { label: "Longest first", href: href(["snapshots", id, "strip"]), on: descending },
            { label: "Shortest first", href: href(["snapshots", id, "strip"], { order: "length" }), on: !descending }]} />
          <a href={href(["sizes"], { snapshot: id })}>size map</a></>}
        chain={<>Every text of the snapshot as it holds it, one after another, sorted by its length in characters; its size is in the margin, with a Refuse button. A click on a text opens it as it is now, to edit, admit or refuse it. Refusing it changes no snapshot: the next snapshot leaves it out.</>} />
      {!ready && <OrderProgress answer={order} onAgain={again} />}
      {!order && <Skeleton lines={6} label="Reading the snapshot" />}
      {ready && <Blob readKey={`${id} ${descending}`} read={read} first={ready} label={`The snapshot ${name}, one text after another`}
        ended="The end of the snapshot." onOpen={setOpen} short />}
      {open && <TextPane id={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

// One text as it is now, in a pane over the right of the page: the Texts page of that text alone, in a frame, so it
// can be edited, admitted or refused there as anywhere, and the page under it keeps its place.
export function TextPane({ id, onClose }: { id: string; onClose: () => void }) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);
  return (
    <aside className="text-pane" aria-label={`The text ${id}`}>
      <div className="text-pane-head small">
        <span className="ledger">{id}</span>
        <a href={href(["texts", id], { id })} target="_blank" rel="noreferrer">Open in a new tab</a>
        <Button variant="quiet" onClick={onClose}>Close<span className="key-hint">(Esc)</span></Button>
      </div>
      <iframe key={id} title={`The text ${id}`} src={`${window.location.pathname}${href(["texts", id], { id })}`} />
    </aside>
  );
}
