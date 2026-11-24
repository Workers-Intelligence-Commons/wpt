// The blob: many texts read as one continuous strip, each text after a thin
// rule that names it, its size in characters in the margin at its left (a
// size in tokens needs a vocabulary, which a strip of the depot's texts has
// not chosen). The Texts page reads what is open, in the playlist's order,
// through POST /api/texts/passages (textsReader); a snapshot's strip reads
// the whole snapshot, its longest texts first, through POST
// /api/snapshots/ID/strip (pages/strip.tsx). Either way the texts come a
// page at a time, bounded by a count of texts (FIRST for the first page, so
// the first screen comes sooner, TEXTS after) and by characters, the next
// page asked only when the end of the last one scrolls into view, so the
// whole is never read at once. A text longer than a page, or, read by size,
// than EACH characters, shows its start and says how much is left. It is read only: a click on a passage, or on the
// name over it, opens that text in the editor of the Texts page, where it can
// be edited, refused or admitted. A click on a fold inside a transcript or on
// a link opens the fold or follows the link instead. A page the server cut at
// its reading bound can hold no text; the next is then read on a click. A
// conversation read whole shows as a transcript (transcript.tsx); one cut by
// the page shows its start as it is stored. A text refused or put back under
// review since the page was read is marked so in the margin. Each text has a
// Refuse button in its margin (refuse.tsx: at once, then Undo), so a big junk
// text found in the strip is refused in one click. The excerpts are short
// (three lines) or long (the start the page read): short by default when the
// texts are read by size, where the point is to see many at once, long
// otherwise; the switch on top changes it and this browser keeps the choice.

import { Button, Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useRef, useState, type MouseEvent } from "react";
import { post, type Answer } from "../api";
import { body, type Order, type Place, type TextFilter } from "../filter";
import { plural, PLAIN, Problems, textName } from "../parts";
import { codePoints } from "../reader";
import { isConversation, Transcript } from "../transcript";
import { RefuseRow } from "./refuse";

// A passage's content as a conversation, when it is one read whole, else null.
function conversationOf(r: Passage): unknown {
  if (r.kind === PLAIN || codePoints(r.passage) < r.length) return null;
  try {
    const content = JSON.parse(r.passage);
    return isConversation(content) ? content : null;
  } catch { return null; }
}

// status: the text's status now, when the reader gives it (a snapshot's strip)
export type Passage = { id: string; kind: string; title: string | null; length: number; passage: string; status?: string | null };
// next: where the next page starts (opaque to the blob), null after the last
export type StripPage<P> = { rows: Passage[]; next: P | null };
// reads the page after a place (null: the first)
export type StripReader<P> = (after: P | null) => Promise<Answer<StripPage<P>>>;

export const TEXTS = 50; // texts a page holds at most
export const FIRST = 20; // texts the first page holds at most: more than a screen shows, and read in a third of the time
export const CHARACTERS = 200_000; // characters a page holds at most
// characters each text shows at most when the texts are read by size, so the longest do not fill the strip alone
export const EACH = 1_500;

// A size in characters, short, for the margin: 950, 12.3k, 3.4M.
export const characters = (n: number) => n < 1000 ? `${n}` : n < 999_500 ? `${(n / 1000).toFixed(n < 9_950 ? 1 : 0)}k` : `${(n / 1e6).toFixed(1)}M`;

// What the Texts page has open, in the order asked; read by length, the start of each text alone (EACH).
export const textsReader = (filter: TextFilter, order: Order): StripReader<Place> => (after) =>
  post<StripPage<Place>>("/api/texts/passages", { filter: body(filter), ...(after ? { after } : {}), limit: after ? TEXTS : FIRST, characters: CHARACTERS,
    order: order.by, descending: order.descending, ...(order.by === "length" ? { each: EACH } : {}) });

// where the next page starts: the first, after a place, or nowhere (the end)
type Next<P> = { after: P | null } | null;

// Whether the excerpts are short, as this browser last chose, or null when it never chose.
const SHORT = "wpt-depot-blob-short";
function readShort(): boolean | null {
  try { const v = window.localStorage.getItem(SHORT); return v === null ? null : v === "yes"; } catch { return null; }
}
function keepShort(on: boolean) {
  try { window.localStorage.setItem(SHORT, on ? "yes" : "no"); } catch { /* not kept: the default comes back next time */ }
}

// readKey: what is read, in words a change of which starts the strip over; first: its first page when the caller
// read it already; onOpen: the text clicked; ended: what the end of the strip says; short: whether the excerpts are
// short unless this browser chose; refused: whether every text read is refused already (no Refuse button then).
export function Blob<P>({ readKey, read, first, onOpen, ended = "The end of the open texts.", label = "The open texts, one after another", short: shortFirst = false, refused = false }: {
  readKey: string; read: StripReader<P>; first?: StripPage<P>; onOpen: (id: string) => void; ended?: string; label?: string; short?: boolean; refused?: boolean;
}) {
  const [chosen, setChosen] = useState(readShort);
  const short = chosen ?? shortFirst;
  const choose = (on: boolean) => { setChosen(on); keepShort(on); };
  const start = (): [{ rows: Passage[] }[], Next<P>] => first
    ? [[{ rows: first.rows }], first.next !== null ? { after: first.next } : null] : [[], { after: null }];
  const [pages, setPages] = useState<{ rows: Passage[] }[]>(() => start()[0]);
  const [next, setNext] = useState<Next<P>>(() => start()[1]);
  const [reading, setReading] = useState(false);
  const [failed, setFailed] = useState<Answer<StripPage<P>> | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const asked = useRef(readKey);
  const shown = useRef(readKey); // what the pages hold now
  useEffect(() => {
    asked.current = readKey;
    if (shown.current === readKey) return; // the first render already holds it
    shown.current = readKey;
    const [p, n] = start();
    setPages(p); setNext(n); setFailed(null); setReading(false);
  }, [readKey]);
  const more = async () => {
    if (next === null || reading) return;
    setReading(true);
    const a = await read(next.after);
    if (asked.current !== readKey) return; // what is read changed meanwhile
    setReading(false);
    if (!a.ok) { setFailed(a); return; }
    setPages((p) => [...p, { rows: a.data.rows }]);
    setNext(a.data.next !== null ? { after: a.data.next } : null);
  };
  // a page cut at the server's reading bound can hold no text: the next one is then read on a click, never
  // one after another on their own while nothing shows
  const stalled = pages.length > 0 && pages[pages.length - 1].rows.length === 0 && next !== null;
  useEffect(() => {
    const sentinel = end.current;
    if (!sentinel || next === null || failed || stalled) return;
    const watcher = new IntersectionObserver((seen) => { if (seen.some((s) => s.isIntersecting)) more(); }, { rootMargin: "800px" });
    watcher.observe(sentinel);
    return () => watcher.disconnect();
  });
  const opened = (id: string) => (e: MouseEvent) => {
    if (window.getSelection()?.isCollapsed === false) return; // a passage being selected, not a click
    if (e.target instanceof Element && e.target.closest("summary, details, a, button, input")) return;
    onOpen(id);
  };
  return (
    <div className={`blob${short ? " short" : ""}`} aria-label={label}>
      <div className="blob-tools small" role="group" aria-label="How much of each text shows">
        <span className="aside">Excerpts:</span>
        <span className="switch">
          <button type="button" className={`choice wic-focus${short ? " on" : ""}`} aria-pressed={short} onClick={() => choose(true)} title="Three lines of each text, so many texts fit on a screen">3 lines</button>
          <button type="button" className={`choice wic-focus${!short ? " on" : ""}`} aria-pressed={!short} onClick={() => choose(false)} title="The start of each text as read">long</button>
        </span>
      </div>
      {pages.length === 0 && next !== null && !failed && <Skeleton lines={8} label="Reading the first texts" />}
      {next === null && pages.length > 0 && pages.every((page) => page.rows.length === 0) && <p>No text matches the filter.</p>}
      {pages.flatMap((page) => page.rows.map((r) => (
        <article key={r.id} className="blob-text" aria-label={r.title ?? r.id}>
          <div className="blob-margin ledger" title={`${r.length.toLocaleString("en")} characters`}>
            <span className="blob-size">{characters(r.length)} <span className="aside small">chars</span></span>
            {r.status && r.status !== "admitted" && <span className="blob-status small">{r.status} now</span>}
            {!refused && <RefuseRow key={r.id} id={r.id} title={textName(r)} refused={r.status === "refused"} />}
          </div>
          <div className="blob-body">
            <h2 className="blob-rule small">
              <button type="button" className="blob-name wic-focus" onClick={() => onOpen(r.id)} title="Open it in the editor, to edit, admit or refuse it">
                <span className="cut">{textName(r)}</span> <span className="ledger aside">{r.id}</span>
              </button>
            </h2>
            {(() => {
              const conversation = conversationOf(r);
              return isConversation(conversation)
                ? <div className="blob-transcript" onClick={opened(r.id)}><Transcript content={conversation} /></div>
                : <pre className="blob-passage" onClick={opened(r.id)}>{r.passage}</pre>;
            })()}
            {!short && codePoints(r.passage) < r.length && <p className="aside small">{plural(r.length - codePoints(r.passage), "more character")} in this text: open it to read them.</p>}
          </div>
        </article>
      )))}
      {failed && !failed.ok && <><Problems problems={failed.problems} /><Button variant="quiet" onClick={() => { setFailed(null); setReading(false); }}>Try again</Button></>}
      <div ref={end} className="blob-end small aside" role="status">
        {next === null ? pages.some((page) => page.rows.length > 0) && ended
          : reading ? "Reading the next texts"
          : stalled && <>No match among the texts read so far. <Button variant="quiet" onClick={more}>Read on</Button></>}
      </div>
    </div>
  );
}
