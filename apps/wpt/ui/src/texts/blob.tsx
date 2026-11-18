// The blob: the open collection read as one continuous text, in the
// playlist's order, each text after a thin rule that names it. It is read a
// page at a time through POST /api/texts/passages, whose pages are bounded
// by a count of texts and by characters, the next page asked only when the
// end of the last one scrolls into view, so the collection is never read
// whole. A text longer than a page shows its start and says how much is
// left. It is read only: a click on a passage, or on the name over it,
// opens that text in the editor, the playlist's page starting at it. A click
// on a fold inside a transcript or on a link opens the fold or follows the
// link instead. A page the server cut at its reading bound can hold no text;
// the next is then read on a click. A conversation read whole shows as a transcript
// (transcript.tsx); one cut by the page shows its start as it is stored.

import { Button, Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useRef, useState, type MouseEvent } from "react";
import { post, type Answer } from "../api";
import { body, type Order, type Place, type TextFilter } from "../filter";
import { plural, Problems, textName } from "../parts";
import { codePoints } from "../reader";
import { isConversation, Transcript } from "../transcript";

// A passage's content as a conversation, when it is one read whole, else null.
function conversationOf(r: Passage): unknown {
  if (r.kind === null || codePoints(r.passage) < r.length) return null;
  try {
    const content = JSON.parse(r.passage);
    return isConversation(content) ? content : null;
  } catch { return null; }
}

type Passage = { id: string; kind: string | null; title: string | null; length: number; passage: string };
type Page = { rows: Passage[]; next: Place | null };

const TEXTS = 50; // texts a page holds at most
const CHARACTERS = 200_000; // characters a page holds at most

// where the next page starts: the first, after a place, or nowhere (the end)
type Next = { after: Place | null } | null;
const FIRST: Next = { after: null };

// onOpen: the text clicked.
export function Blob({ filter, order, onOpen }: { filter: TextFilter; order: Order; onOpen: (id: string) => void }) {
  const [pages, setPages] = useState<{ after: Place | null; rows: Passage[] }[]>([]);
  const [next, setNext] = useState<Next>(FIRST);
  const [reading, setReading] = useState(false);
  const [failed, setFailed] = useState<Answer<Page> | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const key = JSON.stringify([filter, order]);
  const asked = useRef(key);
  useEffect(() => { asked.current = key; setPages([]); setNext(FIRST); setFailed(null); setReading(false); }, [key]);
  const more = async () => {
    if (next === null || reading) return;
    setReading(true);
    const from = next.after;
    const a = await post<Page>("/api/texts/passages", { filter: body(filter), ...(from ? { after: from } : {}), limit: TEXTS, characters: CHARACTERS, order: order.by, descending: order.descending });
    if (asked.current !== key) return; // the filter or the order changed while it was read
    setReading(false);
    if (!a.ok) { setFailed(a); return; }
    setPages((p) => [...p, { after: from, rows: a.data.rows }]);
    setNext(a.data.next ? { after: a.data.next } : null);
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
    <div className="blob" aria-label="The open texts, one after another">
      {pages.length === 0 && next !== null && !failed && <Skeleton lines={8} label="Reading the first texts" />}
      {next === null && pages.length > 0 && pages.every((page) => page.rows.length === 0) && <p>No text matches the filter.</p>}
      {pages.flatMap((page) => page.rows.map((r) => (
        <article key={r.id} className="blob-text" aria-label={r.title ?? r.id}>
          <h2 className="blob-rule small">
            <button type="button" className="blob-name wic-focus" onClick={() => onOpen(r.id)} title="Open it in the editor">
              <span className="cut">{textName(r)}</span> <span className="ledger aside">{r.id}</span>
            </button>
          </h2>
          {(() => {
            const conversation = conversationOf(r);
            return isConversation(conversation)
              ? <div className="blob-transcript" onClick={opened(r.id)}><Transcript content={conversation} /></div>
              : <pre className="blob-passage" onClick={opened(r.id)}>{r.passage}</pre>;
          })()}
          {codePoints(r.passage) < r.length && <p className="aside small">{plural(r.length - codePoints(r.passage), "more character")} in this text: open it to read them.</p>}
        </article>
      )))}
      {failed && !failed.ok && <><Problems problems={failed.problems} /><Button variant="quiet" onClick={() => { setFailed(null); setReading(false); }}>Try again</Button></>}
      <div ref={end} className="blob-end small aside" role="status">
        {next === null ? pages.some((page) => page.rows.length > 0) && "The end of the open texts."
          : reading ? "Reading the next texts"
          : stalled && <>No match among the texts read so far. <Button variant="quiet" onClick={more}>Read on</Button></>}
      </div>
    </div>
  );
}
