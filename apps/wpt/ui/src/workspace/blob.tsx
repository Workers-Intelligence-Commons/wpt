// The blob: the open collection read as one continuous text, in the
// playlist's order, each text after a thin rule that names it. It is read a
// page at a time through POST /api/texts/passages, whose pages are bounded
// by a count of texts and by characters, the next page asked only when the
// end of the last one scrolls into view, so the collection is never read
// whole. A text longer than a page shows its start and says how much is
// left. It is read only: a click on a passage, or on the name over it,
// opens that text in the editor, at its place in the playlist.

import { Button, Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useRef, useState } from "react";
import { post, type Answer } from "../api";
import { type Order, type TextFilter } from "../filter";
import { plural, Problems } from "../parts";
import { codePoints } from "../reader";

type Passage = { id: string; kind: string | null; title: string | null; length: number; passage: string };
type Page = { rows: Passage[]; next: number | null };

const TEXTS = 50; // texts a page holds at most
const CHARACTERS = 200_000; // characters a page holds at most

const body = (f: TextFilter) => ({ ...f, where: f.where.trim() === "" ? null : f.where });

// onOpen: the text clicked, with its place in the order (from 0).
export function Blob({ filter, order, onOpen }: { filter: TextFilter; order: Order; onOpen: (id: string, place: number) => void }) {
  const [pages, setPages] = useState<{ offset: number; rows: Passage[] }[]>([]);
  const [next, setNext] = useState<number | null>(0);
  const [reading, setReading] = useState(false);
  const [failed, setFailed] = useState<Answer<Page> | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const key = JSON.stringify([filter, order]);
  const asked = useRef(key);
  useEffect(() => { asked.current = key; setPages([]); setNext(0); setFailed(null); setReading(false); }, [key]);
  const more = async () => {
    if (next === null || reading) return;
    setReading(true);
    const from = next;
    const a = await post<Page>("/api/texts/passages", { filter: body(filter), offset: from, limit: TEXTS, characters: CHARACTERS, order: order.by, descending: order.descending });
    if (asked.current !== key) return; // the filter or the order changed while it was read
    setReading(false);
    if (!a.ok) { setFailed(a); return; }
    setPages((p) => [...p, { offset: from, rows: a.data.rows }]);
    setNext(a.data.next);
  };
  useEffect(() => {
    const sentinel = end.current;
    if (!sentinel || next === null || failed) return;
    const watcher = new IntersectionObserver((seen) => { if (seen.some((s) => s.isIntersecting)) more(); }, { rootMargin: "800px" });
    watcher.observe(sentinel);
    return () => watcher.disconnect();
  });
  const opened = (id: string, place: number) => () => {
    if (window.getSelection()?.isCollapsed === false) return; // a passage being selected, not a click
    onOpen(id, place);
  };
  return (
    <div className="blob" aria-label="The open texts, one after another">
      {pages.length === 0 && next !== null && !failed && <Skeleton lines={8} label="Reading the first texts" />}
      {pages.length > 0 && pages[0].rows.length === 0 && <p>No text matches the filter.</p>}
      {pages.flatMap((page) => page.rows.map((r, i) => (
        <article key={r.id} className="blob-text" aria-label={r.title ?? r.id}>
          <h2 className="blob-rule small">
            <button type="button" className="blob-name wic-focus" onClick={() => onOpen(r.id, page.offset + i)} title="Open it in the editor">
              <span className="cut">{r.title ?? (r.kind ? `an item of kind ${r.kind}` : "untitled text")}</span> <span className="ledger aside">{r.id}</span>
            </button>
          </h2>
          <pre className="blob-passage" onClick={opened(r.id, page.offset + i)}>{r.passage}</pre>
          {codePoints(r.passage) < r.length && <p className="aside small">{plural(r.length - codePoints(r.passage), "more character")} in this text: open it to read them.</p>}
        </article>
      )))}
      {failed && !failed.ok && <><Problems problems={failed.problems} /><Button variant="quiet" onClick={() => { setFailed(null); setReading(false); }}>Try again</Button></>}
      <div ref={end} className="blob-end small aside" role="status">
        {next === null ? pages.length > 0 && pages[0].rows.length > 0 && "The end of the open texts." : reading && "Reading the next texts"}
      </div>
    </div>
  );
}
