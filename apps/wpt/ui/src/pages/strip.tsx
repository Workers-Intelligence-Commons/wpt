// Read whole: #/strip/LIBRARY reads a library as one strip of text, its texts
// one after another in the numeric order of their ids, each after a thin rule
// that names it (GET and POST /api/libraries/ID/strip). The strip can hold
// more characters than any page could: its text comes from the server in
// reads of READ characters around where the reader is, kept in a buffer of
// about KEPT characters and KEPT_BLOCKS blocks (the reads farthest away dropped), and only the
// blocks of whole lines, about BLOCK characters, that cover the screen and MARGIN screens on each
// side are on the page: those that scroll out are taken off, those that
// scroll in are put on, the view held where it was. The page's own scrollbar
// stands for the whole strip in characters: dragging it, a click on its
// track, a position typed (a number of characters, or a percentage), Home
// and End jump there, the server finding the text at that position from the
// library's map (made in the background the first time, until then only the
// start and the end are reached). A click on a text's name opens it.

import { Button, Skeleton } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { get, post, type Answer } from "../api";
import { plural, Problems, textName, type Library } from "../parts";
import { codePoints } from "../reader";
import { href } from "../route";
import type { Page } from "./index";

const READ = 65_536; // characters one read asks the server for
const KEPT = 1_048_576; // characters the buffer keeps, about: it drops reads past it, far from the screen
const KEPT_BLOCKS = 2_000; // blocks the buffer keeps, about, whatever their characters (empty texts hold none)
const BLOCK = 2_000; // characters of a block on the page at most, but for its first line: blocks end with a line
const LINE = 32_768; // characters past which a line runs on in the next block, its cut showing as a line end
const MARGIN = 2; // screens of text the page holds at least above and below the screen
const FAR = 3 * MARGIN; // screens past which a block leaves the page
const NEAR = 24; // blocks of the buffer left past the page's end when the buffer reads on there

type Piece = { id: string; title: string | null; kind: string | null; length: number; start: number; text: string };
type Read = { pieces: Piece[]; edge: boolean; position: number | null };
type Shape = { texts: number; characters: number; ready: boolean; fresh: boolean; walked: number; failed: string | null };
// a block of the buffer: seq orders the buffer; start, the character of its text it starts at; pos, the strip's
// characters before it; chars, its characters; head, whether it starts its text
type Block = { seq: number; id: string; title: string | null; kind: string | null; length: number; start: number; text: string; pos: number; chars: number; head: boolean };
type Buffer = { blocks: Block[]; first: boolean; last: boolean }; // first, last: whether it holds the strip's start, its end

// A read's pieces as blocks, numbered from seq on, the first at the strip's position pos.
function blocksOf(pieces: Piece[], seq: number, pos: number): Block[] {
  const found: Block[] = [];
  for (const p of pieces) {
    let at = 0, start = p.start;
    do {
      let end = Math.min(p.text.length, at + BLOCK);
      if (end < p.text.length) {
        const before = p.text.lastIndexOf("\n", end - 1), after = p.text.indexOf("\n", end);
        if (before >= at) end = before + 1; // a block ends with its last line
        else if (after >= 0 && after < at + LINE) end = after + 1; // or with a line longer than a block
        else {
          end = Math.min(p.text.length, at + LINE); // a line longer than LINE is cut, at a space when one is near
          const space = p.text.lastIndexOf(" ", end);
          if (space > at + LINE / 2) end = space + 1;
          else if (/[\uDC00-\uDFFF]/.test(p.text[end] ?? "")) end -= 1; // never between the halves of one character
        }
      }
      const text = p.text.slice(at, end);
      const chars = codePoints(text);
      found.push({ seq: seq++, id: p.id, title: p.title, kind: p.kind, length: p.length, start, text, pos, chars, head: start === 0 });
      start += chars; pos += chars; at = end;
    } while (at < p.text.length);
  }
  return found;
}

const sum = (blocks: Block[]) => blocks.reduce((n, b) => n + b.chars, 0);
const EMPTY: Buffer = { blocks: [], first: false, last: false };

// a typed position: characters (spaces, commas and points between digits allowed), or a percentage of total
function typed(words: string, total: number): number | null {
  const percent = words.trim().match(/^(\d+(?:[.,]\d+)?)\s*%$/);
  if (percent) return Math.round((Number(percent[1].replace(",", ".")) / 100) * total);
  const digits = words.replace(/[\s,.'_]/g, "");
  return /^\d+$/.test(digits) ? Number(digits) : null;
}

function Strip({ library }: { library: string }) {
  const [shape, setShape] = useState<Answer<Shape> | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [buffer, setBuffer] = useState<Buffer>(EMPTY);
  const [shown, setShown] = useState<[number, number]>([0, -1]); // the seqs of the first and last blocks on the page
  const [failed, setFailed] = useState<string[] | null>(null);
  const [at, setAt] = useState<{ pos: number; block: Block | null }>({ pos: 0, block: null });
  const [dragging, setDragging] = useState<number | null>(null);
  const [goTo, setGoTo] = useState("");
  const view = useRef<HTMLDivElement>(null);
  const track = useRef<HTMLDivElement>(null);
  const generation = useRef(0); // a jump starts a new one: reads of an older one are dropped
  const reading = useRef({ forward: false, backward: false });
  const held = useRef<{ seq: number; delta: number } | null>(null); // the block the view stays on while the page changes
  const landing = useRef<"top" | "bottom" | null>(null);
  const fromEnd = useRef(false); // the buffer's positions count back from the strip's end, as its size said then
  const total = shape?.ok ? shape.data.characters : 0;
  const ready = shape?.ok === true && shape.data.ready;

  // the library's name, and its shape until its map is made
  useEffect(() => { get<Library[]>("/api/libraries").then((a) => a.ok && setName(a.data.find((l) => l.id === library)?.name ?? library)); }, [library]);
  useEffect(() => {
    let stop = false, timer = 0;
    const ask = async () => {
      const a = await get<Shape>(`/api/libraries/${encodeURIComponent(library)}/strip`);
      if (stop) return;
      setShape(a);
      if (a.ok && !a.data.fresh && !a.data.failed) timer = window.setTimeout(ask, 2000);
    };
    ask();
    return () => { stop = true; window.clearTimeout(timer); };
  }, [library]);

  // look and settle once the browser has laid the page out, at most once a frame
  const frame = useRef(0);
  const later = useRef<() => void>(() => {});
  const scrolled = () => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => { frame.current = 0; later.current(); });
  };
  useEffect(() => () => cancelAnimationFrame(frame.current), []);

  const hold = () => {
    const box = view.current;
    if (!box) return;
    for (const child of Array.from(box.querySelectorAll<HTMLElement>("[data-seq]"))) {
      if (child.offsetTop + child.offsetHeight > box.scrollTop) {
        held.current = { seq: Number(child.dataset.seq), delta: child.offsetTop - box.scrollTop };
        return;
      }
    }
  };

  const jump = useCallback(async (where: number | "end") => {
    const mine = ++generation.current;
    reading.current = { forward: false, backward: false };
    fromEnd.current = where === "end";
    setFailed(null);
    setBuffer(EMPTY);
    setShown([0, -1]);
    const body = where === "end" ? { end: true, characters: READ } : { at: Math.max(0, where), characters: READ };
    const a = await post<Read>(`/api/libraries/${encodeURIComponent(library)}/strip`, body);
    if (mine !== generation.current) return;
    if (!a.ok) { setFailed(a.problems); return; }
    const blocks = blocksOf(a.data.pieces, 0, a.data.position ?? 0);
    const end = where === "end";
    setBuffer({ blocks, first: (a.data.position ?? 0) === 0, last: end || a.data.edge });
    const count = Math.min(blocks.length, 8);  // a first few, and the page settles from there
    setShown(end ? [blocks.length - count, blocks.length - 1] : [0, count - 1]);
    landing.current = end ? "bottom" : "top";
  }, [library]);

  useEffect(() => { jump(0); }, [jump]);

  // a size that changes (the map made, of other figures than the counts) moves positions counted back from the end
  const size = useRef(0);
  useEffect(() => {
    const delta = total - size.current;
    if (size.current > 0 && delta !== 0 && fromEnd.current)
      setBuffer((b) => ({ ...b, blocks: b.blocks.map((x) => ({ ...x, pos: x.pos + delta })) }));
    size.current = total;
  }, [total]);

  const readOn = useCallback(async (forward: boolean) => {
    const way = forward ? "forward" : "backward";
    if (reading.current[way] || buffer.blocks.length === 0) return;
    const edge = forward ? buffer.blocks[buffer.blocks.length - 1] : buffer.blocks[0];
    reading.current[way] = true;
    const mine = generation.current;
    const a = await post<Read>(`/api/libraries/${encodeURIComponent(library)}/strip`, forward
      ? { text: edge.id, offset: edge.start + edge.chars, characters: READ }
      : { text: edge.id, offset: edge.start, backward: true, characters: READ });
    if (mine !== generation.current) return;
    reading.current[way] = false;
    if (!a.ok) { setFailed(a.problems); return; }
    if (!forward) hold(); // the blocks before the page change its first block's rule and the start's note
    setBuffer((b) => {
      const now = forward ? b.blocks[b.blocks.length - 1] : b.blocks[0];
      if (!now || now.seq !== edge.seq || now.chars !== edge.chars) return b; // its end changed while it read
      if (forward) {
        const tail = now;
        return { ...b, blocks: [...b.blocks, ...blocksOf(a.data.pieces, tail.seq + 1, tail.pos + tail.chars)], last: a.data.edge };
      }
      const head = now;
      const chars = a.data.pieces.reduce((n, p) => n + codePoints(p.text), 0);
      const before = blocksOf(a.data.pieces, 0, head.pos - chars);
      const shift = head.seq - before.length;
      return { ...b, blocks: [...before.map((x) => ({ ...x, seq: x.seq + shift })), ...b.blocks], first: a.data.edge };
    });
  }, [buffer, library]);

  // what the screen shows: the block at its top and the strip's position there
  const look = useCallback(() => {
    const box = view.current;
    if (!box) return;
    for (const child of Array.from(box.querySelectorAll<HTMLElement>("[data-seq]"))) {
      if (child.offsetTop + child.offsetHeight > box.scrollTop) {
        const block = buffer.blocks.find((b) => b.seq === Number(child.dataset.seq)) ?? null;
        if (block) {
          const into = Math.max(0, Math.min(1, (box.scrollTop - child.offsetTop) / Math.max(1, child.offsetHeight)));
          const pos = block.pos + Math.round(block.chars * into);
          setAt((was) => (was.pos === pos && was.block?.seq === block.seq ? was : { pos, block }));
        }
        return;
      }
    }
  }, [buffer]);

  // the page's blocks follow the screen: those past FAR screens leave it, and while it holds less than MARGIN screens
  // on a side it takes as many more blocks there as the blocks on it, of their average height, make up; the buffer
  // reads on when its end comes within NEAR blocks of the page's, and what is past KEPT characters, far from the
  // screen, leaves it
  const settle = useCallback(() => {
    const box = view.current;
    const blocks = buffer.blocks;
    if (!box || blocks.length === 0 || shown[1] < shown[0]) return;
    const firstSeq = blocks[0].seq, lastSeq = blocks[blocks.length - 1].seq;
    const screen = box.clientHeight, top = box.scrollTop;
    const children = Array.from(box.querySelectorAll<HTMLElement>("[data-seq]"));
    if (children.length === 0) return;
    const average = Math.max(8, children.reduce((n, c) => n + c.offsetHeight, 0) / children.length);
    let [lo, hi] = shown;
    let gone = 0;
    while (gone < children.length - 1 && children[gone].offsetTop + children[gone].offsetHeight < top - screen * FAR) gone++;
    lo += gone;
    let cut = 0;
    while (cut < children.length - 1 - gone && children[children.length - 1 - cut].offsetTop > top + screen * (1 + FAR)) cut++;
    hi -= cut;
    const below = box.scrollHeight - top - screen, above = top;
    if (below < screen * MARGIN && hi < lastSeq) hi = Math.min(lastSeq, hi + Math.ceil((screen * MARGIN * 1.5 - below) / average));
    if (above < screen * MARGIN && lo > firstSeq) lo = Math.max(firstSeq, lo - Math.ceil((screen * MARGIN * 1.5 - above) / average));
    if (lo !== shown[0] || hi !== shown[1]) { hold(); setShown([lo, hi]); return; }
    if (failed) return; // no read after a failed one, until "Try again"
    if (lastSeq - hi < NEAR && !buffer.last) readOn(true);
    if (lo - firstSeq < NEAR && !buffer.first) readOn(false);
    const extra = sum(blocks) - KEPT, extraBlocks = blocks.length - KEPT_BLOCKS;
    if (extra > 0 || extraBlocks > 0) {
      const dropTail = lastSeq - hi > lo - firstSeq;
      let drop = 0, chars = 0;
      if (dropTail) {
        for (let i = blocks.length - 1; i >= 0 && (chars < extra || drop < extraBlocks) && blocks[i].seq > hi + NEAR * 2; i--) { chars += blocks[i].chars; drop++; }
        if (drop) setBuffer((b) => ({ ...b, blocks: b.blocks.slice(0, b.blocks.length - drop), last: false }));
      } else {
        for (let i = 0; i < blocks.length && (chars < extra || drop < extraBlocks) && blocks[i].seq < lo - NEAR * 2; i++) { chars += blocks[i].chars; drop++; }
        if (drop) setBuffer((b) => ({ ...b, blocks: b.blocks.slice(drop), first: false }));
      }
    }
  }, [buffer, shown, readOn, failed]);

  later.current = () => { look(); settle(); };

  // after each change of the page: the view lands where a jump asked, or stays on the block it was on, then settles
  useLayoutEffect(() => {
    const box = view.current;
    if (!box) return;
    if (landing.current) {
      box.scrollTop = landing.current === "top" ? 0 : box.scrollHeight;
      landing.current = null;
      held.current = null;
    } else if (held.current) {
      const element = box.querySelector<HTMLElement>(`[data-seq="${held.current.seq}"]`);
      if (element) box.scrollTop = element.offsetTop - held.current.delta;
      held.current = null;
    }
    look();
    scrolled(); // the next frame, never in this one: each change of the page settles after the browser laid it out
  });

  // the scrollbar: a position for a place on its track, shown while dragged, jumped to when let go
  const positionAt = (y: number) => {
    const box = track.current?.getBoundingClientRect();
    if (!box || total === 0) return 0;
    return Math.round(Math.max(0, Math.min(1, (y - box.top) / box.height)) * total);
  };
  const pressed = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!ready) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(positionAt(e.clientY));
  };
  const moved = (e: ReactPointerEvent<HTMLDivElement>) => { if (dragging !== null) setDragging(positionAt(e.clientY)); };
  const released = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (dragging === null) return;
    const target = positionAt(e.clientY);
    setDragging(null);
    jump(target >= total ? "end" : target);
  };
  const keyed = (e: KeyboardEvent) => {
    if (e.key === "Home") { e.preventDefault(); jump(0); }
    else if (e.key === "End") { e.preventDefault(); jump("end"); }
  };
  const went = (e: FormEvent) => {
    e.preventDefault();
    const found = typed(goTo, total);
    if (found === null) return;
    jump(found >= total ? "end" : found);
  };

  const pos = dragging ?? at.pos;
  const share = total > 0 ? Math.min(1, pos / total) : 0;
  const onPage = buffer.blocks.filter((b) => b.seq >= shown[0] && b.seq <= shown[1]);
  return (
    <div className="strip-frame">
      <div className="strip-top">
        <h1 className="title strip-title">{name ?? library}</h1>
        <span className="small aside">read whole</span>
        <a className="small" href={href(["texts"], { library })}>back to its texts</a>
        <span className="strip-where small ledger" role="status">
          {total > 0 ? <>{pos.toLocaleString("en")} of {plural(total, "character")} · {(share * 100).toFixed(share < 0.001 && share > 0 ? 4 : 2)}%</> : shape?.ok ? "an empty library" : ""}
          {at.block && dragging === null && <> · in <a href={href(["texts", at.block.id], { library })}>{textName(at.block)}</a></>}
        </span>
        <form className="strip-go small" onSubmit={went}>
          <label>go to <input value={goTo} onChange={(e) => setGoTo(e.target.value)} placeholder={ready ? "50% or 1 000 000" : "after the map"} disabled={!ready} size={14} /></label>
          <Button variant="quiet" type="submit" disabled={!ready}>Go</Button>
          <Button variant="quiet" type="button" onClick={() => jump(0)}>Start</Button>
          <Button variant="quiet" type="button" onClick={() => jump("end")}>End</Button>
        </form>
      </div>
      {shape?.ok && shape.data.ready && !shape.data.fresh && !shape.data.failed && (
        <p className="small aside strip-note">The library changed since its map was made: making it again ({plural(shape.data.walked, "text")} of {shape.data.texts.toLocaleString("en")} walked). Jumps land near their position meanwhile.</p>
      )}
      {shape?.ok && !shape.data.ready && (
        <p className="small aside strip-note">{shape.data.failed ? <>The map of the library failed: {shape.data.failed}. Its start and its end still read.</>
          : <>Making the map of the library, which jumps to any position: {plural(shape.data.walked, "text")} of {shape.data.texts.toLocaleString("en")} walked. Its start and its end read already.</>}</p>
      )}
      {shape && !shape.ok && <Problems problems={shape.problems} />}
      {failed && <><Problems problems={failed} /><Button variant="quiet" onClick={() => jump(at.pos)}>Try again</Button></>}
      <div className="strip-body">
        <div ref={view} className="strip-view" tabIndex={0} onScroll={scrolled} onKeyDown={keyed} aria-label={`The library ${name ?? library}, read whole`}>
          {buffer.blocks.length === 0 && !failed && (shape?.ok && shape.data.texts === 0 ? <p>The library holds no text.</p> : <Skeleton lines={10} label="Reading the strip" />)}
          {buffer.first && shown[0] === buffer.blocks[0]?.seq && <p className="strip-edge small aside">The start of the library.</p>}
          {onPage.map((b, i) => (
            <div key={b.seq} data-seq={b.seq} className="strip-block">
              {(b.head || (i === 0 && b.seq === buffer.blocks[0].seq)) && (
                <h2 className="blob-rule small">
                  <a className="blob-name" href={href(["texts", b.id], { library })} title="Open it">
                    <span className="cut">{textName(b)}</span> <span className="ledger aside">{b.id} · {plural(b.length, "character")}{!b.head && ` · from character ${b.start.toLocaleString("en")}`}</span>
                  </a>
                </h2>
              )}
              {b.text && <pre className="strip-text">{b.text}</pre>}
            </div>
          ))}
          {buffer.last && shown[1] === buffer.blocks[buffer.blocks.length - 1]?.seq && <p className="strip-edge small aside">The end of the library.</p>}
        </div>
        <div ref={track} className={`strip-bar${ready ? "" : " waiting"}`} onPointerDown={pressed} onPointerMove={moved} onPointerUp={released}
          role="scrollbar" aria-orientation="vertical" aria-valuemin={0} aria-valuemax={total} aria-valuenow={pos} aria-label="Where in the library" title={ready ? "Drag or click to jump" : "The map is being made: only the start and the end read"}>
          <div className="strip-thumb" style={{ top: `calc(${share * 100}% - ${share * 28}px)` }} />
        </div>
      </div>
    </div>
  );
}

export const strip: Page = {
  path: "strip", title: "Read whole", group: null, under: "libraries", full: true,
  Component: ({ rest }) => (rest[0] ? <Strip key={rest[0]} library={rest[0]} /> : <p>Name a library: #/strip/LIBRARY.</p>),
};
