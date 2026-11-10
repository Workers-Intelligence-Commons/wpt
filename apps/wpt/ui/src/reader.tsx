// A text's reader and the passage chosen in it, shared by the page of a text
// and the Erasures page: a plain text a page of PAGE characters at a time, or
// an item's content as JSON, the marked passage highlighted, and the passage
// the mouse selects. A place counts characters (Unicode code points) from 0,
// the end excluded, as the depot's needles do; in an item, within one string
// of its content, named by its JSON Pointer.

import { useMemo, useRef, useState, type ReactNode } from "react";
import { Pager } from "./parts";

const PAGE = 20000; // characters of a plain text shown at once

// Characters (code points) in a string.
export function codePoints(text: string): number {
  let n = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c < 0xd800 || c > 0xdbff) n++;
  }
  return n;
}

// The index in UTF-16 units of the character at a code point position.
function unitAt(text: string, point: number): number {
  let i = 0;
  for (let n = 0; n < point && i < text.length; n++) i += (text.codePointAt(i) ?? 0) > 0xffff ? 2 : 1;
  return i;
}

export const slice = (text: string, start: number, end: number) => text.slice(unitAt(text, start), unitAt(text, end));

// The string at a JSON Pointer (RFC 6901), or null.
export function resolve(value: unknown, pointer: string): string | null {
  if (pointer !== "" && !pointer.startsWith("/")) return null;
  let at: unknown = value;
  for (const raw of pointer.split("/").slice(1)) {
    const part = raw.replace(/~1/g, "/").replace(/~0/g, "~");
    if (Array.isArray(at) && /^\d+$/.test(part)) at = at[Number(part)];
    else if (at && typeof at === "object" && part in (at as object)) at = (at as Record<string, unknown>)[part];
    else return null;
  }
  return typeof at === "string" ? at : null;
}

const escape = (key: string) => key.replace(/~/g, "~0").replace(/\//g, "~1");

// The selection's place inside one element whose text is a run of the
// passage's string starting at the character base: [start, end), or null
// when nothing is selected inside it.
function selectedIn(element: Element, base: number): [number, number] | null {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return null;
  const range = selection.getRangeAt(0);
  if (!element.contains(range.startContainer) || !element.contains(range.endContainer)) return null;
  const before = document.createRange();
  before.selectNodeContents(element);
  before.setEnd(range.startContainer, range.startOffset);
  const start = base + codePoints(before.toString());
  return [start, start + codePoints(range.toString())];
}

// The element of the string a node is in, in an item's JSON view.
const stringOf = (node: Node) => (node instanceof Element ? node : node.parentElement)?.closest("[data-pointer]") ?? null;

// A passage: in a plain text (pointer null), or in the string of an item's content at pointer.
export type Mark = { pointer: string | null; start: number; end: number } | null;

// A run of a string, with the marked passage of it highlighted.
function Marked({ text, base, mark }: { text: string; base: number; mark: Mark }) {
  if (!mark) return <>{text}</>;
  const length = codePoints(text);
  const start = Math.max(0, Math.min(length, mark.start - base));
  const end = Math.max(start, Math.min(length, mark.end - base));
  if (start === end) return <>{text}</>;
  return <>{slice(text, 0, start)}<mark>{slice(text, start, end)}</mark>{slice(text, end, length)}</>;
}

// An item's content as JSON, each string in an element that names its pointer.
function JsonView({ value, pointer, mark }: { value: unknown; pointer: string; mark: Mark }): ReactNode {
  if (typeof value === "string")
    return <span className="json-string">"<span data-pointer={pointer}><Marked text={value} base={0} mark={mark?.pointer === pointer ? mark : null} /></span>"</span>;
  if (Array.isArray(value))
    return (
      <>[<div className="json-nest">{value.map((v, i) => <div key={i}><JsonView value={v} pointer={`${pointer}/${i}`} mark={mark} />{i < value.length - 1 && ","}</div>)}</div>]</>
    );
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <>{"{"}<div className="json-nest">{entries.map(([k, v], i) => <div key={k}><span className="json-key">{JSON.stringify(k)}</span>: <JsonView value={v} pointer={`${pointer}/${escape(k)}`} mark={mark} />{i < entries.length - 1 && ","}</div>)}</div>{"}"}</>
    );
  }
  return <span className="json-other">{JSON.stringify(value)}</span>;
}

// Where the page of a plain text starts, in UTF-16 units, every PAGE characters.
function pageStarts(text: string): number[] {
  const starts = [0];
  let n = 0;
  for (let i = 0; i < text.length;) {
    i += (text.codePointAt(i) ?? 0) > 0xffff ? 2 : 1;
    n++;
    if (n % PAGE === 0 && i < text.length) starts.push(i);
  }
  return starts;
}

// The text, a page of PAGE characters at a time, or the item's content as
// JSON, with the marked passage highlighted. onMark, when given, gets the
// passage the mouse selects.
export function TextViewer({ item, content, text, mark, onMark }: { item: boolean; content: unknown; text: string; mark: Mark; onMark?: (m: Mark) => void }) {
  const starts = useMemo(() => (item ? [0] : pageStarts(text)), [item, text]);
  const [page, setPage] = useState(0);
  const viewer = useRef<HTMLDivElement>(null);
  const pageText = item ? "" : text.slice(starts[page], starts[page + 1] ?? text.length);
  const length = item ? 0 : codePoints(text);
  const took = () => {
    const box = viewer.current;
    const selection = window.getSelection();
    if (!onMark || !box || !selection || selection.rangeCount === 0 || selection.isCollapsed) return;
    if (item) {
      const range = selection.getRangeAt(0);
      const from = stringOf(range.startContainer);
      if (!from || from !== stringOf(range.endContainer) || !box.contains(from)) return;
      const found = selectedIn(from, 0);
      if (found) onMark({ pointer: from.getAttribute("data-pointer"), start: found[0], end: found[1] });
    } else {
      const pre = box.querySelector("pre");
      const found = pre && selectedIn(pre, page * PAGE);
      if (found) onMark({ pointer: null, start: found[0], end: found[1] });
    }
  };
  return (
    <>
      <div ref={viewer} className="viewer" onMouseUp={took} onKeyUp={took}>
        {item ? <div className="json ledger"><JsonView value={content} pointer="" mark={mark} /></div> : (
          <pre className="text-page"><Marked text={pageText} base={page * PAGE} mark={mark} /></pre>
        )}
      </div>
      {!item && length > PAGE && <Pager offset={page * PAGE} limit={PAGE} total={length} onOffset={(o) => { setPage(Math.floor(o / PAGE)); viewer.current?.scrollTo(0, 0); }} label="Pages of the text" />}
    </>
  );
}
