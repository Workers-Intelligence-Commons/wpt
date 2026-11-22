// The passage an excerpt block quotes (docs/chat-format.md, "Excerpts"),
// selected in the text (with a mouse or a finger) instead of typed. The picker opens on the text the
// block already quotes, else on the text most recently quoted in the
// conversation, else on a list of the depot's plain texts, the newest first,
// with one button a tag and one for the texts this conversation quotes, so a
// text is found by clicks (words of its title can narrow the list too). In
// the text's reader the person selects the passage; the
// picker trims the spaces around it, takes whole a word the selection cuts at
// either end (a mouse rarely lands between two letters exactly), and fills the block with the passage as
// exact and the 32 characters (Unicode code points, as the depot counts)
// right before and right after it as prefix and suffix, widened 32 at a time
// on both sides until the three together stand at one place of the text
// only, the rule the depot checks (wpt_depot.quotes.find). Only plain texts
// are quoted, as the depot requires.

import { Button, Dialog, TextInput } from "@workers-intelligence-commons/components";
import { useEffect, useMemo, useState } from "react";
import { get } from "./api";
import { EVERY, search, type Found, type Row } from "./filter";
import { PLAIN, plural } from "./parts";
import { codePoints, TextViewer, type Mark } from "./reader";

export type Selector = { type?: string; exact: string; prefix?: string; suffix?: string };
type Quoting = { type: string; source?: string; selector?: Selector; content?: Quoting[] };

export const CONTEXT = 32; // characters of prefix and of suffix, and how many more each widening takes

// UTF-16 index n code points after (or before, n < 0) the index at.
function step(text: string, at: number, n: number): number {
  let i = at;
  for (let k = 0; k < Math.abs(n); k++) {
    if (n > 0) {
      if (i >= text.length) break;
      i += (text.codePointAt(i) ?? 0) > 0xffff ? 2 : 1;
    } else {
      if (i <= 0) break;
      const c = text.charCodeAt(i - 1);
      i -= c >= 0xdc00 && c <= 0xdfff && i >= 2 ? 2 : 1;
    }
  }
  return i;
}

// How many places of the text match the selector, as the depot counts them (overlapping), at most most.
export function matchCount(text: string, sel: Selector, most = 2): number {
  if (!sel.exact) return 0;
  const pattern = (sel.prefix ?? "") + sel.exact + (sel.suffix ?? "");
  let n = 0;
  for (let at = text.indexOf(pattern); at !== -1 && n < most; at = text.indexOf(pattern, at + 1)) n++;
  return n;
}

// Where the passage a selector matches stands, in code points [start, end), when it matches one place only.
export function placeOf(text: string, sel: Selector | undefined): [number, number] | null {
  if (!sel || matchCount(text, sel) !== 1) return null;
  const at = text.indexOf((sel.prefix ?? "") + sel.exact + (sel.suffix ?? "")) + (sel.prefix ?? "").length;
  const start = codePoints(text.slice(0, at));
  return [start, start + codePoints(sel.exact)];
}

const WORD = /[\p{L}\p{N}\p{M}_'’]/u; // a character inside a word: a mouse selection that cuts one takes it whole

// The selector of the passage [start, end) of the text (code points), its
// spaces around trimmed and a word cut at either end taken whole, or null
// when nothing but spaces is selected.
export function selectorFor(text: string, start: number, end: number): Selector | null {
  let from = step(text, 0, start);
  let to = step(text, from, end - start);
  while (from < to && /\s/.test(text[from])) from++;
  while (to > from && /\s/.test(text[to - 1])) to--;
  if (from >= to) return null;
  while (from > 0 && WORD.test(text[from]) && WORD.test(text[from - 1])) from--;
  while (to < text.length && WORD.test(text[to - 1]) && WORD.test(text[to])) to++;
  const exact = text.slice(from, to);
  for (let wide = CONTEXT; ; wide += CONTEXT) {
    const before = step(text, from, -wide);
    const after = step(text, to, wide);
    const sel: Selector = { type: "TextQuoteSelector", exact, prefix: text.slice(before, from), suffix: text.slice(to, after) };
    // the whole text, prefix to suffix, stands at one place only, so the widening always ends
    if (matchCount(text, sel) === 1 || (before === 0 && after === text.length)) {
      if (!sel.prefix) delete sel.prefix;
      if (!sel.suffix) delete sel.suffix;
      return sel;
    }
  }
}

// The texts a conversation quotes, the most recently quoted (the last excerpt) first.
export function quotedIn(messages: { content: Quoting[] }[]): string[] {
  const seen: string[] = [];
  const each = (b: Quoting) => {
    if (b.type === "excerpt" && b.source?.trim()) seen.push(b.source.trim());
    b.content?.forEach(each);
  };
  messages.forEach((m) => m.content.forEach(each));
  return [...new Set(seen.reverse())];
}

// A text's title by its id, read once a page load, for the excerpt blocks.
const titles = new Map<string, Promise<string | null>>();
export function useTitle(id: string | undefined): string | null {
  const [title, setTitle] = useState<string | null>(null);
  useEffect(() => {
    setTitle(null);
    if (!id) return;
    if (!titles.has(id)) titles.set(id, search({ ...EVERY, ids: [id] }, {}, 1).then((a) => (a.ok ? a.data.rows[0]?.title ?? null : null)));
    let live = true;
    titles.get(id)!.then((t) => live && setTitle(t));
    return () => { live = false; };
  }, [id]);
  return title;
}

type Tag = { tag: string; texts: number };
// What the picker reads of a text (GET /api/texts/ID).
type Shown = { text: { id: string; kind: string; title: string | null; text: string | null } };
const showText = (id: string) => get<Shown>(`/api/texts/${encodeURIComponent(id)}`);
type Shelf = { what: "newest" } | { what: "quoted" } | { what: "tag"; tag: string };
const MOST_TAGS = 12; // tags shown as buttons before "every tag"

// The depot's plain texts, a page at a time, as buttons: the newest, those a tag holds, or those the conversation quotes.
function TextList({ quoted, onPick }: { quoted: string[]; onPick: (id: string) => void }) {
  const [shelf, setShelf] = useState<Shelf>(quoted.length ? { what: "quoted" } : { what: "newest" });
  const [words, setWords] = useState("");
  const [typed, setTyped] = useState("");
  const [tags, setTags] = useState<Tag[]>([]);
  const [allTags, setAllTags] = useState(false);
  const [rows, setRows] = useState<Row[]>([]);
  const [next, setNext] = useState<Found["next"]>(null);
  const [state, setState] = useState<"reading" | "done" | string>("reading");
  useEffect(() => { get<Tag[]>("/api/tags").then((a) => a.ok && setTags([...a.data].sort((x, y) => y.texts - x.texts))); }, []);
  useEffect(() => { const t = setTimeout(() => setWords(typed), 400); return () => clearTimeout(t); }, [typed]);
  const filter = useMemo(() => ({
    ...EVERY, kinds: [PLAIN], search: words,
    ids: shelf.what === "quoted" ? quoted : null, tags: shelf.what === "tag" ? [shelf.tag] : [],
  }), [shelf, words, quoted]);
  const read = async (after: Found["next"], live: () => boolean) => {
    setState("reading");
    // a page in the order of ids, descending: the newest texts first
    const a = await search(filter, after ? { after } : {}, 20, { by: "id", descending: true });
    if (!live()) return;
    if (!a.ok) { setState(a.problems.join(" ") || "The texts could not be read."); return; }
    setRows((was) => (after ? [...was, ...a.data.rows] : a.data.rows));
    setNext(a.data.next);
    setState("done");
  };
  useEffect(() => {
    let live = true;
    setRows([]);
    read(null, () => live);
    return () => { live = false; };
  }, [filter]); // eslint-disable-line react-hooks/exhaustive-deps
  const shownTags = allTags ? tags : tags.slice(0, MOST_TAGS);
  const pressed = (s: Shelf) => JSON.stringify(s) === JSON.stringify(shelf);
  return (
    <div className="passage-list">
      <div className="chips" role="group" aria-label="Which texts">
        {quoted.length > 0 && <button type="button" className="chip-button" aria-pressed={pressed({ what: "quoted" })} onClick={() => setShelf({ what: "quoted" })}>Quoted here ({quoted.length})</button>}
        <button type="button" className="chip-button" aria-pressed={pressed({ what: "newest" })} onClick={() => setShelf({ what: "newest" })}>Newest</button>
        {shownTags.map((t) => <button key={t.tag} type="button" className="chip-button" aria-pressed={pressed({ what: "tag", tag: t.tag })}
          onClick={() => setShelf({ what: "tag", tag: t.tag })} title={plural(t.texts, "text")}>{t.tag}</button>)}
        {tags.length > MOST_TAGS && <Button variant="quiet" onClick={() => setAllTags(!allTags)}>{allTags ? "Fewer tags" : `Every tag (${tags.length})`}</Button>}
      </div>
      <TextInput label="Words of its title (optional)" value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="rivers" />
      <ul className="passage-texts">
        {rows.map((r) => (
          <li key={r.id}>
            <button type="button" className="passage-text" onClick={() => onPick(r.id)}>
              <span className="passage-title">{r.title || r.id}</span>
              <span className="small aside">{r.id} · {plural(r.length, "character")}{r.tags.length ? ` · ${r.tags.join(", ")}` : ""}</span>
            </button>
          </li>
        ))}
      </ul>
      {state === "reading" ? <p className="small aside" role="status">Reading the texts…</p>
        : state !== "done" ? <p className="small refusal" role="alert">{state}</p>
        : !rows.length && !next ? <p className="small aside">No plain text here.</p> : null}
      {state === "done" && next && <Button variant="quiet" onClick={() => read(next, () => true)}>More texts</Button>}
    </div>
  );
}

// The context of a quote on one line: each run of line breaks, with the
// spaces around it, shown as one pilcrow between spaces, so a paragraph break
// is no blank gap inside the quote.
const oneLine = (context: string) => context.replace(/[ \t]*(?:\r?\n[ \t]*)+/g, " ¶ ");

// A quote as an excerpt shows it: the passage marked, the context before and after it muted, on one line.
export function QuotePreview({ selector }: { selector: Selector }) {
  return (
    <blockquote className="passage-quote">
      {selector.prefix && <span className="aside">…{oneLine(selector.prefix)}</span>}<mark>{selector.exact}</mark>{selector.suffix && <span className="aside">{oneLine(selector.suffix)}…</span>}
    </blockquote>
  );
}

// The dialog that picks a passage and gives the excerpt block it fills.
export function PassagePicker({ open, onClose, source, selector, quoted, onPick }: {
  open: boolean; onClose: () => void; source: string | undefined; selector: Selector | undefined; quoted: string[];
  onPick: (source: string, selector: Selector) => void;
}) {
  const [id, setId] = useState<string | null>(null);
  const [shown, setShown] = useState<Shown | null>(null);
  const [trouble, setTrouble] = useState<string | null>(null);
  const [mark, setMark] = useState<Mark>(null);
  // each opening starts on the text the block quotes, else the one quoted last
  useEffect(() => {
    if (!open) return;
    setId(source?.trim() || quoted[0] || null);
    setTrouble(null);
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    setShown(null); setMark(null);
    if (!id) return;
    let live = true;
    showText(id).then((a) => {
      if (!live) return;
      if (!a.ok) { setTrouble(`The text ${id} could not be read: ${a.problems.join(" ")}`); setId(null); return; }
      if (a.data.text.kind !== PLAIN) { setTrouble(`${id} is a ${a.data.text.kind}, not a plain text: only a plain text is quoted.`); setId(null); return; }
      setShown(a.data);
      const at = id === source?.trim() ? placeOf(a.data.text.text ?? "", selector) : null;
      setMark(at ? { pointer: null, start: at[0], end: at[1] } : null);
    });
    return () => { live = false; };
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps
  const text = shown?.text.text ?? "";
  const picked = useMemo(() => (shown && mark && mark.start < mark.end ? selectorFor(text, mark.start, mark.end) : null), [shown, mark, text]);
  const choose = (m: Mark) => { setMark(m); setTrouble(null); };
  const title = shown ? shown.text.title || shown.text.id : null;
  return (
    <Dialog open={open} onClose={onClose} title={shown ? `Quote ${title}` : "Quote a passage"}>
      <div className="passage-picker">
        {trouble && <p className="small refusal" role="alert">{trouble}</p>}
        {!id && <><p className="small">Pick the plain text to quote.</p><TextList quoted={quoted} onPick={(picked) => { setTrouble(null); setId(picked); }} /></>}
        {id && !shown && <p className="small aside" role="status">Reading {id}…</p>}
        {shown && (
          <>
            <div className="row tight passage-head">
              <span className="small aside">{shown.text.id} · select the passage in the text.</span>
              <Button variant="quiet" onClick={() => { setId(null); setTrouble(null); }}>Another text</Button>
            </div>
            <TextViewer item={false} content={null} text={text} mark={mark} onMark={choose} />
            <div className="passage-chosen">
              {picked ? (
                <>
                  <QuotePreview selector={picked} />
                  <p className="small aside">
                    {plural(codePoints(picked.exact), "character")} quoted, with {plural(codePoints(picked.prefix ?? ""), "character")} before
                    and {plural(codePoints(picked.suffix ?? ""), "character")} after, so the quote stands at one place of the text.
                  </p>
                </>
              ) : <p className="small aside">{mark ? "Only spaces are selected: select words." : "No passage selected yet."}</p>}
              <Button variant="primary" disabled={!picked} onClick={() => { if (picked && shown) { onPick(shown.text.id, picked); onClose(); } }}>Quote this passage</Button>
            </div>
          </>
        )}
      </div>
    </Dialog>
  );
}
