// A plain text too long to open whole in the editor (GET /api/texts/ID came cut: it can hold billions of characters),
// read only, a passage at a time from GET /api/texts/ID/passage. The view holds a few passages in a row, in one
// flow of text so a line cut between two reads on; scrolling near its end reads the next passage and drops the first,
// scrolling near its start reads the one before and drops the last, the place in view kept still. Each passage held
// besides the one in view is the overlap that keeps scrolling smooth. A box jumps to any character. Translate reads
// the passage in view translated (translated.tsx).

import { Button } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { get } from "../api";
import { Translated } from "../translation/translated";

const PASSAGE = 200_000; // characters read at a time
const HELD = 3; // passages held in the view at most
const NEAR = 1500; // pixels from an end of the view that read the next passage

type Passage = { from: number; text: string };
type Read = { id: string; version: number; length: number; from: number; text: string };

const readPassage = (id: string, version: number, from: number) =>
  get<Read>(`/api/texts/${encodeURIComponent(id)}/passage?from=${from}&characters=${PASSAGE}&version=${version}`);

export function LongText({ id, version, length, tags, translating }: {
  id: string; version: number; length: number; tags: string[]; translating: boolean;
}) {
  const [held, setHeld] = useState<Passage[]>([]);
  const [problem, setProblem] = useState<string | null>(null);
  const [reading, setReading] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  // what the last change did at the start of the view, so the place in view stays still: the height and scroll before
  const kept = useRef<{ height: number; top: number } | null>(null);
  const busy = useRef(false);
  const [top, setTop] = useState(0); // where the passage at the top of the view starts, which Translate reads

  const read = useCallback(async (from: number) => {
    if (busy.current) return null;
    busy.current = true;
    setReading(true);
    const answer = await readPassage(id, version, from);
    busy.current = false;
    setReading(false);
    if (!answer.ok) { setProblem(answer.problems.join(" ")); return null; }
    setProblem(null);
    return { from: answer.data.from, text: answer.data.text };
  }, [id, version]);

  // the passage holding the character at, alone in the view, from its start (or its end, toEnd)
  const jump = useCallback(async (at: number, toEnd = false) => {
    const from = Math.max(0, Math.min(length - 1, Math.floor(at / PASSAGE) * PASSAGE));
    const one = await read(from);
    if (!one) return;
    kept.current = null;
    setTop(one.from);
    setHeld([one]);
    requestAnimationFrame(() => box.current?.scrollTo({ top: toEnd ? box.current.scrollHeight : 0 }));
  }, [read, length]);
  useEffect(() => { void jump(0); }, [jump]);

  // after each change: the place in view put back where it was, then the passage past HELD dropped at the end away
  // from the one just read (the first after reading on, the last after reading back)
  const drop = useRef<"first" | "last" | null>(null);
  useLayoutEffect(() => {
    const el = box.current;
    if (el && kept.current) el.scrollTop = kept.current.top + (el.scrollHeight - kept.current.height);
    kept.current = null;
    if (held.length <= HELD || !el) return;
    if (drop.current === "first") {
      kept.current = { height: el.scrollHeight, top: el.scrollTop };
      setHeld((now) => now.slice(1));
    } else setHeld((now) => now.slice(0, HELD));
    drop.current = null;
  }, [held]);

  const onScroll = async () => {
    const el = box.current;
    if (!el || !held.length) return;
    const spans = [...el.querySelectorAll<HTMLElement>("[data-from]")];
    const atTop = spans.filter((s) => s.getBoundingClientRect().top <= el.getBoundingClientRect().top + 1).at(-1) ?? spans[0];
    if (atTop) setTop(Number(atTop.dataset.from));
    if (busy.current) return;
    const first = held[0], last = held[held.length - 1];
    const end = last.from + PASSAGE;
    if (el.scrollTop + el.clientHeight > el.scrollHeight - NEAR && end < length) {
      const next = await read(end);
      if (!next) return;
      drop.current = "first";
      setHeld((now) => [...now, next]);
    } else if (el.scrollTop < NEAR && first.from > 0) {
      const before = await read(first.from - PASSAGE);
      if (!before) return;
      const view = box.current!;
      kept.current = { height: view.scrollHeight, top: view.scrollTop };
      drop.current = "last";
      setHeld((now) => [before, ...now]);
    }
  };

  const shownFrom = held[0]?.from ?? 0;
  const shownTo = held.length ? held[held.length - 1].from + held[held.length - 1].text.length : 0;
  const [typed, setTyped] = useState("");
  const inView = held.find((p) => p.from === top) ?? held[0];
  return (
    <>
      <div className="long-line small" role="status">
        <span>Too long to edit here: read only, {PASSAGE.toLocaleString("en")} characters at a time as you scroll.</span>
        <span className="aside">{held.length ? `Characters ${(shownFrom + 1).toLocaleString("en")} to ${shownTo.toLocaleString("en")} of ${length.toLocaleString("en")}` : "Reading…"}{reading && held.length ? ", reading on…" : ""}</span>
        <form className="row tight" onSubmit={(e) => { e.preventDefault(); const n = Number(typed.replace(/[\s,_]/g, "")); if (Number.isFinite(n) && n >= 1) void jump(n - 1); }}>
          <input className="wic-input wic-focus long-jump" inputMode="numeric" aria-label="Go to the character" placeholder="Go to character…" value={typed} onChange={(e) => setTyped(e.target.value)} />
          <Button variant="quiet" type="submit">Go</Button>
          <Button variant="quiet" type="button" onClick={() => void jump(0)}>Start</Button>
          <Button variant="quiet" type="button" onClick={() => void jump(length - 1, true)}>End</Button>
        </form>
      </div>
      {problem && <p className="act small" role="alert">Not read: {problem}</p>}
      <div ref={box} className="viewer content-scroll" hidden={translating} onScroll={() => void onScroll()} aria-label="The text, read only">
        <p className="text-page">{held.map((p) => <span key={p.from} data-from={p.from}>{p.text}</span>)}</p>
      </div>
      {translating && inView && <Translated id={id} text={inView.text} tags={tags} />}
    </>
  );
}
