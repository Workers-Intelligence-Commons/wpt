// The form that asks to erase a passage of one text or item, or all of it,
// shared by the page of the text and the Erasures page: the text in its
// reader, where a selection with the mouse fills the place (or the place is
// typed), the passage it names, the reason, and the confirm that sends the
// request (POST /api/erasures), under review.

import { Checkbox, TextArea, TextInput } from "@workers-intelligence-commons/components";
import { useMemo, useState, type ReactNode } from "react";
import { post, type Answer } from "./api";
import { Confirm, only, Outcome, required, RequiredNote, type Declaration } from "./parts";
import { codePoints, resolve, slice, TextViewer, type Mark } from "./reader";
import { href } from "./route";

// What the form needs of a text: its id, and its text, or its kind and content as JSON text.
export type Erasable = { id: string; kind: string | null; text: string | null; content: string | null };

export function EraseForm({ row, onDone }: { row: Erasable; onDone: (words: ReactNode) => void }) {
  const item = row.kind !== null;
  const content = useMemo(() => (item ? JSON.parse(row.content ?? "null") : null), [item, row.content]);
  const text = row.text ?? "";
  const length = item ? 0 : codePoints(text);
  // The place as typed: the field pointer and the two positions are kept as
  // the user wrote them, and read as numbers only for the check and the request.
  const [place, setPlace] = useState<{ pointer: string | null; from: string; to: string } | null>(null);
  const [whole, setWhole] = useState(false);
  const [reason, setReason] = useState("");
  const [answer, setAnswer] = useState<Answer<{ id: string }> | null>(null);
  const position = (v: string) => (/^\s*\d+\s*$/.test(v) ? Number(v) : null);
  const start = place === null ? null : position(place.from);
  const end = place === null ? null : position(place.to);
  const mark: Mark = place !== null && start !== null && end !== null ? { pointer: place.pointer, start, end } : null;
  const choose = (m: Mark) => setPlace(m === null ? null : { pointer: m.pointer, from: String(m.start), to: String(m.end) });
  const value = item ? (mark?.pointer != null ? resolve(content, mark.pointer) : null) : text;
  const size = value === null ? 0 : item ? codePoints(value) : length;
  const fits = mark !== null && value !== null && mark.start < mark.end && mark.end <= size;
  const passage = fits && value !== null ? slice(value, mark!.start, mark!.end) : null;
  const ready = whole || fits;
  const send = async (declaration: Declaration) => {
    const needle = whole ? { text: row.id } : { text: row.id, start: mark!.start, end: mark!.end, field: item ? mark!.pointer : null };
    const a = await post<{ id: string }>("/api/erasures", { reason, needles: [needle], declaration });
    setAnswer(a);
    if (a.ok) onDone(<>Erasure request <a href={href(["erasures"])}>{a.data.id}</a>: asked, under review.</>);
  };
  return (
    <>
      <RequiredNote />
      <div className="erase-layout">
        {!whole && (
          <div>
            <p className="form-note small">Select the passage in the text with the mouse: its place fills the character fields.</p>
            <TextViewer item={item} content={content} text={text} mark={mark} onMark={choose} />
          </div>
        )}
        <div className="action-form">
          <Checkbox label={item ? "Erase the whole item" : "Erase the whole text"} checked={whole} onChange={(e) => setWhole(e.target.checked)} />
          {!whole && (
            <div className="fields">
              {item && (
                <TextInput label={required(only("Field, a JSON Pointer", "for an item"))} required value={place?.pointer ?? ""} placeholder="/messages/0/content/0/text"
                  onChange={(e) => setPlace({ pointer: e.target.value, from: place?.from ?? "", to: place?.to ?? "" })} />
              )}
              <TextInput label={required(only("From character", "for a passage"))} required type="number" min={0} value={place?.from ?? ""}
                onChange={(e) => setPlace({ pointer: place?.pointer ?? null, from: e.target.value, to: place?.to ?? "" })} />
              <TextInput label={required(only("To character, excluded", "for a passage"))} required type="number" min={0} value={place?.to ?? ""}
                onChange={(e) => setPlace({ pointer: place?.pointer ?? null, from: place?.from ?? "", to: e.target.value })} />
            </div>
          )}
          {!whole && (passage !== null ? (
            <div className="passage small">
              <p>Characters {mark!.start} to {mark!.end}, the end excluded: {mark!.end - mark!.start} characters{item ? <> of the string at <span className="ledger">{mark!.pointer || "(the root)"}</span></> : null}.</p>
              <pre className="ledger">{passage}</pre>
            </div>
          ) : (
            <p className="small">
              {place === null ? "No passage chosen yet." : mark === null ? "Give both positions as whole numbers." : item && value === null ? "The field names no string of the content." : `The place is not one: 0 ≤ from < to ≤ ${size}.`}
            </p>
          ))}
          <TextArea label={required("Reason")} required value={reason} onChange={(e) => setReason(e.target.value)} hint={`Public, with the request. Once it is admitted, every place where the passage appears in ${item ? "a string of this item" : "this text"} is erased.`} />
          <Confirm label="Confirm and ask" disabled={!ready || !reason.trim()} onConfirm={send} />
          <Outcome answer={answer} done={() => ""} />
        </div>
      </div>
    </>
  );
}
