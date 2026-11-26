// The kind of a text as cards, one a kind, each with one short plain line
// (SHORT; a kind not named there shows the depot's meaning), the depot's
// full meaning of each behind the "?" beside the picker's label, chosen
// with one click (radio buttons, so arrow keys move between them too). The
// order is fixed: a plain text first, then the depot's other kinds as
// GET /api/kinds lists them, so a click never lands on a card that moved.
// On the Add page the last kind picked is remembered in this browser's
// localStorage (lastKind) and picked again next time; a browser that
// refuses storage simply starts on a plain text. A text's kind is fixed
// when it is added: on an open text its own card alone shows (FixedKind),
// small, greyed and disabled, in the line of its facts. useKinds reads the
// kinds once a page load, since the depot's kinds do not change while it runs.

import { useEffect, useId, useState } from "react";
import { get } from "./api";
import { Help, PLAIN, required } from "./parts";
import type { Schema } from "./schema-form";

export type Kind = { id: string; meaning: string; schema: Schema };

let kindsRead: Promise<Kind[] | null> | null = null;

export function useKinds(): Kind[] | null {
  const [kinds, setKinds] = useState<Kind[] | null>(null);
  useEffect(() => {
    kindsRead ??= get<Kind[]>("/api/kinds").then((a) => (a.ok ? a.data : null));
    let live = true;
    kindsRead.then((found) => {
      if (found === null) kindsRead = null; // read again next time
      if (live) setKinds(found);
    });
    return () => { live = false; };
  }, []);
  return kinds;
}

const KEY = "wpt.add.kind";

export const lastKind = (): string | null => {
  try { return window.localStorage.getItem(KEY); } catch { return null; }
};

const remember = (kind: string) => {
  try { window.localStorage.setItem(KEY, kind); } catch { /* not remembered: the page works the same */ }
};

// a plain text first, the other kinds after in the order given
const ordered = (kinds: Kind[]) => [...kinds.filter((k) => k.id === PLAIN), ...kinds.filter((k) => k.id !== PLAIN)];
const nameOf = (id: string) => (id === PLAIN ? "Plain text" : id.replace(/^./, (c) => c.toUpperCase()));
const sentence = (meaning: string) => `${meaning.replace(/^./, (c) => c.toUpperCase())}.`;
// What each kind is, in a few plain words, for its card.
const SHORT: Record<string, string> = {
  plain: "Prose, code or any text file",
  conversation: "Chat turns between people and an assistant",
  task: "A job for an agent, with its files and its check",
  probe: "A test question for a model, never trained on",
  rating: "A person's pass or fail on a model's attempt",
};

// value: the kind's id (PLAIN for a plain text).
export function KindPicker({ kinds, value, onChange }: { kinds: Kind[]; value: string; onChange: (kind: string) => void }) {
  const name = useId();
  return (
    <fieldset className="kind-picker">
      <legend className="wic-label">{required("What it is")}<Help label="the kinds">
        <span className="kind-help">{ordered(kinds).map((k) => <span key={k.id}><strong>{nameOf(k.id)}</strong>: {sentence(k.meaning)} </span>)}A text keeps its kind once added.</span>
      </Help></legend>
      <div className="kind-cards">
        {ordered(kinds).map((k) => (
          <label key={k.id} className="kind-card">
            <input type="radio" name={name} value={k.id} checked={value === k.id}
              onChange={() => { remember(k.id); onChange(k.id); }} />
            <span className="kind-name">{nameOf(k.id)}</span>
            <span className="kind-meaning">{SHORT[k.id] ?? sentence(k.meaning)}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

// The kind of a text already added: its card alone, small and greyed, in the line of the text's facts.
// A disabled radio button, checked, so a screen reader says it is a choice that cannot change.
export function FixedKind({ kind, meaning }: { kind: string; meaning?: string }) {
  const why = `${meaning ? sentence(meaning) + " " : ""}Fixed when it was added: a text keeps its kind.`;
  return (
    <label className="kind-card kind-fixed" title={why}>
      <input type="radio" checked disabled readOnly aria-label={`Its kind: ${nameOf(kind)}. ${why}`} />
      <span className="kind-name">{nameOf(kind)}</span>
    </label>
  );
}
