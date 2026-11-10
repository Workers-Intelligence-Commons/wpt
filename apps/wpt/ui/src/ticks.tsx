// Several values chosen at once: a list of ticks, with a box above it that
// keeps only the options holding every word typed, and the same list folded
// into a compact menu for a filter bar, beside a menu of one value of a few.

import { Button } from "@workers-intelligence-commons/components";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";

export type Option = { value: string; label: ReactNode; words?: string; count?: number };

const wordsOf = (o: Option) => (o.words ?? `${o.value} ${typeof o.label === "string" ? o.label : ""}`).toLowerCase();
const SHOWN = 80; // ticks shown at once; typing finds the others

// The ticks, the ticked ones first, under their label (a legend, with its
// markers) and a quiet hint; legend: the label in plain words, for the box
// and the group's name when label is not a string; search: a box to find
// options by typing; empty: what to say, with the words typed, when no
// option holds them (a way to add one, for instance).
export function TickList({ label, legend, hint, options, chosen, onChange, search = false, empty }: {
  label: ReactNode; legend?: string; hint?: ReactNode; options: Option[]; chosen: string[]; onChange: (values: string[]) => void; search?: boolean;
  empty?: (typed: string) => ReactNode;
}) {
  const words = legend ?? (typeof label === "string" ? label : "options");
  const [typed, setTyped] = useState("");
  const wanted = typed.toLowerCase().split(/\s+/).filter(Boolean);
  const found = options.filter((o) => wanted.every((w) => wordsOf(o).includes(w)));
  const ordered = [...found.filter((o) => chosen.includes(o.value)), ...found.filter((o) => !chosen.includes(o.value))];
  const shown = ordered.slice(0, Math.max(SHOWN, chosen.length));
  const flip = (value: string, on: boolean) => onChange(on ? [...chosen, value] : chosen.filter((c) => c !== value));
  return (
    <fieldset className="ticks">
      <legend className="wic-label">{label}</legend>
      {hint && <p className="wic-field-hint">{hint}</p>}
      {search && (
        <input className="wic-input wic-focus tick-search" type="search" value={typed} placeholder={`Find ${words.toLowerCase()} by typing`}
          aria-label={`Find ${words.toLowerCase()}`} onChange={(e) => setTyped(e.target.value)} />
      )}
      <div className="tick-options" role="group" aria-label={words}>
        {shown.map((o) => (
          <label key={o.value} className="tick">
            <input type="checkbox" className="wic-check-box wic-focus" checked={chosen.includes(o.value)} onChange={(e) => flip(o.value, e.target.checked)} />
            <span className="tick-label">{o.label}</span>
            {o.count !== undefined && <span className="aside tick-count">{o.count.toLocaleString("en")}</span>}
          </label>
        ))}
        {ordered.length > shown.length && <p className="aside">{ordered.length - shown.length} more: type to find them.</p>}
        {found.length === 0 && (empty ? empty(typed.trim()) : <p className="aside">{typed ? `None holds "${typed}".` : "None."}</p>)}
      </div>
      {chosen.length > 0 && <Button variant="quiet" onClick={() => onChange([])}>Untick all</Button>}
    </fieldset>
  );
}

// A menu of a filter bar: a button that names it and says what is chosen
// (said, whole on hover in title), and the panel it opens below it, which
// closes on Escape or a click outside.
function Fold({ label, said, title, children }: { label: string; said: string; title?: string; children: ReactNode }) {
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const away = (e: MouseEvent | KeyboardEvent) => {
      const menu = ref.current;
      if (!menu?.open) return;
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !menu.contains(e.target as Node)) {
        menu.open = false;
        if (e instanceof KeyboardEvent) menu.querySelector("summary")?.focus();
      }
    };
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", away);
    return () => { document.removeEventListener("mousedown", away); document.removeEventListener("keydown", away); };
  }, []);
  return (
    <details ref={ref} className="tick-menu">
      <summary className="wic-input wic-focus">
        <span className="cue">{label}:</span> <span className="tick-said" title={title ?? said}>{said}</span>
      </summary>
      <div className="tick-panel">{children}</div>
    </details>
  );
}

// A TickList folded into a menu ("Tags: any", "Tags: 3 ticked").
export function TickMenu({ label, options, chosen, onChange, search }: { label: string; options: Option[]; chosen: string[]; onChange: (values: string[]) => void; search?: boolean }) {
  const said = chosen.length === 0 ? "any" : chosen.length === 1 ? chosen[0] : `${chosen.length} ticked`;
  return (
    <Fold label={label} said={said} title={chosen.join(", ") || said}>
      <TickList label={label} options={options} chosen={chosen} onChange={onChange} search={search} />
    </Fold>
  );
}

// One value of a few, folded into a menu the same way ("In a locked
// library: either"); choosing one applies it and closes the menu.
export function ChoiceMenu({ label, options, chosen, onChange }: { label: string; options: { value: string; label: string }[]; chosen: string; onChange: (value: string) => void }) {
  const name = useId();
  return (
    <Fold label={label} said={options.find((o) => o.value === chosen)?.label ?? chosen}>
      <div className="tick-options" role="radiogroup" aria-label={label}>
        {options.map((o) => (
          <label key={o.value} className="tick">
            <input type="radio" className="wic-focus" name={name} checked={o.value === chosen}
              onChange={(e) => { onChange(o.value); (e.target.closest("details") as HTMLDetailsElement | null)?.removeAttribute("open"); }} />
            <span className="tick-label">{o.label}</span>
          </label>
        ))}
      </div>
    </Fold>
  );
}
