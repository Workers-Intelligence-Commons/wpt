// The review panel beside the open text (text-panel.tsx): everything a
// review clicks, in one column whose buttons never move from one text to the
// next. From the top: a row per score name the depot knows (GET
// /api/scores), with its choices (choices.ts), − and + that step by a
// twentieth of its range, and its value, which opens a number pad beside the
// panel; the depot's most used tags as fixed toggles, the text's other tags
// after them with a cross, and a box for a new tag or a new score name; the
// moves and decisions the Texts page hands in (actions); the presets, each a
// set of scores and of tags to add and take away, and one that keeps the
// text's own as a new preset; then the text's facts, its other actions in a
// menu. A click changes the panel at once and is sent a moment after the
// last one (WAIT), or as soon as the text is left: the scores as one change
// (POST /api/texts/scores), the tags as one edit of its credits (POST
// /api/texts/ID/credits), each a decision under review under the
// declaration this browser keeps, the text keeping its status and libraries.
// What is waiting is outlined until it is sent. A text of the upstream shows
// its scores and tags and changes none.

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { get, post } from "../api";
import { useAbout } from "../about";
import { declared, useDeclarant } from "../declarant";
import { href } from "../route";
import { plural, settingsHref, type Library, type Status } from "../parts";
import type { Score } from "../scores";
import { forgetPreset, forgetScale, keepRecent, presetWords, readRecent, round, savePreset, shortValue, usePresets, useScales, type Preset, type Scale } from "./choices";

const WAIT = 2000; // milliseconds after the last click before what changed is sent
const TAGS_SHOWN = 9; // the depot's most used tags, as toggles

export type PanelText = {
  id: string; tags: string[]; scores: Score[] | null; content_version: number; licenses: string[]; authors: string[] | null;
  address: string | null; revision: number; kind: string | null;
};

// Read once a page: the score names and the tags the depot has, with how many texts have each.
let namesRead: Promise<{ name: string; texts: number }[]> | null = null;
let tagsRead: Promise<{ tag: string; texts: number }[]> | null = null;
const scoreNames = () => (namesRead ??= get<{ name: string; texts: number }[]>("/api/scores").then((a) => (a.ok ? a.data : [])));
const depotTags = () => (tagsRead ??= get<{ tag: string; texts: number }[]>("/api/tags").then((a) => (a.ok ? a.data : [])));
// a name or tag this browser added: the next panels show it at once
const addedNames = new Set<string>();

function useOnce<T>(read: () => Promise<T>, fallback: T): T {
  const [found, setFound] = useState<T>(fallback);
  useEffect(() => { let live = true; read().then((v) => { if (live) setFound(v); }); return () => { live = false; }; }, [read]);
  return found;
}

// What waits to be sent: each score's new value (null: removed), and the tags as they are to be (null: unchanged).
type Waiting = { scores: Record<string, number | null>; tags: string[] | null };
const NOTHING: Waiting = { scores: {}, tags: null };

export function ReviewPanel({ text, status, decisions, libraries, readOnly, actions, more, onSaved, onRecord }: {
  text: PanelText; status: Status; decisions: number; libraries: Library[]; readOnly: boolean;
  actions: ReactNode; more: ReactNode; onSaved: (words: ReactNode) => void; onRecord: () => void;
}) {
  const names = useOnce(scoreNames, []);
  const tags = useOnce(depotTags, []);
  const presets = usePresets();
  const who = useDeclarant();
  const { about } = useAbout();
  // what the depot holds, as this panel last sent or read it
  const [held, setHeld] = useState(() => ({ scores: text.scores ?? [], tags: text.tags }));
  const [waiting, setWaiting] = useState<Waiting>(NOTHING);
  const [problem, setProblem] = useState<ReactNode>(null);
  const [recent, setRecent] = useState(readRecent);
  const [extraNames, setExtraNames] = useState<string[]>([]);
  const allNames = useMemo(() => [...new Set([...names.map((n) => n.name), ...held.scores.map((s) => s.name), ...addedNames, ...extraNames])].sort(),
    [names, held.scores, extraNames]);
  const scales = useScales(allNames);
  const shownTags = useMemo(() => [...tags].sort((a, b) => b.texts - a.texts || a.tag.localeCompare(b.tag)).slice(0, TAGS_SHOWN).map((t) => t.tag), [tags]);

  const heldScore = (name: string) => held.scores.find((s) => s.name === name)?.value ?? null;
  const scoreNow = (name: string) => (name in waiting.scores ? waiting.scores[name] : heldScore(name));
  const tagsNow = waiting.tags ?? held.tags;

  // send what waits; called after WAIT and when the text is left (every move and decision leaves it).
  // What is sent counts as held at once, so a click made while it is on its way builds on it; a part
  // the depot refuses goes back to what it held. A problem is said in the panel and in the page's line,
  // which stays when the panel has gone.
  const state = useRef({ waiting, held, text, who, about, onSaved });
  state.current = { waiting, held, text, who, about, onSaved };
  const send = useCallback(async () => {
    const { waiting: w, held: h, text: t, who: me, about: ab } = state.current;
    const set = Object.entries(w.scores).filter(([name, v]) => v !== null && v !== (h.scores.find((s) => s.name === name)?.value ?? null))
      .map(([name, v]) => ({ name, value: v as number, content_version: t.content_version }));
    const remove = Object.entries(w.scores).filter(([name, v]) => v === null && h.scores.some((s) => s.name === name)).map(([name]) => name);
    const tagsChanged = w.tags !== null && JSON.stringify(w.tags) !== JSON.stringify(h.tags);
    if (!set.length && !remove.length && !tagsChanged) { setWaiting(NOTHING); return; }
    const failed = (words: ReactNode) => { setProblem(words); state.current.onSaved(words); };
    if (!declared(me)) { failed(<>Not saved: set a name and email in <a href={settingsHref()}>Settings</a>, then click again.</>); return; }
    if (!ab) { failed(<>Not saved: the depot's requirements could not be read yet; click again in a moment.</>); return; }
    const declaration = { name: me.name.trim(), email: me.email.trim(), accepted: true };
    const scoresTo = [...h.scores.filter((s) => !remove.includes(s.name) && !set.some((n) => n.name === s.name)), ...set];
    const tagsTo = tagsChanged ? w.tags! : h.tags;
    setWaiting(NOTHING);
    setHeld({ scores: scoresTo, tags: tagsTo });
    const words: ReactNode[] = [];
    if (set.length || remove.length) {
      const a = await post<{ decision: string }>("/api/texts/scores", { texts: [{ id: t.id, scores: set, remove }], declaration });
      if (!a.ok) {
        setHeld(h);
        failed(<>Not saved: {[...set.map((s) => `${s.name} ${shortValue(s.value)}`), ...remove.map((n) => `${n} removed`), ...(tagsChanged ? ["the tags"] : [])].join(", ")} on {t.id}: {a.problems.join(" ")}</>);
        return;
      }
      words.push(<span key="s">scores {[...set.map((s) => `${s.name} ${shortValue(s.value)}`), ...remove.map((n) => `${n} removed`)].join(", ")} (<a href={href(["decisions", a.data.decision])}>{a.data.decision}</a>)</span>);
    }
    if (tagsChanged) {
      const a = await post<{ decision: string }>(`/api/texts/${encodeURIComponent(t.id)}/credits`, { tags: tagsTo, declaration });
      if (!a.ok) {
        setHeld((now) => ({ ...now, tags: h.tags }));
        failed(<>{words.length > 0 && <>Saved {words}; </>}not saved: the tags of {t.id}: {a.problems.join(" ")}</>);
        return;
      }
      const added = tagsTo.filter((x) => !h.tags.includes(x)), gone = h.tags.filter((x) => !tagsTo.includes(x));
      words.push(<span key="t">tags {[...added.map((x) => `+${x}`), ...gone.map((x) => `−${x}`)].join(" ")} (<a href={href(["decisions", a.data.decision])}>{a.data.decision}</a>)</span>);
    }
    setProblem(null);
    state.current.onSaved(<>Saved, {words.map((w, i) => <span key={i}>{i > 0 && "; "}{w}</span>)}.</>);
  }, []);
  const pending = Object.keys(waiting.scores).length > 0 || waiting.tags !== null;
  useEffect(() => {
    if (!pending) return;
    const timer = window.setTimeout(send, WAIT);
    // a reload or a closed tab within the wait asks first, since what waits would be lost
    const leaving = (e: BeforeUnloadEvent) => { e.preventDefault(); };
    window.addEventListener("beforeunload", leaving);
    return () => { window.clearTimeout(timer); window.removeEventListener("beforeunload", leaving); };
  }, [waiting, pending, send]);
  useEffect(() => () => { void send(); }, [send]); // leaving the text sends what waits

  const setScore = (name: string, value: number | null) => {
    setWaiting((w) => ({ ...w, scores: { ...w.scores, [name]: value } }));
    if (value !== null) {
      keepRecent(name, value);
      setRecent(readRecent());
      const scale = scales.get(name);
      if (scale && (value < scale.low || value > scale.high)) forgetScale(name);
    }
  };
  const setTags = (next: string[]) => setWaiting((w) => ({ ...w, tags: next }));
  const toggleTag = (tag: string) => setTags(tagsNow.includes(tag) ? tagsNow.filter((t) => t !== tag) : [...tagsNow, tag]);
  const apply = (p: Preset) => {
    setWaiting((w) => ({ scores: { ...w.scores, ...p.scores }, tags: [...new Set([...(w.tags ?? held.tags).filter((t) => !p.remove.includes(t)), ...p.add])] }));
    Object.entries(p.scores).forEach(([name, v]) => keepRecent(name, v));
    setRecent(readRecent());
  };

  const [pad, setPad] = useState<{ name: string; top: number; left: number } | null>(null);
  const [typing, setTyping] = useState<"tag" | "score" | "preset" | null>(null);
  const others = tagsNow.filter((t) => !shownTags.includes(t));
  const ownPreset = (): Preset => ({
    name: "", scores: Object.fromEntries(allNames.flatMap((n) => { const v = scoreNow(n); return v === null ? [] : [[n, v]]; })),
    add: [...tagsNow], remove: [],
  });
  const declaredHere = declared(who);

  return (
    <aside className="review-panel small" aria-label="Review the open text">
      <section className="panel-scores" aria-label="Scores" onScroll={() => setPad(null)}>
        {allNames.map((name) => (
          <ScoreRow key={name} name={name} value={scoreNow(name)} waiting={name in waiting.scores} recent={recent[name] ?? null} scale={scales.get(name) ?? null}
            readOnly={readOnly} onSet={(v) => setScore(name, v)} padOpen={pad?.name === name}
            onPad={(el) => { if (pad?.name === name) { setPad(null); return; } const r = el.getBoundingClientRect(); setPad({ name, top: r.top, left: r.left }); }} />
        ))}
        {!allNames.length && <p className="aside">No score yet: add one below.</p>}
      </section>
      <section className="panel-tags" aria-label="Tags">
        <div className="tag-grid">
          {shownTags.map((tag) => (
            <button key={tag} type="button" className={`choice wic-focus${tagsNow.includes(tag) ? " on" : ""}${waiting.tags && waiting.tags.includes(tag) !== held.tags.includes(tag) ? " waiting" : ""}`}
              aria-pressed={tagsNow.includes(tag)} disabled={readOnly} onClick={() => toggleTag(tag)} title={tagsNow.includes(tag) ? `Take the tag ${tag} away` : `Tag it ${tag}`}>{tag}</button>
          ))}
        </div>
        <div className="panel-line">
          {others.length > 0 && <span>also {others.map((t) => (
            <span key={t} className={`also${waiting.tags && !held.tags.includes(t) ? " waiting" : ""}`}>{t}{!readOnly && <button type="button" className="forget wic-focus" aria-label={`Take the tag ${t} away`} onClick={() => toggleTag(t)}>✕</button>}</span>
          ))}</span>}
          {!readOnly && <>
            {typing === "tag" ? <Typed label="A new tag" onDone={(t) => { setTyping(null); if (t && !tagsNow.includes(t)) setTags([...tagsNow, t]); }} />
              : <button type="button" className="link-button wic-focus" onClick={() => setTyping("tag")}>+ tag</button>}
            {typing === "score" ? <Typed label="A new score's name" onDone={(n) => { setTyping(null); if (n) { addedNames.add(n); setExtraNames([...extraNames, n]); } }} />
              : <button type="button" className="link-button wic-focus" onClick={() => setTyping("score")}>+ score</button>}
          </>}
        </div>
      </section>
      <section className="panel-actions" aria-label="Move and decide">
        {actions}
        <p className={`panel-note${pending ? " waiting-note" : ""}`} role="status">
          {problem ?? (pending ? "Saving in a moment…" : declaredHere ? <>Decides and saves in public as {who.name}{about && <> (<a href={about.requirements} target="_blank" rel="noreferrer">requirements</a>)</>}</>
            : <>No name and email: <a href={settingsHref()}>set them</a> to save and decide.</>)}
        </p>
      </section>
      {!readOnly && (
        <section className="panel-presets" aria-label="Presets">
          {presets.map((p) => (
            <span key={p.name} className="preset">
              <button type="button" className="choice wic-focus" onClick={() => apply(p)} title={`Sets ${presetWords(p)}`}>
                <span className="preset-name">{p.name}</span><span className="preset-words">{presetWords(p) || "nothing"}</span>
              </button>
              <button type="button" className="forget wic-focus" aria-label={`Forget the preset ${p.name}`} title={`Forget the preset ${p.name}`} onClick={() => forgetPreset(p.name)}>✕</button>
            </span>
          ))}
          {typing === "preset" ? <Typed label="The preset's name" onDone={(n) => { setTyping(null); if (n && !savePreset({ ...ownPreset(), name: n })) setProblem("This browser keeps nothing, so the preset is not kept."); }} />
            : <button type="button" className="choice add-preset wic-focus" onClick={() => setTyping("preset")} title="Keep this text's scores and tags as a preset">+ preset from this text</button>}
        </section>
      )}
      <section className="panel-facts" aria-label="Facts">
        <dl className="mini-facts">
          <dt>license</dt><dd className="ledger">{text.licenses.map((l) => <a key={l} href={href(["licenses", l])}>{l}</a>).reduce<ReactNode[]>((all, one, i) => (i ? [...all, ", ", one] : [one]), [])}</dd>
          {text.authors?.length ? <><dt>authors</dt><dd>{text.authors.join(", ")}</dd></> : null}
          {text.address && <><dt>address</dt><dd className="ledger cut" title={text.address}>{/^https?:\/\//.test(text.address) ? <a href={text.address} target="_blank" rel="noreferrer">{text.address}</a> : text.address}</dd></>}
          <dt>decisions</dt><dd><button type="button" className="link-button wic-focus" onClick={onRecord} title="Its whole record">{status}, {plural(decisions, "decision")}</button></dd>
          <dt>libraries</dt><dd>{libraries.length ? libraries.map((l) => l.name).join(", ") : "none"}</dd>
          <dt>id</dt><dd className="ledger">{text.id} · revision {text.revision}</dd>
        </dl>
        {more}
      </section>
      {pad && <NumberPad name={pad.name} at={pad} scale={scales.get(pad.name) ?? null} value={scoreNow(pad.name)}
        onDone={(v) => { setPad(null); if (v !== undefined) setScore(pad.name, v); }} />}
    </aside>
  );
}

// One score: its choices, − and +, and its value, which opens the pad. A choice clicked again takes the score away.
function ScoreRow({ name, value, waiting, recent, scale, readOnly, onSet, onPad, padOpen }: {
  name: string; value: number | null; waiting: boolean; recent: number | null; scale: Scale | null; readOnly: boolean;
  onSet: (v: number | null) => void; onPad: (el: HTMLElement) => void; padOpen: boolean;
}) {
  const step = (by: 1 | -1) => {
    if (!scale) return;
    const from = value ?? recent ?? (by > 0 ? scale.low - scale.step : scale.high + scale.step);
    onSet(round(Math.min(scale.high, Math.max(scale.low, from + by * scale.step))));
  };
  return (
    <div className="score-row">
      <span className="score-name" title={name}>{name}</span>
      <span className="score-choices">
        {(scale?.values ?? []).map((v) => (
          <button key={v} type="button" disabled={readOnly}
            className={`choice wic-focus${value !== null && round(value) === round(v) ? " on" : ""}${recent !== null && round(recent) === round(v) ? " recent" : ""}${waiting && value !== null && round(value) === round(v) ? " waiting" : ""}`}
            aria-label={`${name} ${shortValue(v)}`} aria-pressed={value !== null && round(value) === round(v)} onClick={() => onSet(value !== null && round(value) === round(v) ? null : v)}>{shortValue(v)}</button>
        ))}
      </span>
      <span className="score-nudge">
        <button type="button" className="choice wic-focus" disabled={readOnly || !scale} aria-label={`${name} down by ${scale ? shortValue(scale.step) : ""}`} onClick={() => step(-1)}>−</button>
        <button type="button" className={`choice value wic-focus${padOpen ? " on" : ""}${waiting ? " waiting" : ""}`} disabled={readOnly} aria-label={`${name}: ${value === null ? "none" : shortValue(value)}; type a value`}
          title="Type a value with the pad" onClick={(e) => onPad(e.currentTarget)}>{value === null ? "⌨" : shortValue(value)}</button>
        <button type="button" className="choice wic-focus" disabled={readOnly || !scale} aria-label={`${name} up by ${scale ? shortValue(scale.step) : ""}`} onClick={() => step(1)}>+</button>
      </span>
    </div>
  );
}

const KEYS = ["7", "8", "9", "⌫", "4", "5", "6", ".", "1", "2", "3", "✓", "0", "−", "none", "✕"];

// A number pad beside the panel, at the height of the value it types, over the text: nothing in the panel moves.
// ✓ sets what was typed, "none" takes the score away, ✕ or Escape closes it; the keyboard's digits work too.
function NumberPad({ name, at, scale, value, onDone }: { name: string; at: { top: number; left: number }; scale: Scale | null; value: number | null; onDone: (v?: number | null) => void }) {
  const [typed, setShown] = useState("");
  // what was typed so far, kept outside the render so presses quicker than it build on each other
  const now = useRef("");
  const setTyped = (t: string) => { now.current = t; setShown(t); };
  const press = (k: string) => {
    const t = now.current;
    if (k === "✕") return onDone();
    if (k === "none") return onDone(null);
    if (k === "✓") return t !== "" && Number.isFinite(Number(t)) ? onDone(Number(t)) : onDone();
    if (k === "⌫") return setTyped(t.slice(0, -1));
    if (k === "−") return setTyped(t.startsWith("-") ? t.slice(1) : `-${t}`);
    if (k === ".") return setTyped(t.includes(".") ? t : `${t || "0"}.`);
    setTyped(t + k);
  };
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => { box.current?.focus(); }, []);
  const outside = typed !== "" && scale && Number.isFinite(Number(typed)) && (Number(typed) < scale.low || Number(typed) > scale.high);
  return (
    <div ref={box} className="number-pad" role="dialog" aria-label={`Type the value of ${name}`} tabIndex={-1}
      style={{ top: Math.max(8, at.top - 8), left: Math.max(8, at.left - 220) }}
      onKeyDown={(e) => {
        const map: Record<string, string> = { Enter: "✓", Escape: "✕", Backspace: "⌫", "-": "−", ",": "." };
        const k = map[e.key] ?? (/^[0-9.]$/.test(e.key) ? e.key : null);
        if (k) { e.preventDefault(); e.stopPropagation(); press(k); }
      }}>
      <div className="pad-head"><span>{name}</span><span className="ledger">{typed || (value === null ? "…" : shortValue(value))}</span></div>
      <div className="pad-keys">
        {KEYS.map((k) => <button key={k} type="button" className={`choice wic-focus${k === "✓" ? " ok" : ""}`} onClick={() => press(k)}>{k}</button>)}
      </div>
      {scale && <p className={`pad-range${outside ? " act" : ""}`}>{outside ? "Outside " : "From "}{shortValue(scale.low)} to {shortValue(scale.high)}; − and + step {shortValue(scale.step)}</p>}
    </div>
  );
}

// One line typed: Enter keeps it, Escape or leaving the box drops it.
function Typed({ label, onDone }: { label: string; onDone: (value: string | null) => void }) {
  const [value, setValue] = useState("");
  return (
    <input className="wic-input wic-focus typed" autoFocus aria-label={label} placeholder={label} value={value}
      onChange={(e) => setValue(e.target.value)} onBlur={() => onDone(null)}
      onKeyDown={(e) => {
        if (e.key === "Enter") { e.preventDefault(); onDone(value.trim() || null); }
        else if (e.key === "Escape") { e.preventDefault(); onDone(null); }
      }} />
  );
}
