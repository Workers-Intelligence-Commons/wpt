// Pieces every page uses: a page's head and its sections, the markers of
// form fields, what an operation answered, a status, the pagers, a value cut to
// one line, and the confirm of every decision.

import { Badge, Button, type ButtonVariant } from "@workers-intelligence-commons/components";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { useAbout } from "./about";
import type { Answer } from "./api";
import { declared, useDeclarant } from "./declarant";
import { href } from "./route";

// A page's head: its one title, with what the page is for behind a "?"
// beside it (chain: the explanation and the links to its neighbours, never
// shown until asked for), a line of facts (meta), and its actions, the main
// one emphasized and the others quiet, with a short note about them (note)
// right under them.
export function PageHead({ title, meta, chain, actions, note }: { title: ReactNode; meta?: ReactNode; chain?: ReactNode; actions?: ReactNode; note?: ReactNode }) {
  return (
    <header className="page-head">
      <div className="page-head-main">
        <div className="title-row">
          <h1 className="title page-title">{title}</h1>
          {chain && <Help label="this page">{chain}</Help>}
        </div>
        {meta && <div className="page-meta small">{meta}</div>}
      </div>
      {(actions || note) && (
        <div className="head-actions">
          {actions && <div className="head-buttons">{actions}</div>}
          {note && <p className="head-note small">{note}</p>}
        </div>
      )}
    </header>
  );
}

// A block of a page: its title, the explanation behind a "?" beside it
// (help), optional tools beside the title (what acts on the whole block),
// then its body. Without a title it is only the block that sets the body
// apart.
export function Section({ title, help, tools, children, className }: { title?: ReactNode; help?: ReactNode; tools?: ReactNode; children: ReactNode; className?: string }) {
  const id = useId();
  return (
    <section className={`section ${className ?? ""}`} aria-labelledby={title ? id : undefined}>
      {(title || tools) && (
        <div className="section-head">
          {title && (
            <div className="title-row">
              <h2 id={id} className="cue section-title">{title}</h2>
              {help && <Help label={typeof title === "string" ? title : "this part"}>{help}</Help>}
            </div>
          )}
          {tools}
        </div>
      )}
      {children}
    </section>
  );
}

// A small "?" that opens an explanation under it and closes on a second
// click, on Escape (which then closes nothing else, such as the dialog
// around it), on a click elsewhere or when the focus leaves it. label: what
// it explains, for the button's name ("About Most tokens"). It sits beside a
// heading or in a line of text, never inside a heading or a field's label,
// whose name would take its words; a field's explanation is its hint. It
// opens toward the side of the screen with room.
export function Help({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState<null | "left" | "right">(null);
  const ref = useRef<HTMLSpanElement>(null);
  const id = useId();
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(null); };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [open]);
  const toggle = () => {
    const box = ref.current?.getBoundingClientRect();
    setOpen(open ? null : box && box.left > window.innerWidth / 2 ? "right" : "left");
  };
  return (
    <span className="help" ref={ref} onBlur={(e) => { if (open && e.relatedTarget && !ref.current?.contains(e.relatedTarget)) setOpen(null); }} onKeyDown={(e) => { if (open && e.key === "Escape") { e.preventDefault(); e.stopPropagation(); setOpen(null); } }}>
      <button type="button" className="help-button wic-focus" aria-expanded={Boolean(open)} aria-controls={open ? id : undefined} aria-label={`About ${label}`} title={`About ${label}`} onClick={toggle}>?</button>
      {open && <span id={id} role="note" className={`help-pop small help-${open}`}>{children}</span>}
    </span>
  );
}

// The less used actions behind one button (label), as a list that opens
// under it toward side, and closes again once one is chosen or a click lands
// elsewhere.
export function MoreMenu({ label, side = "left", children }: { label: string; side?: "left" | "right"; children: ReactNode }) {
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const away = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) ref.current.removeAttribute("open"); };
    document.addEventListener("click", away);
    return () => document.removeEventListener("click", away);
  }, []);
  return (
    <details className="more-menu" ref={ref}>
      <summary className="wic-focus small">{label}</summary>
      <div className={`more-items more-${side}`} onClick={() => ref.current?.removeAttribute("open")}>{children}</div>
    </details>
  );
}

// A number set big, with what it counts under it: the figures a page leads with.
export function Stat({ value, label, href: to }: { value: ReactNode; label: ReactNode; href?: string }) {
  const body = <><span className="stat-value">{value}</span><span className="stat-label small">{label}</span></>;
  return to ? <a className="stat" href={to}>{body}</a> : <div className="stat">{body}</div>;
}

// The markers of form fields, the same in every form: a required field's
// label ends with a red star, hidden from assistive technology since the
// control itself is marked required; an optional field says so in its label;
// a field shown only in some case names that case. Each form shows the note
// once.
export const required = (label: ReactNode) => <>{label}<span className="required-mark" aria-hidden="true">*</span></>;
export const optional = (label: string) => `${label} (optional)`;
export const only = (label: string, when: string) => `${label} (only ${when})`;
export function RequiredNote() {
  return <p className="form-note small" aria-hidden="true"><span className="required-mark">*</span>required</p>;
}
// The legend of a group of ticks that needs one ticked: the star for the
// eye, the word for assistive technology, which a group cannot be marked
// required for.
export const requiredGroup = (label: string) => <>{label}<span className="required-mark" aria-hidden="true">*</span><span className="visually-hidden"> (required)</span></>;

export type Status = "under review" | "admitted" | "refused";
export const STATUSES: Status[] = ["under review", "admitted", "refused"];

// A text's or an erasure's status: its glyph and its word, never colour alone.
export function StatusBadge({ status }: { status: Status }) {
  return status === "under review" ? <Badge status="pending">under review</Badge> : <Badge status={status} />;
}

// A message as a sentence: its first letter a capital, a full stop at its end.
export const sentence = (text: string) => {
  const t = text.trim();
  return t ? `${t[0].toUpperCase()}${t.slice(1)}${/[.!?:]$/.test(t) ? "" : "."}` : t;
};

// One of the depot's messages in plain words. The server names the kind of
// failure before each message ("BadQuery: SQLError: near ..."); the kind
// says what to do about some of them, and is otherwise left out.
export function plainProblem(problem: string): string {
  const found = /^([A-Z][A-Za-z]+): ([\s\S]*)$/.exec(problem);
  if (!found) return sentence(problem);
  const [, kind, rest] = found;
  const sql = /^SQLError: ([\s\S]*)$/.exec(rest);
  if (sql) return `This SQL condition is not valid (${sql[1].trim()}). Correct it, or clear it.`;
  if (kind === "TimeLimitReached") return `That took longer than the server allows: ${rest.trim()} Narrow the filter or the query.`;
  if (kind === "NoReaderFree") return "The server is busy reading for other requests: try again in a moment.";
  if (kind === "Busy") return `${sentence(rest)} Wait for it on the Jobs page.`;
  return sentence(rest);
}

// The depot's own messages for a refused operation, in plain words, each
// whole as the server gave it on hover.
export function Problems({ problems }: { problems: string[] }) {
  if (problems.length === 0) return null;
  return (
    <div className="refusal cue" role="alert">
      {problems.map((p) => (
        <p key={p} title={p}>
          <span aria-hidden="true">✕ </span>
          {plainProblem(p)}
        </p>
      ))}
    </div>
  );
}

// What an operation answered: its problems, or the words given for success.
export function Outcome<T>({ answer, done }: { answer: Answer<T> | null; done: (data: T) => ReactNode }) {
  if (!answer) return null;
  return answer.ok ? <p className="done cue" role="status">{done(answer.data)}</p> : <Problems problems={answer.problems} />;
}

export const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString("en")} ${n === 1 ? one : many}`;

// what a build left out, as its record counts it (docs/data-selection.md, "Conversations that quote texts")
export type LeftOut = { "quoted text left out"?: number; "quote unmatched"?: number; "kinds not taken"?: Record<string, number> };

// The conversations a build left out over their quotes, and the texts of the kinds no step reads, in words; null when none
export const leftOutWords = (l: LeftOut | null | undefined): string | null => {
  if (!l) return null;
  const said = [];
  if (l["quoted text left out"]) said.push(`${plural(l["quoted text left out"], "conversation")} quoting a text the corpus does not hold`);
  if (l["quote unmatched"]) said.push(`${plural(l["quote unmatched"], "conversation")} whose quote matches no single place of the text`);
  const kinds = Object.entries(l["kinds not taken"] ?? {}).filter(([, n]) => n > 0);
  if (kinds.length) said.push(`${kinds.map(([kind, n]) => plural(n, `text of kind ${kind}`, `texts of kind ${kind}`)).join(", ")}, which no training step reads`);
  return said.length ? said.join("; ") : null;
};

// What a corpus took, counted one way everywhere: its plain texts, then its
// conversations and tasks, from texts (every text and item it took), and of
// them its conversations and tasks.
export function takenWords(texts: number, conversations = 0, tasks = 0) {
  const items = [conversations ? plural(conversations, "conversation") : null, tasks ? plural(tasks, "task") : null].filter(Boolean).join(" and ");
  return `${plural(Math.max(0, texts - conversations - tasks), "plain text")}${items ? ` and ${items}` : ""}`;
}

// The kind of a text read as it is (wpt_depot.kinds): every other kind is
// an item's, JSON that matches its kind's schema.
export const PLAIN = "plain";

// A text's name as every page gives it: its title, else for an item its
// kind with an article ("a conversation"), else "an untitled text".
export const textName = (r: { title: string | null; kind: string | null }) =>
  r.title ?? (r.kind && r.kind !== PLAIN ? `${/^[aeiou]/.test(r.kind) ? "an" : "a"} ${r.kind}` : "an untitled text");

export const SIZES = [25, 50, 100, 200];

// Where a list of rows stands, the page to jump to, and, when onLimit is
// given, how many rows a page holds.
export function Pager({ offset, limit, total, onOffset, onLimit, label }: { offset: number; limit: number; total: number; onOffset: (offset: number) => void; onLimit?: (limit: number) => void; label: string }) {
  const pages = Math.max(1, Math.ceil(total / limit));
  const page = Math.floor(offset / limit) + 1;
  const [typed, setTyped] = useState(String(page));
  const id = useId();
  useEffect(() => setTyped(String(page)), [page]);
  if (total === 0) return null;
  const go = (n: number) => onOffset((Math.min(pages, Math.max(1, n)) - 1) * limit);
  return (
    <nav className="pager small" aria-label={label}>
      <span className="aside">{(offset + 1).toLocaleString("en")} to {Math.min(offset + limit, total).toLocaleString("en")} of {total.toLocaleString("en")}</span>
      {pages > 1 && (
        <span className="row tight">
          <Button variant="quiet" disabled={page === 1} onClick={() => go(page - 1)}>Previous</Button>
          <form className="row tight" onSubmit={(e) => { e.preventDefault(); go(Number(typed) || 1); }}>
            <label className="aside" htmlFor={`${id}-page`}>page</label>
            <input id={`${id}-page`} className="wic-input wic-focus page-number" type="number" min={1} max={pages} value={typed}
              onChange={(e) => setTyped(e.target.value)} onBlur={() => go(Number(typed) || 1)} />
            <span className="aside">of {pages.toLocaleString("en")}</span>
          </form>
          <Button variant="quiet" disabled={page === pages} onClick={() => go(page + 1)}>Next</Button>
        </span>
      )}
      {onLimit && (
        <span className="row tight">
          <label className="aside" htmlFor={`${id}-size`}>rows per page</label>
          <select id={`${id}-size`} className="wic-input wic-select wic-focus page-size" value={limit} onChange={(e) => onLimit(Number(e.target.value))}>
            {SIZES.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </span>
      )}
    </nav>
  );
}

// Where a list read one page after another stands (filter.tsx: a page of
// texts starts after a row, not at a number): what is known of its count
// (counted), the first page, the page before and the page after, each when
// there is one, and, when onLimit is given, how many rows a page holds.
export function PlacePager({ onFirst, onPrevious, onNext, limit, onLimit, label, counted }: {
  onFirst: () => void; onPrevious: (() => void) | null; onNext: (() => void) | null; limit: number;
  onLimit?: (limit: number) => void; label: string; counted?: ReactNode;
}) {
  const id = useId();
  return (
    <nav className="pager small" aria-label={label}>
      {counted && <span className="aside">{counted}</span>}
      {(onPrevious || onNext) && (
        <span className="row tight">
          <Button variant="quiet" disabled={!onPrevious} onClick={onFirst}>First</Button>
          <Button variant="quiet" disabled={!onPrevious} onClick={() => onPrevious?.()}>Previous</Button>
          <Button variant="quiet" disabled={!onNext} onClick={() => onNext?.()}>Next</Button>
        </span>
      )}
      {onLimit && (
        <span className="row tight">
          <label className="aside" htmlFor={`${id}-size`}>rows per page</label>
          <select id={`${id}-size`} className="wic-input wic-select wic-focus page-size" value={limit} onChange={(e) => onLimit(Number(e.target.value))}>
            {SIZES.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </span>
      )}
    </nav>
  );
}

// A value on one line, cut with an ellipsis when it is too long, whole on hover.
export function Cut({ text, className }: { text: string; className?: string }) {
  return <span className={`cut ${className ?? ""}`} title={text}>{text}</span>;
}


export type Declaration = { name: string; email: string; accepted: true };

// The end of every form that sends a decision: one button with a short
// label ("Confirm and admit"), and beside it, once, the full confirmation
// sentence the server gives, the link to the requirements and the name and
// email the decision records. Only a click on that button sends the
// decision, with accepted true. Without a name or an email, or before the
// server's confirmation is read, the button is off and says why.
export function Confirm({ label, disabled, onConfirm, variant = "primary", note }: { label: string; disabled?: boolean; onConfirm: (declaration: Declaration) => void; variant?: ButtonVariant; note?: ReactNode }) {
  const { about, problems } = useAbout();
  const who = useDeclarant();
  const ready = declared(who);
  return (
    <div className="confirm-area">
      <Button variant={variant} disabled={!ready || !about || disabled}
        onClick={() => about && ready && onConfirm({ name: who.name.trim(), email: who.email.trim(), accepted: true })}>
        {label}
      </Button>
      <div className="confirm-words small">
        {note && <p className="act">{note}</p>}
        {about ? (
          <p>
            By clicking you confirm: <q>{about.confirmation}</q> <a href={about.requirements} target="_blank" rel="noreferrer">The requirements</a>.
          </p>
        ) : problems.length > 0 ? (
          <Problems problems={["The server's confirmation could not be read, so no decision can be sent.", ...problems]} />
        ) : <p className="aside">Reading the confirmation</p>}
        {ready ? (
          <p>Recorded as {who.name}, <span className="ledger">{who.email}</span>. <a href={settingsHref()}>Change</a></p>
        ) : (
          <p className="act"><strong>No name and email set.</strong> <a href={settingsHref()}>Set them</a> to decide.</p>
        )}
      </div>
    </div>
  );
}

// The address of Settings, carrying the page it was opened from, which it links back to once saved.
export const settingsHref = () => href(["settings"], { back: window.location.hash || null });

// Facts as a list of italic labels and their values; empty values are left out.
export function Facts({ rows }: { rows: [string, ReactNode][] }) {
  const shown = rows.filter(([, value]) => value !== null && value !== undefined && value !== "" && !(Array.isArray(value) && value.length === 0));
  return (
    <dl className="facts">
      {shown.map(([label, value]) => [<dt key={`${label}-t`} className="aside">{label}</dt>, <dd key={`${label}-d`}>{value}</dd>])}
    </dl>
  );
}

// Several values separated by commas, each as given.
export function Joined({ values, render }: { values: string[]; render?: (value: string) => ReactNode }) {
  return (
    <>
      {values.map((v, i) => (
        <span key={v}>
          {i > 0 && ", "}
          {render ? render(v) : v}
        </span>
      ))}
    </>
  );
}
