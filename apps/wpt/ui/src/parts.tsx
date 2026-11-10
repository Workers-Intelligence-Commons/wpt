// Pieces every page uses: a page's head and its sections, the markers of
// form fields, what an operation answered, a status, a pager, a value cut to
// one line, the libraries, and the confirm of every decision.

import { Badge, Button, type ButtonVariant } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useId, useState, type ReactNode } from "react";
import { useAbout } from "./about";
import { get, type Answer } from "./api";
import { declared, useDeclarant } from "./declarant";
import { href } from "./route";

// A page's head: its one title, a line of facts (meta), one short line that
// places the page in the chain of the depot with links to its neighbours
// (chain), and its actions, the main one emphasized and the others quiet,
// with a quiet note about them (note) right under them.
export function PageHead({ title, meta, chain, actions, note }: { title: ReactNode; meta?: ReactNode; chain?: ReactNode; actions?: ReactNode; note?: ReactNode }) {
  return (
    <header className="page-head">
      <div className="page-head-main">
        <h1 className="title page-title">{title}</h1>
        {meta && <div className="page-meta small">{meta}</div>}
        {chain && <p className="chain small">{chain}</p>}
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

// A section of a page under a rule: its title, optional tools beside the
// title (what acts on the whole section), then its body. Without a title it
// is only the rule that sets the body apart.
export function Section({ title, tools, children, className }: { title?: ReactNode; tools?: ReactNode; children: ReactNode; className?: string }) {
  const id = useId();
  return (
    <section className={`section ${className ?? ""}`} aria-labelledby={title ? id : undefined}>
      {(title || tools) && (
        <div className="section-head">
          {title && <h2 id={id} className="cue section-title">{title}</h2>}
          {tools}
        </div>
      )}
      {children}
    </section>
  );
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

// The depot's own messages for a refused operation.
export function Problems({ problems }: { problems: string[] }) {
  if (problems.length === 0) return null;
  return (
    <div className="refusal cue" role="alert">
      {problems.map((p) => (
        <p key={p}>
          <span aria-hidden="true">✕ </span>
          {p}
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

// A value on one line, cut with an ellipsis when it is too long, whole on hover.
export function Cut({ text, className }: { text: string; className?: string }) {
  return <span className={`cut ${className ?? ""}`} title={text}>{text}</span>;
}

export type Problem = { texts: string[]; problem: string; decision: string };
// upstream: it belongs to the upstream this depot follows, which alone changes it
export type Library = { id: string; name: string; forked_from: string | null; problems: Problem[] | null; locked_at: string | null; texts: number; upstream: boolean };

// Every library, read again by reload.
export function useLibraries(): { libraries: Library[]; loaded: boolean; problems: string[]; reload: () => void } {
  const [libraries, setLibraries] = useState<Library[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [problems, setProblems] = useState<string[]>([]);
  const reload = useCallback(() => {
    get<Library[]>("/api/libraries").then((a) => {
      setLibraries(a.ok ? a.data : []);
      setProblems(a.ok ? [] : a.problems);
      setLoaded(true);
    });
  }, []);
  useEffect(reload, [reload]);
  return { libraries, loaded, problems, reload };
}

// A library's name, linked to its page, and the word "locked" when it is.
export function LibraryLink({ library }: { library: Library | { id: string; name?: string; locked_at?: string | null } }) {
  return (
    <span>
      <a href={href(["libraries", library.id])}>{library.name ?? library.id}</a>
      {library.locked_at ? <span className="aside"> locked</span> : null}
    </span>
  );
}

export type DatasetRef = { depot: string | null; library: string; change: number };
export type Dataset = { id: string; revision: number; name: string; tokenizer: string; libraries: DatasetRef[]; shares: { tag: string; share: number }[] | null; upstream: boolean; here: string[] };

// Every dataset, read again by reload.
export function useDatasets(): { datasets: Answer<Dataset[]> | null; reload: () => void } {
  const [datasets, setDatasets] = useState<Answer<Dataset[]> | null>(null);
  const reload = useCallback(() => { get<Dataset[]>("/api/datasets").then(setDatasets); }, []);
  useEffect(reload, [reload]);
  return { datasets, reload };
}

// Whether a dataset reads its library ref here, as the server answers it
// (here: a library this depot holds, named by this depot's address in any
// form, or by the upstream's for a library of the upstream it follows).
export const local = (dataset: Dataset, ref: DatasetRef) => dataset.here.includes(ref.library);

// The datasets that read a library of this depot, by this depot's address
// or, for a library of the upstream it follows, by the upstream's (the
// server's here).
export const readers = (datasets: Dataset[], library: string) => datasets.filter((d) => d.here.includes(library));

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
          <p>Recorded as {who.name}, <span className="ledger">{who.email}</span>. <a href={href(["settings"])}>Change</a></p>
        ) : (
          <p className="act"><strong>No legal name and email are set in this browser.</strong> <a href={href(["settings"])}>Set them in the settings</a> to decide.</p>
        )}
      </div>
    </div>
  );
}

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
