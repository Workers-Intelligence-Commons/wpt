// Pieces every page uses: a page's head and its sections, the markers of
// form fields, what an operation answered, a status, the pagers, a value cut to
// one line, the libraries, and the confirm of every decision.

import { Badge, Button, type ButtonVariant } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useId, useState, type ReactNode } from "react";
import { useAbout } from "./about";
import { get, part, post, type Answer } from "./api";
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

// What a corpus took, counted one way everywhere: its plain texts, then its
// conversations and tasks, from texts (every text and item it took), and of
// them its conversations and tasks; held: how many conversations and tasks
// the dataset holds, when known, so a small corpus says it took some of them.
export function takenWords(texts: number, conversations = 0, tasks = 0, held: number | null = null) {
  const items = [conversations ? plural(conversations, "conversation") : null, tasks ? plural(tasks, "task") : null].filter(Boolean).join(" and ");
  const of = held !== null && held > conversations + tasks ? ` (of the dataset's ${held.toLocaleString("en")})` : "";
  return `${plural(Math.max(0, texts - conversations - tasks), "plain text")}${items ? ` and ${items}${of}` : ""}`;
}

// A text's name as every page gives it: its title, else its kind with an
// article ("a conversation"), else "an untitled text".
export const textName = (r: { title: string | null; kind: string | null }) =>
  r.title ?? (r.kind ? `${/^[aeiou]/.test(r.kind) ? "an" : "a"} ${r.kind}` : "an untitled text");

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


export type Problem = { texts: string[]; problem: string; decision: string };
// upstream: it belongs to the upstream this depot follows, which alone changes it
export type Library = { id: string; name: string; forked_from: string | null; problems: Problem[] | null; locked_at: string | null; upstream: boolean };

// A library's size (GET /api/libraries/ID/size, from the depot's counts,
// which it keeps counted): how many texts it holds; version: raised after a
// change, so it is read again.
export function LibrarySize({ id, version = 0 }: { id: string; version?: number }) {
  const [size, setSize] = useState<Answer<{ texts: number }> | null>(null);
  useEffect(() => { get<{ texts: number }>(`/api/libraries/${encodeURIComponent(id)}/size`).then(setSize); }, [id, version]);
  if (!size) return <span className="aside">counting its texts</span>;
  if (!size.ok) return <span className="not-allowed">{size.problems.join(" ")}</span>;
  return <span>{plural(size.data.texts, "text")}</span>;
}

// What libraries hold now (POST /api/libraries/measure, computed by the
// server from the index of their memberships), each text once: every text,
// the plain texts, the conversations and tasks (items) and the items of
// other kinds, each in texts, characters and bytes; with shares, the plain
// texts each share takes (the first share whose tag a text has) and the mix
// of the corpus of every token (what each share keeps, the share that
// limits the others, the share of the bytes the conversations and tasks
// hold then).
export type Held = { texts: number; characters: number; bytes: number };
export type ShareHeld = Held & { tag: string; share?: number; kept_bytes?: number; kept_characters?: number; limits?: boolean };
export type Measure = Held & {
  plain: Held; items: Held; other: Held; shares: ShareHeld[]; unshared: Held | null;
  mix: { kept_bytes: number; kept_characters: number; items_share: number } | null;
};

// The measure of libraries, with shares when given: null while it is read,
// the measure once read, or the problem in words. It reads every membership
// of the libraries, so it is read once for each set of libraries and shares,
// never again after a change. Nothing is read without a library.
export function useMeasure(libraries: string[], shares: { tag: string; share: number }[] | null = null): Measure | string | null {
  const [found, setFound] = useState<{ key: string; value: Measure | string } | null>(null);
  const key = JSON.stringify([libraries, shares]);
  useEffect(() => {
    if (!libraries.length) return;
    let live = true;
    post<Measure>("/api/libraries/measure", { libraries, ...(shares?.length ? { shares } : {}) })
      .then((a) => live && setFound({ key, value: a.ok ? a.data : plainProblem(a.problems[0] ?? "") }));
    return () => { live = false; };
  }, [key]);
  return found?.key === key ? found.value : null;
}

// How many characters the texts of libraries hold now (useMeasure): null
// while it is read, a number once read, or the problem in words.
export function useCharacters(libraries: string[]): number | string | null {
  const found = useMeasure(libraries);
  return found === null || typeof found === "string" ? found : found.characters;
}

// A number of characters in words: "about 5.4 million characters".
export const characters = (n: number) => n < 10_000 ? plural(n, "character")
  : n < 1e6 ? `about ${Math.round(n / 1000).toLocaleString("en")} thousand characters`
  : n < 1e9 ? `about ${(n / 1e6).toFixed(n < 1e7 ? 1 : 0)} million characters` : `about ${(n / 1e9).toFixed(1)} billion characters`;

// What useCharacters found, in words.
export function CharactersNote({ found }: { found: number | string | null }) {
  if (found === null) return <span className="aside">measuring their size</span>;
  if (typeof found === "string") return <span className="aside" title={found}>size not measured</span>;
  return <span title={plural(found, "character")}>{characters(found)}</span>;
}

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
export type Dataset = { id: string; revision: number; name: string; tokenizer: string; libraries: DatasetRef[]; shares: { tag: string; share: number }[] | null; max_tokens: number | null; vocabulary: string | null; upstream: boolean; here: string[] };

// Sets the vocabulary a dataset publishes (a sha256, or null for none) as a
// new revision of it, the rest of its row as read (POST /api/datasets/ID,
// which keeps each library it reads as stored). The revision read goes with
// it, so the server refuses the change when someone changed the dataset
// after it was read, rather than undo their change.
export const publishVocabulary = (dataset: Dataset, vocabulary: string | null) => post(`/api/datasets/${part(dataset.id)}`, {
  name: dataset.name, libraries: dataset.libraries.map((r) => ({ depot: r.depot, library: r.library })),
  shares: dataset.shares, max_tokens: dataset.max_tokens, vocabulary, revision: dataset.revision,
});

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

// The libraries of this depot a dataset reads, by id: its here when the
// answer gives it, else the libraries it names with no depot (the
// download and export routes give the dataset's bare row).
export const ownLibraries = (dataset: Dataset) => dataset.libraries.filter((r) => (dataset.here ? dataset.here.includes(r.library) : r.depot === null)).map((r) => r.library);

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
