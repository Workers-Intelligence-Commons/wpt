// The workspace's right column: one text or item as it is now, in two tabs.
// "Text" is its editor: a plain text in CodeMirror (code-editor.tsx), with
// find and replace inside this text, an item through the form of its
// kind's schema; saving is save only, a new content version through POST
// /api/texts/ID/content, which the depot records as a public decision under
// review, so the text goes back under review and leaves every open library
// until it is admitted again, and the confirm says so before the click.
// "Record" is its record: credits, its revision and content version and how
// many it has of each, every decision with its reason (those before its
// latest license or content change marked as no longer counting), its
// libraries, each locked one with the revision it pinned, its reported
// problems and erasure requests. Above both stand its actions, each opening
// its form in a dialog: decide on it, change its licenses, edit its credits
// (a new revision; a locked library keeps the revision it pinned), take it
// out of an open library that holds it (it stays admitted), report a
// problem to a library that holds it, and ask to erase a passage
// (erase-form.tsx). A text that belongs to the upstream this depot follows
// says so, is shown read only, and keeps only the report to a library of
// this depot's own and the take-out from one. Every decision, tag, license
// and problem it names links to its own view.

import { Badge, Button, Dialog, Select, Skeleton, TextArea, TextInput } from "@workers-intelligence-commons/components";
import type { Text } from "@codemirror/state";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { get, post, type Answer } from "../api";
import type { EditorHandle } from "../code-editor";
import { RelicenseForm, relicensed, type Relicensed } from "../decide-forms";
import { EraseForm } from "../erase-form";
import { Confirm, Cut, Facts, Joined, LibraryLink, optional, Outcome, plural, Problems, required, RequiredNote, Section, StatusBadge, type Declaration, type Library, type Status } from "../parts";
import { codePoints, TextViewer } from "../reader";
import { useAbout } from "../about";
import { declared, useDeclarant } from "../declarant";
import { href } from "../route";
import { SchemaForm, type Schema } from "../schema-form";

// counts: whether the policy counts it (none made before the text's latest license or content change)
type Decision = { id: string; status: Status; name: string; email: string; made_at: string; reason: string | null; counts: boolean };
type Row = {
  id: string; text: string | null; kind: string | null; content: string | null; licenses: string[]; notice: string | null;
  title: string | null; authors: string[] | null; tags: string[]; address: string | null; decisions: string[]; libraries: string[];
  revision: number; content_version: number;
};
// pins: each locked library holding it, with the revision it pinned; upstream: it belongs to the upstream this depot follows
export type Shown = { text: Row; status: Status; notice: string | null; decisions: Decision[]; libraries: Library[]; revisions: number;
  pins: { library: string; name: string; revision: number }[]; content_versions: number; upstream: boolean };

export const showText = (id: string) => get<Shown>(`/api/texts/${encodeURIComponent(id)}`);

// id: the text; version: raised by the workspace whenever the text may have changed elsewhere (a key decided on it);
// onChanged: after any change made here; onDirty: whether its editor holds unsaved changes; handle: the editor's;
// onShown: the id of the text it shows now, or null while it shows none (the keys act only on the text shown).
// Only the answer for the id asked last is kept, so a text read slowly never replaces the one asked after it.
export function TextPanel({ id, version, onChanged, onDirty, handle, onShown }: {
  id: string; version: number; onChanged: () => void; onDirty: (dirty: boolean) => void; handle: (h: EditorHandle | null) => void;
  onShown: (id: string | null) => void;
}) {
  const [shown, setShown] = useState<{ id: string; answer: Answer<Shown> } | null>(null);
  const asked = useRef(id);
  asked.current = id;
  const load = useCallback(() => { showText(id).then((answer) => { if (asked.current === id) setShown({ id, answer }); }); }, [id]);
  useEffect(load, [load, version]);
  const here = shown !== null && shown.id === id && shown.answer.ok;
  useEffect(() => { onShown(here ? id : null); }, [here, id, onShown]);
  useEffect(() => () => onShown(null), [onShown]);
  if (!shown || shown.id !== id) return <Skeleton lines={6} label="Reading the text" />;
  if (!shown.answer.ok) return <Problems problems={shown.answer.problems} />;
  return <Loaded key={id} shown={shown.answer.data} reload={() => { load(); onChanged(); }} onDirty={onDirty} handle={handle} />;
}

type Dialogs = "decide" | "relicense" | "credits" | "take-out" | "report" | "erase";

function Loaded({ shown, reload, onDirty, handle }: { shown: Shown; reload: () => void; onDirty: (dirty: boolean) => void; handle: (h: EditorHandle | null) => void }) {
  const row = shown.text;
  const item = row.kind !== null;
  const [tab, setTab] = useState<"text" | "record">("text");
  const [open, setOpen] = useState<Dialogs | null>(null);
  const [said, setSaid] = useState<ReactNode>(null);
  const done = (words: ReactNode) => { setOpen(null); setSaid(words); reload(); };
  const title = row.title ?? (item ? `An item of kind ${row.kind}` : "Untitled text");
  const theirs = shown.upstream;
  const reportable = shown.libraries.filter((l) => !l.upstream); // only the depot's own libraries take a problem
  const leavable = reportable.filter((l) => !l.locked_at); // and only its own open ones let a text out
  const more = useRef<HTMLDetailsElement>(null); // the less used actions, closed again once one is chosen or a click lands elsewhere
  useEffect(() => {
    const away = (e: MouseEvent) => { if (more.current && !more.current.contains(e.target as Node)) more.current.removeAttribute("open"); };
    document.addEventListener("click", away);
    return () => document.removeEventListener("click", away);
  }, []);
  return (
    <div className="text-panel">
      <header className="text-head">
        <h2 className="title text-title">{title}</h2>
        <div className="page-meta small">
          <StatusBadge status={shown.status} /><span className="ledger">{row.id}</span>
          {theirs && <span className="aside">the upstream's</span>}
          {item && <span>an item of kind <span className="ledger">{row.kind}</span></span>}
        </div>
        <div className="head-buttons text-actions">
          {!theirs && <Button variant="quiet" onClick={() => setOpen("decide")}>Decide</Button>}
          {!theirs && <Button variant="quiet" onClick={() => setOpen("relicense")}>Change licenses</Button>}
          {!theirs && <Button variant="quiet" onClick={() => setOpen("credits")}>Edit credits</Button>}
          {(leavable.length > 0 || reportable.length > 0 || !theirs) && (
            <details className="more-menu" ref={more}>
              <summary className="wic-btn wic-btn-quiet wic-focus">More ▾</summary>
              <div className="more-items" onClick={() => more.current?.removeAttribute("open")}>
                {leavable.length > 0 && <Button variant="quiet" onClick={() => setOpen("take-out")}>Take out of a library</Button>}
                {reportable.length > 0 && <Button variant="quiet" onClick={() => setOpen("report")}>Report a problem</Button>}
                {!theirs && <Button variant="quiet" onClick={() => setOpen("erase")}>Ask to erase</Button>}
              </div>
            </details>
          )}
        </div>
        {theirs && <p className="head-note-left small">It belongs to the upstream this depot follows: only the upstream decides on it, changes its licenses, content or credits, or erases it, and a pull brings what it did here.</p>}
      </header>
      {said && <p className="done cue" role="status">{said}</p>}
      <div className="tabs" role="tablist" aria-label="The text and its record">
        <button type="button" role="tab" className="tab wic-focus" aria-selected={tab === "text"} onClick={() => setTab("text")}>{item ? "Content" : "Text"}</button>
        <button type="button" role="tab" className="tab wic-focus" aria-selected={tab === "record"} onClick={() => setTab("record")}>Record <span className="aside">{plural(shown.decisions.length, "decision")}</span></button>
      </div>
      <div role="tabpanel" hidden={tab !== "text"}>
        {theirs ? <ReadOnly row={row} /> : <EditContent key={`${row.id}:${row.content_version}`} row={row} libraries={shown.libraries} onDone={done} onDirty={onDirty} handle={handle} />}
      </div>
      <div role="tabpanel" hidden={tab !== "record"}>
        <Record shown={shown} />
      </div>
      <Dialog open={open === "decide"} onClose={() => setOpen(null)} title="Decide on this text">
        {open === "decide" && <DecideOn id={row.id} onDone={done} />}
      </Dialog>
      <Dialog open={open === "relicense"} onClose={() => setOpen(null)} title="Change the licenses of this text">
        {open === "relicense" && <Relicense row={row} onDone={done} />}
      </Dialog>
      <Dialog open={open === "credits"} onClose={() => setOpen(null)} title="Edit the credits of this text">
        {open === "credits" && <Credits row={row} notice={shown.notice} onDone={done} />}
      </Dialog>
      <Dialog open={open === "take-out"} onClose={() => setOpen(null)} title="Take this text out of a library">
        {open === "take-out" && leavable.length > 0 && <TakeOut id={row.id} libraries={leavable} onDone={done} />}
      </Dialog>
      <Dialog open={open === "report"} onClose={() => setOpen(null)} title="Report a problem with this text">
        {open === "report" && reportable.length > 0 && <Report id={row.id} libraries={reportable} onDone={done} />}
      </Dialog>
      <Dialog open={open === "erase"} onClose={() => setOpen(null)} title="Ask to erase a passage">
        {open === "erase" && <div className="wide"><EraseForm row={row} onDone={done} /></div>}
      </Dialog>
    </div>
  );
}

// An upstream text, read only: its text or its content as JSON.
function ReadOnly({ row }: { row: Row }) {
  const item = row.kind !== null;
  const content = useMemo(() => (item ? JSON.parse(row.content ?? "null") : null), [item, row.content]);
  return <TextViewer item={item} content={content} text={row.text ?? ""} mark={null} />;
}

function Record({ shown }: { shown: Shown }) {
  const row = shown.text;
  const item = row.kind !== null;
  const [allErasures, setAllErasures] = useState<{ id: string; status: Status; needles: { text: string }[] }[]>([]);
  useEffect(() => { get<typeof allErasures>("/api/erasures").then((a) => a.ok && setAllErasures(a.data)); }, [row.id]);
  const erasures = allErasures.filter((e) => e.needles.some((n) => n.text === row.id));
  const problems = shown.libraries.flatMap((l) => (l.problems ?? []).filter((p) => p.texts.includes(row.id)).map((p) => ({ library: l, problem: p })));
  return (
    <div className="record small">
      {shown.pins.length > 0 && <p className="aside">A locked library holds it ({shown.pins.map((p) => p.name).join(", ")}): a decision, a license change, a content change or an edit of its credits makes a new revision, and the locked library keeps reading the one it pinned.</p>}
      <Section title="Credits">
        <Facts rows={[
          ["revision", <span className="ledger">{row.revision} of {shown.revisions}</span>],
          ["content", <span className="ledger">version {row.content_version} of {shown.content_versions}</span>],
          ["characters", item ? null : codePoints(row.text ?? "").toLocaleString("en")],
          ["authors", row.authors?.join(", ")],
          ["licenses", row.licenses.length ? <span className="row tight">{row.licenses.map((l) => <a key={l} href={href(["licenses", l])} title={`The license ${l}`}><Badge status="license">{l}</Badge></a>)}</span> : null],
          ["tags", row.tags.length ? <Joined values={row.tags} render={(t) => <a href={href(["tags", t])}>{t}</a>} /> : null],
          ["address", row.address && (/^https?:\/\//.test(row.address) ? <a href={row.address} target="_blank" rel="noreferrer" className="ledger cut" title={row.address}>{row.address}</a> : <Cut className="ledger" text={row.address} />)],
          ["notice", shown.notice && <span className="notice-text">{shown.notice}</span>],
        ]} />
      </Section>
      <Section title="Libraries">
        {shown.libraries.length ? (
          <ul className="plain">
            {shown.libraries.map((l) => {
              const pin = shown.pins.find((p) => p.library === l.id);
              return <li key={l.id}><LibraryLink library={l} />{pin && <span className={pin.revision < row.revision ? "aside" : "quiet"}> reads revision {pin.revision}{pin.revision < row.revision ? ", an older one" : ""}</span>}</li>;
            })}
          </ul>
        ) : <p className="aside">None holds it{shown.status === "admitted" ? "" : ": only an admitted text joins one"}.</p>}
      </Section>
      <Section title="Decisions">
        <ol className="decisions">
          {shown.decisions.map((d) => (
            <li key={d.id} className={d.counts ? undefined : "uncounted"}>
              <span className="row tight"><StatusBadge status={d.status} /><time dateTime={d.made_at} className="ledger">{d.made_at.replace("T", " ").slice(0, 16)}</time>{!d.counts && <span className="aside">no longer counts</span>}</span>
              {d.reason && <span>{d.reason}</span>}
              <span className="quiet"><a className="ledger" href={href(["decisions", d.id])}>{d.id}</a>, {d.name}, <span className="ledger">{d.email}</span></span>
            </li>
          ))}
        </ol>
      </Section>
      {problems.length > 0 && (
        <Section title="Problems reported">
          <ul className="plain stacked">
            {problems.map(({ library, problem }) => <li key={problem.decision}><a href={href(["libraries", library.id, "problems", problem.decision])}>{problem.problem}</a> <span className="quiet">to <LibraryLink library={library} />, <a className="ledger" href={href(["decisions", problem.decision])}>{problem.decision}</a></span></li>)}
          </ul>
        </Section>
      )}
      {erasures.length > 0 && (
        <Section title="Erasure requests">
          <ul className="plain">
            {erasures.map((e) => <li key={e.id}><a className="ledger" href={href(["erasures"], { request: e.id })} title={`What ${e.id} asks to erase`}>{e.id}</a> <StatusBadge status={e.status} /></li>)}
          </ul>
        </Section>
      )}
    </div>
  );
}

function DecideOn({ id, onDone }: { id: string; onDone: (words: ReactNode) => void }) {
  const [status, setStatus] = useState<"admitted" | "refused">("admitted");
  const [reason, setReason] = useState("");
  const [answer, setAnswer] = useState<Answer<{ decision: string }> | null>(null);
  const send = async (declaration: Declaration) => {
    const a = await post<{ decision: string }>("/api/decisions", { status, texts: [id], ...(reason.trim() ? { reason: reason.trim() } : {}), declaration });
    setAnswer(a);
    if (a.ok) onDone(`Decision ${a.data.decision}: ${status}.`);
  };
  return (
    <div className="action-form">
      <RequiredNote />
      <div className="fields">
        <Select label={required("Decision")} required value={status} onChange={(e) => setStatus(e.target.value as "admitted" | "refused")}>
          <option value="admitted">Admit it</option>
          <option value="refused">Refuse it</option>
        </Select>
        <TextInput label={optional("Reason")} hint="Public, with the decision." value={reason} onChange={(e) => setReason(e.target.value)} />
      </div>
      <Confirm label={status === "admitted" ? "Confirm and admit" : "Confirm and refuse"} onConfirm={send}
        note={status === "refused" ? "A refusal takes it out of every open library." : undefined} />
      <Outcome answer={answer} done={() => ""} />
    </div>
  );
}

function Relicense({ row, onDone }: { row: Row; onDone: (words: ReactNode) => void }) {
  const [answer, setAnswer] = useState<Answer<Relicensed> | null>(null);
  return (
    <>
      <p>It names {row.licenses.map((l, i) => <span key={l}>{i > 0 && " or "}<Badge status="license">{l}</Badge></span>)} now.</p>
      <RelicenseForm chosen={[row.id]} subject="this text" onDone={(a, licenses) => { setAnswer(a); if (a.ok) onDone(relicensed(a.data, licenses)); }} />
      <Outcome answer={answer} done={() => ""} />
    </>
  );
}

// The text in the editor, or an item's content through the form of its
// kind's schema, prefilled; saved only once it differs, as a new content version.
function EditContent({ row, libraries, onDone, onDirty, handle }: {
  row: Row; libraries: Library[]; onDone: (words: ReactNode) => void; onDirty: (dirty: boolean) => void; handle: (h: EditorHandle | null) => void;
}) {
  const item = row.kind !== null;
  const original = useMemo(() => (item ? JSON.parse(row.content ?? "null") : null), [item, row.content]);
  const [doc, setDoc] = useState<{ text: Text; changed: boolean } | null>(null); // the editor's, once edited
  const [content, setContent] = useState<unknown>(original);
  const [schema, setSchema] = useState<Schema | null>(null);
  const [answer, setAnswer] = useState<Answer<{ decision: string | null; version: number | null }> | null>(null);
  const [editor, setEditor] = useState<EditorHandle | null>(null);
  useEffect(() => {
    if (item) get<{ id: string; schema: Schema }[]>("/api/kinds").then((a) => a.ok && setSchema(a.data.find((k) => k.id === row.kind)?.schema ?? null));
  }, [item, row.kind]);
  const changed = item ? JSON.stringify(content) !== JSON.stringify(original) : Boolean(doc?.changed);
  useEffect(() => { onDirty(changed); }, [changed, onDirty]);
  useEffect(() => () => onDirty(false), [onDirty]);
  const handed = useCallback((h: EditorHandle | null) => { setEditor(h); handle(h); }, [handle]);
  const open = libraries.filter((l) => !l.locked_at && !l.upstream);
  const locked = libraries.filter((l) => l.locked_at);
  const who = useDeclarant();
  const { about } = useAbout();
  // Ctrl+S (Cmd+S) saves as the button does, once something changed and a name and email are set
  useEffect(() => {
    const press = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "s") return;
      e.preventDefault();
      if (changed && about && declared(who)) send({ name: who.name.trim(), email: who.email.trim(), accepted: true });
    };
    window.addEventListener("keydown", press);
    return () => window.removeEventListener("keydown", press);
  });
  const send = async (declaration: Declaration) => {
    const a = await post<{ decision: string | null; version: number | null }>(`/api/texts/${encodeURIComponent(row.id)}/content`, { ...(item ? { content } : { text: doc!.text.toString() }), declaration });
    setAnswer(a);
    if (!a.ok) return;
    onDirty(false);
    if (a.data.decision === null) onDone("Nothing changed: its content is this already.");
    else onDone(<>Saved as content version {a.data.version}, recorded by decision <a href={href(["decisions", a.data.decision])}>{a.data.decision}</a>: the text is under review again{open.length > 0 && <> and left {open.map((l, i) => <span key={l.id}>{i > 0 && ", "}<LibraryLink library={l} /></span>)}</>}. Admit it again, then put it back, to have it read.</>);
  };
  return (
    <div className="edit-content">
      {item ? (
        schema ? <SchemaForm value={content} schema={schema} root={schema} name={row.kind!} need="required" onChange={setContent} /> : <Skeleton lines={4} label="Reading the kind's schema" />
      ) : (
        <>
          <div className="editor-tools small">
            <Button variant="quiet" disabled={!editor} onClick={() => editor?.openFind()} title='Ctrl+F inside the text too; tick "regexp" for a regular expression. Only this text is searched.'>Find and replace</Button>
            <span className="aside editor-count">{plural(doc ? doc.text.lines : lineCount(row.text ?? ""), "line")}{changed && ", changed"}</span>
          </div>
          <Suspense fallback={<Skeleton lines={8} label="Opening the editor" />}>
            <CodeEditor initial={row.text ?? ""} onChange={(text, edited) => setDoc({ text, changed: edited })} label={`The text of ${row.title ?? row.id}`} handle={handed} />
          </Suspense>
        </>
      )}
      {changed ? (
        <Confirm label="Save, under review" onConfirm={send}
          note={<>Saving (or Ctrl+S) is a public decision under review and nothing else: it makes content version {row.content_version + 1}, the decisions before it no longer count, so the text goes back under review and leaves {open.length ? <>its open {open.length === 1 ? "library" : "libraries"} ({open.map((l) => l.name).join(", ")})</> : "every open library"}. It is read again only once it is admitted again, separately, and put back.{locked.length > 0 && <> A locked library ({locked.map((l) => l.name).join(", ")}) keeps reading the content of the revision it pinned.</>}</>} />
      ) : <p className="save-idle small aside">Nothing changed yet. Saving makes a new content version and puts the text back under review.</p>}
      <Outcome answer={answer} done={() => ""} />
    </div>
  );
}

// The lines of a text, as the editor counts them: one more than its line breaks.
const lineCount = (text: string) => text.split("\n").length;

// CodeMirror, read only when a plain text is first shown in the editor, so the other pages load without it.
const CodeEditor = lazy(() => import("../code-editor").then((m) => ({ default: m.CodeEditor })));

// One value per line, without the blank lines.
const lines = (text: string) => text.split("\n").map((l) => l.trim()).filter(Boolean);

// The credits of the text, prefilled; only the credits changed are sent, an emptied one as none.
function Credits({ row, notice, onDone }: { row: Row; notice: string | null; onDone: (words: ReactNode) => void }) {
  const [title, setTitle] = useState(row.title ?? "");
  const [authors, setAuthors] = useState((row.authors ?? []).join("\n"));
  const [tags, setTags] = useState(row.tags.join("\n"));
  const [address, setAddress] = useState(row.address ?? "");
  const [noticeText, setNoticeText] = useState(notice ?? "");
  const [answer, setAnswer] = useState<Answer<{ decision: string; reason: string }> | null>(null);
  const changes: Record<string, unknown> = {};
  if (title.trim() !== (row.title ?? "")) changes.title = title.trim() || null;
  if (JSON.stringify(lines(authors)) !== JSON.stringify(row.authors ?? [])) changes.authors = lines(authors).length ? lines(authors) : null;
  if (JSON.stringify(lines(tags)) !== JSON.stringify(row.tags)) changes.tags = lines(tags);
  if (address.trim() !== (row.address ?? "")) changes.address = address.trim() || null;
  if (noticeText.trim() !== (notice ?? "").trim()) changes.notice = noticeText.trim() ? noticeText : null;
  const send = async (declaration: Declaration) => {
    const a = await post<{ decision: string; reason: string }>(`/api/texts/${encodeURIComponent(row.id)}/credits`, { ...changes, declaration });
    setAnswer(a);
    if (a.ok) onDone(<>Decision <a href={href(["decisions", a.data.decision])}>{a.data.decision}</a>, under review: {a.data.reason}.</>);
  };
  return (
    <div className="action-form">
      <div className="fields">
        <TextInput label={optional("Title")} value={title} onChange={(e) => setTitle(e.target.value)} />
        <TextInput label={optional("Address")} hint="Where it came from." value={address} onChange={(e) => setAddress(e.target.value)} />
      </div>
      <div className="fields">
        <TextArea label={optional("Authors")} hint="One per line, in order." rows={3} value={authors} onChange={(e) => setAuthors(e.target.value)} />
        <TextArea label={optional("Tags")} hint="One per line." rows={3} value={tags} onChange={(e) => setTags(e.target.value)} />
      </div>
      <TextArea label={optional("Notice")} hint="The copyright notice exactly as the source gives it, or why the work is public domain." rows={4} value={noticeText} onChange={(e) => setNoticeText(e.target.value)} />
      <Confirm label="Confirm and save the credits" disabled={Object.keys(changes).length === 0} onConfirm={send}
        note={Object.keys(changes).length === 0 ? "Nothing is changed yet."
          : <>Saved as a decision under review whose reason names what changed ({Object.keys(changes).join(", ")}). The decisions before it keep counting: the text keeps its status and its libraries, and a locked library keeps the revision it pinned.</>} />
      <Outcome answer={answer} done={() => ""} />
    </div>
  );
}

// Out of one of the depot's own open libraries that hold it; it stays admitted.
function TakeOut({ id, libraries, onDone }: { id: string; libraries: Library[]; onDone: (words: ReactNode) => void }) {
  const [library, setLibrary] = useState(libraries[0].id);
  const [answer, setAnswer] = useState<Answer<{ taken_out: string[] }> | null>(null);
  const name = libraries.find((l) => l.id === library)?.name ?? library;
  const send = async () => {
    const a = await post<{ taken_out: string[]; not_held: string[] }>(`/api/libraries/${encodeURIComponent(library)}/taken-out`, { texts: [id] });
    setAnswer(a);
    if (a.ok) onDone(a.data.taken_out.length ? `Taken out of ${name}; it keeps its status.` : `${name} did not hold it.`);
  };
  return (
    <div className="action-form">
      <RequiredNote />
      <Select label={required("Out of the library")} required value={library} onChange={(e) => setLibrary(e.target.value)}
        hint="Only this depot's own open libraries let a text out; a locked one is corrected by a fork without it.">
        {libraries.map((l) => <option key={l.id} value={l.id}>{l.name} ({l.id})</option>)}
      </Select>
      <p className="small">It leaves {name} and keeps its status and its decisions: it stays admitted, and can be put back. No declaration is needed: it changes a library, not the text.</p>
      <Button variant="primary" onClick={send}>Take it out</Button>
      <Outcome answer={answer} done={() => ""} />
    </div>
  );
}

function Report({ id, libraries, onDone }: { id: string; libraries: Library[]; onDone: (words: ReactNode) => void }) {
  const [library, setLibrary] = useState(libraries[0].id);
  const [problem, setProblem] = useState("");
  const [answer, setAnswer] = useState<Answer<{ decision: string }> | null>(null);
  const send = async (declaration: Declaration) => {
    const a = await post<{ decision: string }>(`/api/libraries/${encodeURIComponent(library)}/problems`, { problem, texts: [id], declaration });
    setAnswer(a);
    if (a.ok) onDone(`Problem reported, recorded by decision ${a.data.decision}.`);
  };
  return (
    <div className="action-form">
      <RequiredNote />
      <Select label={required("To the library")} required value={library} onChange={(e) => setLibrary(e.target.value)}>
        {libraries.map((l) => <option key={l.id} value={l.id}>{l.name} ({l.id}){l.locked_at ? ", locked" : ""}</option>)}
      </Select>
      <TextArea label={required("The problem")} required value={problem} onChange={(e) => setProblem(e.target.value)} hint="Public. It names this text and changes no text and no membership." />
      <Confirm label="Confirm and report" disabled={!problem.trim()} onConfirm={send} />
      <Outcome answer={answer} done={() => ""} />
    </div>
  );
}
