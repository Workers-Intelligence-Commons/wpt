// The open text of the Texts page: its content, always in view and always
// editable, under one line of facts: a plain text in CodeMirror
// (code-editor.tsx), with find and replace inside this text, or an item
// through the form of its kind's schema, a conversation read first as a
// transcript. Saving is save only, with Save or Ctrl+S, a new content
// version through POST /api/texts/ID/content, which the depot records as a
// public decision under review, so the text goes back under review and
// leaves every open library until it is admitted again; an admitted text
// asks once more. Beside it, the review panel (panel.tsx): its scores, its
// tags, the moves and decisions, the presets, its facts and the menu of its
// other actions, each opening its form in a dialog: its record (credits,
// revision and content version and how many it has of each, every decision
// with its reason, those that no longer count marked, a put back shown as a
// correction, its libraries, each locked one with the revision it pinned,
// its reported problems and erasure requests), a decision with a reason,
// its licenses, its credits (a new revision; a locked library keeps the
// revision it pinned), taking it out of an open library that holds it (it
// stays admitted), a problem reported to a library that holds it, and a
// request to erase a passage (erase-form.tsx). A text that belongs to the
// upstream this depot follows says so, is shown read only, and keeps only
// the report to a library of this depot's own and the take-out from one.
// The texts read are kept a little, and the Texts page asks for the next
// ones ahead (prefetch), so moving on shows the next one at once.

import { Badge, Button, Dialog, Select, Skeleton, TextArea, TextInput } from "@workers-intelligence-commons/components";
import type { Text } from "@codemirror/state";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { get, post, write, type Answer } from "../api";
import type { EditorHandle } from "../code-editor";
import type { Deciding } from "../blocked";
import { RelicenseForm, relicensed, type Decided, type Relicensed } from "../decide-forms";
import { named, sent } from "../filter";
import { EraseForm } from "../erase-form";
import { Confirm, Cut, Facts, Help, Joined, LibraryLink, MoreMenu, optional, Outcome, plural, Problems, required, RequiredNote, Section, settingsHref, StatusBadge, textName, type Declaration, type Library, type Status } from "../parts";
import { codePoints, TextViewer } from "../reader";
import { useAbout } from "../about";
import { declared, useDeclarant } from "../declarant";
import { Correction } from "../pages/decisions";
import { href } from "../route";
import { KIND_WORDS } from "../kind-words";
import { SchemaForm, type Schema } from "../schema-form";
import { type Score } from "../scores";
import { ReviewPanel } from "./panel";
import { isConversation, Transcript } from "../transcript";

// counts: whether the policy counts it (none made before the text's latest license or content change, or its latest putting back under review)
// put_back: it put the text back under review, a correction of the decisions before it
type Decision = { id: string; status: Status; name: string; email: string; made_at: string; reason: string | null; counts: boolean; put_back: boolean };
type Row = {
  id: string; text: string | null; kind: string | null; content: string | null; licenses: string[]; notice: string | null;
  title: string | null; authors: string[] | null; tags: string[]; address: string | null; decisions: string[]; libraries: string[];
  revision: number; content_version: number; scores: Score[] | null;
};
// pins: each locked library holding it, with the revision it pinned; upstream: it belongs to the upstream this depot follows
export type Shown = { text: Row; status: Status; notice: string | null; decisions: Decision[]; libraries: Library[]; revisions: number;
  pins: { library: string; name: string; revision: number }[]; content_versions: number; upstream: boolean };

export const showText = (id: string) => get<Shown>(`/api/texts/${encodeURIComponent(id)}`);

// Texts already read, or asked for ahead (prefetch), by id and the version of the page they were read at, so
// moving to the next text shows it at once; a few at most.
const read = new Map<string, Promise<Answer<Shown>>>();
const readAt = (id: string, version: number) => {
  const key = `${version} ${id}`;
  if (!read.has(key)) {
    read.set(key, showText(id));
    while (read.size > 8) read.delete(read.keys().next().value!);
  }
  return read.get(key)!;
};
// A text read again when next opened: its panel saved scores or tags, which the version does not follow.
const forgetText = (id: string) => [...read.keys()].filter((key) => key.endsWith(` ${id}`)).forEach((key) => read.delete(key));
// Ask for texts before they are opened (the Texts page asks for the next ones).
export const prefetch = (ids: string[], version: number) => ids.forEach((id) => readAt(id, version));

// id: the text; actions: the moves and decisions of the Texts page, which the panel shows under the tags; version:
// raised by the Texts page whenever the text may have changed elsewhere (a key decided on it); onChanged: after any
// change made here but scores and tags; onSaved: after scores or tags were saved, or failed to be, with what happened; onDirty:
// whether its editor holds unsaved changes; handle: the editor's; onShown: the id of the text it shows now, or null
// while it shows none (the keys act only on the text shown); find: the words the search looked for inside the
// texts, marked in it, the first one scrolled to. Only the answer for the id asked last is kept, so a text read
// slowly never replaces the one asked after it.
export function TextPanel({ id, actions, version, onChanged, onSaved, onDirty, handle, onShown, find }: {
  id: string; actions: ReactNode; version: number; onChanged: () => void; onSaved: (words: ReactNode) => void; onDirty: (dirty: boolean) => void;
  handle: (h: EditorHandle | null) => void; onShown: (id: string | null) => void; find?: string;
}) {
  const [shown, setShown] = useState<{ id: string; answer: Answer<Shown> } | null>(null);
  const asked = useRef(id);
  asked.current = id;
  const load = useCallback((again = false) => {
    if (again) read.clear();
    readAt(id, version).then((answer) => { if (asked.current === id) setShown({ id, answer }); });
  }, [id, version]);
  useEffect(() => load(), [load]);
  useEffect(() => () => forgetText(id), [id]); // a text left may have been saved on leaving: read it again on return
  const here = shown !== null && shown.id === id && shown.answer.ok;
  useEffect(() => { onShown(here ? id : null); }, [here, id, onShown]);
  useEffect(() => () => onShown(null), [onShown]);
  if (!shown || shown.id !== id) return <div className="text-column"><Skeleton lines={6} label="Reading the text" /></div>;
  if (!shown.answer.ok) return <div className="text-column"><Problems problems={shown.answer.problems} /></div>;
  return <Loaded key={id} actions={actions} shown={shown.answer.data} reload={() => { load(true); onChanged(); }} onSaved={(words) => { forgetText(id); onSaved(words); }} onDirty={onDirty} handle={handle} find={find} />;
}

type Dialogs = "decide" | "relicense" | "credits" | "take-out" | "report" | "erase" | "record";

// The open text in two parts: its content, always in view, under one line of facts, and the review panel beside
// it (panel.tsx), whose menu opens the other actions, each in a dialog.
function Loaded({ shown, actions, reload, onSaved, onDirty, handle, find }: {
  shown: Shown; actions: ReactNode; reload: () => void; onSaved: (words: ReactNode) => void; onDirty: (dirty: boolean) => void;
  handle: (h: EditorHandle | null) => void; find?: string;
}) {
  const row = shown.text;
  const item = row.kind !== null;
  const [open, setOpen] = useState<Dialogs | null>(null);
  const [said, setSaid] = useState<ReactNode>(null);
  const done = (words: ReactNode) => { setOpen(null); setSaid(words); reload(); };
  const named = textName(row);
  const title = row.title ? named : named[0].toUpperCase() + named.slice(1); // a real title (often a file name) stays as written
  const theirs = shown.upstream;
  const reportable = shown.libraries.filter((l) => !l.upstream); // only the depot's own libraries take a problem
  const leavable = reportable.filter((l) => !l.locked_at); // and only its own open ones let a text out
  const facts = (
    <>
      <strong className="content-title" title={title}>{title}</strong>
      <StatusBadge status={shown.status} />
      <span className="ledger aside">v{row.content_version}{item ? <> · {row.kind}</> : <> · {codePoints(row.text ?? "").toLocaleString("en")} chars</>}</span>
      {theirs && <span className="aside" title="Only the upstream this depot follows decides on it, changes its licenses, content or credits, or erases it">the upstream's, read only</span>}
    </>
  );
  const more = (
    <MoreMenu label="More on this text">
      <Button variant="quiet" onClick={() => setOpen("record")}>Its record</Button>
      {!theirs && <Button variant="quiet" onClick={() => setOpen("decide")}>Decide, with a reason</Button>}
      {!theirs && <Button variant="quiet" onClick={() => setOpen("relicense")}>Change licenses</Button>}
      {!theirs && <Button variant="quiet" onClick={() => setOpen("credits")}>Edit credits</Button>}
      {leavable.length > 0 && <Button variant="quiet" onClick={() => setOpen("take-out")}>Take out of a library</Button>}
      {reportable.length > 0 && <Button variant="quiet" onClick={() => setOpen("report")}>Report a problem</Button>}
      {!theirs && <Button variant="quiet" onClick={() => setOpen("erase")}>Ask to erase</Button>}
    </MoreMenu>
  );
  return (
    <>
      <div className="text-column">
        {said && <p className="done cue content-said" role="status">{said}</p>}
        {theirs ? <><div className="content-line small">{facts}</div><ReadOnly row={row} find={find} /></>
          : <EditContent key={`${row.id}:${row.content_version}`} row={row} status={shown.status} facts={facts} libraries={shown.libraries} onDone={done} onDirty={onDirty} handle={handle} find={find} />}
      </div>
      <ReviewPanel key={`${row.id}:${row.revision}`} text={row} status={shown.status} decisions={shown.decisions.length} libraries={shown.libraries} readOnly={theirs}
        actions={actions} more={more} onSaved={onSaved} onRecord={() => setOpen("record")} />
      <Dialog open={open === "record"} onClose={() => setOpen(null)} title={`The record of ${title}`}>
        {open === "record" && <div className="wide"><Record shown={shown} /></div>}
      </Dialog>
      <Dialog open={open === "decide"} onClose={() => setOpen(null)} title="Decide on this text">
        {open === "decide" && <DecideOn id={row.id} status={shown.status} onDone={done} />}
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
    </>
  );
}

// An upstream text, read only: its text, a conversation as a transcript, or another item's content as JSON.
function ReadOnly({ row, find }: { row: Row; find?: string }) {
  const item = row.kind !== null;
  const content = useMemo(() => (item ? JSON.parse(row.content ?? "null") : null), [item, row.content]);
  if (isConversation(content)) return <div className="viewer"><Transcript content={content} find={find} /></div>;
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
              <span className="row tight"><StatusBadge status={d.status} /><Correction d={d} /><time dateTime={d.made_at} className="ledger">{d.made_at.replace("T", " ").slice(0, 16)}</time>{!d.counts && <span className="aside">no longer counts</span>}</span>
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

// Admit, refuse, or, for a text admitted or refused, put it back under
// review: a decision that undoes the admission or refusal, after which the
// decisions before it no longer count.
function DecideOn({ id, status: now, onDone }: { id: string; status: Status; onDone: (words: ReactNode) => void }) {
  const [status, setStatus] = useState<Deciding>("admitted");
  const [reason, setReason] = useState("");
  const [answer, setAnswer] = useState<Answer<Decided> | null>(null);
  const send = async (declaration: Declaration) => {
    const a = await write<Decided>("/api/decisions", { status, selection: sent(named([id])), ...(reason.trim() ? { reason: reason.trim() } : {}), declaration });
    setAnswer(a);
    if (a.ok) onDone(`Decision ${a.data.decision}: ${status === "under review" ? "put back under review" : status}.`);
  };
  return (
    <div className="action-form">
      <RequiredNote />
      <div className="fields">
        <Select label={required("Decision")} required value={status} onChange={(e) => setStatus(e.target.value as Deciding)}>
          <option value="admitted">Admit it</option>
          <option value="refused">Refuse it</option>
          {now !== "under review" && <option value="under review">Put it back under review (undo its {now === "admitted" ? "admission" : "refusal"})</option>}
        </Select>
        <TextInput label={optional("Reason")} hint="Public, with the decision." value={reason} onChange={(e) => setReason(e.target.value)} />
      </div>
      <Confirm label={status === "admitted" ? "Confirm and admit" : status === "refused" ? "Confirm and refuse" : "Confirm and put back under review"} onConfirm={send}
        note={status === "refused" ? "A refusal takes it out of every open library."
          : status === "under review" ? "A public decision that puts it back under review: the decisions before it no longer count, and it leaves every open library until it is admitted again; a locked library keeps the revision it pinned." : undefined} />
      <Outcome answer={answer} done={() => ""} />
    </div>
  );
}

function Relicense({ row, onDone }: { row: Row; onDone: (words: ReactNode) => void }) {
  const [answer, setAnswer] = useState<Answer<Relicensed> | null>(null);
  return (
    <>
      <p>It names {row.licenses.map((l, i) => <span key={l}>{i > 0 && " or "}<Badge status="license">{l}</Badge></span>)} now.</p>
      <RelicenseForm selection={named([row.id])} subject="this text" onDone={(a, licenses) => { setAnswer(a); if (a.ok) onDone(relicensed(a.data, licenses)); }} />
      <Outcome answer={answer} done={() => ""} />
    </>
  );
}

// The text in the editor, always editable, or an item's content through the form of its kind's schema,
// prefilled; a conversation is read as a transcript first, and its form opens on "Edit" or the E key (the
// handle's focus). Above it, the facts line with "Find" and its count of lines. Once it differs, a bar under it
// saves (Save, or Ctrl+S) or reverts: saving makes a new content version, a public decision under review, so
// the text goes back under review and leaves every open library; an admitted text asks once more before that.
function EditContent({ row, status, facts, libraries, onDone, onDirty, handle, find }: {
  row: Row; status: Status; facts: ReactNode; libraries: Library[]; onDone: (words: ReactNode) => void; onDirty: (dirty: boolean) => void;
  handle: (h: EditorHandle | null) => void; find?: string;
}) {
  const item = row.kind !== null;
  const original = useMemo(() => (item ? JSON.parse(row.content ?? "null") : null), [item, row.content]);
  const conversation = isConversation(original);
  const [editing, setEditing] = useState(!conversation);
  useEffect(() => {
    if (!conversation || editing) return;
    handle({ focus: () => setEditing(true), openFind: () => {} });
    return () => handle(null);
  }, [conversation, editing, handle]);
  const [doc, setDoc] = useState<{ text: Text; changed: boolean } | null>(null); // the editor's, once edited
  const [content, setContent] = useState<unknown>(original);
  const [schema, setSchema] = useState<Schema | null>(null);
  const [answer, setAnswer] = useState<Answer<{ decision: string | null; version: number | null }> | null>(null);
  const [editor, setEditor] = useState<EditorHandle | null>(null);
  const [fresh, setFresh] = useState(0); // raised by Revert: the editor starts again from the text as saved
  const [asking, setAsking] = useState(false); // an admitted text's save, asked once more
  useEffect(() => {
    if (item) get<{ id: string; schema: Schema }[]>("/api/kinds").then((a) => a.ok && setSchema(a.data.find((k) => k.id === row.kind)?.schema ?? null));
  }, [item, row.kind]);
  const changed = item ? JSON.stringify(content) !== JSON.stringify(original) : Boolean(doc?.changed);
  useEffect(() => { onDirty(changed); if (!changed) setAsking(false); }, [changed, onDirty]);
  useEffect(() => () => onDirty(false), [onDirty]);
  const handed = useCallback((h: EditorHandle | null) => { setEditor(h); handle(h); }, [handle]);
  const open = libraries.filter((l) => !l.locked_at && !l.upstream);
  const locked = libraries.filter((l) => l.locked_at);
  const who = useDeclarant();
  const { about } = useAbout();
  const ready = declared(who) && about !== null;
  const save = () => {
    if (!changed || !ready) return;
    if (status === "admitted" && !asking) { setAsking(true); return; }
    send({ name: who.name.trim(), email: who.email.trim(), accepted: true });
  };
  const revert = () => { setDoc(null); setContent(original); setFresh((n) => n + 1); setAsking(false); };
  // Ctrl+S (Cmd+S) saves as the button does
  useEffect(() => {
    const press = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "s") return;
      e.preventDefault();
      save();
    };
    window.addEventListener("keydown", press);
    return () => window.removeEventListener("keydown", press);
  });
  const send = async (declaration: Declaration) => {
    setAsking(false);
    const a = await post<{ decision: string | null; version: number | null }>(`/api/texts/${encodeURIComponent(row.id)}/content`, { ...(item ? { content } : { text: doc!.text.toString() }), declaration });
    setAnswer(a);
    if (!a.ok) return;
    onDirty(false);
    if (a.data.decision === null) onDone("Nothing changed: its content is this already.");
    else onDone(<>Saved as content version {a.data.version} (<a href={href(["decisions", a.data.decision])}>{a.data.decision}</a>): under review again{open.length > 0 && <>, out of {open.map((l, i) => <span key={l.id}>{i > 0 && ", "}<LibraryLink library={l} /></span>)}</>}.</>);
  };
  return (
    <div className="edit-content">
      <div className="content-line small">
        {facts}
        {!item && <span className="content-tools">
          <span className="aside">{plural(doc ? doc.text.lines : lineCount(row.text ?? ""), "line")}</span>
          <Button variant="quiet" disabled={!editor} onClick={() => editor?.openFind()} title='Find and replace in this text (Ctrl+F); tick "regexp" for a regular expression'>Find</Button>
        </span>}
        {conversation && !editing && <span className="content-tools"><span className="aside">{plural(original.messages.length, "message")}</span>
          <Button variant="quiet" onClick={() => setEditing(true)} aria-keyshortcuts="E" title="Opens the form of its messages, roles and blocks (E)">Edit</Button></span>}
        {conversation && editing && !changed && <span className="content-tools"><Button variant="quiet" onClick={() => setEditing(false)}>Read it</Button></span>}
      </div>
      {conversation && !editing ? <div className="viewer content-scroll"><Transcript content={original} find={find} /></div>
        : item ? (
          <div className="content-scroll">{schema ? <SchemaForm key={fresh} value={content} schema={schema} root={schema} name={row.kind!} words={KIND_WORDS[row.kind!]} need="required" onChange={setContent} /> : <Skeleton lines={4} label="Reading the kind's schema" />}</div>
        ) : (
          <Suspense fallback={<Skeleton lines={8} label="Opening the editor" />}>
            <CodeEditor key={fresh} initial={row.text ?? ""} onChange={(text, edited) => setDoc({ text, changed: edited })} label={`The text of ${row.title ?? row.id}`} handle={handed} find={find} />
          </Suspense>
        )}
      {changed && (
        <div className={`save-bar small${asking ? " asking" : ""}`} role="group" aria-label="Unsaved changes">
          {asking ? <span className="act">It is admitted: saving puts it back under review{open.length > 0 && <> and out of {open.map((l) => l.name).join(", ")}</>}.</span>
            : <span>Edited, not saved.</span>}
          <Button variant="primary" disabled={!ready} onClick={save}>{asking ? "Save anyway" : "Save"}<span className="key-hint">(Ctrl S)</span></Button>
          <Button variant="quiet" onClick={asking ? () => setAsking(false) : revert}>{asking ? "Keep editing" : "Revert"}</Button>
          <span className="aside save-words">{ready
            ? <>A public decision as {who.name}: content version {row.content_version + 1}, the text back under review{locked.length > 0 && <>; {locked.map((l) => l.name).join(", ")} keeps the version it pinned</>}. It confirms: <q>{about!.confirmation}</q> <a href={about!.requirements} target="_blank" rel="noreferrer">The requirements</a>.</>
            : declared(who) ? <>Reading the depot's requirements before it can save…</>
            : <>Set a name and email in <a href={settingsHref()}>Settings</a> to save.</>}</span>
        </div>
      )}
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
        note={Object.keys(changes).length === 0 ? "Nothing changed yet."
          : <>Changes: {Object.keys(changes).join(", ")}.<Help label="saving credits">Saved as a decision under review whose reason names what changed. The decisions before it keep counting: the text keeps its status and its libraries, and a locked library keeps the revision it pinned.</Help></>} />
      <Outcome answer={answer} done={() => ""} />
    </div>
  );
}

// Out of one of the depot's own open libraries that hold it; it stays admitted.
function TakeOut({ id, libraries, onDone }: { id: string; libraries: Library[]; onDone: (words: ReactNode) => void }) {
  const [library, setLibrary] = useState(libraries[0].id);
  const [answer, setAnswer] = useState<Answer<{ taken_out: number }> | null>(null);
  const name = libraries.find((l) => l.id === library)?.name ?? library;
  const send = async () => {
    const a = await write<{ taken_out: number; not_held: number }>(`/api/libraries/${encodeURIComponent(library)}/taken-out`, { selection: sent(named([id])) });
    setAnswer(a);
    if (a.ok) onDone(a.data.taken_out ? `Taken out of ${name}; it keeps its status.` : `${name} did not hold it.`);
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
    const a = await write<{ decision: string }>(`/api/libraries/${encodeURIComponent(library)}/problems`, { problem, selection: sent(named([id])), declaration });
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
