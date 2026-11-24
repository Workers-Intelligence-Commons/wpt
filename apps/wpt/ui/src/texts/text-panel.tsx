// The open text of the Texts page: its content, always in view and always
// editable, under one line of facts: a plain text in CodeMirror
// (code-editor.tsx), with find and replace inside this text and a Translate
// button that shows it translated, read only, in the editor's place
// (translation/translated.tsx), or an item
// through the form of its kind's schema, a conversation read first as a
// transcript and edited as a chat (conversation-editor.tsx), its kind shown in the facts line, fixed. An edit has no
// button: it is saved when the Texts page moves to another text, before Admit & next admits it, with Ctrl+S, and when
// the text is left any other way, as a new content version through POST /api/texts/ID/content, which the depot
// records as a public decision under review, so the text goes back under review (a snapshot made before keeps the
// revision it holds); an edit that cannot be saved when the text is left that way is said, and kept in this tab
// until the text opens again; a reload or a closed tab with an edit not saved asks first. Beside it, the review panel (panel.tsx): its scores, its
// tags, the moves and decisions, the presets, its facts and the menu of its
// other actions, each opening its form in a dialog: its record (credits,
// revision and content version and how many it has of each, every decision
// with its reason, those that no longer count marked, a put back shown as a
// correction, and its erasure requests), a decision with a reason, its
// licenses, its credits (a new revision), and a request to erase a passage
// (erase-form.tsx). A text that belongs to the upstream this depot follows
// says so and is shown read only, with its record.
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
import { Confirm, Cut, Facts, Help, Joined, MoreMenu, optional, Outcome, PLAIN, plural, Problems, required, RequiredNote, Section, settingsHref, StatusBadge, textName, type Declaration, type Status } from "../parts";
import { codePoints, TextViewer } from "../reader";
import { useAbout } from "../about";
import { declared, useDeclarant } from "../declarant";
import { Correction } from "../pages/decisions";
import { href } from "../route";
import { KIND_WORDS } from "../kind-words";
import { FixedKind, useKinds } from "../kind-picker";
import { SchemaForm } from "../schema-form";
import { type Score } from "../scores";
import { ReviewPanel } from "./panel";
import { isConversation, Transcript } from "../transcript";
import { ConversationEditor, type Conversation } from "../conversation-editor";
import { tell } from "../notice";
import { canonical, EditSaves, type Content, type Made, type Saved } from "./edit-saving";
import { Translated } from "../translation/translated";

// counts: whether the policy counts it (none made before the text's latest license or content change, or its latest putting back under review)
// put_back: it put the text back under review, a correction of the decisions before it
type Decision = { id: string; status: Status; name: string; email: string; made_at: string; reason: string | null; counts: boolean; put_back: boolean };
type Row = {
  id: string; text: string | null; kind: string; content: string | null; licenses: string[]; notice: string | null;
  title: string | null; authors: string[] | null; tags: string[]; address: string | null; decisions: string[];
  revision: number; content_version: number; scores: Score[] | null;
};
// upstream: it belongs to the upstream this depot follows
export type Shown = { text: Row; status: Status; notice: string | null; decisions: Decision[]; revisions: number;
  content_versions: number; upstream: boolean };

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

// What the Texts page asks of the editor before it leaves the text or decides on it: whether it holds an edit not
// saved, and the save of that edit, which answers whether it was saved (words: what happened, said on the page).
export type SaveResult = { ok: boolean; words: ReactNode };
export type Saver = { dirty: () => boolean; save: () => Promise<SaveResult> };

// id: the text; actions: the moves and decisions of the Texts page, which the panel shows under the tags; version:
// raised by the Texts page whenever the text may have changed elsewhere (a key decided on it); onChanged: after any
// change made here but scores and tags; onSaved: after scores or tags were saved, or failed to be, with what happened; saver: the editor's save, for the page (Saver); handle: the editor's; onShown: the id of the text it shows now, or null
// while it shows none (the keys act only on the text shown); find: the words the search looked for inside the
// texts, marked in it, the first one scrolled to. Only the answer for the id asked last is kept, so a text read
// slowly never replaces the one asked after it.
export function TextPanel({ id, actions, version, onChanged, onSaved, saver, handle, onShown, find }: {
  id: string; actions: ReactNode; version: number; onChanged: () => void; onSaved: (words: ReactNode) => void;
  saver: (s: Saver | null) => void; handle: (h: EditorHandle | null) => void; onShown: (id: string | null) => void; find?: string;
}) {
  const [shown, setShown] = useState<{ id: string; answer: Answer<Shown> } | null>(null);
  // a plain text read translated in place of its editor; kept from one text to the next, so moving on translates the
  // next one too
  const [translating, setTranslating] = useState(false);
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
  return <Loaded key={id} actions={actions} shown={shown.answer.data} reload={() => { load(true); onChanged(); }} onSaved={(words) => { forgetText(id); onSaved(words); }}
    saver={saver} handle={handle} find={find} translating={translating} setTranslating={setTranslating} />;
}

type Dialogs = "decide" | "relicense" | "credits" | "erase" | "record";

// The open text in two parts: its content, always in view, under one line of facts, and the review panel beside
// it (panel.tsx), whose menu opens the other actions, each in a dialog.
function Loaded({ shown, actions, reload, onSaved, saver, handle, find, translating, setTranslating }: {
  shown: Shown; actions: ReactNode; reload: () => void; onSaved: (words: ReactNode) => void;
  saver: (s: Saver | null) => void; handle: (h: EditorHandle | null) => void; find?: string;
  translating: boolean; setTranslating: (t: boolean) => void;
}) {
  const row = shown.text;
  const item = row.kind !== PLAIN;
  const [open, setOpen] = useState<Dialogs | null>(null);
  const [said, setSaid] = useState<ReactNode>(null);
  const done = (words: ReactNode) => { setOpen(null); setSaid(words); reload(); };
  const named = textName(row);
  const title = row.title ? named : named[0].toUpperCase() + named.slice(1); // a real title (often a file name) stays as written
  const theirs = shown.upstream;
  const kinds = useKinds();
  const facts = (
    <>
      <strong className="content-title" title={title}>{title}</strong>
      <StatusBadge status={shown.status} />
      <FixedKind kind={row.kind ?? PLAIN} meaning={kinds?.find((k) => k.id === (row.kind ?? PLAIN))?.meaning} />
      <span className="ledger aside">v{row.content_version}{!item && <> · {codePoints(row.text ?? "").toLocaleString("en")} chars</>}</span>
      {theirs && <span className="aside" title="Only the upstream this depot follows decides on it, changes its licenses, content or credits, or erases it">the upstream's, read only</span>}
    </>
  );
  const more = (
    <MoreMenu label="More on this text">
      <Button variant="quiet" onClick={() => setOpen("record")}>Its record</Button>
      {!theirs && <Button variant="quiet" onClick={() => setOpen("decide")}>Decide, with a reason</Button>}
      {!theirs && <Button variant="quiet" onClick={() => setOpen("relicense")}>Change licenses</Button>}
      {!theirs && <Button variant="quiet" onClick={() => setOpen("credits")}>Edit credits</Button>}
      {!theirs && <Button variant="quiet" onClick={() => setOpen("erase")}>Ask to erase</Button>}
    </MoreMenu>
  );
  return (
    <>
      <div className="text-column">
        {said && <p className="done cue content-said" role="status">{said}</p>}
        {theirs ? <><div className="content-line small">{facts}</div><ReadOnly row={row} find={find} /></>
          : <EditContent key={row.id} row={row} status={shown.status} facts={facts} onDone={done} onLeftSaved={onSaved}
              saver={saver} handle={handle} find={find} translating={translating} setTranslating={setTranslating} />}
      </div>
      <ReviewPanel key={`${row.id}:${row.revision}`} text={row} status={shown.status} decisions={shown.decisions.length} readOnly={theirs}
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
      <Dialog open={open === "erase"} onClose={() => setOpen(null)} title="Ask to erase a passage">
        {open === "erase" && <div className="wide"><EraseForm row={row} onDone={done} /></div>}
      </Dialog>
    </>
  );
}

// An upstream text, read only: its text, a conversation as a transcript, or another item's content as JSON.
function ReadOnly({ row, find }: { row: Row; find?: string }) {
  const item = row.kind !== PLAIN;
  const content = useMemo(() => (item ? JSON.parse(row.content ?? "null") : null), [item, row.content]);
  if (isConversation(content)) return <div className="viewer"><Transcript content={content} find={find} /></div>;
  return <TextViewer item={item} content={content} text={row.text ?? ""} mark={null} />;
}

function Record({ shown }: { shown: Shown }) {
  const row = shown.text;
  const item = row.kind !== PLAIN;
  const [allErasures, setAllErasures] = useState<{ id: string; status: Status; needles: { text: string }[] }[]>([]);
  useEffect(() => { get<typeof allErasures>("/api/erasures").then((a) => a.ok && setAllErasures(a.data)); }, [row.id]);
  const erasures = allErasures.filter((e) => e.needles.some((n) => n.text === row.id));
  return (
    <div className="record small">
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
        note={status === "under review" ? "A public decision that puts it back under review: the decisions before it no longer count; a snapshot made before keeps the revision it holds." : undefined} />
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
// handle's focus). Above it, the facts line, with its kind (kind-picker.tsx) greyed since a text keeps the
// kind it was added with, "Find" and its count of lines. Once it differs, a mark in the facts line, above the
// editor and so in view while typing on a phone too, says it is not saved yet, and a bar under it, in the same
// colour, says when it is saved (Saver: on the next move or decision of the Texts page, or Ctrl+S) and offers Revert;
// saving makes a new content version, a public decision under review, so the text goes back under review, an
// admitted one too. What that decision records and confirms is said in full the first time this browser shows the
// bar (SEEN), then in one line with a "?" that opens it. The saves (edit-saving.ts) send what was typed while a save
// was on its way too, and the editor keeps it when the text is read again after its own save. An edit still waiting
// when the editor closes (the browser's back button, a filter that opens another text) is saved then, and said
// through onLeftSaved; one that cannot be saved then is said too, and kept in this tab, so the text opens again with
// it, unsaved; a reload or a closed tab asks first.
function EditContent({ row, status, facts, onDone, onLeftSaved, saver, handle, find, translating, setTranslating }: {
  row: Row; status: Status; facts: ReactNode; onDone: (words: ReactNode) => void; onLeftSaved: (words: ReactNode) => void;
  saver: (s: Saver | null) => void; handle: (h: EditorHandle | null) => void; find?: string;
  translating: boolean; setTranslating: (t: boolean) => void;
}) {
  const item = row.kind !== PLAIN;
  const original = useMemo(() => (item ? JSON.parse(row.content ?? "null") : null), [item, row.content]);
  const conversation = isConversation(original);
  const [editing, setEditing] = useState(!conversation);
  useEffect(() => {
    if (!conversation || editing) return;
    handle({ focus: () => setEditing(true), openFind: () => {} });
    return () => handle(null);
  }, [conversation, editing, handle]);
  // an edit this tab could not save when the editor closed, kept to start from when the text opens again unchanged
  const [restored] = useState(() => {
    const one = unsaved.get(row.id);
    unsaved.delete(row.id);
    return one && one.version === row.content_version ? one : null;
  });
  useEffect(() => { if (restored) tell(null); }, [restored]); // the edit the line said was kept is in view again
  // what the plain editor starts from: the text, or the edit kept; again on Revert and when the text changed elsewhere
  const [start, setStart] = useState<string>(() => (restored && !item ? restored.text! : row.text ?? ""));
  const [doc, setDoc] = useState<{ text: Text; changed: boolean } | null>(null); // the editor's, once edited
  const [content, setContent] = useState<unknown>(() => (restored && item ? JSON.parse(restored.text!) : original));
  const [editor, setEditor] = useState<EditorHandle | null>(null);
  const [fresh, setFresh] = useState(0); // raised by Revert: the editor starts again from the text as saved
  const [saving, setSaving] = useState(false);
  const [problem, setProblem] = useState<ReactNode>(null); // why the last save saved nothing
  const [chatProblems, setChatProblems] = useState<string[]>([]); // what the depot would refuse in the conversation edited
  const kinds = useKinds();
  const schema = (item && kinds?.find((k) => k.id === row.kind)?.schema) || null;
  const handed = useCallback((h: EditorHandle | null) => { setEditor(h); handle(h); }, [handle]);
  const who = useDeclarant();
  const { about } = useAbout();
  // what a save sends and needs, read when it runs, so the page's saver, a save on its way and the save on closing see
  // the latest edit; snap: the content compared with what the depot holds, a plain text's document by identity
  // (the editor's own Text, or start while it holds start), an item as JSON with its keys in order
  const snapOf = (): Content<Snap> => item ? { snap: canonical(content), body: { content } }
    : { snap: doc?.changed ? doc.text : start, body: { text: doc?.changed ? doc.text.toString() : start } };
  const title = textName(row);
  const latest = useRef({ snapOf, who, about, chatProblems, conversation, title });
  latest.current = { snapOf, who, about, chatProblems, conversation, title };
  const told = useRef({ onDone, onLeftSaved }); // the latest callbacks, so the saves keep one identity
  told.current = { onDone, onLeftSaved };
  const mounted = useRef(true);
  const saves = useRef<EditSaves<Snap> | null>(null);
  if (!saves.current) {
    saves.current = new EditSaves<Snap>(item ? canonical(original) : row.text ?? "", {
      now: () => latest.current.snapOf(),
      same: (a, b) => a === b,
      refusal: () => {
        const now = latest.current;
        if (now.conversation && now.chatProblems.length) return now.chatProblems[0];
        if (!declared(now.who)) return "set a name and email in Settings to save the edit.";
        if (!now.about) return "the depot's requirements are not read yet; try again in a moment.";
        return null;
      },
      send: (body) => {
        const who = latest.current.who;
        return post<{ decision: string | null; version: number | null }>(`/api/texts/${encodeURIComponent(row.id)}/content`,
          { ...body, declaration: { name: who.name.trim(), email: who.email.trim(), accepted: true } satisfies Declaration })
          .then((a) => (a.ok ? { ok: true as const, ...a.data } : a));
      },
      onStart: () => { if (mounted.current) { setSaving(true); setProblem(null); } },
      onEnd: (saved) => {
        const words = savedWords(saved, mounted.current, latest.current.title, row.id);
        if (mounted.current) {
          setSaving(false);
          if (!saved.ok) setProblem(words);
          if (saved.sent) told.current.onDone(saved.ok ? words : <>{madeWords(saved.made)} {words}</>);
        } else {
          // one not saved is said on whatever page is open now, which may not be the Texts page
          if (!saved.ok) { keepUnsaved(row.id, saves.current!.version ?? row.content_version, latest.current.snapOf()); tell(words); }
          if (saved.sent) told.current.onLeftSaved(saved.ok ? words : madeWords(saved.made));
        }
      },
    });
  }
  const changed = saves.current.dirty();
  const save = useCallback(async (): Promise<SaveResult> => {
    const saved = await saves.current!.save();
    return { ok: saved.ok, words: saved.ok && !saved.sent ? null : savedWords(saved, true, title, row.id) };
  }, [title, row.id]);
  useEffect(() => {
    saver({ dirty: () => saves.current!.dirty(), save });
    return () => saver(null);
  }, [saver, save]);
  // the text read again: when its content version is one these saves made (or older, a read that left before the
  // save answered), the editor keeps what it holds, words typed since the save left included; when it changed
  // elsewhere, the editor starts over from it
  const seenVersion = useRef(row.content_version);
  useEffect(() => {
    if (seenVersion.current === row.content_version) return;
    seenVersion.current = row.content_version;
    const mine = saves.current!.version;
    if (saves.current!.saving() || (mine !== null && row.content_version <= mine)) return;
    saves.current!.restart(item ? canonical(original) : row.text ?? "");
    setStart(row.text ?? ""); setDoc(null); setContent(original); setFresh((n) => n + 1);
  }, [row.content_version, row.text, item, original]);
  // an edit still waiting when the editor closes is saved then, or kept in this tab when it cannot be, and said
  // either way through onLeftSaved; a reload or a closed tab asks first
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      const saves_ = saves.current!;
      if (saves_.dirty() || saves_.saving()) void saves_.save();
    };
  }, []);
  useEffect(() => {
    if (!changed) return;
    const leaving = (e: BeforeUnloadEvent) => { e.preventDefault(); };
    window.addEventListener("beforeunload", leaving);
    return () => window.removeEventListener("beforeunload", leaving);
  }, [changed]);
  // back to what the depot holds as this editor knows it: the text as opened, or as its last save left it
  const revert = () => {
    const held = saves.current!.held;
    if (item) setContent(JSON.parse(held as string));
    else setStart(typeof held === "string" ? held : held.toString());
    if (!item) saves.current!.restart(typeof held === "string" ? held : held.toString());
    setDoc(null); setProblem(null); setFresh((n) => n + 1);
  };
  // Ctrl+S (Cmd+S) saves at once, the text staying open
  useEffect(() => {
    const press = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "s") return;
      e.preventDefault();
      void save();
    };
    window.addEventListener("keydown", press);
    return () => window.removeEventListener("keydown", press);
  }, [save]);
  const ready = declared(who) && about !== null && !(conversation && chatProblems.length);
  // the whole of what a saved edit records, said in full once in this browser, then behind a "?"
  const [seenBefore] = useState(readSeen);
  useEffect(() => { if (changed && ready) keepSeen(); }, [changed, ready]);
  return (
    <div className="edit-content">
      <div className="content-line small">
        {facts}
        {changed && <span className={`edited-mark${saving ? " saving" : ""}`} role="status">{saving ? "saving…" : "● edited, not saved"}</span>}
        {!item && <span className="content-tools">
          <span className="aside">{plural(doc ? doc.text.lines : lineCount(start), "line")}</span>
          <Button variant="quiet" disabled={!editor || translating} onClick={() => editor?.openFind()} title='Find and replace in this text (Ctrl+F); tick "regexp" for a regular expression'>Find</Button>
          <Button variant="quiet" aria-pressed={translating} onClick={() => setTranslating(!translating)}
            title="Read it translated, read only, by Mozilla's models in this browser: nothing of the translation is saved">{translating ? "Back to the text" : "Translate"}</Button>
        </span>}
        {conversation && !editing && <span className="content-tools"><span className="aside">{plural(original.messages.length, "message")}</span>
          <Button variant="quiet" onClick={() => setEditing(true)} aria-keyshortcuts="E" title="Opens the form of its messages, roles and blocks (E)">Edit</Button></span>}
        {conversation && editing && !changed && <span className="content-tools"><Button variant="quiet" onClick={() => setEditing(false)}>Read it</Button></span>}
      </div>
      {conversation && !editing ? <div className="viewer content-scroll"><Transcript content={original} find={find} /></div>
        : item ? (
          <div className="content-scroll">{conversation && isConversation(content)
            ? <ConversationEditor key={fresh} value={content as Conversation} onChange={setContent} onProblems={setChatProblems} />
            : schema ? <SchemaForm key={fresh} value={content} schema={schema} root={schema} name={row.kind!} words={KIND_WORDS[row.kind!]} need="required" onChange={setContent} /> : <Skeleton lines={4} label="Reading the kind's schema" />}</div>
        ) : (
          <>
            {/* hidden, not closed, while the translation shows: an edit not saved and its undo stay as they were; a
                browser's own page translation must not rewrite the editor's text, which would then be saved */}
            <div className="editor-slot" hidden={translating} translate="no">
              <Suspense fallback={<Skeleton lines={8} label="Opening the editor" />}>
                <CodeEditor key={fresh} initial={start} onChange={(text, edited) => setDoc({ text, changed: edited })} label={`The text of ${row.title ?? row.id}`} handle={handed} find={find} />
              </Suspense>
            </div>
            {translating && !item && <Translated id={row.id} text={doc?.changed ? doc.text.toString() : start} tags={row.tags} />}
          </>
        )}
      {changed && (
        <div className={`save-bar small${status === "admitted" ? " asking" : ""}`} role="status" aria-label="Edit not saved yet">
          <span className={saving ? "act" : undefined}>{saving ? "Saving the edit…" : <><strong className="edited-word">Not saved yet:</strong> saved when you move on (Next, ‹, Admit &amp; next) or with Ctrl+S.</>}
            {status === "admitted" && !saving && <> It is admitted: saving puts it back under review, and Admit &amp; next admits the new version.</>}</span>
          <Button variant="quiet" disabled={saving} onClick={revert}>Revert</Button>
          <span className="aside save-words">{conversation && chatProblems.length ? <>{chatProblems[0]}</>
            : ready && seenBefore
            ? <>A public decision as {who.name}, the text back under review.<Help label="what saving records">Content version {row.content_version + 1}, the text back under review; a snapshot made before keeps the version it holds. It confirms: <q>{about!.confirmation}</q> <a href={about!.requirements} target="_blank" rel="noreferrer">The requirements</a>.</Help></>
            : ready
            ? <>A public decision as {who.name}: content version {row.content_version + 1}, the text back under review; a snapshot made before keeps the version it holds. It confirms: <q>{about!.confirmation}</q> <a href={about!.requirements} target="_blank" rel="noreferrer">The requirements</a>.</>
            : declared(who) ? <>Reading the depot's requirements before it can save…</>
            : <>Set a name and email in <a href={settingsHref()}>Settings</a> to save.</>}</span>
        </div>
      )}
      {problem && changed && <p className="act small" role="alert">{problem}</p>}
    </div>
  );
}

type Snap = Text | string;

// What a save did, in words: the page's line while the editor is open; once it closed (left by a way the page does not
// guard), the same with the text named, and for an edit it could not save, that the edit is kept in this tab with a
// link back to the text, where it opens again unsaved.
function savedWords(saved: Saved, open: boolean, title: string, id: string): ReactNode {
  const name = <a href={href(["texts", id])}>{title}</a>;
  if (!saved.ok) {
    const why = saved.before && saved.problems[0].startsWith("set a name and email")
      ? <>set a name and email in <a href={settingsHref()}>Settings</a> to save the edit.</> : saved.problems.join(" ");
    return open ? <>Not saved, so it stays open: {why}</>
      : <>The edit of {name} was not saved: {why} It is kept in this tab: open {name} again to save it.</>;
  }
  const done = saved.made ? madeWords(saved.made) : <>Nothing to save: its content is this already.</>;
  return open ? done : <>{name}: {done}</>;
}
const madeWords = (made: Made | null): ReactNode => made
  ? <>Edit saved as content version {made.version} (<a href={href(["decisions", made.decision])}>{made.decision}</a>): under review again.</> : null;

// The edits this tab could not save when their editor closed, by text: the content version they were made on and the
// edit (a plain text, or an item's JSON), taken back when the text opens again at that version; in memory only.
const unsaved = new Map<string, { version: number; text: string }>();
function keepUnsaved(id: string, version: number, now: Content<Snap>) {
  unsaved.set(id, { version, text: "content" in now.body ? JSON.stringify(now.body.content) : String(now.body.text) });
}

// Whether this browser showed in full what a saved edit records (localStorage; none kept: shown in full each time).
const SEEN = "wpt-depot-edit-terms-seen";
function readSeen(): boolean {
  try { return window.localStorage.getItem(SEEN) === "yes"; } catch { return false; }
}
function keepSeen() {
  try { window.localStorage.setItem(SEEN, "yes"); } catch { /* not kept: said in full again next time */ }
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
          : <>Changes: {Object.keys(changes).join(", ")}.<Help label="saving credits">Saved as a decision under review whose reason names what changed. The decisions before it keep counting: the text keeps its status, and a snapshot made before keeps the revision it holds.</Help></>} />
      <Outcome answer={answer} done={() => ""} />
    </div>
  );
}
