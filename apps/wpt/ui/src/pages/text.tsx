// One text or item as it is now: its text (or its content, as JSON), its
// record (credits, its revision and how many it has, its content version and
// how many it has, every decision with its reason, those before its latest
// license or content change marked as no longer counting, its libraries,
// each locked one with the revision it pinned), and six buttons, each
// opening its form in a dialog: decide on it, change its licenses, edit its
// content (a plain text in a text area, an item through the form of its
// kind's schema), edit its credits (each a new revision; a locked library
// keeps the revision it pinned), report a problem to a library that holds
// it, and ask to erase a passage
// (erase-form.tsx, the text in its reader again, where a passage is chosen
// with the mouse or typed). A text that belongs to the upstream this depot
// follows says so instead, and only the upstream decides on it, changes it
// or erases it: of the six, it keeps only the report to a library of this
// depot's own that holds it. Every decision, tag, license and problem it
// names links to its own view.

import { Badge, Button, Dialog, Select, Skeleton, TextArea, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { get, post, type Answer } from "../api";
import { Confirm, Cut, Facts, Joined, LibraryLink, optional, Outcome, PageHead, Problems, required, RequiredNote, Section, StatusBadge, type Declaration, type Library, type Status } from "../parts";
import { RelicenseForm, relicensed, type Relicensed } from "../decide-forms";
import { EraseForm } from "../erase-form";
import { codePoints, TextViewer } from "../reader";
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
type Shown = { text: Row; status: Status; notice: string | null; decisions: Decision[]; libraries: Library[]; revisions: number;
  pins: { library: string; name: string; revision: number }[]; content_versions: number; upstream: boolean };

export function Text({ id }: { id: string }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  const load = useCallback(() => { get<Shown>(`/api/texts/${encodeURIComponent(id)}`).then(setShown); }, [id]);
  useEffect(load, [load]);
  if (!shown) return <Skeleton lines={6} label="Reading the text" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  return <Loaded key={id} shown={shown.data} reload={load} />;
}

type Dialogs = "decide" | "relicense" | "content" | "credits" | "report" | "erase";

function Loaded({ shown, reload }: { shown: Shown; reload: () => void }) {
  const row = shown.text;
  const item = row.kind !== null;
  const content = useMemo(() => (item ? JSON.parse(row.content ?? "null") : null), [item, row.content]);
  const text = row.text ?? "";
  const [open, setOpen] = useState<Dialogs | null>(null);
  const [said, setSaid] = useState<ReactNode>(null);
  const done = (words: ReactNode) => { setOpen(null); setSaid(words); reload(); };
  const title = row.title ?? (item ? `An item of kind ${row.kind}` : "Untitled text");
  const [allErasures, setAllErasures] = useState<{ id: string; status: Status; needles: { text: string }[] }[]>([]);
  useEffect(() => { get<typeof allErasures>("/api/erasures").then((a) => a.ok && setAllErasures(a.data)); }, [row.id]);
  const erasures = allErasures.filter((e) => e.needles.some((n) => n.text === row.id));
  const theirs = shown.upstream;
  const reportable = shown.libraries.filter((l) => !l.upstream); // only the depot's own libraries take a problem
  const problems = shown.libraries.flatMap((l) => (l.problems ?? []).filter((p) => p.texts.includes(row.id)).map((p) => ({ library: l, problem: p })));
  return (
    <>
      <PageHead title={title}
        meta={<><StatusBadge status={shown.status} /><span className="ledger">{row.id}</span>{theirs && <span className="aside">the upstream's</span>}{item && row.title !== null && <span>an item of kind <span className="ledger">{row.kind}</span></span>}</>}
        chain={<>One of the <a href={href(["texts"])}>texts</a>. Admitted, it can join a <a href={href(["libraries"])}>library</a>; an <a href={href(["erasures"])}>erasure request</a> can cut a passage of it.</>}
        actions={<>
          {!theirs && <Button variant="primary" onClick={() => setOpen("decide")}>Decide</Button>}
          {!theirs && <Button variant="quiet" onClick={() => setOpen("relicense")}>Change licenses</Button>}
          {!theirs && <Button variant="quiet" onClick={() => setOpen("content")}>Edit content</Button>}
          {!theirs && <Button variant="quiet" onClick={() => setOpen("credits")}>Edit credits</Button>}
          {reportable.length > 0 && <Button variant="quiet" onClick={() => setOpen("report")}>Report a problem</Button>}
          {!theirs && <Button variant="quiet" onClick={() => setOpen("erase")}>Ask to erase</Button>}
        </>}
        note={theirs
          ? <>It belongs to the upstream this depot follows: only the upstream decides on it, changes its licenses, content or credits, or erases it, and a pull brings what it did here. {reportable.length > 0 ? "A problem can still be reported to a library of this depot that holds it." : "A library of this depot can still hold it once it is admitted."}</>
          : shown.pins.length > 0
          ? <>A locked library holds it ({shown.pins.map((p) => p.name).join(", ")}): a decision, a license change, a content change or an edit of its credits makes a new revision, and the locked library keeps reading the one it pinned.</>
          : reportable.length === 0 ? `No library${shown.libraries.length ? " of this depot" : ""} holds it, so no problem can be reported on it.` : undefined} />
      {said && <p className="done cue" role="status">{said}</p>}
      <div className="text-layout">
        <section aria-label={item ? "Its content, as JSON" : "Its text"}>
          <TextViewer item={item} content={content} text={text} mark={null} />
        </section>
        <aside className="record small" aria-label="Its record">
          <Section title="Record">
            <Facts rows={[
              ["revision", <span className="ledger">{row.revision} of {shown.revisions}</span>],
              ["content", <span className="ledger">version {row.content_version} of {shown.content_versions}</span>],
              ["characters", item ? null : codePoints(text).toLocaleString("en")],
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
        </aside>
      </div>
      <Dialog open={open === "decide"} onClose={() => setOpen(null)} title="Decide on this text">
        {open === "decide" && <DecideOn id={row.id} onDone={done} />}
      </Dialog>
      <Dialog open={open === "relicense"} onClose={() => setOpen(null)} title="Change the licenses of this text">
        {open === "relicense" && <Relicense row={row} onDone={done} />}
      </Dialog>
      <Dialog open={open === "content"} onClose={() => setOpen(null)} title={item ? "Edit the content of this item" : "Edit the content of this text"}>
        {open === "content" && <div className="wide"><EditContent row={row} libraries={shown.libraries} onDone={done} /></div>}
      </Dialog>
      <Dialog open={open === "credits"} onClose={() => setOpen(null)} title="Edit the credits of this text">
        {open === "credits" && <Credits row={row} notice={shown.notice} onDone={done} />}
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

// The text in a text area, or an item's content through the form of its kind's schema, prefilled; sent only once it differs.
function EditContent({ row, libraries, onDone }: { row: Row; libraries: Library[]; onDone: (words: ReactNode) => void }) {
  const item = row.kind !== null;
  const original = useMemo(() => (item ? JSON.parse(row.content ?? "null") : null), [item, row.content]);
  const [text, setText] = useState(row.text ?? "");
  const [content, setContent] = useState<unknown>(original);
  const [schema, setSchema] = useState<Schema | null>(null);
  const [answer, setAnswer] = useState<Answer<{ decision: string | null; version: number | null }> | null>(null);
  useEffect(() => {
    if (item) get<{ id: string; schema: Schema }[]>("/api/kinds").then((a) => a.ok && setSchema(a.data.find((k) => k.id === row.kind)?.schema ?? null));
  }, [item, row.kind]);
  const changed = item ? JSON.stringify(content) !== JSON.stringify(original) : text !== (row.text ?? "");
  const open = libraries.filter((l) => !l.locked_at && !l.upstream);
  const locked = libraries.filter((l) => l.locked_at);
  const send = async (declaration: Declaration) => {
    const a = await post<{ decision: string | null; version: number | null }>(`/api/texts/${encodeURIComponent(row.id)}/content`, { ...(item ? { content } : { text }), declaration });
    setAnswer(a);
    if (!a.ok) return;
    if (a.data.decision === null) onDone("Nothing changed: its content is this already.");
    else onDone(<>Decision <a href={href(["decisions", a.data.decision])}>{a.data.decision}</a>, under review: content version {a.data.version}.{open.length > 0 && <> It left {open.map((l, i) => <span key={l.id}>{i > 0 && ", "}<LibraryLink library={l} /></span>)}; put it back once it is admitted.</>}</>);
  };
  return (
    <div className="action-form">
      <RequiredNote />
      {item
        ? schema ? <SchemaForm value={content} schema={schema} root={schema} name={row.kind!} need="required" onChange={setContent} /> : <Skeleton lines={4} label="Reading the kind's schema" />
        : <TextArea label={required("Text")} required rows={14} value={text} onChange={(e) => setText(e.target.value)} />}
      <Confirm label="Confirm and save the content" disabled={!changed} onConfirm={send}
        note={!changed ? "Nothing is changed yet."
          : <>Saved as content version {row.content_version + 1}, with a decision under review whose reason names the old and new versions. The decisions before it no longer count: the text goes back under review and leaves {open.length ? <>its open {open.length === 1 ? "library" : "libraries"} ({open.map((l) => l.name).join(", ")})</> : "every open library"}, to be put back after its admission.{locked.length > 0 && <> A locked library ({locked.map((l) => l.name).join(", ")}) keeps reading the content of the revision it pinned.</>}</>} />
      <Outcome answer={answer} done={() => ""} />
    </div>
  );
}

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
