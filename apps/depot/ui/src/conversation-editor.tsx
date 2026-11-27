// A conversation (an item of kind conversation, docs/chat-format.md) edited
// as a chat, in the design system's messages, as the transcript reads it:
// each message under the name of its role, always shown, with one button a
// role to change it (system, user, assistant, tool). A message changes role
// only when the new role can hold what it holds (conversation.ts, recast): a
// tool's results unwrap to their blocks and blocks wrap into a result, an empty
// text or thinking block is dropped, and a button that would lose something is
// off and says why. A new message takes the role that usually follows the last
// one: the assistant after the user, the user after the assistant, a tool after
// an assistant's call not answered yet (its result answering that call), the
// assistant after a tool, so most messages need no role click; the other roles
// stand beside it. An assistant message says how its turn ended with one click
// (stop, tool calls, limit), or says nothing. Each block has a box of its own
// (a text, a thinking block, a call's arguments as JSON, or the raw text of a
// call that could not be parsed; tools defined as JSON; an excerpt's passage is
// selected in its text, passage-picker.tsx, or typed in a fold), and a message
// offers only the blocks the depot takes in its role: text and excerpt for the
// user, text, thinking and tool call for the assistant, text and tools defined
// for the system, a tool's result for a tool, whose content takes text, excerpt
// and tools defined. A text outside an assistant message can be marked as added
// by the harness. A tool call gets the next free id (c1, c2...), its tool from
// one click on a tool known at that point (offered from the start, or defined
// in a tools block before it), and its form with one click: its arguments, its
// arguments not parsed, or the whole call not read. A result answers a call
// picked with one click, the oldest one not answered first, and may keep what
// the tool returned (any JSON value) and the id of the depot conversation of
// the sub-agent that produced it, each added and removed with a button; the
// editor says what that id names, and the depot checks it on save. Messages and
// blocks move up and down and are deleted with a button each. Blocks the editor
// does not write (a JSON value, a file) are kept as they are, shown, and can be
// deleted. A message a model drafted is marked edited once changed. The harness
// the conversation was recorded from (its name and version) and the tools
// offered from the start (as JSON) are edited in a fold; the conversation's
// metadata is kept as it is. What the depot would refuse is listed
// (onProblems), so the page can hold its save until it is fixed.

import { Button, Checkbox, Message, TextArea, TextInput } from "@workers-intelligence-commons/components";
import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import {
  ADDS, BLOCK_NAME, CALL_FORM_LABEL, CALL_FORM_MEANING, CALL_FORMS, END_LABEL, END_MEANING, ENDS, ROLE_LABEL, ROLES, RESULT_ADDS,
  answered, calls, formOf, freeId, harnessOf, nextRole, problemsOf, recast, reform, result, startFor, toolsAt, toolsProblem, withField,
  type Block, type Conversation, type Role, type Tool, type Turn,
} from "./conversation";
import { search, EVERY } from "./filter";
import { plural } from "./parts";
import { PassagePicker, QuotePreview, quotedIn, useTitle, type Selector } from "./passage-picker";
import { href } from "./route";

export type { Block, Conversation, Role, Turn } from "./conversation";
export { nextRole, problemsOf, ROLE_LABEL, ROLES } from "./conversation";

// Reports a box whose text is not yet what it must be: its words, or null once it is.
type OnBad = (key: string, words: string | null) => void;

const rows = (value: string, least = 2) => Math.min(14, Math.max(least, value.split("\n").length + 1));
const shown = (v: unknown) => JSON.stringify(v, null, 2);

// Up, down and delete, for a message or a block.
function Moves({ what, at, count, onMove, onDelete, deleteOff, word }: {
  what: string; at: number; count: number; onMove: (to: number) => void; onDelete: () => void; deleteOff?: string; word?: string;
}) {
  return (
    <span className="chat-moves">
      {word && <span className="small aside">{word}</span>}
      <Button variant="quiet" disabled={at === 0} onClick={() => onMove(at - 1)} aria-label={`Move ${what} up`} title="Move it up">↑</Button>
      <Button variant="quiet" disabled={at === count - 1} onClick={() => onMove(at + 1)} aria-label={`Move ${what} down`} title="Move it down">↓</Button>
      <Button variant="quiet" disabled={Boolean(deleteOff)} onClick={onDelete} aria-label={`Delete ${what}`} title={deleteOff ?? "Delete it"}>✕</Button>
    </span>
  );
}

const moved = <T,>(list: T[], from: number, to: number) => {
  const next = [...list];
  const [item] = next.splice(from, 1);
  next.splice(to, 0, item);
  return next;
};

// A JSON value in a box, kept as typed while it is not what it must be (read
// says why), the value keeping its last good form meanwhile; a change from
// outside (a move, a revert) shows the value it brings. empty: the text of
// the value an empty box stands for, when it stands for one; blank: that value
// is shown as an empty box.
function JsonBox({ label, value, read, words, onChange, onBad, hint, placeholder, empty, blank = false }: {
  label: string; value: unknown; read: (v: unknown) => string | null; words: string; onChange: (v: unknown) => void; onBad: OnBad;
  hint?: string; placeholder?: string; empty?: string; blank?: boolean;
}) {
  const key = useId();
  const show = (v: unknown) => (blank && JSON.stringify(v) === empty ? "" : shown(v));
  const [draft, setDraft] = useState(() => show(value));
  const [bad, setBad] = useState<string | null>(null);
  useEffect(() => {
    let same = false;
    try { same = JSON.stringify(JSON.parse(draft.trim() || (empty ?? ""))) === JSON.stringify(value); } catch { /* the draft does not parse */ }
    if (!same && !bad) setDraft(show(value));
  }, [JSON.stringify(value)]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { onBad(key, bad ? words : null); }, [bad, words]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => onBad(key, null), []); // eslint-disable-line react-hooks/exhaustive-deps
  const typed = (v: string) => {
    setDraft(v);
    let parsed: unknown;
    try { parsed = JSON.parse(v.trim() || (empty ?? "")); } catch { setBad("Not JSON yet"); return; }
    const why = read(parsed);
    if (why) { setBad(why); return; }
    setBad(null); onChange(parsed);
  };
  return <TextArea label={label} className="code" rows={rows(draft)} value={draft} onChange={(e) => typed(e.target.value)} hint={hint} placeholder={placeholder}
    error={bad ? `${bad}: it keeps its last value that was.` : undefined} />;
}

const asObject = (v: unknown) => (v && typeof v === "object" && !Array.isArray(v) ? null : "Not a JSON object yet");
const TOOLS_EXAMPLE = '[{"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}, {"name": "shell"}]';
const TOOLS_HINT = "Each with its name, and its description and the JSON Schema of its parameters when they are known: a tool may be named alone.";

// One click among a few choices, each with its meaning on hover, or why it is off.
function Choices<T extends string>({ label, choices, on, words, meaning, off, onPick }: {
  label: string; choices: T[]; on: T | undefined; words: Record<T, string>; meaning: Record<T, string>; off?: (c: T) => string | undefined; onPick: (c: T) => void;
}) {
  return (
    <div className="chips chat-roles" role="group" aria-label={label}>
      <span className="small aside">{label}</span>
      {choices.map((c) => {
        const why = off?.(c);
        return <button key={c} type="button" className="chip-button" aria-pressed={on === c} disabled={Boolean(why)} title={why ?? meaning[c]}
          onClick={() => onPick(c)}>{words[c]}</button>;
      })}
    </div>
  );
}

// An excerpt: the passage it quotes and the text it is from, selected in the
// PassagePicker (opened at once on a block just added), or typed
// by hand in a fold.
function ExcerptEditor({ block, messages, autoOpen, onChange }: { block: Block; messages: Turn[]; autoOpen: boolean; onChange: (b: Block) => void }) {
  const [picking, setPicking] = useState(autoOpen);
  const sel = block.selector ?? { type: "TextQuoteSelector", exact: "" };
  const title = useTitle(block.source?.trim() || undefined);
  const set = (s: Partial<Selector>) => {
    const next = { ...sel, ...s, type: "TextQuoteSelector" };
    if (!next.prefix) delete next.prefix;
    if (!next.suffix) delete next.suffix;
    onChange({ ...block, selector: next });
  };
  return (
    <div className="chat-fields">
      {sel.exact ? <QuotePreview selector={sel} /> : <p className="small aside">No passage quoted yet.</p>}
      {block.source && <p className="small aside">From {title ? <>{title} ({block.source})</> : block.source}</p>}
      <div className="row tight">
        <Button variant={sel.exact ? "quiet" : "primary"} onClick={() => setPicking(true)}>{sel.exact ? "Pick another passage" : "Pick the passage"}</Button>
      </div>
      <PassagePicker open={picking} onClose={() => setPicking(false)} source={block.source} selector={block.selector} quoted={quotedIn(messages)}
        onPick={(source, selector) => onChange({ ...block, source, selector })} />
      <details className="small chat-by-hand">
        <summary>Type it by hand</summary>
        <div className="chat-fields">
          <TextInput label="Quoted text (its id)" value={block.source ?? ""} placeholder="my-depot-17" onChange={(e) => onChange({ ...block, source: e.target.value })} />
          <TextArea label="The passage, character for character" rows={rows(sel.exact, 2)} value={sel.exact} onChange={(e) => set({ exact: e.target.value })} />
          <div className="chat-pair">
            <TextInput label="Right before it (optional)" value={sel.prefix ?? ""} onChange={(e) => set({ prefix: e.target.value })} />
            <TextInput label="Right after it (optional)" value={sel.suffix ?? ""} onChange={(e) => set({ suffix: e.target.value })} />
          </div>
        </div>
      </details>
    </div>
  );
}

// What the id typed names in the depot, read a moment after the typing stops:
// undefined while it is read, null when no text has it.
type Named = { title: string | null; kind: string } | null | undefined;
function useNamed(id: string): Named {
  const [named, setNamed] = useState<Named>(undefined);
  useEffect(() => {
    setNamed(undefined);
    if (!id) return;
    let live = true;
    const wait = setTimeout(() => {
      search({ ...EVERY, ids: [id] }, {}, 1).then((a) => {
        if (!live) return;
        const row = a.ok ? a.data.rows[0] : undefined;
        setNamed(row ? { title: row.title, kind: row.kind ?? "plain" } : a.ok ? null : undefined);
      });
    }, 300);
    return () => { live = false; clearTimeout(wait); };
  }, [id]);
  return named;
}

// The depot conversation of the sub-agent that produced a result, and what its id names.
function SubAgent({ id, onChange, onRemove }: { id: string; onChange: (id: string) => void; onRemove: () => void }) {
  const named = useNamed(id.trim());
  const said = !id.trim() ? null
    : named === undefined ? <span className="aside">Looking for it…</span>
    : named === null ? <span className="chat-warning">No text of the depot has this id: the depot refuses it, unless the version you change named it already.</span>
    : named.kind !== "conversation" ? <span className="chat-warning">It is a text of kind {named.kind}: the depot takes only a conversation here.</span>
    : <>Names <a href={href(["texts", id.trim()])} target="_blank" rel="noreferrer">{named.title ?? id.trim()}</a>, a conversation of the depot.</>;
  return (
    <div className="chat-optional">
      <div className="chat-optional-head small"><span className="turn-label">The sub-agent's conversation</span>
        <Button variant="quiet" onClick={onRemove} aria-label="Remove the sub-agent's conversation" title="Remove it">✕</Button></div>
      <TextInput label="Its id in the depot" value={id} placeholder="my-depot-41" onChange={(e) => onChange(e.target.value)}
        hint={said ?? "The conversation must be in the depot already: add the sub-agent's conversation first."} />
    </div>
  );
}

function ToolCallEditor({ block, known, onChange, onBad }: { block: Block; known: Tool[]; onChange: (b: Block) => void; onBad: OnBad }) {
  const form = formOf(block);
  const cast = (f: (typeof CALL_FORMS)[number]) => reform(block, f);
  return (
    <div className="chat-fields">
      <Choices label="Kept as" choices={CALL_FORMS} on={form} words={CALL_FORM_LABEL} meaning={CALL_FORM_MEANING}
        off={(f) => { const r = cast(f); return typeof r === "string" ? r : undefined; }}
        onPick={(f) => { const r = cast(f); if (typeof r !== "string") onChange(r); }} />
      {form !== "whole" && (
        <>
          <div className="chat-pair">
            <TextInput label={<>Tool <span className="aside">call {block.id}</span></>} value={block.name ?? ""} onChange={(e) => onChange({ ...block, name: e.target.value })} />
          </div>
          {known.length > 0 && (
            <div className="chips" aria-label="Tools known here">
              {known.map((t) => <button key={t.name} type="button" className="chip-button" aria-pressed={block.name === t.name} title={t.description ?? "Known by its name alone"}
                onClick={() => onChange({ ...block, name: t.name })}>{t.name}</button>)}
            </div>
          )}
        </>
      )}
      {form === "arguments"
        ? <JsonBox label="Arguments, as a JSON object" value={block.arguments ?? {}} read={asObject} empty="{}"
            words={`The arguments of the call ${block.id} are not a JSON object.`} onChange={(v) => onChange({ ...block, arguments: v as Record<string, unknown> })} onBad={onBad} />
        : <TextArea label={form === "named" ? <>The text of its arguments, as the model wrote it <span className="aside">call {block.id}</span></>
              : <>The whole call, as the model wrote it <span className="aside">call {block.id}</span></>}
            className="code" rows={rows(block.raw ?? "")} value={block.raw ?? ""} onChange={(e) => onChange({ ...block, raw: e.target.value })} />}
    </div>
  );
}

function ResultEditor({ block, role, messages, at, value, onChange, onBad }: {
  block: Block; role: Role; messages: Turn[]; at: number; value: Conversation; onChange: (b: Block) => void; onBad: OnBad;
}) {
  const before = calls(messages.slice(0, at));
  const done = answered(messages);
  const hasValue = "value" in block;
  const hasLink = block.conversation !== undefined;
  return (
    <div className="chat-fields">
      <div className="chips" role="group" aria-label="The call it answers">
        <span className="small aside">Answers</span>
        {before.length === 0 && <span className="small aside">no call before it</span>}
        {before.map((c) => <button key={c.id} type="button" className="chip-button" aria-pressed={block.call === c.id}
          onClick={() => onChange({ ...block, call: c.id })}>{c.id} {c.name ?? "(not read)"}{done.has(c.id) && block.call !== c.id ? " (answered)" : ""}</button>)}
      </div>
      <Checkbox label="The call failed" checked={Boolean(block.error)} onChange={(e) => onChange(withField(block, "error", e.target.checked ? true : undefined))} />
      {hasValue && (
        <div className="chat-optional">
          <div className="chat-optional-head small"><span className="turn-label">What the tool returned, structured</span>
            <Button variant="quiet" onClick={() => onChange(withField(block, "value", undefined))} aria-label="Remove what the tool returned" title="Remove it">✕</Button></div>
          <JsonBox label="Any JSON value, never read as tokens" value={block.value} read={() => null}
            words={`What the tool returned for the call ${block.call} is not JSON.`} onChange={(v) => onChange({ ...block, value: v })} onBad={onBad} />
        </div>
      )}
      {hasLink && <SubAgent id={block.conversation ?? ""} onChange={(id) => onChange({ ...block, conversation: id })} onRemove={() => onChange(withField(block, "conversation", undefined))} />}
      <Blocks blocks={block.content ?? []} role={role} inResult messages={messages} at={at} value={value} onChange={(content) => onChange({ ...block, content })} onBad={onBad}
        more={<>
          {!hasValue && <Button variant="quiet" onClick={() => onChange({ ...block, value: {} })} title="What the tool returned in structured form, such as an exit code">+ structured value</Button>}
          {!hasLink && <Button variant="quiet" onClick={() => onChange({ ...block, conversation: "" })} title="The depot conversation of the sub-agent that produced this result">+ sub-agent's conversation</Button>}
        </>} />
    </div>
  );
}

function BlockEditor({ block, role, inResult, messages, at, value, autoOpen, onChange, onBad }: {
  block: Block; role: Role; inResult: boolean; messages: Turn[]; at: number; value: Conversation; autoOpen: boolean; onChange: (b: Block) => void; onBad: OnBad;
}) {
  switch (block.type) {
    case "text":
    case "thinking": {
      // the block's head already names it (text, thinking): the box needs no second label on screen
      const box = <textarea className="wic-input wic-textarea wic-focus chat-text" aria-label={block.type === "text" ? "Text" : "Thinking"} rows={rows(block.text ?? "", 3)} value={block.text ?? ""}
        onChange={(e) => onChange({ ...block, text: e.target.value })} />;
      // an assistant's text is the model's own: the mark is offered elsewhere, and kept where it is to be taken off
      if (block.type === "thinking" || (role === "assistant" && !inResult && !block.by)) return box;
      return (
        <div className="chat-fields">
          {box}
          <Checkbox label="Added by the harness, not by the message's author" checked={Boolean(block.by)}
            onChange={(e) => onChange(withField(block, "by", e.target.checked ? "harness" : undefined))} />
        </div>
      );
    }
    case "excerpt":
      return <ExcerptEditor block={block} messages={messages} autoOpen={autoOpen} onChange={onChange} />;
    case "tool_call":
      return <ToolCallEditor block={block} known={toolsAt(value, at)} onChange={onChange} onBad={onBad} />;
    case "tool_result":
      return <ResultEditor block={block} role={role} messages={messages} at={at} value={value} onChange={onChange} onBad={onBad} />;
    case "tools":
      return <JsonBox label="The tools defined from here on, as JSON" value={block.tools ?? []} read={(v) => toolsProblem(v, 1)} hint={TOOLS_HINT} placeholder={TOOLS_EXAMPLE}
        words={`Message ${at + 1}: the tools defined are not a list of tools yet.`} onChange={(v) => onChange({ ...block, tools: v as Tool[] })} onBad={onBad} />;
    default:
      return <pre className="turn-json ledger">{JSON.stringify(block.type === "json" ? block.value : block, null, 2)}</pre>;
  }
}

const headOf = (b: Block) => (b.type === "text" && b.by ? "text, added by the harness" : BLOCK_NAME[b.type] ?? b.type);

// The blocks of a message, or of a tool's result, each with its moves, and the buttons that add one.
function Blocks({ blocks, role, inResult = false, messages, at, value, onChange, onBad, more }: {
  blocks: Block[]; role: Role; inResult?: boolean; messages: Turn[]; at: number; value: Conversation;
  onChange: (blocks: Block[]) => void; onBad: OnBad; more?: ReactNode;
}) {
  const adds = inResult ? RESULT_ADDS : ADDS[role];
  const [fresh, setFresh] = useState<number | null>(null); // the excerpt just added, whose picker opens at once
  const add = (type: string) => {
    setFresh(type === "excerpt" ? blocks.length : null);
    const known = toolsAt(value, at);
    const block: Block = type === "tool_call" ? { type, id: freeId(messages), name: known.length === 1 ? known[0].name : "", arguments: {} }
      : type === "tool_result" ? result(messages, at)
      : type === "excerpt" ? { type, source: "", selector: { type: "TextQuoteSelector", exact: "" } }
      : type === "tools" ? { type, tools: [] }
      : { type, text: "" };
    onChange([...blocks, block]);
  };
  return (
    <div className={inResult ? "chat-blocks chat-inner" : "chat-blocks"}>
      {blocks.map((b, i) => (
        <div key={i} className={`chat-block chat-block-${b.type}${b.by ? " chat-block-harness" : ""}`}>
          <div className="chat-block-head small">
            <span className="turn-label">{headOf(b)}</span>
            {blocks.length > 1 || b.type !== "text" || inResult
              ? <Moves what={`the ${BLOCK_NAME[b.type] ?? b.type} block`} at={i} count={blocks.length}
                  onMove={(to) => { setFresh(null); onChange(moved(blocks, i, to)); }} onDelete={() => { setFresh(null); onChange(blocks.filter((_, j) => j !== i)); }} />
              : null}
          </div>
          <BlockEditor block={b} role={role} inResult={inResult} messages={messages} at={at} value={value} onBad={onBad} autoOpen={fresh === i}
            onChange={(nb) => onChange(blocks.map((x, j) => (j === i ? nb : x)))} />
        </div>
      ))}
      <div className="chips chat-adds small" aria-label={inResult ? "Add to the result" : "Add a block"}>
        {adds.map((t) => <Button key={t} variant="quiet" onClick={() => add(t)}>+ {BLOCK_NAME[t]}</Button>)}
        {more}
      </div>
    </div>
  );
}

export function ConversationEditor({ value, onChange, onProblems }: {
  value: Conversation; onChange: (c: Conversation) => void; onProblems?: (problems: string[]) => void;
}) {
  const messages = value.messages;
  const [bad, setBad] = useState<Record<string, string>>({});
  const problems = useMemo(() => problemsOf(value, Object.values(bad)), [value, bad]);
  useEffect(() => { onProblems?.(problems); }, [problems]); // eslint-disable-line react-hooks/exhaustive-deps
  const onBad: OnBad = (key, words) => setBad((was) => {
    if ((was[key] ?? null) === words) return was;
    const { [key]: _, ...rest } = was;
    return words === null ? rest : { ...rest, [key]: words };
  });
  const box = useRef<HTMLDivElement>(null);
  const [focusAt, setFocusAt] = useState<number | null>(null);
  useEffect(() => {
    if (focusAt === null) return;
    box.current?.querySelectorAll<HTMLElement>(".chat-turn")[focusAt]?.querySelector<HTMLElement>("textarea, input[type=text]")?.focus();
    setFocusAt(null);
  }, [focusAt]);
  const setMessages = (next: Turn[]) => onChange({ ...value, messages: next });
  const setTurn = (i: number, m: Turn) => {
    const was = messages[i];
    const draft = m.metadata?.draft;
    // a drafted message changed by hand is an edited draft
    if (draft && !draft.edited && JSON.stringify(m.content) !== JSON.stringify(was.content)) m = { ...m, metadata: { ...m.metadata, draft: { ...draft, edited: true } } };
    setMessages(messages.map((x, j) => (j === i ? m : x)));
  };
  const addTurn = (role: Role) => {
    setMessages([...messages, { role, content: startFor(role, messages, messages.length) }]);
    setFocusAt(messages.length);
  };
  const next = nextRole(messages);
  return (
    <div className="conversation-editor" ref={box}>
      <div className="transcript">
        {messages.map((m, i) => {
          const draft = m.metadata?.draft;
          return (
            <div key={i} className={`chat-turn chat-turn-${m.role}`}>
              <Message from={m.role === "user" ? "user" : "assistant"} label={ROLE_LABEL[m.role] ?? m.role}
                details={draft ? `drafted by the model ${draft.model}${draft.edited ? ", then edited" : ", unchanged"}` : undefined}
                actions={<Moves word="This message" what={`message ${i + 1}`} at={i} count={messages.length} onMove={(to) => setMessages(moved(messages, i, to))}
                  onDelete={() => setMessages(messages.filter((_, j) => j !== i))} deleteOff={messages.length === 1 ? "A conversation keeps at least one message." : undefined} />}>
                <div className="chips chat-roles" role="group" aria-label={`Who speaks in message ${i + 1}`}>
                  {ROLES.map((r) => {
                    const cast = recast(messages, i, r);
                    const why = typeof cast === "string" ? cast : undefined;
                    return <button key={r} type="button" className="chip-button" aria-pressed={m.role === r} disabled={Boolean(why)} title={why}
                      onClick={() => typeof cast !== "string" && setTurn(i, cast)}>{ROLE_LABEL[r]}</button>;
                  })}
                </div>
                <Blocks blocks={m.content} role={m.role} messages={messages} at={i} value={value} onBad={onBad}
                  onChange={(content) => setTurn(i, { ...m, content })} />
                {(m.role === "assistant" || m.end) && (
                  <div className="chips chat-roles chat-end" role="group" aria-label={`How the turn of message ${i + 1} ended`}>
                    <span className="small aside">Its turn ended</span>
                    <button type="button" className="chip-button" aria-pressed={!m.end} title="Not recorded" onClick={() => setTurn(i, withField(m, "end", undefined))}>not said</button>
                    {ENDS.map((e) => <button key={e} type="button" className="chip-button" aria-pressed={m.end === e} title={END_MEANING[e]}
                      onClick={() => setTurn(i, { ...m, end: e })}>{END_LABEL[e]}</button>)}
                  </div>
                )}
              </Message>
            </div>
          );
        })}
      </div>
      <div className="chat-new row tight">
        <Button variant="quiet" onClick={() => addTurn(next)}>Add {next === "assistant" ? "an" : "a"} {ROLE_LABEL[next]} message</Button>
        <span className="small aside">or a message of</span>
        {ROLES.filter((r) => r !== next).map((r) => <Button key={r} variant="quiet" onClick={() => addTurn(r)}>{ROLE_LABEL[r]}</Button>)}
      </div>
      <Extras value={value} onChange={onChange} onBad={onBad} />
      {problems.length > 0 && (
        <div className="chat-problems small" role="status">
          <strong>The depot would refuse it as it is:</strong>
          <ul>{problems.map((p) => <li key={p}>{p}</li>)}</ul>
        </div>
      )}
    </div>
  );
}

// The harness it was recorded from and the tools offered from the start, in a
// fold, and the conversation's metadata, kept as it is.
function Extras({ value, onChange, onBad }: { value: Conversation; onChange: (c: Conversation) => void; onBad: OnBad }) {
  const tools = value.tools ?? [];
  const harness = value.harness;
  const setHarness = (name: string, version: string) => onChange(withField(value, "harness", harnessOf(name, version)));
  const meta = value.metadata && Object.keys(value.metadata).length ? value.metadata : null;
  const said = [
    harness ? `recorded from ${[harness.name || "a harness with no name", harness.version].filter(Boolean).join(" ")}` : "no harness named",
    tools.length ? `${plural(tools.length, "tool")} offered (${tools.map((t) => t.name).join(", ")})` : "no tool offered",
    meta ? "its metadata" : null,
  ].filter(Boolean).join("; ");
  return (
    <details className="turn-extras small chat-extras" open={Boolean(harness && !harness.name) || undefined}>
      <summary>{said.charAt(0).toUpperCase() + said.slice(1)}</summary>
      <div className="chat-pair">
        <TextInput label="Harness it was recorded from (optional)" value={harness?.name ?? ""} placeholder="opencode"
          onChange={(e) => setHarness(e.target.value, harness?.version ?? "")} />
        <TextInput label="Its version (optional)" value={harness?.version ?? ""} placeholder="1.19.6"
          onChange={(e) => setHarness(harness?.name ?? "", e.target.value)} />
      </div>
      <p className="aside">The agent harness whose forms the conversation follows; never read as tokens. Clear both to name none.</p>
      <JsonBox label="Tools offered from the start, as JSON" value={tools} empty="[]" blank read={(v) => toolsProblem(v)} placeholder={TOOLS_EXAMPLE}
        hint={`${TOOLS_HINT} Their names then stand as one-click choices in each tool call; a tools block in a system message or a tool's result defines more from there on.`}
        words="The tools offered are not a list of tools in JSON." onBad={onBad}
        onChange={(v) => onChange(withField(value, "tools", (v as Tool[]).length ? (v as Tool[]) : undefined))} />
      {meta && <><p className="aside">Its metadata, never read as tokens, kept as it is:</p><pre className="turn-json ledger">{JSON.stringify(meta, null, 2)}</pre></>}
    </details>
  );
}
