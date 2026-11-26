// A conversation (an item of kind conversation, docs/chat-format.md) edited
// as a chat, in the design system's messages, as the transcript reads it:
// each message under the name of its role, always shown, with one button a
// role to change it (system, user, assistant, tool). A message changes role
// only when the new role can hold what it holds: a tool's results unwrap to
// their blocks and blocks wrap into a result, an empty text or thinking block
// is dropped, and a button that would lose something is off and says why.
// A new message takes the role that usually follows the last one: the
// assistant after the user, the user after the assistant, a tool after an
// assistant's call not answered yet (its result answering that call), the
// assistant after a tool, so most messages need no role click; the other
// roles stand beside it. Each block has a text box of its own (a text, a
// thinking block, a call's arguments as JSON; an excerpt's passage is
// selected in its text, passage-picker.tsx, or typed in a fold), and a
// message offers only the blocks the depot takes in its role: text and
// excerpt for the user, text, thinking and tool call for the assistant, text
// for the system, a tool's result for a tool, whose content takes text and
// excerpt. A tool call gets the next free id (c1, c2...) and its tool from
// one click on a tool offered; a result answers a call picked with one click,
// the oldest one not answered first. Messages and blocks move up and down
// and are deleted with a button each. Blocks the editor does not write (a
// JSON value, a file) are kept as they are, shown, and can be deleted. A
// message a model drafted is marked edited once changed. The tools offered
// are edited as JSON in a fold; the conversation's metadata is kept as it
// is. What the depot would refuse is listed (onProblems), so the page can
// hold its save until it is fixed.

import { Button, Checkbox, Message, TextArea, TextInput } from "@workers-intelligence-commons/components";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { plural } from "./parts";
import { PassagePicker, QuotePreview, quotedIn, useTitle, type Selector } from "./passage-picker";

export type Role = "system" | "user" | "assistant" | "tool";
export type Block = {
  type: string; text?: string; id?: string; name?: string; arguments?: Record<string, unknown>; call?: string; error?: boolean;
  content?: Block[]; value?: unknown; sha256?: string; media_type?: string; source?: string; selector?: Selector;
};
type Draft = { model: string; weights: Record<string, string>; sha256: string; edited: boolean };
export type Turn = { role: Role; content: Block[]; metadata?: { draft?: Draft } & Record<string, unknown> };
type Tool = { name: string; description: string; parameters: Record<string, unknown> };
export type Conversation = { messages: Turn[]; tools?: Tool[]; metadata?: Record<string, unknown> };

export const ROLES: Role[] = ["system", "user", "assistant", "tool"];
// The name each role shows above its message; the design system names only the user and the assistant.
export const ROLE_LABEL: Record<Role, string> = { system: "system", user: "user", assistant: "assistant", tool: "tool" };

// The blocks a message of each role may hold (the kind's schema), and those the editor offers to add.
const HOLDS: Record<Role, string[]> = {
  system: ["text", "json", "file"], user: ["text", "json", "file", "excerpt"],
  assistant: ["text", "thinking", "tool_call", "json", "file"], tool: ["tool_result"],
};
const ADDS: Record<Role, string[]> = { system: ["text"], user: ["text", "excerpt"], assistant: ["text", "thinking", "tool_call"], tool: ["tool_result"] };
const RESULT_HOLDS = ["text", "json", "file", "excerpt"];
const RESULT_ADDS = ["text", "excerpt"];
const BLOCK_NAME: Record<string, string> = {
  text: "text", thinking: "thinking", tool_call: "tool call", tool_result: "tool result", excerpt: "excerpt", json: "JSON value", file: "file",
};

const text = (): Block => ({ type: "text", text: "" });
const empty = (b: Block) => (b.type === "text" || b.type === "thinking") && !(b.text ?? "").trim();
const calls = (messages: Turn[]) => messages.flatMap((m) => (m.role === "assistant" ? m.content.filter((b) => b.type === "tool_call") : []));
const answered = (messages: Turn[]) => new Set(messages.flatMap((m) => m.content.filter((b) => b.type === "tool_result").map((b) => b.call)));
// The calls made before message `at` that no result answers yet, oldest first.
const open = (messages: Turn[], at: number) => {
  const done = answered(messages);
  return calls(messages.slice(0, at)).filter((c) => !done.has(c.id));
};
const freeId = (messages: Turn[]) => {
  const used = new Set(calls(messages).map((c) => c.id));
  let n = 1;
  while (used.has(`c${n}`)) n += 1;
  return `c${n}`;
};

// The role that usually follows the last message.
export function nextRole(messages: Turn[]): Role {
  const last = messages.at(-1);
  if (!last || last.role === "system") return "user";
  if (last.role === "user" || last.role === "tool") return "assistant";
  return open(messages, messages.length).length ? "tool" : "user";
}

const result = (messages: Turn[], at: number, content: Block[] = [text()]): Block =>
  ({ type: "tool_result", call: open(messages, at)[0]?.id ?? "", content });

const startFor = (role: Role, messages: Turn[], at: number): Block[] => (role === "tool" ? [result(messages, at)] : [text()]);

// The message in another role, or why it cannot take it without losing something.
function recast(messages: Turn[], at: number, role: Role): Turn | string {
  const m = messages[at];
  if (m.role === role) return m;
  if (m.metadata?.draft && role !== "assistant") return "A model drafted it; only an assistant message records a draft.";
  let blocks = m.content.filter((b) => !empty(b));
  if (role === "tool") {
    const loose = blocks.filter((b) => b.type !== "tool_result");
    const lost = loose.find((b) => !RESULT_HOLDS.includes(b.type));
    if (lost) return `It holds a ${BLOCK_NAME[lost.type] ?? lost.type} block, which a tool's result cannot hold.`;
    blocks = [...blocks.filter((b) => b.type === "tool_result"), ...(loose.length ? [result(messages, at, loose)] : [])];
  } else {
    blocks = blocks.flatMap((b) => (b.type === "tool_result" ? (b.content ?? []).filter((c) => !empty(c)) : [b]));
    const lost = blocks.find((b) => !HOLDS[role].includes(b.type));
    if (lost) return `It holds a ${BLOCK_NAME[lost.type] ?? lost.type} block, which a ${ROLE_LABEL[role]} message cannot hold.`;
  }
  if (!blocks.length) blocks = startFor(role, messages, at);
  return { ...m, role, content: blocks };
}

// What the depot would refuse in the conversation, in words, and the JSON boxes that do not parse.
export function problemsOf(c: Conversation, badJson: string[] = []): string[] {
  const found: string[] = [];
  if (!c.messages.length) found.push("A conversation holds at least one message.");
  c.messages.forEach((m, i) => {
    const where = `Message ${i + 1} (${ROLE_LABEL[m.role] ?? m.role})`;
    if (m.role !== "assistant" && !m.content.length) found.push(`${where} holds no block; only an assistant message may be empty.`);
    const each = (b: Block) => {
      if (b.type === "tool_call" && !(b.name ?? "").trim()) found.push(`${where}: a tool call names no tool.`);
      if (b.type === "tool_result" && !b.call) found.push(`${where}: a tool result answers no call.`);
      if (b.type === "excerpt" && !(b.source ?? "").trim()) found.push(`${where}: an excerpt names no quoted text.`);
      if (b.type === "excerpt" && !b.selector?.exact) found.push(`${where}: an excerpt quotes no passage.`);
      b.content?.forEach(each);
    };
    m.content.forEach(each);
  });
  badJson.forEach((key) => found.push(key === "tools" ? "The tools offered are not a list of tools in JSON." : `The arguments of the call ${key} are not a JSON object.`));
  return found;
}

const rows = (value: string, least = 2) => Math.min(14, Math.max(least, value.split("\n").length + 1));

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

// A tool call's arguments as JSON, kept as typed while it does not parse.
function Arguments({ block, onChange, onBad }: { block: Block; onChange: (args: Record<string, unknown>) => void; onBad: (bad: boolean) => void }) {
  const shown = (args: unknown) => JSON.stringify(args ?? {}, null, 2);
  const [draft, setDraft] = useState(shown(block.arguments));
  const [bad, setBad] = useState(false);
  // a change from outside (a move, a revert) shows the arguments it brings
  useEffect(() => {
    let same = false;
    try { same = JSON.stringify(JSON.parse(draft)) === JSON.stringify(block.arguments ?? {}); } catch { /* the draft does not parse */ }
    if (!same && !bad) setDraft(shown(block.arguments));
  }, [block.arguments]); // eslint-disable-line react-hooks/exhaustive-deps
  const typed = (value: string) => {
    setDraft(value);
    try {
      const parsed = JSON.parse(value || "{}");
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("not an object");
      setBad(false); onBad(false); onChange(parsed);
    } catch { setBad(true); onBad(true); }
  };
  useEffect(() => () => onBad(false), []); // eslint-disable-line react-hooks/exhaustive-deps
  return <TextArea label="Arguments, as a JSON object" className="code" rows={rows(draft)} value={draft} onChange={(e) => typed(e.target.value)}
    error={bad ? "Not a JSON object yet: the call keeps its last arguments that were one." : undefined} />;
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

function BlockEditor({ block, role, messages, at, tools, autoOpen, onChange, onBad }: {
  block: Block; role: Role; messages: Turn[]; at: number; tools: Tool[]; autoOpen: boolean; onChange: (b: Block) => void; onBad: (id: string, bad: boolean) => void;
}) {
  switch (block.type) {
    case "text":
    case "thinking":
      // the block's head already names it (text, thinking): the box needs no second label on screen
      return <textarea className="wic-input wic-textarea wic-focus chat-text" aria-label={block.type === "text" ? "Text" : "Thinking"} rows={rows(block.text ?? "", 3)} value={block.text ?? ""}
        onChange={(e) => onChange({ ...block, text: e.target.value })} />;
    case "excerpt":
      return <ExcerptEditor block={block} messages={messages} autoOpen={autoOpen} onChange={onChange} />;
    case "tool_call":
      return (
        <div className="chat-fields">
          <div className="chat-pair">
            <TextInput label={<>Tool <span className="aside">call {block.id}</span></>} value={block.name ?? ""} onChange={(e) => onChange({ ...block, name: e.target.value })} />
          </div>
          {tools.length > 0 && (
            <div className="chips" aria-label="Tools offered">
              {tools.map((t) => <button key={t.name} type="button" className="chip-button" aria-pressed={block.name === t.name} title={t.description}
                onClick={() => onChange({ ...block, name: t.name })}>{t.name}</button>)}
            </div>
          )}
          <Arguments block={block} onChange={(args) => onChange({ ...block, arguments: args })} onBad={(bad) => onBad(block.id ?? "", bad)} />
        </div>
      );
    case "tool_result": {
      const before = calls(messages.slice(0, at));
      const done = answered(messages);
      const inner = block.content ?? [];
      const setInner = (content: Block[]) => onChange({ ...block, content });
      return (
        <div className="chat-fields">
          <div className="chips" role="group" aria-label="The call it answers">
            <span className="small aside">Answers</span>
            {before.length === 0 && <span className="small aside">no call before it</span>}
            {before.map((c) => <button key={c.id} type="button" className="chip-button" aria-pressed={block.call === c.id}
              onClick={() => onChange({ ...block, call: c.id })}>{c.id} {c.name}{done.has(c.id) && block.call !== c.id ? " (answered)" : ""}</button>)}
          </div>
          <Checkbox label="The call failed" checked={Boolean(block.error)} onChange={(e) => {
            const { error: _, ...rest } = block;
            onChange(e.target.checked ? { ...rest, error: true } : rest);
          }} />
          <Blocks blocks={inner} role={role} inResult messages={messages} at={at} tools={tools} onChange={setInner} onBad={onBad} />
        </div>
      );
    }
    default:
      return <pre className="turn-json ledger">{JSON.stringify(block.type === "json" ? block.value : block, null, 2)}</pre>;
  }
}

// The blocks of a message, or of a tool's result, each with its moves, and the buttons that add one.
function Blocks({ blocks, role, inResult = false, messages, at, tools, onChange, onBad }: {
  blocks: Block[]; role: Role; inResult?: boolean; messages: Turn[]; at: number; tools: Tool[];
  onChange: (blocks: Block[]) => void; onBad: (id: string, bad: boolean) => void;
}) {
  const adds = inResult ? RESULT_ADDS : ADDS[role];
  const [fresh, setFresh] = useState<number | null>(null); // the excerpt just added, whose picker opens at once
  const add = (type: string) => {
    setFresh(type === "excerpt" ? blocks.length : null);
    const block: Block = type === "tool_call" ? { type, id: freeId(messages), name: tools.length === 1 ? tools[0].name : "", arguments: {} }
      : type === "tool_result" ? result(messages, at)
      : type === "excerpt" ? { type, source: "", selector: { type: "TextQuoteSelector", exact: "" } }
      : { type, text: "" };
    onChange([...blocks, block]);
  };
  return (
    <div className={inResult ? "chat-blocks chat-inner" : "chat-blocks"}>
      {blocks.map((b, i) => (
        <div key={i} className={`chat-block chat-block-${b.type}`}>
          <div className="chat-block-head small">
            <span className="turn-label">{BLOCK_NAME[b.type] ?? b.type}</span>
            {blocks.length > 1 || b.type !== "text" || inResult
              ? <Moves what={`the ${BLOCK_NAME[b.type] ?? b.type} block`} at={i} count={blocks.length}
                  onMove={(to) => { setFresh(null); onChange(moved(blocks, i, to)); }} onDelete={() => { setFresh(null); onChange(blocks.filter((_, j) => j !== i)); }} />
              : null}
          </div>
          <BlockEditor block={b} role={role} messages={messages} at={at} tools={tools} onBad={onBad} autoOpen={fresh === i}
            onChange={(nb) => onChange(blocks.map((x, j) => (j === i ? nb : x)))} />
        </div>
      ))}
      <div className="chips chat-adds small" aria-label={inResult ? "Add to the result" : "Add a block"}>
        {adds.map((t) => <Button key={t} variant="quiet" onClick={() => add(t)}>+ {BLOCK_NAME[t]}</Button>)}
      </div>
    </div>
  );
}

export function ConversationEditor({ value, onChange, onProblems }: {
  value: Conversation; onChange: (c: Conversation) => void; onProblems?: (problems: string[]) => void;
}) {
  const messages = value.messages;
  const tools = value.tools ?? [];
  const [bad, setBad] = useState<string[]>([]);
  const problems = useMemo(() => problemsOf(value, bad), [value, bad]);
  useEffect(() => { onProblems?.(problems); }, [problems]); // eslint-disable-line react-hooks/exhaustive-deps
  const onBad = (id: string, isBad: boolean) => setBad((was) => (isBad ? (was.includes(id) ? was : [...was, id]) : was.filter((x) => x !== id)));
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
                <Blocks blocks={m.content} role={m.role} messages={messages} at={i} tools={tools} onBad={onBad}
                  onChange={(content) => setTurn(i, { ...m, content })} />
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
      <Extras value={value} onChange={onChange} onBad={(b) => onBad("tools", b)} />
      {problems.length > 0 && (
        <div className="chat-problems small" role="status">
          <strong>The depot would refuse it as it is:</strong>
          <ul>{problems.map((p) => <li key={p}>{p}</li>)}</ul>
        </div>
      )}
    </div>
  );
}

// The tools offered, as JSON in a fold, and the conversation's metadata, kept as it is.
function Extras({ value, onChange, onBad }: { value: Conversation; onChange: (c: Conversation) => void; onBad: (bad: boolean) => void }) {
  const tools = value.tools ?? [];
  const [draft, setDraft] = useState(() => (tools.length ? JSON.stringify(tools, null, 2) : ""));
  const [error, setError] = useState<ReactNode>(null);
  const typed = (v: string) => {
    setDraft(v);
    if (!v.trim()) { const { tools: _, ...rest } = value; setError(null); onBad(false); onChange(rest); return; }
    try {
      const parsed = JSON.parse(v);
      if (!Array.isArray(parsed) || parsed.some((t) => !t || typeof t !== "object" || typeof t.name !== "string" || !t.name
        || typeof t.description !== "string" || !t.parameters || typeof t.parameters !== "object")) {
        throw new Error("each tool needs a name, a description and its parameters (a JSON Schema object)");
      }
      setError(null); onBad(false); onChange({ ...value, tools: parsed });
    } catch (e) { setError(`Not a list of tools yet: ${e instanceof Error ? e.message : String(e)}.`); onBad(true); }
  };
  const meta = value.metadata && Object.keys(value.metadata).length ? value.metadata : null;
  return (
    <details className="turn-extras small chat-extras">
      <summary>{tools.length ? `${plural(tools.length, "tool")} offered (${tools.map((t) => t.name).join(", ")})` : "No tool offered"}{meta ? "; its metadata" : ""}</summary>
      <TextArea label="Tools offered, as JSON" className="code" rows={rows(draft, 4)} value={draft} onChange={(e) => typed(e.target.value)}
        placeholder={'[{"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]'}
        hint="Each with its name, description and the JSON Schema of its parameters. Their names then stand as one-click choices in each tool call."
        error={error ?? undefined} />
      {meta && <><p className="aside">Its metadata, never read as tokens, kept as it is:</p><pre className="turn-json ledger">{JSON.stringify(meta, null, 2)}</pre></>}
    </details>
  );
}
