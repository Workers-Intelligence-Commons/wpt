// The rules of a conversation (an item of kind conversation, docs/chat-format.md),
// as its schema says them (packages/depot/src/depot/kinds/conversation.schema.json),
// that the chat editor (conversation-editor.tsx) and the transcript (transcript.tsx)
// follow, with no page: the blocks each role holds, the role a new message takes,
// a message changed to another role, the three forms of a tool call, the tools
// known at a message, how a turn ended, and what the depot would refuse. Run its
// tests with `bun test` in apps/depot/ui.

import type { Selector } from "./passage-picker";

export type Role = "system" | "user" | "assistant" | "tool";
// a tool: its name, and its description and the JSON Schema of its parameters when they are known
export type Tool = { name: string; description?: string; parameters?: Record<string, unknown> };
export type Block = {
  type: string; text?: string; by?: "harness"; id?: string; name?: string; arguments?: Record<string, unknown>; raw?: string;
  call?: string; error?: boolean; value?: unknown; conversation?: string; tools?: Tool[]; content?: Block[];
  sha256?: string; media_type?: string; source?: string; selector?: Selector;
};
export type End = "stop" | "tool_calls" | "limit";
type Draft = { model: string; weights: Record<string, string>; sha256: string; edited: boolean };
export type Turn = { role: Role; content: Block[]; end?: End; metadata?: { draft?: Draft } & Record<string, unknown> };
export type Harness = { name: string; version?: string };
export type Conversation = { messages: Turn[]; tools?: Tool[]; harness?: Harness; metadata?: Record<string, unknown> };

export const ROLES: Role[] = ["system", "user", "assistant", "tool"];
// The name each role shows above its message; the design system names only the user and the assistant.
export const ROLE_LABEL: Record<Role, string> = { system: "system", user: "user", assistant: "assistant", tool: "tool" };

// The blocks a message of each role may hold (the kind's schema), and those the editor offers to add.
export const HOLDS: Record<Role, string[]> = {
  system: ["text", "json", "file", "tools"], user: ["text", "json", "file", "excerpt"],
  assistant: ["text", "thinking", "tool_call", "json", "file"], tool: ["tool_result"],
};
export const ADDS: Record<Role, string[]> = {
  system: ["text", "tools"], user: ["text", "excerpt"], assistant: ["text", "thinking", "tool_call"], tool: ["tool_result"],
};
export const RESULT_HOLDS = ["text", "json", "file", "excerpt", "tools"];
export const RESULT_ADDS = ["text", "excerpt", "tools"];
export const BLOCK_NAME: Record<string, string> = {
  text: "text", thinking: "thinking", tool_call: "tool call", tool_result: "tool result", excerpt: "excerpt", json: "JSON value",
  file: "file", tools: "tools defined",
};

// How an assistant's turn ended, in the words the editor's buttons and the transcript use.
export const ENDS: End[] = ["stop", "tool_calls", "limit"];
export const END_LABEL: Record<End, string> = { stop: "stop", tool_calls: "tool calls", limit: "limit" };
export const END_MEANING: Record<End, string> = {
  stop: "The model closed its turn with no tool call.",
  tool_calls: "The model closed its turn holding tool calls, whether they ran or not.",
  limit: "A limit cut the turn before the model closed it.",
};

export const text = (): Block => ({ type: "text", text: "" });
export const empty = (b: Block) => (b.type === "text" || b.type === "thinking") && !(b.text ?? "").trim();
const isObject = (v: unknown): v is Record<string, unknown> => Boolean(v) && typeof v === "object" && !Array.isArray(v);
export const calls = (messages: Turn[]) => messages.flatMap((m) => (m.role === "assistant" ? m.content.filter((b) => b.type === "tool_call") : []));
export const answered = (messages: Turn[]) => new Set(messages.flatMap((m) => m.content.filter((b) => b.type === "tool_result").map((b) => b.call)));
// The calls made before message `at` that no result answers yet, oldest first.
export const open = (messages: Turn[], at: number) => {
  const done = answered(messages);
  return calls(messages.slice(0, at)).filter((c) => !done.has(c.id));
};
export const freeId = (messages: Turn[]) => {
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

export const result = (messages: Turn[], at: number, content: Block[] = [text()]): Block =>
  ({ type: "tool_result", call: open(messages, at)[0]?.id ?? "", content });

export const startFor = (role: Role, messages: Turn[], at: number): Block[] => (role === "tool" ? [result(messages, at)] : [text()]);

// What a tool's result holds besides its blocks, which unwrapping it to its blocks would lose.
function resultKeeps(b: Block): string | null {
  if ("value" in b) return `its result of the call ${b.call} keeps what the tool returned`;
  if (b.conversation !== undefined) return `its result of the call ${b.call} names a sub-agent's conversation`;
  if (b.error) return `its result of the call ${b.call} says the call failed`;
  return null;
}

// The message in another role, or why it cannot take it without losing something.
export function recast(messages: Turn[], at: number, role: Role): Turn | string {
  const m = messages[at];
  if (m.role === role) return m;
  if (m.metadata?.draft && role !== "assistant") return "A model drafted it; only an assistant message records a draft.";
  if (m.end && role !== "assistant") return "It says how its turn ended; only an assistant message records that.";
  let blocks = m.content.filter((b) => !empty(b));
  if (role === "tool") {
    const loose = blocks.filter((b) => b.type !== "tool_result");
    const lost = loose.find((b) => !RESULT_HOLDS.includes(b.type));
    if (lost) return `It holds a ${BLOCK_NAME[lost.type] ?? lost.type} block, which a tool's result cannot hold.`;
    blocks = [...blocks.filter((b) => b.type === "tool_result"), ...(loose.length ? [result(messages, at, loose)] : [])];
  } else {
    const keeps = blocks.filter((b) => b.type === "tool_result").map(resultKeeps).find(Boolean);
    if (keeps) return `In it, ${keeps}, which only a tool's result holds.`;
    blocks = blocks.flatMap((b) => (b.type === "tool_result" ? (b.content ?? []).filter((c) => !empty(c)) : [b]));
    const lost = blocks.find((b) => !HOLDS[role].includes(b.type));
    if (lost) return `It holds a ${BLOCK_NAME[lost.type] ?? lost.type} block, which a ${ROLE_LABEL[role]} message cannot hold.`;
    if (role === "assistant" && blocks.some((b) => b.type === "text" && b.by)) {
      return "It holds text the harness added, which an assistant message cannot hold: its text is the model's own.";
    }
  }
  if (!blocks.length) blocks = startFor(role, messages, at);
  return { ...m, role, content: blocks };
}

// A tool call has one of three forms: its arguments as a JSON object; the text
// of its arguments as the model wrote them, kept raw under its name, when they
// could not be parsed; or the whole call as the model wrote it, kept raw with
// no name, when not even its name could be read.
export type CallForm = "arguments" | "named" | "whole";
export const CALL_FORMS: CallForm[] = ["arguments", "named", "whole"];
export const CALL_FORM_LABEL: Record<CallForm, string> = { arguments: "arguments", named: "arguments not parsed", whole: "call not read" };
export const CALL_FORM_MEANING: Record<CallForm, string> = {
  arguments: "Its arguments, as a JSON object.",
  named: "Its arguments could not be parsed: the text of its arguments as the model wrote them, under its name.",
  whole: "Not even its name could be read: the whole call as the model wrote it.",
};
export const formOf = (b: Block): CallForm => (b.raw === undefined ? "arguments" : b.name === undefined ? "whole" : "named");

// A whole call is written as the chat template writes a call, so a name and its
// arguments made into one text split back into the same name and arguments.
const wholeOf = (name: string, args: string) => `{"name": ${JSON.stringify(name)}, "arguments": ${args}}`;
function splitWhole(raw: string): { name: string; args: string } | null {
  const found = /^\{"name": ("(?:[^"\\]|\\.)*"), "arguments": ([\s\S]*)\}$/.exec(raw);
  if (!found) return null;
  try { return { name: JSON.parse(found[1]) as string, args: found[2] }; } catch { return null; }
}
const parsedObject = (raw: string): Record<string, unknown> | null => {
  try { const v: unknown = JSON.parse(raw.trim() || "{}"); return isObject(v) ? v : null; } catch { return null; }
};

// The call in another form, or why it cannot take it without losing what the model wrote.
export function reform(b: Block, form: CallForm): Block | string {
  const now = formOf(b);
  if (now === form) return b;
  const { arguments: args, raw, name, ...rest } = b;
  const argsText = args && Object.keys(args).length ? JSON.stringify(args) : "{}";
  if (form === "arguments") {
    const split = now === "whole" ? splitWhole(raw ?? "") : { name: name ?? "", args: raw ?? "" };
    if (!split) return "Its raw text is not a call with a name and arguments: keep it raw, or write it as one first.";
    const parsed = parsedObject(split.args);
    if (!parsed) return "Its raw arguments are not a JSON object: keep them raw, or correct them first.";
    return { ...rest, name: split.name, arguments: parsed };
  }
  if (form === "named") {
    if (now === "arguments") return { ...rest, name: name ?? "", raw: argsText };
    const split = splitWhole(raw ?? "");
    return split ? { ...rest, name: split.name, raw: split.args } : { ...rest, name: "", raw: raw ?? "" };
  }
  return { ...rest, raw: wholeOf(name ?? "", now === "arguments" ? argsText : raw ?? "") };
}

// What is wrong with a list of tools, or null: each tool a name, and a
// description and the JSON Schema of its parameters when they are known; a
// tools block names at least one (least 1).
export function toolsProblem(tools: unknown, least = 0): string | null {
  if (!Array.isArray(tools)) return "they are not a list";
  if (tools.length < least) return "they name no tool";
  for (const [i, t] of tools.entries()) {
    const which = `tool ${i + 1}`;
    if (!isObject(t)) return `${which} is not an object`;
    if (typeof t.name !== "string" || !t.name) return `${which} has no name`;
    if ("description" in t && typeof t.description !== "string") return `the description of ${t.name} is not a text`;
    if ("parameters" in t && !isObject(t.parameters)) return `the parameters of ${t.name} are not a JSON Schema object`;
    const other = Object.keys(t).find((k) => !["name", "description", "parameters"].includes(k));
    if (other) return `${t.name} holds ${other}, which a tool does not hold (only name, description and parameters)`;
  }
  return null;
}

// The tools known when message `at` is written: those offered from the start,
// then those each tools block before it defines (in a system message or a
// tool's result), a later definition of a tool completing an earlier one.
export function toolsAt(c: Conversation, at: number): Tool[] {
  const known = new Map<string, Tool>();
  const learn = (tools: Tool[] | undefined) => tools?.forEach((t) => t?.name && known.set(t.name, { ...known.get(t.name), ...t }));
  learn(c.tools);
  const each = (b: Block) => {
    if (b.type === "tools") learn(b.tools);
    b.content?.forEach(each);
  };
  c.messages.slice(0, at).forEach((m) => m.content.forEach(each));
  return [...known.values()];
}

// The harness with a name and a version typed, or none when both are blank;
// a version without a name stays, for problemsOf to name.
export function harnessOf(name: string, version: string): Harness | undefined {
  if (!name && !version) return undefined;
  return version ? { name, version } : { name };
}

// The conversation with a field set, or removed when the value is undefined.
export function withField<T extends object, K extends keyof T>(thing: T, key: K, value: T[K] | undefined): T {
  const { [key]: _, ...rest } = thing;
  return (value === undefined ? rest : { ...rest, [key]: value }) as T;
}

// What the depot would refuse in the conversation, in words, then the words of
// each box whose text is not what it must be yet (bad).
export function problemsOf(c: Conversation, bad: string[] = []): string[] {
  const found: string[] = [];
  if (c.harness && !c.harness.name) found.push("The harness has a version but no name: name it, or clear its version.");
  const offered = toolsProblem(c.tools ?? []);
  if (offered) found.push(`The tools offered are not a list of tools: ${offered}.`);
  if (!c.messages.length) found.push("A conversation holds at least one message.");
  c.messages.forEach((m, i) => {
    const where = `Message ${i + 1} (${ROLE_LABEL[m.role] ?? m.role})`;
    if (m.role !== "assistant" && !m.content.length) found.push(`${where} holds no block; only an assistant message may be empty.`);
    if (m.role !== "assistant" && m.end) found.push(`${where} says how its turn ended; only an assistant message does.`);
    const each = (b: Block) => {
      if (b.type === "text" && b.by && m.role === "assistant") found.push(`${where}: text the harness added stands in an assistant message, whose text is the model's own.`);
      if (b.type === "tool_call") {
        const form = formOf(b);
        if (form !== "whole" && !(b.name ?? "").trim()) found.push(`${where}: a tool call names no tool.`);
        if (form === "arguments" && !isObject(b.arguments)) found.push(`${where}: the tool call ${b.id} holds neither arguments nor raw text.`);
      }
      if (b.type === "tool_result" && !b.call) found.push(`${where}: a tool result answers no call.`);
      if (b.type === "tool_result" && b.conversation !== undefined && !b.conversation.trim()) found.push(`${where}: the result of the call ${b.call} names no sub-agent's conversation: name one or remove the field.`);
      if (b.type === "tools") {
        const wrong = toolsProblem(b.tools, 1);
        if (wrong) found.push(`${where}: the tools defined are not a list of tools: ${wrong}.`);
      }
      if (b.type === "excerpt" && !(b.source ?? "").trim()) found.push(`${where}: an excerpt names no quoted text.`);
      if (b.type === "excerpt" && !b.selector?.exact) found.push(`${where}: an excerpt quotes no passage.`);
      b.content?.forEach(each);
    };
    m.content.forEach(each);
  });
  return [...found, ...bad];
}
