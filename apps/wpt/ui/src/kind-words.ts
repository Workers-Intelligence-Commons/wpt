// The plain words the content form (schema-form.tsx) uses for the fields
// of an item's kind, by kind, in place of the names of its JSON Schema
// (docs/chat-format.md for a conversation). A kind not named here shows its
// schema's own names. Only what the form shows changes: the content sent
// keeps the schema's names.

import type { Words } from "./schema-form";

const TEXT_BLOCK = { type: "text", text: "" };

// who speaks next in a conversation: the user after anyone but the user, the assistant after the user
const nextRole = (messages: unknown[]) => {
  const last = messages.at(-1) as { role?: string } | undefined;
  return last?.role === "user" ? "assistant" : "user";
};

const CONVERSATION: Words = {
  names: {
    messages: "Messages", role: "Who speaks", content: "What it holds", type: "Kind of block", text: "Text",
    tools: "Tools offered", metadata: "Where it was recorded", name: "Name", description: "Description",
    parameters: "Parameters", id: "Call id", arguments: "Arguments", call: "Call id it answers", error: "Is an error",
    value: "Value",
  },
  entries: { messages: "message", content: "block", tools: "tool" },
  values: {
    system: "system", user: "user", assistant: "assistant", tool: "tool",
    text: "text", thinking: "thinking", tool_call: "a tool call", tool_result: "a tool's result", json: "JSON", file: "a file", excerpt: "an excerpt",
  },
  // a new message speaks after the last one and starts with one text block
  start: (field, entries) => (field === "messages" ? { role: nextRole(entries), content: [{ ...TEXT_BLOCK }] } : undefined),
};

export const KIND_WORDS: Record<string, Words> = { conversation: CONVERSATION };

// The start of a new item of a kind: one message of the user with one text
// block for a conversation, else the schema's own start (given).
export const kindStart = (kind: string, given: unknown) =>
  kind === "conversation" ? { messages: [{ role: "user", content: [{ ...TEXT_BLOCK }] }] } : given;
