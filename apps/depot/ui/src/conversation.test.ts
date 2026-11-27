// The rules the chat editor follows (conversation.ts), as the kind's schema
// says them: run with `bun test` in apps/depot/ui.

import { expect, test } from "bun:test";
import { formOf, harnessOf, problemsOf, recast, reform, toolsAt, toolsProblem, withField, type Block, type Conversation, type Turn } from "./conversation";

const user = (t: string): Turn => ({ role: "user", content: [{ type: "text", text: t }] });
const call = (id: string, name: string, args: Record<string, unknown> = {}): Block => ({ type: "tool_call", id, name, arguments: args });

test("a tool call takes each of its three forms and comes back to its arguments as they were", () => {
  const parsed = call("c1", "shell", { command: "ls" });
  expect(formOf(parsed)).toBe("arguments");
  const named = reform(parsed, "named") as Block;
  expect(named).toEqual({ type: "tool_call", id: "c1", name: "shell", raw: '{"command":"ls"}' });
  expect(formOf(named)).toBe("named");
  const whole = reform(named, "whole") as Block;
  expect(whole).toEqual({ type: "tool_call", id: "c1", raw: '{"name": "shell", "arguments": {"command":"ls"}}' });
  expect(formOf(whole)).toBe("whole");
  expect(reform(whole, "named")).toEqual(named);
  expect(reform(whole, "arguments")).toEqual(parsed);
  expect(reform(named, "arguments")).toEqual(parsed);
});

test("raw text that does not parse stays raw, and a whole call that names nothing keeps its text under a name to fill in", () => {
  const broken: Block = { type: "tool_call", id: "c1", name: "shell", raw: '{"command": "bun test' };
  expect(typeof reform(broken, "arguments")).toBe("string");
  const unread: Block = { type: "tool_call", id: "c2", raw: "<call>shell ls" };
  expect(typeof reform(unread, "arguments")).toBe("string");
  expect(reform(unread, "named")).toEqual({ type: "tool_call", id: "c2", name: "", raw: "<call>shell ls" });
});

test("a call kept raw needs a name only under its name, and one with neither arguments nor raw is refused", () => {
  const c = (b: Block): Conversation => ({ messages: [user("go"), { role: "assistant", content: [b] }] });
  expect(problemsOf(c({ type: "tool_call", id: "c1", raw: "{oops" }))).toEqual([]);
  expect(problemsOf(c({ type: "tool_call", id: "c1", name: "shell", raw: "{oops" }))).toEqual([]);
  expect(problemsOf(c({ type: "tool_call", id: "c1", name: "", raw: "{oops" }))).toEqual(["Message 2 (assistant): a tool call names no tool."]);
  expect(problemsOf(c({ type: "tool_call", id: "c1", name: "shell" }))).toEqual(["Message 2 (assistant): the tool call c1 holds neither arguments nor raw text."]);
});

test("a tool may be named alone, and a list of tools holds nothing else", () => {
  expect(toolsProblem([{ name: "shell" }, { name: "read", description: "Read.", parameters: { type: "object" } }])).toBeNull();
  expect(toolsProblem([], 1)).toBe("they name no tool");
  expect(toolsProblem([{ name: "" }])).toBe("tool 1 has no name");
  expect(toolsProblem([{ name: "shell", parameters: "x" }])).toBe("the parameters of shell are not a JSON Schema object");
  expect(toolsProblem([{ name: "shell", kind: "x" }])).toContain("shell holds kind");
  expect(toolsProblem({ name: "shell" })).toBe("they are not a list");
});

test("the tools known at a message are those offered, completed by every tools block before it", () => {
  const c: Conversation = {
    tools: [{ name: "find_tools", description: "Load tools." }, { name: "shell" }],
    messages: [
      { role: "system", content: [{ type: "tools", tools: [{ name: "read", description: "Read a file." }] }] },
      user("go"),
      { role: "assistant", content: [call("c1", "find_tools")] },
      { role: "tool", content: [{ type: "tool_result", call: "c1", content: [{ type: "tools", tools: [{ name: "shell", description: "Run a command." }] }] }] },
      { role: "assistant", content: [] },
    ],
  };
  expect(toolsAt(c, 0).map((t) => t.name)).toEqual(["find_tools", "shell"]);
  expect(toolsAt(c, 2).map((t) => t.name)).toEqual(["find_tools", "shell", "read"]);
  expect(toolsAt(c, 2).find((t) => t.name === "shell")?.description).toBeUndefined();
  expect(toolsAt(c, 4).find((t) => t.name === "shell")?.description).toBe("Run a command.");
  expect(problemsOf(c)).toEqual([]);
});

test("a tools block names at least one tool", () => {
  const c: Conversation = { messages: [{ role: "system", content: [{ type: "tools", tools: [] }] }, user("go")] };
  expect(problemsOf(c)).toEqual(["Message 1 (system): the tools defined are not a list of tools: they name no tool."]);
});

test("text the harness added stands anywhere but in an assistant message", () => {
  const harness: Block = { type: "text", text: "A reminder.", by: "harness" };
  const ok: Conversation = { messages: [{ role: "user", content: [{ type: "text", text: "go" }, harness] }, { role: "assistant", content: [call("c1", "shell")] },
    { role: "tool", content: [{ type: "tool_result", call: "c1", content: [{ type: "text", text: "done" }, harness] }] }] };
  expect(problemsOf(ok)).toEqual([]);
  const wrong: Conversation = { messages: [user("go"), { role: "assistant", content: [harness] }] };
  expect(problemsOf(wrong)).toEqual(["Message 2 (assistant): text the harness added stands in an assistant message, whose text is the model's own."]);
  // a user message holding it cannot become the assistant's
  expect(recast(ok.messages, 0, "assistant")).toContain("the harness added");
  expect(recast(ok.messages, 0, "system")).toMatchObject({ role: "system" });
});

test("only an assistant message says how its turn ended, and it keeps it to change role", () => {
  const messages: Turn[] = [user("go"), { role: "assistant", end: "stop", content: [{ type: "text", text: "done" }] }];
  expect(problemsOf({ messages })).toEqual([]);
  expect(recast(messages, 1, "user")).toContain("how its turn ended");
  expect(recast([user("go"), withField(messages[1], "end", undefined)], 1, "user")).toMatchObject({ role: "user" });
  expect(problemsOf({ messages: [{ ...user("go"), end: "limit" }] })).toEqual(["Message 1 (user) says how its turn ended; only an assistant message does."]);
});

test("a result keeps what the tool returned and its sub-agent's conversation, which unwrapping it would lose", () => {
  const answer = (extra: Partial<Block>): Turn[] => [user("go"), { role: "assistant", content: [call("c1", "subagent")] },
    { role: "tool", content: [{ type: "tool_result", call: "c1", content: [{ type: "text", text: "found it" }], ...extra }] }];
  expect(problemsOf({ messages: answer({ value: { exitCode: 1 }, conversation: "my-depot-41" }) })).toEqual([]);
  expect(problemsOf({ messages: answer({ value: null }) })).toEqual([]);
  expect(recast(answer({ value: 0 }), 2, "user")).toContain("keeps what the tool returned");
  expect(recast(answer({ conversation: "my-depot-41" }), 2, "user")).toContain("sub-agent's conversation");
  expect(recast(answer({ error: true }), 2, "user")).toContain("the call failed");
  expect(recast(answer({}), 2, "user")).toMatchObject({ role: "user", content: [{ type: "text", text: "found it" }] });
  expect(problemsOf({ messages: answer({ conversation: " " }) })).toEqual([
    "Message 3 (tool): the result of the call c1 names no sub-agent's conversation: name one or remove the field."]);
});

test("a harness is named, with or without its version, or not at all", () => {
  expect(harnessOf("", "")).toBeUndefined();
  expect(harnessOf("opencode", "")).toEqual({ name: "opencode" });
  expect(harnessOf("opencode", "1.19.6")).toEqual({ name: "opencode", version: "1.19.6" });
  const c: Conversation = { harness: harnessOf("", "1.19.6"), messages: [user("go")] };
  expect(problemsOf(c)).toEqual(["The harness has a version but no name: name it, or clear its version."]);
  expect(withField(c, "harness", undefined)).toEqual({ messages: [user("go")] });
});

test("the boxes not yet what they must be come after the depot's refusals", () => {
  expect(problemsOf({ messages: [] }, ["The arguments of the call c1 are not a JSON object."])).toEqual([
    "A conversation holds at least one message.", "The arguments of the call c1 are not a JSON object."]);
});
