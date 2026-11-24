// The saves of the open text's edit (edit-saving.ts): run with `bun test` in apps/wpt/ui.

import { expect, test } from "bun:test";
import { canonical, EditSaves, ROUNDS, type Answered } from "./edit-saving";

// An editor whose content is a string, a depot that answers each request when told to, and what it received.
function editor(opened: string, refusal: () => string | null = () => null) {
  const state = { content: opened, depot: opened, version: 1 };
  const sent: string[] = [];
  const waiting: (() => void)[] = [];
  const ended: unknown[] = [];
  const saves = new EditSaves<string>(opened, {
    now: () => ({ snap: state.content, body: { text: state.content } }),
    same: (a, b) => a === b,
    refusal,
    send: (body) => new Promise<Answered>((resolve) => {
      const text = body.text as string;
      sent.push(text);
      waiting.push(() => {
        if (text === state.depot) return resolve({ ok: true, decision: null, version: null });
        state.depot = text;
        state.version += 1;
        resolve({ ok: true, decision: `d${state.version}`, version: state.version });
      });
    }),
    onEnd: (saved) => ended.push(saved),
  });
  const answer = async () => { while (!waiting.length) await Promise.resolve(); waiting.shift()!(); };
  return { state, sent, saves, answer, ended };
}

test("words typed while a save is on its way are sent before the save answers, so nothing typed is lost", async () => {
  const e = editor("one");
  e.state.content = "one two";
  const saving = e.saves.save();
  e.state.content = "one two three"; // typed after the request left
  await e.answer();
  await e.answer();
  const saved = await saving;
  expect(e.sent).toEqual(["one two", "one two three"]);
  expect(saved).toEqual({ ok: true, sent: true, made: { decision: "d3", version: 3 } });
  expect(e.state.depot).toBe("one two three");
  expect(e.saves.dirty()).toBe(false);
  expect(e.saves.version).toBe(3);
  expect(e.ended).toHaveLength(1);
});

test("after a save the depot answered with nothing to save, the next edit is saved", async () => {
  const e = editor("same");
  e.state.content = "same "; // seen as changed by the page, which the depot holds already
  e.state.depot = "same ";
  const nothing = e.saves.save();
  await e.answer();
  expect(await nothing).toEqual({ ok: true, sent: true, made: null });
  expect(e.saves.dirty()).toBe(false);
  e.state.content = "a real edit";
  expect(e.saves.dirty()).toBe(true);
  const real = e.saves.save();
  await e.answer();
  expect((await real).made?.version).toBe(2);
  expect(e.state.depot).toBe("a real edit");
});

test("a save refused before sending or by the depot leaves the edit dirty, to save again", async () => {
  let why: string | null = "set a name and email";
  const e = editor("x", () => why);
  e.state.content = "y";
  const refused = await e.saves.save();
  expect(refused).toEqual({ ok: false, sent: false, made: null, problems: ["set a name and email"], before: true });
  expect(e.sent).toEqual([]);
  expect(e.saves.dirty()).toBe(true);
  why = null;
  const saving = e.saves.save();
  await e.answer();
  expect((await saving).ok).toBe(true);
});

test("a save asked while one is on its way is that same save", async () => {
  const e = editor("a");
  e.state.content = "b";
  const first = e.saves.save();
  const second = e.saves.save();
  expect(second).toBe(first);
  await e.answer();
  await first;
  expect(e.sent).toEqual(["b"]);
});

test("content that keeps changing under every request ends the save as not done, still dirty", async () => {
  const e = editor("0");
  e.state.content = "1";
  const saving = e.saves.save();
  for (let i = 0; i < ROUNDS; i++) { e.state.content = `${i + 2}`; await e.answer(); }
  const saved = await saving;
  expect(saved.ok).toBe(false);
  expect(saved.sent).toBe(true);
  expect(e.sent).toHaveLength(ROUNDS);
  expect(e.saves.dirty()).toBe(true);
});

test("items compare by value, whatever the order of their keys", () => {
  expect(canonical({ role: "user", content: [{ type: "text", text: "hi" }] }))
    .toBe(canonical({ content: [{ text: "hi", type: "text" }], role: "user" }));
  expect(canonical({ a: 1 })).not.toBe(canonical({ a: 2 }));
});
