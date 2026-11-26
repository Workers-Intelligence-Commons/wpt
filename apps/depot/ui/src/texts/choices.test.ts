// The choices the review panel offers for a score, from the values the depot
// holds (GET /api/scores/NAME/values): run with `bun test` in apps/depot/ui.

import { expect, test } from "bun:test";
import { isGraded, scaleOf, tagButtons, type Values } from "./choices";

const values = (common: [number, number][], others = 0): Values => {
  const texts = common.reduce((sum, [, n]) => sum + n, 0) + others;
  const all = common.map(([v]) => v);
  return { name: "quality", texts, low: Math.min(...all), high: Math.max(...all), common: common.map(([value, n]) => ({ value, texts: n })) };
};

test("a person's grades from 0 to 10 are offered whole, every one of them, even those no text holds yet", () => {
  // mostly 6 to 8, a few 2 and 10: the buttons are still 0 to 10, and − and + step by 1
  const scale = scaleOf(values([[2, 3], [6, 40], [7, 80], [8, 30], [10, 2]]));
  expect(scale.values).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  expect([scale.low, scale.high, scale.step]).toEqual([0, 10, 1]);
});

test("grades stay whole although a few texts hold values of another kind", () => {
  const scale = scaleOf(values([[0.37, 2], [0.81, 3], [3, 50], [5, 40], [9, 30]]));
  expect(scale.values).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
});

test("a measure between 0 and 1, often exactly 0 or 1, keeps buttons with decimals", () => {
  const scale = scaleOf(values([[0, 30], [0.25, 10], [0.5, 10], [0.75, 10], [1, 40]], 20));
  expect(scale.values.some((v) => !Number.isInteger(v))).toBe(true);
  expect([scale.low, scale.high]).toEqual([0, 1]);
});

test("a score with no value yet offers 0 to 1", () => {
  expect(scaleOf(null).values).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
});

test("only a scale of whole grades, one button each, keeps the pad to its range", () => {
  expect(isGraded(scaleOf(values([[2, 3], [6, 40], [7, 80], [8, 30], [10, 2]])))).toBe(true);
  expect(isGraded(scaleOf(values([[0, 30], [0.25, 10], [0.5, 10], [0.75, 10], [1, 40]], 20)))).toBe(false);
  expect(isGraded(scaleOf(null))).toBe(false);
});

test("the tags put on a text last take the first tag buttons, the depot's most used the rest", () => {
  const used = [{ tag: "book", texts: 50 }, { tag: "en", texts: 90 }, { tag: "poem", texts: 50 }, { tag: "fr", texts: 3 }];
  expect(tagButtons([], used, 3)).toEqual(["en", "book", "poem"]);
  expect(tagButtons(["fr", "new-source"], used, 3)).toEqual(["fr", "new-source", "en"]);
  expect(tagButtons(["en"], used, 3)).toEqual(["en", "book", "poem"]); // shown once
  expect(tagButtons(["a", "b", "c", "d", "e"], used, 9)).toEqual(["a", "b", "c", "d", "en", "book", "poem", "fr"]);
});
