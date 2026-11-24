import { describe, expect, test } from "bun:test";
import { guess, route, type Pair } from "./languages";

const pair = (from: string, to: string): Pair => ({ from, to, version: "1.0", files: {} });
const PAIRS = [pair("de", "en"), pair("en", "fr"), pair("en", "de"), pair("ru", "en"), pair("en", "zh-Hans")];

describe("route", () => {
  test("a pair with its own model takes that model alone", () => {
    expect(route(PAIRS, "de", "en")).toEqual([pair("de", "en")]);
  });
  test("other pairs go through English", () => {
    expect(route(PAIRS, "de", "fr")).toEqual([pair("de", "en"), pair("en", "fr")]);
  });
  test("no route when a side has no model, or the languages are the same", () => {
    expect(route(PAIRS, "fr", "de")).toBeNull();
    expect(route(PAIRS, "ru", "ja")).toBeNull();
    expect(route(PAIRS, "de", "de")).toBeNull();
  });
});

describe("guess", () => {
  const known = ["de", "en", "fr", "ru", "zh-Hans"];
  test("a tag that is a language's code names the text's language", () => {
    expect(guess(["novel", "ru", "19th-century"], known)).toBe("ru");
  });
  test("a tag names a language written with a script or region too", () => {
    expect(guess(["zh"], known)).toBe("zh-Hans");
  });
  test("no language without a tag naming one", () => {
    expect(guess(["novel", "war"], known)).toBeNull();
  });
});
