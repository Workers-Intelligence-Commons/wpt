import { describe, expect, test } from "bun:test";
import { guess, route, source, type Pair } from "./languages";

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

describe("source", () => {
  const known = ["de", "en", "fr", "nb", "he", "zh-Hans"];
  test("a language the detector is sure of wins over the tags", () => {
    expect(source({ language: "de", confident: true }, ["fr"], known)).toEqual({ language: "de", how: "detected" });
  });
  test("the detector's own codes become the models'", () => {
    expect(source({ language: "no", confident: true }, [], known)).toEqual({ language: "nb", how: "detected" });
    expect(source({ language: "iw", confident: true }, [], known)).toEqual({ language: "he", how: "detected" });
    expect(source({ language: "zh", confident: true }, [], known)).toEqual({ language: "zh-Hans", how: "detected" });
  });
  test("a language detected surely that no model knows is kept, not replaced by a tag", () => {
    expect(source({ language: "la", confident: true }, ["fr"], known)).toEqual({ language: "la", how: "detected" });
  });
  test("an unsure detection, or none, leaves it to the tags, then to the person", () => {
    expect(source({ language: "en", confident: false }, ["fr"], known)).toEqual({ language: "fr", how: "tags" });
    expect(source(null, ["fr"], known)).toEqual({ language: "fr", how: "tags" });
    expect(source({ language: "un", confident: true }, [], known)).toBeNull();
  });
});
