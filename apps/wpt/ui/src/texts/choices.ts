// What the review panel offers without typing (panel.tsx): for each score
// name the depot knows, its choices, read once a page from GET
// /api/scores/NAME/values (the values it has across every current text,
// from the depot's counts); the
// value each score was last set to from this browser; and the presets,
// named sets of scores and of tags to add and take away, that this browser
// keeps. Recent values and presets live in localStorage and never reach the
// depot; a browser that refuses storage keeps none, and the panel works
// without them.

import { useEffect, useState } from "react";
import { get, type Answer } from "../api";

// The choices of one score: the values offered as buttons, from smallest to
// largest, the range they span, and the step of − and +, a twentieth of it
// (at least 1 for a score of whole numbers).
export type Scale = { values: number[]; low: number; high: number; step: number };
export type Values = { name: string; texts: number; low: number | null; high: number | null; common: { value: number; texts: number }[] };

const SHOWN = 6; // buttons a score's row has room for beside its name
const GRADES = 10; // the widest span of whole numbers offered one button each, on a line of their own (0 to 10)

// A number cut to what a person reads: at most 4 significant digits, no trailing zeros.
export const round = (n: number) => Number(n.toPrecision(4));
// A value as a button shows it: .25 for 0.25, -1.5, 4.
export const shortValue = (n: number) => String(round(n)).replace(/^(-?)0\./, "$1.");

// The scale of a score from its values across the depot: a scale of whole
// grades when most of its texts (80 %) hold whole values, of three grades
// at least or every text whole, within GRADES of each other, every whole number of it, from 0 when they are all from 0 to
// GRADES (a person's 0 to 10, even when no text holds 0 yet, and whatever
// values of another kind a few texts hold); else the common values
// themselves when few enough cover most of the texts seen, else
// SHOWN values evenly spread over the range: whole numbers when every value
// seen is whole, else over the range widened to round numbers (0 to 0.3 for
// values from 0.013 to 0.29), so each button reads at a glance.
// With no value seen, 0 to 1; with one, 0 to it, or it to 0 when negative.
export function scaleOf(found: Values | null): Scale {
  const common = found?.common ?? [];
  const grades = common.filter((one) => Number.isInteger(one.value));
  const wholeTexts = grades.reduce((sum, one) => sum + one.texts, 0);
  // most texts hold whole values, of several grades, or every text does (a measure whose share is often exactly 0
  // or 1 keeps its decimals)
  if (found && grades.length && wholeTexts >= 0.8 * found.texts && (grades.length >= 3 || wholeTexts === found.texts)) {
    let first = Math.min(...grades.map((one) => one.value)), last = Math.max(...grades.map((one) => one.value));
    if (first > 0 && last <= GRADES) first = 0;
    if (last - first <= GRADES) {
      if (first === last) last = first + 1;
      return { values: Array.from({ length: last - first + 1 }, (_, i) => first + i), low: first, high: last, step: 1 };
    }
  }
  const covered = common.reduce((sum, one) => sum + one.texts, 0);
  let low = found?.low ?? 0, high = found?.high ?? 1;
  if (low === high) [low, high] = low > 0 ? [0, low] : low < 0 ? [low, 0] : [0, 1];
  // a score whose every value seen is whole keeps to whole numbers: its buttons, and a step of at least 1
  const whole = Boolean(found?.texts) && Number.isInteger(low) && Number.isInteger(high) && common.every((one) => Number.isInteger(one.value));
  const step = whole ? Math.max(1, Math.round((high - low) / 20)) : round((high - low) / 20);
  if (common.length >= 2 && common.length <= SHOWN && found && covered >= 0.8 * found.texts)
    return { values: common.map((one) => one.value).sort((a, b) => a - b), low, high, step };
  if (whole) {
    const values = high - low < SHOWN ? Array.from({ length: high - low + 1 }, (_, i) => low + i)
      : [...new Set(Array.from({ length: SHOWN }, (_, i) => Math.round(low + ((high - low) * i) / (SHOWN - 1))))];
    return { values, low, high, step };
  }
  const unit = 10 ** Math.floor(Math.log10(high - low));
  low = round(Math.floor(low / unit) * unit);
  high = round(Math.ceil(high / unit) * unit);
  return { values: Array.from({ length: SHOWN }, (_, i) => round(low + ((high - low) * i) / (SHOWN - 1))), low, high, step: round((high - low) / 20) };
}

// A scale of whole grades with one button each, more than fit beside the score's name (0 to 10): the review panel
// offers its grades alone, with a pad kept to them for a value between two.
export const isGraded = (scale: Scale) => scale.step === 1 && scale.values.length === scale.high - scale.low + 1 && scale.values.length > SHOWN;

// One read per score name and page, its answer kept for every later panel.
const read = new Map<string, Promise<Answer<Values>>>();
export function useScales(names: string[]): Map<string, Scale> {
  const key = names.join("\n");
  const [found, setFound] = useState<Map<string, Scale>>(new Map());
  useEffect(() => {
    let live = true;
    const wanted = key ? key.split("\n") : [];
    Promise.all(wanted.map(async (name) => {
      if (!read.has(name)) read.set(name, get<Values>(`/api/scores/${encodeURIComponent(name)}/values`));
      const a = await read.get(name)!;
      return [name, scaleOf(a.ok ? a.data : null)] as const;
    })).then((all) => { if (live) setFound(new Map(all)); });
    return () => { live = false; };
  }, [key]);
  return found;
}

// A scale read again after this browser set a score outside it, so the next panel offers it.
export const forgetScale = (name: string) => { read.delete(name); };

function kept<T>(key: string, fallback: T, valid: (v: unknown) => v is T): T {
  try {
    const found = JSON.parse(window.localStorage.getItem(key) ?? "null");
    return valid(found) ? found : fallback;
  } catch {
    return fallback;
  }
}
function keep(key: string, value: unknown): boolean {
  try { window.localStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; }
}

// The value each score was last set to here, which its row marks.
const RECENT = "wpt-depot-recent-scores";
const isRecent = (v: unknown): v is Record<string, number> =>
  Boolean(v) && typeof v === "object" && !Array.isArray(v) && Object.values(v as object).every((n) => typeof n === "number");
export const readRecent = () => kept(RECENT, {}, isRecent);
export const keepRecent = (name: string, value: number) => keep(RECENT, { ...readRecent(), [name]: value });

// A preset: scores to set and tags to add and take away, in one click.
export type Preset = { name: string; scores: Record<string, number>; add: string[]; remove: string[] };
const PRESETS = "wpt-depot-presets";
const CHANGED = "wpt-depot-presets-changed";
const strings = (v: unknown) => Array.isArray(v) && v.every((s) => typeof s === "string");
const isPresets = (v: unknown): v is Preset[] => Array.isArray(v) && v.every((p) =>
  p && typeof p.name === "string" && isRecent(p.scores) && strings(p.add) && strings(p.remove));
export const readPresets = () => kept(PRESETS, [], isPresets);
function writePresets(all: Preset[]): boolean {
  const ok = keep(PRESETS, all);
  window.dispatchEvent(new Event(CHANGED));
  return ok;
}
// Keep a preset under its name, replacing one of the same name; whether the browser kept it.
export const savePreset = (preset: Preset) => writePresets([...readPresets().filter((p) => p.name !== preset.name), preset]);
export const forgetPreset = (name: string) => writePresets(readPresets().filter((p) => p.name !== name));
export function usePresets(): Preset[] {
  const [all, setAll] = useState(readPresets);
  useEffect(() => {
    const changed = () => setAll(readPresets());
    window.addEventListener(CHANGED, changed);
    window.addEventListener("storage", changed);
    return () => { window.removeEventListener(CHANGED, changed); window.removeEventListener("storage", changed); };
  }, []);
  return all;
}

// What a preset does, in a few characters: q4 e2 +code −recipes.
export const presetWords = (p: Preset) => [
  ...Object.entries(p.scores).map(([name, value]) => `${name.slice(0, 1)}${shortValue(value)}`),
  ...p.add.map((t) => `+${t}`), ...p.remove.map((t) => `−${t}`),
].join(" ");
