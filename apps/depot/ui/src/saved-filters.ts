// Saved filters: named filters of the texts (filter.tsx's TextFilter, the
// SQL condition included) that this browser keeps in its localStorage, so
// the Texts page can open them again. They never reach the depot. A browser
// that refuses storage keeps none, and saving says so.

import { useEffect, useState } from "react";
import { EVERY, type TextFilter } from "./filter";

// gone: the parts a filter saved before snapshots held that limited it to
// libraries (libraries, outside, locked); such a filter is listed but not
// opened, since without them it would match more texts than it was saved for.
export type SavedFilter = { name: string; filter: TextFilter; gone: string[] };

const KEY = "depot-saved-filters";
const CHANGED = "depot-saved-filters-changed"; // tells the open views of this tab

// The parts of a filter saved before snapshots that limited it to libraries.
function goneParts(f: Record<string, unknown>): string[] {
  const held = (v: unknown) => Array.isArray(v) && v.length > 0;
  return [...(held(f.libraries) ? ["libraries"] : []), ...(held(f.outside) ? ["outside"] : []),
    ...(typeof f.locked === "boolean" ? ["locked"] : [])];
}

// A filter read back from storage, each part checked, the parts it lacks as
// EVERY has them.
function checked(found: unknown): TextFilter | null {
  if (!found || typeof found !== "object") return null;
  const f = found as Record<string, unknown>;
  const strings = (v: unknown) => (Array.isArray(v) && v.every((s) => typeof s === "string") ? (v as string[]) : []);
  return {
    ...EVERY, tags: strings(f.tags), licenses: strings(f.licenses), statuses: strings(f.statuses) as TextFilter["statuses"],
    kinds: strings(f.kinds),
    decision: typeof f.decision === "string" ? f.decision : null, search: typeof f.search === "string" ? f.search : "", where: typeof f.where === "string" ? f.where : "",
    inText: f.inText === true, wholeWords: f.wholeWords === true,
  };
}

export function readSavedFilters(): SavedFilter[] {
  try {
    const found = JSON.parse(window.localStorage.getItem(KEY) ?? "[]");
    if (!Array.isArray(found)) return [];
    return found.flatMap((one) => {
      const filter = checked(one?.filter);
      // gone is kept with the filter, so rewriting the list (another filter saved) keeps it refused
      const kept = Array.isArray(one?.gone) ? one.gone.filter((g: unknown): g is string => typeof g === "string") : [];
      const gone = [...new Set([...goneParts(one.filter), ...kept])];
      return typeof one?.name === "string" && filter ? [{ name: one.name, filter, gone }] : [];
    });
  } catch {
    return []; // no storage, or a value this page did not write
  }
}

// Whether the browser kept them.
function write(all: SavedFilter[]): boolean {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(all));
  } catch {
    return false;
  }
  window.dispatchEvent(new Event(CHANGED));
  return true;
}

// Keep a filter under a name; a filter saved under the same name is replaced.
export const saveFilter = (name: string, filter: TextFilter) =>
  write([...readSavedFilters().filter((s) => s.name !== name), { name, filter: { ...filter, ids: null }, gone: [] }]);

export const forgetFilter = (name: string) => write(readSavedFilters().filter((s) => s.name !== name));

export function useSavedFilters(): SavedFilter[] {
  const [all, setAll] = useState(readSavedFilters);
  useEffect(() => {
    const changed = () => setAll(readSavedFilters());
    window.addEventListener(CHANGED, changed);
    window.addEventListener("storage", changed); // another tab saved
    return () => {
      window.removeEventListener(CHANGED, changed);
      window.removeEventListener("storage", changed);
    };
  }, []);
  return all;
}
