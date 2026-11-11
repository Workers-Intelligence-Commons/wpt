// Saved filters: named filters of the texts (filter.tsx's TextFilter, the
// SQL condition included) that this browser keeps in its localStorage, so
// the workspace can open them again. They never reach the depot. A browser
// that refuses storage keeps none, and saving says so.

import { useEffect, useState } from "react";
import { EVERY, type TextFilter } from "./filter";

export type SavedFilter = { name: string; filter: TextFilter };

const KEY = "wpt-depot-saved-filters";
const CHANGED = "wpt-depot-saved-filters-changed"; // tells the open views of this tab

// A filter read back from storage, each part checked, the parts it lacks as EVERY has them.
function checked(found: unknown): TextFilter | null {
  if (!found || typeof found !== "object") return null;
  const f = found as Record<string, unknown>;
  const strings = (v: unknown) => (Array.isArray(v) && v.every((s) => typeof s === "string") ? (v as string[]) : []);
  return {
    ...EVERY, tags: strings(f.tags), licenses: strings(f.licenses), statuses: strings(f.statuses) as TextFilter["statuses"],
    libraries: strings(f.libraries), locked: typeof f.locked === "boolean" ? f.locked : null, where: typeof f.where === "string" ? f.where : "",
  };
}

export function readSavedFilters(): SavedFilter[] {
  try {
    const found = JSON.parse(window.localStorage.getItem(KEY) ?? "[]");
    if (!Array.isArray(found)) return [];
    return found.flatMap((one) => {
      const filter = checked(one?.filter);
      return typeof one?.name === "string" && filter ? [{ name: one.name, filter }] : [];
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
  write([...readSavedFilters().filter((s) => s.name !== name), { name, filter: { ...filter, ids: null } }]);

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
