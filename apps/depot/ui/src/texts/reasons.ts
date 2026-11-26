// The reasons a refusal offers as chips once it is made (texts.tsx): the
// depot's usual ones, then those this browser keeps, each added by a click
// from a reason typed once. They live in localStorage and never reach the
// depot until one is attached to a refusal; a browser that refuses storage
// keeps none, and the usual ones stay.

import { useEffect, useState } from "react";

export const USUAL = ["bad quality", "bad license", "license not compatible"];
const KEY = "depot-refusal-reasons";
const CHANGED = "depot-refusal-reasons-changed";

function readOwn(): string[] {
  try {
    const found = JSON.parse(window.localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(found) ? found.filter((one): one is string => typeof one === "string" && one.trim() !== "") : [];
  } catch {
    return [];
  }
}
function writeOwn(all: string[]): boolean {
  let ok = true;
  try { window.localStorage.setItem(KEY, JSON.stringify(all)); } catch { ok = false; }
  window.dispatchEvent(new Event(CHANGED));
  return ok;
}

// Keep a reason as a chip, after the usual ones and those kept before; whether the browser kept it.
export const keepReason = (reason: string) =>
  USUAL.includes(reason.trim()) || readOwn().includes(reason.trim()) || writeOwn([...readOwn(), reason.trim()]);
export const forgetReason = (reason: string) => writeOwn(readOwn().filter((one) => one !== reason));

// The reasons this browser keeps, kept up to date when another page or tab changes them.
export function useOwnReasons(): string[] {
  const [all, setAll] = useState(readOwn);
  useEffect(() => {
    const changed = () => setAll(readOwn());
    window.addEventListener(CHANGED, changed);
    window.addEventListener("storage", changed);
    return () => { window.removeEventListener(CHANGED, changed); window.removeEventListener("storage", changed); };
  }, []);
  return all;
}
