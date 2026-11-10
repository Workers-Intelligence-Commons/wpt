// Who decides: the legal name and email this browser keeps in its
// localStorage, which every decision form sends. A browser that refuses
// storage keeps nothing, and the forms then say that no name is set.

import { useEffect, useState } from "react";

export type Declarant = { name: string; email: string };

const KEY = "wpt-depot-declarant";
const CHANGED = "wpt-depot-declarant-changed"; // tells the open pages of this tab
const NOBODY: Declarant = { name: "", email: "" };

export function readDeclarant(): Declarant {
  try {
    const found = JSON.parse(window.localStorage.getItem(KEY) ?? "null");
    if (found && typeof found.name === "string" && typeof found.email === "string") return { name: found.name, email: found.email };
  } catch {
    // no storage, or a value this page did not write
  }
  return NOBODY;
}

// Whether the browser kept it.
export function saveDeclarant(who: Declarant): boolean {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(who));
  } catch {
    return false;
  }
  window.dispatchEvent(new Event(CHANGED));
  return true;
}

export const declared = (who: Declarant) => who.name.trim() !== "" && who.email.trim() !== "";

export function useDeclarant(): Declarant {
  const [who, setWho] = useState(readDeclarant);
  useEffect(() => {
    const changed = () => setWho(readDeclarant());
    window.addEventListener(CHANGED, changed);
    window.addEventListener("storage", changed); // another tab saved
    return () => {
      window.removeEventListener(CHANGED, changed);
      window.removeEventListener("storage", changed);
    };
  }, []);
  return who;
}
