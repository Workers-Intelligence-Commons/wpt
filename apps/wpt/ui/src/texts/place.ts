// The place reached on the Texts page, kept in this browser (localStorage): its whole address, so what is open (the
// filter or search, the order, the mode) and the text open in it. The "Texts" link of the top bar leads back to it
// from any page, and a reload keeps it in the address itself; a new filter or search replaces it as soon as it opens.
// A Texts page shown in a pane of another page (pages/strip.tsx) keeps nothing, so it never replaces it.

const PLACE = "wpt-depot-texts-place";
const FRAMED = (() => { try { return window.self !== window.top; } catch { return true; } })();

export function keepTextsPlace(address: string) {
  if (FRAMED || !address.startsWith("#/texts")) return;
  try { window.localStorage.setItem(PLACE, address); } catch { /* not kept: the Texts link opens every text */ }
}

// The address the Texts link leads to: the place kept, else every text.
export function textsPlace(): string {
  try {
    const found = window.localStorage.getItem(PLACE);
    if (found?.startsWith("#/texts")) return found;
  } catch { /* nothing kept */ }
  return "#/texts";
}
