// Which of Mozilla's translation models take a text from one language to another, and which language a text is in,
// from what the detector found and from its tags (apart from React, so it is tested alone: languages.test.ts).
// Languages are named by the codes the models use (BCP 47: "de", "zh-Hans").

// One model, as GET /api/translations lists it: its files by part (model, lex, and vocab or srcvocab and trgvocab).
export type Pair = { from: string; to: string; version: string; files: Record<string, { sha256: string; size: number }> };

// Nearly every model translates into or out of English, so English links the others.
export const PIVOT = "en";

// The models a translation runs through, in order: the pair's own, else through English, else null when no model
// reaches.
export function route(pairs: Pair[], from: string, to: string): Pair[] | null {
  if (from === to) return null;
  const one = (a: string, b: string) => pairs.find((p) => p.from === a && p.to === b);
  const direct = one(from, to);
  if (direct) return [direct];
  const out = one(from, PIVOT), back = one(PIVOT, to);
  return out && back ? [out, back] : null;
}

// Every language the models know, in either direction.
export function languages(pairs: Pair[]): string[] {
  return [...new Set(pairs.flatMap((p) => [p.from, p.to]))].sort();
}

// The language a text's tags name, among those the models know: a tag that is a language's code, or the code
// without its script or region ("zh" for "zh-Hans"); null when no tag names one.
export function guess(tags: string[], known: string[]): string | null {
  for (const tag of tags) {
    const exact = known.find((code) => code.toLowerCase() === tag.toLowerCase());
    if (exact) return exact;
  }
  for (const tag of tags) {
    const base = known.find((code) => code.split("-")[0].toLowerCase() === tag.toLowerCase());
    if (base) return base;
  }
  return null;
}

// The codes CLD2 (detect.ts) writes otherwise than the models do.
const DETECTOR_CODES: Record<string, string> = { no: "nb", iw: "he", zh: "zh-Hans" };

// Where the text's language comes from: what the detector found when it is sure of it, else a tag that names one,
// else null. A language detected surely is kept even when no model knows it, so the page says so rather than trust a
// tag the text contradicts.
export function source(detected: { language: string; confident: boolean } | null, tags: string[], known: string[]):
  { language: string; how: "detected" | "tags" } | null {
  if (detected?.confident && detected.language !== "un") {
    const code = DETECTOR_CODES[detected.language] ?? detected.language;
    return { language: known.find((k) => k.toLowerCase() === code.toLowerCase()) ?? code, how: "detected" };
  }
  const tagged = guess(tags, known);
  return tagged ? { language: tagged, how: "tags" } : null;
}

// A language's name in English, the words of the interface; its code when the browser does not know it.
export function languageName(code: string): string {
  try {
    return new Intl.DisplayNames(["en"], { type: "language" }).of(code) ?? code;
  } catch {
    return code;
  }
}
