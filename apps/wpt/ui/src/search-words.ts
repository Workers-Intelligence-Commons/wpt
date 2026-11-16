// The words of a plain search, which the filter looks for (filter.tsx) and
// the editor and transcripts mark: what is typed, split at spaces.

export const searchWords = (search: string | undefined) => (search ?? "").trim().split(/\s+/).filter(Boolean);

// The words of a search as one pattern that matches any of them in any case,
// each word as written (no character of it a pattern's), or null for none.
// grouped: the whole pattern in one group, so a split keeps what it matched.
export function wordsPattern(search: string | undefined, grouped = false): RegExp | null {
  const words = searchWords(search);
  if (!words.length) return null;
  const any = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  return new RegExp(grouped ? `(${any})` : any, "gi");
}
