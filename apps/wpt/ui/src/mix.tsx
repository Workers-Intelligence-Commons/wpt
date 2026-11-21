// A dataset's mix, said the same way on every page: the conversations and
// tasks (the items) take their natural fraction of the corpus, and the shares
// split the rest, each given as its part of that rest (the plain text). So
// the items' figure is a part of the whole corpus and the shares' figures are
// parts of the plain text, in whole percents everywhere. Before a build the
// items' fraction is the one the depot's counts give (items_share of POST
// /api/libraries/measure), marked "~" as an expectation; after it, what the
// corpus holds, with the share asked beside a share that came out more than
// UNEVEN off it.

import { figures, takenNames, takenShare, type MeasureState } from "./parts";

// The items' name wherever a page does not know which kinds they hold.
export const ITEMS = "conversations and tasks";
export const UNEVEN = 0.1; // how far, as a part of the plain text, a share may be from its part asked
export const whole = (n: number) => `${Math.round(n * 100)}%`;

export type MixRow = { part: string; value: number; asked?: number | null };
export type Mixed = { items: { part: string; value: number; expected: boolean } | null; shares: MixRow[] };

// The mix a dataset asks for: its shares, each its part of their sum, and the
// items' fraction once measured (null before, and when the libraries hold none).
export function asked(shares: { tag: string; share: number }[] | null, measured: MeasureState): Mixed {
  const found = figures(measured);
  const sum = (shares ?? []).reduce((t, s) => t + s.share, 0) || 1;
  const items = found ? takenShare(found) : 0;
  return {
    items: items > 0 && found ? { part: takenNames(found), value: items, expected: true } : null,
    shares: shares?.length ? shares.map((s) => ({ part: s.tag, value: s.share / sum })) : items > 0 ? [{ part: "plain texts", value: 1 }] : [],
  };
}

// The mix a built corpus holds: the items' part of all its tokens, and each
// share's part of its plain tokens beside the part asked.
// plainTokens: the plain tokens in all, for a corpus without shares.
export function built(shares: { tag: string; share: number }[] | null, got: { tag: string; tokens: number }[] | null | undefined, itemTokens: number, tasks: boolean, plainTokens = 0): Mixed {
  const plain = got?.length ? got.reduce((t, s) => t + s.tokens, 0) : plainTokens;
  const all = plain + itemTokens;
  const sum = (shares ?? []).reduce((t, s) => t + s.share, 0) || 1;
  return {
    items: itemTokens > 0 && all > 0 ? { part: tasks ? ITEMS : "conversations", value: itemTokens / all, expected: false } : null,
    shares: !(plain > 0) ? [] : shares?.length ? (got ?? []).map((s) => ({ part: s.tag, value: s.tokens / plain, asked: (shares.find((a) => a.tag === s.tag)?.share ?? 0) / sum }))
      : itemTokens > 0 ? [{ part: "plain texts", value: 1 }] : [],
  };
}

export const off = (m: Mixed) => m.shares.some((s) => s.asked != null && Math.abs(s.value - s.asked) > UNEVEN);
// A share's part asked, when it came out another whole percent: beside its figure, marked when far off (UNEVEN).
const askedOf = (s: MixRow) => (s.asked != null && whole(s.value) !== whole(s.asked) ? { words: `asked ${whole(s.asked)}`, far: Math.abs(s.value - s.asked) > UNEVEN } : null);

// The mix in one line, for a table's cell: "conversations ~19 % · rest: code 60 %, prose 40 %".
export function mixLine(m: Mixed): string {
  const items = m.items ? `${m.items.part} ${m.items.expected ? "~" : ""}${whole(m.items.value)}` : null;
  const shares = m.shares.map((s) => `${s.part} ${whole(s.value)}${askedOf(s) ? ` (${askedOf(s)!.words})` : ""}`).join(", ");
  return [items, shares && (items ? `rest: ${shares}` : shares)].filter(Boolean).join(" · ") || "every text";
}

// The mix as a bar, for a stage card: each part a piece of the bar as wide as
// its part of the whole corpus, the conversations hatched; under it, the
// conversations' figure on one line, then the shares' figures after "rest"
// (the plain text, which they split).
export function MixBar({ m }: { m: Mixed }) {
  const items = m.items?.value ?? 0;
  const pieces = [
    ...(m.items ? [{ key: "items", kind: "items", grow: items }] : []),
    ...m.shares.map((s, i) => ({ key: s.part, kind: `share-${i % 3}`, grow: (1 - items) * s.value })),
  ];
  return (
    <div className="mix">
      <div className="mix-bar" aria-hidden="true">{pieces.map((p) => <span key={p.key} className={`mix-seg mix-${p.kind}`} style={{ flexGrow: p.grow }} />)}</div>
      <ul className="mix-legend">
        {m.items && <li><span /><span className="mix-part"><span className="mix-swatch mix-items" aria-hidden="true" /><span className="mix-figure">{m.items.expected ? "~" : ""}{whole(items)}</span> <span className="mix-name">{m.items.part}</span></span></li>}
        {m.shares.map((s, i) => (
          <li key={s.part}><span className="mix-rest-word">{m.items && i === 0 ? "rest" : ""}</span>
            <span className="mix-part"><span className={`mix-swatch mix-share-${i % 3}`} aria-hidden="true" /><span className="mix-figure">{whole(s.value)}</span> <span className="mix-name">{s.part}
              {askedOf(s) && <span className={askedOf(s)!.far ? "not-allowed" : "aside"}> {askedOf(s)!.words}</span>}</span></span></li>
        ))}
      </ul>
    </div>
  );
}
