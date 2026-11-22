// A mix, said the same way on every page: the conversations and tasks (the
// items) take their natural fraction of the corpus, and the shares split the
// rest, each given as its part of that rest (the plain text). So the items'
// figure is a part of the whole corpus and the shares' figures are parts of
// the plain text, in whole percents everywhere. An expected figure (before a
// build) is marked "~"; after a build a share that came out another percent
// than asked shows the part asked beside it, marked when more than UNEVEN off.

export const UNEVEN = 0.1; // how far, as a part of the plain text, a share may be from its part asked
export const whole = (n: number) => `${Math.round(n * 100)}%`;

export type MixRow = { part: string; value: number; asked?: number | null; detail?: string };
export type Mixed = { items: { part: string; value: number; expected: boolean; detail?: string } | null; shares: MixRow[] };

// A share's part asked, when it came out another whole percent: beside its figure, marked when far off (UNEVEN).
const askedOf = (s: MixRow) => (s.asked != null && whole(s.value) !== whole(s.asked) ? { words: `asked ${whole(s.asked)}`, far: Math.abs(s.value - s.asked) > UNEVEN } : null);

// The mix as a bar: each part a piece of the bar as wide as its part of the
// whole corpus, the shares first and the conversations last, in a muted
// fill; under it one legend row a part, in the bar's order: its swatch, its
// figure, its name and what it holds (detail), the conversations' figure
// said to be of the whole corpus, since the shares' are of the plain text.
export function MixBar({ m }: { m: Mixed }) {
  const items = m.items?.value ?? 0;
  const pieces = [
    ...m.shares.map((s, i) => ({ key: s.part, kind: `share-${i % 3}`, grow: (1 - items) * s.value })),
    ...(m.items ? [{ key: "items", kind: "items", grow: items }] : []),
  ];
  return (
    <div className="mix">
      <div className="mix-bar" aria-hidden="true">{pieces.map((p) => <span key={p.key} className={`mix-seg mix-${p.kind}`} style={{ flexGrow: p.grow }} />)}</div>
      <ul className="mix-legend">
        {m.shares.map((s, i) => {
          const asked = askedOf(s);
          return (
            <li key={s.part}><span className={`mix-swatch mix-share-${i % 3}`} aria-hidden="true" /><span className="mix-figure">{whole(s.value)}</span>
              <span className="mix-name">{s.part}{asked && <span className={asked.far ? "not-allowed" : "aside"}> {asked.words}</span>}{s.detail && <span className="mix-detail"> · {s.detail}</span>}</span></li>
          );
        })}
        {m.items && (
          <li><span className="mix-swatch mix-items" aria-hidden="true" /><span className="mix-figure">{m.items.expected ? "~" : ""}{whole(items)}</span>
            <span className="mix-name">{m.items.part}{m.shares.length > 0 && <span className="mix-detail"> of the whole</span>}{m.items.detail && <span className="mix-detail"> · {m.items.detail}</span>}</span></li>
        )}
      </ul>
    </div>
  );
}
