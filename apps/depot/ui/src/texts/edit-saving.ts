// The saves of the open text's edit (text-panel.tsx), apart from React so they are tested alone
// (edit-saving.test.ts, `bun test` in apps/depot/ui). It knows what the depot holds of the text as far as this
// editor knows (held), so the edit is dirty whenever the editor's content is not that. A save sends the editor's
// content until the depot holds it: words typed while a save was on its way go in one more request before the save
// answers, so a decision that follows (Admit & next) takes them. An answer of "nothing to save" means the depot
// holds that content too, so the editor can save the next edit. A save refused before sending or by the depot
// keeps the edit dirty.

// snap: the content compared with held (what same reads); body: what POST /api/texts/ID/content takes for it
export type Content<S> = { snap: S; body: Record<string, unknown> };
export type Answered = { ok: true; decision: string | null; version: number | null } | { ok: false; problems: string[] };
// made: the last decision a request of this save recorded, null when every answer was "nothing to save" or none
// was sent; before: refused before anything was sent in its last round (what the page asks is missing)
export type Made = { decision: string; version: number | null };
export type Saved = { ok: true; sent: boolean; made: Made | null }
  | { ok: false; sent: boolean; made: Made | null; problems: string[]; before: boolean };

// the most requests one save sends while the content keeps changing under it
export const ROUNDS = 5;

export class EditSaves<S> {
  held: S;
  version: number | null = null; // the content version this editor's saves made last, null before any
  private inflight: Promise<Saved> | null = null;

  // now: the editor's content now; same: whether two snaps are the same content; send: the request;
  // refusal: why a save cannot be sent now (no name and email, ...), or null; onStart and onEnd: once per save
  constructor(held: S, private readonly hooks: {
    now: () => Content<S>; same: (a: S, b: S) => boolean; send: (body: Record<string, unknown>) => Promise<Answered>;
    refusal: () => string | null; onStart?: () => void; onEnd?: (saved: Saved) => void;
  }) {
    this.held = held;
  }

  dirty(): boolean {
    return !this.hooks.same(this.hooks.now().snap, this.held);
  }

  saving(): boolean {
    return this.inflight !== null;
  }

  // The save on its way, or a new one: what it did once the depot holds the content, or why it does not.
  save(): Promise<Saved> {
    if (this.inflight) return this.inflight;
    if (!this.dirty()) return Promise.resolve({ ok: true, sent: false, made: null });
    this.hooks.onStart?.();
    const running = this.run().then((saved) => {
      this.inflight = null;
      this.hooks.onEnd?.(saved);
      return saved;
    });
    this.inflight = running;
    return running;
  }

  // The depot holds held now, as read again (the editor started over from it, or Revert).
  restart(held: S) {
    this.held = held;
  }

  private async run(): Promise<Saved> {
    let made: Made | null = null;
    let sent = false;
    for (let round = 0; round < ROUNDS && this.dirty(); round++) {
      const refused = this.hooks.refusal();
      if (refused) return { ok: false, sent, made, problems: [refused], before: true };
      const { snap, body } = this.hooks.now();
      let a: Answered;
      try {
        a = await this.hooks.send(body);
      } catch (error) {
        a = { ok: false, problems: [String(error)] };
      }
      if (!a.ok) return { ok: false, sent, made, problems: a.problems, before: false };
      sent = true;
      this.held = snap;
      if (a.decision !== null) {
        made = { decision: a.decision, version: a.version };
        this.version = a.version;
      }
    }
    if (this.dirty()) return { ok: false, sent, made, problems: ["it kept changing while it was saved; save again"], before: false };
    return { ok: true, sent, made };
  }
}

// JSON with every object's keys in order, so two items that differ only in key order compare the same, as the
// depot compares them.
export function canonical(value: unknown): string {
  return JSON.stringify(value, (_, v) => (v && typeof v === "object" && !Array.isArray(v)
    ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, v[k]])) : v));
}
