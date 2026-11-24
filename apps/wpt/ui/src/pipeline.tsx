// Where a model recipe stands on its way from the depot's texts to a corpus,
// read the same way on every page that shows it: snapshot -> recipe ->
// vocabulary -> corpus. A snapshot (GET /api/snapshots) is a name on a change
// number with the set of admitted texts then; a model recipe is a person's
// file, handed over to the Build page (POST /api/recipes/check), whose data
// section names a snapshot, kinds, shares by tag, a size and a vocabulary;
// the vocabulary is trained from a tokenizer recipe (GET /api/tokenizers, GET
// /api/vocabularies); the corpus is what a build of the recipe gives, a cache
// kept by its build key. The first stage not done is the next one, with the
// button or link that does it; the stages after it wait.
//
// This file also keeps what every page reads these through: one request per
// address at a time (shared), read again after a page's own action
// (pipelineChanged) or a move to another address, and the stage card (Stage).

import { useCallback, useEffect, useState, type ReactNode } from "react";
import { get, type Answer, type Job } from "./api";
import { href } from "./route";

// [texts, characters, bytes]
export type Sized = [number, number, number];
// A snapshot as GET /api/snapshots lists it; GET /api/snapshots/ID adds its
// summary and what erasures carried out since did to it.
export type Snapshot = {
  id: string; name: string; owner: string; at_change: number; change: number; made_at: string; origin: { kind: string } & Record<string, unknown>;
  note: string | null; texts: number; characters: number; bytes: number; digest: string;
};
export type Summary = {
  version: number; kinds: Record<string, Sized>; tags: Record<string, Sized>; plain_tagsets: Record<string, Sized>;
  licenses: Record<string, Sized>; owners: Record<string, number>;
};
export type SnapshotShown = Snapshot & {
  summary: Summary; erased_since: { texts: number; erasures: string[] };
};
// What GET /api/snapshots/new offers: the next free name and the cost of making one now.
export type SnapshotEstimate = { texts: number; seconds: number; bytes: number; set_bytes: number; summary_bytes: number; at: string; basis: string; change: number };
export type SnapshotOffer = { name: string; being_made: string[]; estimate: SnapshotEstimate };

// A recipe's data section, of the snapshot shape.
export type Share = { tag: string; share: number };
export type DataSection = {
  depot: string; snapshot: string; change: number; digest: string; kinds: string[]; shares: Share[];
  total_tokens: number | null; vocabulary: { sha256: string; size: number | null };
};
export type RecipeState = "ready" | "dataset" | "snapshot not here" | "snapshot differs" | "no vocabulary" | "vocabulary differs";
// The summary of a built corpus, as the recipes' routes give it.
export type CorpusSummary = {
  id: string; build_key: string; built_at: string; recipe: string | null; texts: number; conversations: number; tasks: number;
  total_tokens: number; items_tokens: number | null; fresh: boolean; stale: { erasures: string[] } | null;
  deleted: { at: string; why: string } | null; code_commit: string | null; link: string;
};
// A recipe handed over, as POST /api/recipes/check reads it: its name (its file's; a series' ends with ":"), its kind.
export type RecipeRow = {
  name: string | null; kind: "model" | "series"; state: RecipeState; why: string | null; data: DataSection; shape: "snapshot" | "dataset";
  snapshot: Snapshot | null; vocabulary: { sha256: string; size: number; recipe: string; version: number } | null;
  build_key: string | null; corpus: CorpusSummary | null; job: Job<unknown> | null;
};

// A tokenizer recipe as GET /api/tokenizers lists it, and a vocabulary a tokenizer recipe trained.
export type Tokenizer = {
  id: string; name: string; version: number; revision: number; sample: { snapshot: string; kinds: string[] | null } | null;
  snapshot_name: string | null; needs_snapshot: boolean; dataset: string; dataset_name: string | null; depot: string | null;
  snapshot_offer?: { id: string; name: string } | null; // for a revision written before snapshots: the snapshot made of its dataset
  sample_bytes: number; vocab_size: number; note: string | null; trained?: number | boolean; job: Job<unknown> | null;
};
export type Trained = { sha256: string; recipe: string; name: string | null; version: number; texts?: number; bytes?: number | null; made_at?: string };
export const trainedWords = (v: Trained) => `${v.name ?? v.recipe} version ${v.version}`;
// A trained vocabulary's short name (mix-tok-v1), and each one's by its sha256 when no other vocabulary bears it.
export const vocabularyName = (v: Trained) => `${v.name ?? v.recipe}-v${v.version}`;
export function vocabularyNames(trained: Trained[]): Map<string, string> {
  return new Map(trained.filter((v) => trained.filter((o) => vocabularyName(o) === vocabularyName(v)).length === 1).map((v) => [v.sha256, vocabularyName(v)]));
}

// What the stages read changes after a page's own action (a job that ends):
// pipelineChanged() tells every part of the page that reads them to read
// again, as does a move to another address, since a page kept for a new
// address would otherwise show what it read first.
//
// The parts of a page that read the same address at the same time share one
// request (shared). A change forgets the requests under way, so what is read
// after it never comes from a request sent before it.
const CHANGED = "wpt-pipeline-changed";
const reading = new Map<string, Promise<Answer<unknown>>>();
export function shared<T>(address: string): Promise<Answer<T>> {
  let answer = reading.get(address);
  if (!answer) {
    const sent: Promise<Answer<unknown>> = get<unknown>(address).finally(() => { if (reading.get(address) === sent) reading.delete(address); });
    reading.set(address, sent);
    answer = sent;
  }
  return answer as Promise<Answer<T>>;
}
export const pipelineChanged = () => { reading.clear(); window.dispatchEvent(new Event(CHANGED)); };
window.addEventListener("hashchange", () => reading.clear(), { capture: true });
export function usePipelineChange(read: () => void) {
  useEffect(() => {
    window.addEventListener(CHANGED, read);
    window.addEventListener("hashchange", read);
    return () => { window.removeEventListener(CHANGED, read); window.removeEventListener("hashchange", read); };
  }, [read]);
}

// One address read once, again on a change (pipelineChanged) and by reload.
function useShared<T>(address: string | null): { answer: Answer<T> | null; reload: () => void } {
  const [answer, setAnswer] = useState<Answer<T> | null>(null);
  const reload = useCallback(() => { if (address) shared<T>(address).then(setAnswer); }, [address]);
  useEffect(reload, [reload]);
  usePipelineChange(reload);
  return { answer, reload };
}
export const useSnapshots = () => useShared<Snapshot[]>("/api/snapshots");

// The tokenizer recipes and the trained vocabularies, read once, again by reload and on a change.
export function useRecipesAndVocabularies(): { recipes: Answer<Tokenizer[]> | null; trained: Answer<Trained[]> | null; reload: () => void } {
  const tokenizers = useShared<Tokenizer[]>("/api/tokenizers");
  const vocabularies = useShared<Trained[]>("/api/vocabularies");
  const reload = useCallback(() => { tokenizers.reload(); vocabularies.reload(); }, [tokenizers.reload, vocabularies.reload]);
  return { recipes: tokenizers.answer, trained: vocabularies.answer, reload };
}

// A snapshot's page, and the words that name it with its change.
export const snapshotHref = (s: { id: string }, query?: Record<string, string | null>) => href(["snapshots", s.id], query);
export const snapshotWords = (s: { name: string; at_change: number }) => `${s.name} (change ${s.at_change.toLocaleString("en")})`;

// One stage of a pipeline, as a card: its name (linked to its page when
// given), its state as a mark of a glyph and a word, never colour alone
// (done ✓, next →, waiting ○, problem ✕, unknown: still read, a grey card
// with no mark, so no state shows before it is true; word: another word for
// the state, such as "training"; mark false: none), its big number and what
// it counts (empty: nothing to count yet, shown small and muted), an
// optional line under them, and at most one button; current marks the stage
// of the page shown with an ink bar under it.
export type StageState = "done" | "next" | "waiting" | "problem" | "unknown";
// done: a tick drawn by .check, since some faces draw "✓" like a root sign
const GLYPH: Record<StageState, string> = { done: "", next: "→", waiting: "○", problem: "✕", unknown: "" };
export function Stage({ name, to, state, word, mark = true, value, empty = false, label, line, action, current = false }: {
  name: string; to?: string; state: StageState; word?: string; mark?: boolean; value?: ReactNode; empty?: boolean; label?: ReactNode; line?: ReactNode; action?: ReactNode; current?: boolean;
}) {
  const Name = value === undefined ? "span" : "h2"; // the small strip's names are no headings of the page
  return (
    <li className={`stage stage-${state}${current ? " stage-current" : ""}`}>
      <div className="stage-top">
        <Name className="cue stage-name">{to ? <a href={to} aria-current={current ? "step" : undefined}>{name}</a> : name}</Name>
        {mark && <span className="stage-mark cue" aria-hidden={state === "unknown" || undefined}><span className={state === "done" ? "check" : undefined} aria-hidden="true">{GLYPH[state]}</span> <span className={word ? undefined : "mark-plain"}>{word ?? state}</span></span>}
      </div>
      {value !== undefined && <div className={`stage-value${empty ? " empty" : ""}`}>{value}</div>}
      {label !== undefined && <div className="stage-label small">{label}</div>}
      {line && <div className="stage-line small">{line}</div>}
      {action && <div className="stage-action">{action}</div>}
    </li>
  );
}

// A recipe's four stages, in order, each with its name, the page it links to
// and its state; next: the step to do now (its words and where it leads),
// done: where its corpus downloads once built.
export const STAGES = ["snapshot", "recipe", "vocabulary", "corpus"] as const;
export type Where = (typeof STAGES)[number];
export type Step = { label: string; href: string };
export type Standing = { stages: Record<Where, { name: string; to: string; state: StageState; word: string }>; next: Step | null; done: Step | null };

export function standing(r: RecipeRow): Standing {
  const page = r.build_key ? href(["build"], { key: r.build_key }) : href(["build"]);
  const s = (name: string, to: string, state: StageState, word: string) => ({ name, to, state, word });
  const snapshotTo = r.snapshot ? snapshotHref(r.snapshot) : href(["snapshots"]);
  const later = (name: string, to: string) => s(name, to, "waiting", "waiting");
  const vocabularyTo = href(["tokenizers"]);
  if (r.state === "dataset")
    return { stages: { snapshot: s("Snapshot", snapshotTo, "unknown", ""), recipe: s("Recipe", page, "problem", "of a dataset"), vocabulary: later("Vocabulary", vocabularyTo), corpus: later("Corpus", page) }, next: null, done: null };
  const recipe = s("Recipe", page, "done", r.kind === "series" ? "a series" : "read");
  if (r.state === "snapshot not here" || r.state === "snapshot differs") {
    const word = r.state === "snapshot differs" ? "differs" : "not here";
    return { stages: { snapshot: s("Snapshot", snapshotTo, "problem", word), recipe, vocabulary: later("Vocabulary", vocabularyTo), corpus: later("Corpus", page) }, next: null, done: null };
  }
  const snapshot = s("Snapshot", snapshotTo, "done", r.snapshot?.name ?? r.data?.snapshot ?? "");
  if (r.state === "no vocabulary")
    return { stages: { snapshot, recipe, vocabulary: s("Vocabulary", vocabularyTo, "next", "to train"), corpus: later("Corpus", page) }, next: { label: "Train a vocabulary", href: vocabularyTo }, done: null };
  if (r.state === "vocabulary differs")
    return { stages: { snapshot, recipe, vocabulary: s("Vocabulary", vocabularyTo, "problem", "size differs"), corpus: later("Corpus", page) }, next: null, done: null };
  const vocabulary = s("Vocabulary", r.vocabulary ? href(["tokenizers", r.vocabulary.recipe]) : vocabularyTo, "done", r.vocabulary ? `v${r.vocabulary.version}` : "held");
  if (r.job) return { stages: { snapshot, recipe, vocabulary, corpus: s("Corpus", page, "waiting", r.job.state === "running" ? "building" : "queued") }, next: null, done: null };
  if (r.corpus?.fresh) return { stages: { snapshot, recipe, vocabulary, corpus: s("Corpus", page, "done", "built") }, next: null, done: { label: "Download", href: page } };
  if (r.corpus?.stale) return { stages: { snapshot, recipe, vocabulary, corpus: s("Corpus", page, "problem", "stale") }, next: { label: "Build again", href: page }, done: null };
  return { stages: { snapshot, recipe, vocabulary, corpus: s("Corpus", page, "next", "to build") }, next: { label: "Build", href: page }, done: null };
}
