// Where a dataset stands on its way from texts to a corpus, read the same way
// on every page that shows it: its tokenizer recipe (GET /api/tokenizers),
// the trained vocabularies (GET /api/vocabularies) and the corpora built from
// it (GET /api/datasets/ID). The first stage not done is the next one, with
// the button or link that does it; the stages after it wait. Texts, library
// and dataset count as done here: the dataset's own page measures them. Only
// a corpus of the dataset's current revision counts, since a corpus page
// serves only those.

import { ButtonLink } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { get, part, post, type Answer, type Job } from "./api";
import { ownLibraries, type Dataset } from "./parts";
import { href } from "./route";

export type Recipe = { name: string; version: number; trained?: number | boolean; job: Job<unknown> | null };
export type Trained = { sha256: string; recipe: string; name: string | null; version: number };
// A corpus as GET /api/datasets/ID lists it, of any revision of the dataset.
export type BuiltCorpus = {
  id: string; revision: number; vocabulary: string; max_tokens: number | null; built_at: string; texts: number; conversations: number;
  total_tokens: number; items_tokens?: number | null; stale: { erasures: string[] } | null; deleted: { at: string; why: string; whole: boolean } | null;
};
// The corpora that count for a dataset now: built from its current revision and not deleted.
export const currentCorpora = (dataset: Dataset, corpora: BuiltCorpus[] | null | undefined) =>
  (corpora ?? []).filter((c) => !c.deleted && c.revision === dataset.revision);

export const trainedWords = (v: Trained) => `${v.name ?? v.recipe} version ${v.version}`;
// A trained vocabulary's name in an address (mix-tok-v1), and each one's by
// its sha256 when no other vocabulary bears it, as a corpus page's address names it.
export const vocabularyName = (v: Trained) => `${v.name ?? v.recipe}-v${v.version}`;
export function vocabularyNames(trained: Trained[]): Map<string, string> {
  return new Map(trained.filter((v) => trained.filter((o) => vocabularyName(o) === vocabularyName(v)).length === 1).map((v) => [v.sha256, vocabularyName(v)]));
}

// The stages after the dataset itself, and the next step: its words, where it
// leads (href) or, for publishing, that the dataset's page does it (publish:
// the vocabulary to publish; its words name that page, since a link elsewhere
// publishes nothing, and the dataset's page shows its own button).
export type Step = { label: string; href: string; publish?: Trained };
export type Standing = {
  vocabulary: { state: StageState; word: string };
  corpus: { state: StageState; word: string };
  next: Step | null; // null: every stage done, or waiting on a job, or not known yet
  done: Step | null; // once built: where its corpora download
  fresh: number; // corpora of the current revision built and not stale
  own: Trained[]; // the vocabularies its own recipe trained, the latest last
};

// recipes, trained: null while read; corpora: undefined while read, null when the store cannot be read here.
export function standing(dataset: Dataset, recipes: Recipe[] | null, trained: Trained[] | null, corpora: BuiltCorpus[] | null | undefined): Standing {
  const own = (trained ?? []).filter((v) => v.name === dataset.tokenizer);
  const live = currentCorpora(dataset, corpora);
  const fresh = live.filter((c) => !c.stale).length;
  const stale = live.length - fresh;
  const corporaPage = href(["download", dataset.id]);
  const waitingCorpus = { state: "waiting" as StageState, word: "waiting" };
  const unknown = { state: "unknown" as StageState, word: "" };
  if (!recipes || !trained) return { vocabulary: unknown, corpus: unknown, next: null, done: null, fresh, own };
  const recipe = recipes.find((r) => r.name === dataset.tokenizer);
  let vocabulary: Standing["vocabulary"];
  let next: Step | null = null;
  if (dataset.vocabulary) {
    vocabulary = trained.some((v) => v.sha256 === dataset.vocabulary) ? { state: "done", word: "published" } : { state: "problem", word: "not held here" };
  } else if (!recipe) {
    vocabulary = { state: "next", word: "no recipe" };
    if (!dataset.upstream) next = { label: "Make the recipe", href: href(["tokenizers"], { new: "1", name: dataset.tokenizer, dataset: dataset.name }) };
  } else if (recipe.job) {
    vocabulary = { state: "waiting", word: "training" };
  } else if (own.length === 0) {
    vocabulary = { state: "next", word: "to train" };
    if (!dataset.upstream) next = { label: "Train it", href: href(["tokenizers", recipe.name], { train: "1" }) };
  } else {
    vocabulary = { state: "next", word: "to publish" };
    if (!dataset.upstream) next = { label: `Open to publish v${own.at(-1)!.version}`, href: href(["datasets", dataset.id]), publish: own.at(-1) };
  }
  let corpus: Standing["corpus"] = waitingCorpus;
  let done: Step | null = null;
  if (vocabulary.state === "done") {
    if (corpora === undefined) corpus = unknown;
    else if (corpora === null) corpus = { state: "problem", word: "no store" };
    else if (fresh) { corpus = { state: "done", word: "built" }; done = { label: "Download", href: corporaPage }; }
    else if (stale) { corpus = { state: "problem", word: "stale" }; next = { label: "Build again", href: corporaPage }; }
    else { corpus = { state: "next", word: "to build" }; next = { label: "Build a corpus", href: corporaPage }; }
  }
  return { vocabulary, corpus, next, done, fresh, own };
}

// What the stages read changes after a page's own action (a job that ends, a
// vocabulary published): pipelineChanged() tells every part of the page that
// reads them to read again, as does a move to another address, since a page
// kept for a new address would otherwise show what it read first.
//
// The parts of a page that read the same address at the same time share one
// request (shared): a page holds several parts that show the stages, and each
// reads through these hooks. A change forgets the requests under way, so what
// is read after it never comes from a request sent before it.
const CHANGED = "wpt-pipeline-changed";
const reading = new Map<string, Promise<Answer<unknown>>>();
function shared<T>(address: string): Promise<Answer<T>> {
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

// The recipes and the trained vocabularies, read once, again by reload and on a change (pipelineChanged).
export function useRecipesAndVocabularies(): { recipes: Answer<Recipe[]> | null; trained: Answer<Trained[]> | null; reload: () => void } {
  const [recipes, setRecipes] = useState<Answer<Recipe[]> | null>(null);
  const [trained, setTrained] = useState<Answer<Trained[]> | null>(null);
  const reload = useCallback(() => {
    shared<Recipe[]>("/api/tokenizers").then(setRecipes);
    shared<Trained[]>("/api/vocabularies").then(setTrained);
  }, []);
  useEffect(reload, [reload]);
  usePipelineChange(reload);
  return { recipes, trained, reload };
}

// The corpora of each dataset asked for, by id: undefined while read, null when the store cannot be read;
// read again on a change (pipelineChanged).
export function useCorpora(ids: string[]): Map<string, BuiltCorpus[] | null> {
  const [found, setFound] = useState<Map<string, BuiltCorpus[] | null>>(new Map());
  const [reading, setReading] = useState(0);
  const again = useCallback(() => setReading((n) => n + 1), []);
  usePipelineChange(again);
  const key = ids.join("\u0000");
  useEffect(() => {
    let live = true;
    for (const id of ids) shared<{ corpora: BuiltCorpus[] | null }>(`/api/datasets/${part(id)}`).then((a) => {
      if (live) setFound((m) => new Map(m).set(id, a.ok ? a.data.corpora : null));
    });
    return () => { live = false; };
  }, [key, reading]);
  return found;
}

// The standing of one dataset, by its id or name, with everything it reads.
export function usePipeline(dataset: Dataset | null): Standing | null {
  const { recipes, trained } = useRecipesAndVocabularies();
  const corpora = useCorpora(dataset ? [dataset.id] : []);
  if (!dataset) return null;
  return standing(dataset, recipes?.ok ? recipes.data : null, trained?.ok ? trained.data : null, corpora.get(dataset.id));
}

// One stage of a pipeline, as a card: its name (linked to its page when
// given), its state as a mark of a glyph and a word, never colour alone
// (done ✓, next →, waiting ○, problem ✕, unknown: still read, a grey card
// with no mark, so no state shows before it is true; word: another word for the state,
// such as "training"; mark false: none, for a stage that is a source rather
// than a step to finish), its big number and what it counts (empty: nothing
// to count yet, shown small and muted), an optional line under them, and at
// most one button; current marks the stage of the page shown with an ink bar
// under it. The same card makes a dataset's stages on every page about it
// (stages.tsx) and, without a number, the strip of a list of datasets (Chain).
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
const Glyph = ({ state }: { state: StageState }) => <span className={`chain-glyph${state === "done" ? " check" : ""}`} aria-hidden="true">{GLYPH[state]}</span>;

// A dataset's five stages, in order (STAGES), each with its name and the page it links
// to, for both the cards (stages.tsx) and the strip (Chain): the library stage
// opens the one library of this depot it reads, else the list of libraries.
export const STAGES = ["texts", "library", "dataset", "vocabulary", "corpus"] as const;
export type Where = (typeof STAGES)[number];
export function stagePlaces(dataset: Dataset): Record<Where, { name: string; to: string }> {
  const mine = ownLibraries(dataset);
  return {
    texts: { name: "Texts", to: href(["texts"]) },
    library: { name: dataset.libraries.length === 1 ? "Library" : "Libraries", to: mine.length === 1 ? href(["texts"], { library: mine[0] }) : href(["libraries"]) },
    dataset: { name: "Dataset", to: href(["datasets", dataset.id]) },
    vocabulary: { name: "Vocabulary", to: href(["tokenizers", dataset.tokenizer]) },
    corpus: { name: "Corpus", to: href(["download", dataset.id]) },
  };
}

// A dataset's stages in one line, for a list of datasets (the overview): the
// stage cards without their numbers, the same five stages marked the same way,
// each a link to its page. With action, the next step's button closes the
// line: a step still to do in red when primary (the page's main one), else
// quiet; a download, once built, always in ink, as every download is.
export function Chain({ dataset, at, action = "none" }: { dataset: Dataset; at: Standing | null; action?: "primary" | "quiet" | "none" }) {
  const places = stagePlaces(dataset);
  const states: Partial<Record<Where, { state: StageState; word?: string }>> = {
    vocabulary: { state: at?.vocabulary.state ?? "unknown", word: at?.vocabulary.word },
    corpus: { state: at?.corpus.state ?? "unknown", word: at?.corpus.word },
  };
  return (
    <div className="chain-row">
      <ol className="stages stages-small" aria-label={`Stages of ${dataset.name}`}>
        {STAGES.map((key) => <Stage key={key} name={places[key].name} to={places[key].to} state={states[key]?.state ?? "done"} word={states[key]?.word} />)}
      </ol>
      {action !== "none" && <StepButton at={at} primary={action === "primary"} />}
    </div>
  );
}

// The button of a dataset's next step, the same on every page and list: a
// step to do in red when it is the view's main one, else quiet; once built,
// Download in ink. Nothing while the standing is read or a job runs.
export function StepButton({ at, primary }: { at: Standing | null; primary: boolean }) {
  if (at?.next) return <ButtonLink variant={primary ? "primary" : "quiet"} href={at.next.href}>{at.next.label}</ButtonLink>;
  if (at?.done) return <ButtonLink variant="ink" href={at.done.href}>{at.done.label}</ButtonLink>;
  return null;
}

// Where a dataset stands, as one mark: the stage it has reached, by its glyph and words ("→ vocabulary: to train").
// While a state is still read: a grey dash, no words.
export function StageMark({ at }: { at: Standing }) {
  const [name, s] = at.vocabulary.state === "done" ? ["corpus", at.corpus] : ["vocabulary", at.vocabulary];
  if (s.state === "unknown") return <span className="stage-word aside" aria-label="reading">…</span>;
  return <span className={`stage-word chain-${s.state}`}><Glyph state={s.state} /> {name}: {s.word}</span>;
}

// Sets the vocabulary a dataset publishes (a sha256, or null for none) as a
// new revision of it, the rest of its row as read (POST /api/datasets/ID,
// which keeps each library it reads as stored). The revision read goes with
// it, so the server refuses the change when someone changed the dataset
// after it was read, rather than undo their change.
export const publishVocabulary = async (dataset: Dataset, vocabulary: string | null) => {
  const a = await post(`/api/datasets/${part(dataset.id)}`, {
    name: dataset.name, libraries: dataset.libraries.map((r) => ({ depot: r.depot, library: r.library })),
    shares: dataset.shares, max_tokens: dataset.max_tokens, vocabulary, revision: dataset.revision,
  });
  if (a.ok) pipelineChanged(); // the stages shown read again
  return a;
};
