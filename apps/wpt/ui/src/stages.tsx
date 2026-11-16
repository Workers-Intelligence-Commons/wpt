// A dataset's stages as cards, from what it reads to what it builds, the
// same five cards on every page that shows them (the overview, the dataset's
// page, its tokenizer recipe's, its corpora and its export): its texts (what
// its current revision reads, as it stores it, each library counted apart and
// the sum marked so when there are several, with how many more or fewer texts
// they hold now; "unknown" when it does not know them), its libraries, the
// dataset itself (its mix, a share that takes no text a problem, unknown when
// it has shares and no figures), its vocabulary and its
// corpora (standing, pipeline.tsx: make its tokenizer recipe, train it,
// publish it, build a corpus, or download one built). The first stage not
// done is the next one, with its button; the stages after it wait.
// Publishing happens on the card itself, the vocabulary its own recipe
// trained last. A dataset of the upstream has no button that changes it.
//
// here: the stage of the page that shows the cards, marked; action: whether
// the next step's button is the view's main one (primary), a quiet one, or
// none (a page whose own button does that step).

import { Button, ButtonLink } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { get, part, type Answer } from "./api";
import { asked, MixBar } from "./mix";
import { characters, Help, plural, Problems, Since, SUMMED_NOTE, type Dataset, type DatasetRef, type RevisionFigures } from "./parts";
import { currentCorpora, publishVocabulary, Stage, stagePlaces, standing, trainedWords, usePipelineChange, useRecipesAndVocabularies, type BuiltCorpus, type Recipe, type StageState, type Trained, type Where } from "./pipeline";
import { href } from "./route";
import { JobProgress } from "./pages/jobs";

// What GET /api/datasets/ID gives.
export type Shown = {
  dataset: Dataset; upstream: boolean; remote: unknown; corpora: BuiltCorpus[] | null; store: string | null;
  libraries: (DatasetRef & { local: boolean; name: string | null; locked_at: string | null })[];
  figures: RevisionFigures;
  read_again: boolean;
};

// The cards of a dataset known by its id, read here: what GET
// /api/datasets/ID gives, read again on a change (pipelineChanged), with what its
// revision reads. onPublished: after a vocabulary published from its card.
export function DatasetStages({ id, here, action = "primary", onPublished }: { id: string; here?: Where; action?: "primary" | "quiet" | "none"; onPublished?: (a: Answer<any>, words: string) => void }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  const { recipes, trained } = useRecipesAndVocabularies();
  const read = useCallback(() => { get<Shown>(`/api/datasets/${part(id)}`).then(setShown); }, [id]);
  useEffect(read, [read]);
  usePipelineChange(read);
  if (!shown) return <ol className="stages stages-reading" aria-label="From texts to corpus" aria-busy="true" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  return <Stages shown={shown.data} recipes={recipes} trained={trained} here={here} action={action} onPublished={(a, words) => { read(); onPublished?.(a, words); }} />;
}

export function Stages({ shown, recipes, trained, here, action: role = "primary", onPublished }: {
  shown: Shown; recipes: Answer<Recipe[]> | null; trained: Answer<Trained[]> | null;
  here?: Where; action?: "primary" | "quiet" | "none"; onPublished?: (a: Answer<any>, words: string) => void;
}) {
  const { dataset, libraries, corpora } = shown;
  const [publishing, setPublishing] = useState(false);
  const found = shown.figures.measure;
  const texts = found?.texts ?? null;
  const missing = found ? found.shares.filter((s) => s.texts === 0).map((s) => s.tag) : [];
  const recipe = recipes?.ok ? recipes.data.find((r) => r.name === dataset.tokenizer) : undefined;
  const at = standing(dataset, recipes?.ok ? recipes.data : null, trained?.ok ? trained.data : null, corpora);
  const published = trained?.ok && dataset.vocabulary ? trained.data.find((v) => v.sha256 === dataset.vocabulary) : undefined;
  // the corpora of its current revision, as standing counts them
  const current = currentCorpora(dataset, corpora);
  const stale = current.filter((c) => c.stale);
  const publish = async (v: Trained) => {
    setPublishing(true);
    const a = await publishVocabulary(dataset, v.sha256);
    setPublishing(false);
    onPublished?.(a, `Published ${trainedWords(v)}.`);
  };

  const textsState: StageState = texts === null ? "unknown" : texts === 0 ? "problem" : "done";
  // a share may take nothing, which only the figures can tell: unknown without them
  const datasetState: StageState = missing.length ? "problem" : !found && dataset.shares?.length ? "unknown" : "done";
  const step = at.next;
  const variant = role === "primary" ? "primary" : "quiet";
  const action: ReactNode = role !== "none" && step && (step.publish
    ? <Button variant={variant} disabled={publishing} onClick={() => publish(step.publish!)}>Publish v{step.publish.version}</Button>
    : <ButtonLink variant={variant} href={step.href}>{step.label}</ButtonLink>);
  // the vocabulary card's figures: not known until the recipes and vocabularies are read
  let vocabulary: { value: ReactNode; label: ReactNode; line?: ReactNode; empty?: boolean };
  if (!recipes || !trained) vocabulary = { value: "…", label: dataset.tokenizer, empty: true };
  else if (!recipes.ok || !trained.ok) vocabulary = { value: "?", label: <Problems problems={[...(recipes.ok ? [] : recipes.problems), ...(trained.ok ? [] : trained.problems)]} /> };
  else if (dataset.vocabulary) vocabulary = published ? { value: `v${published.version}`, label: published.name ?? published.recipe } : { value: "?", label: "published, not held here" };
  else if (!recipe) vocabulary = { value: "none", label: <>{dataset.tokenizer}, not made</>, empty: true };
  else if (recipe.job) vocabulary = { value: `v${recipe.version}`, label: recipe.name, line: <a href={href(["tokenizers", recipe.name])}><JobProgress job={recipe.job} /></a> };
  else if (at.own.length === 0) vocabulary = { value: `v${recipe.version}`, label: recipe.name };
  else vocabulary = { value: `v${at.own.at(-1)!.version}`, label: <>{recipe.name}, trained</> };
  const place = stagePlaces(dataset);
  // the mix it asks for (mix.tsx): the conversations' expected part of the corpus, then its shares of the rest
  const mix = asked(dataset.shares, found);
  // the corpus card's figure: the tokens of the fresh corpus at the dataset's own size, else of the latest fresh one
  const fresh = current.filter((c) => !c.stale);
  const latest = fresh.find((c) => c.max_tokens === dataset.max_tokens) ?? fresh[0];
  const latestTokens = latest ? latest.total_tokens + (latest.items_tokens ?? 0) : null;
  const others = fresh.length - (latest ? 1 : 0);
  return (
    <ol className="stages" aria-label={`Stages of ${dataset.name}`}>
      <Stage name={place.texts.name} to={place.texts.to} state={textsState} word={texts === 0 ? "none" : undefined} current={here === "texts"}
        value={texts === null ? "unknown" : texts.toLocaleString("en")} empty={texts === null}
        label={found ? <>{characters(found.characters)}{libraries.length > 1 && <Help label="summed texts">{SUMMED_NOTE}</Help>}</> : <span title="This revision cannot say what its libraries held at the change it reads them at">for this revision</span>}
        line={<Since since={shown.figures.since} />} />
      <Stage name={place.library.name} to={place.library.to} state="done" current={here === "library"}
        value={libraries.length === 1 ? (libraries[0].local ? libraries[0].name ?? libraries[0].library : libraries[0].library) : libraries.length.toLocaleString("en")}
        label={libraries.length === 1 ? (libraries[0].local ? (libraries[0].locked_at ? "locked" : "open") : "of another depot") : libraries.map((r) => r.local ? r.name ?? r.library : r.library).join(", ")} />
      <Stage name={place.dataset.name} to={place.dataset.to} state={datasetState} word={missing.length ? "empty share" : undefined} current={here === "dataset"}
        value={mix.items || mix.shares.length ? <MixBar m={mix} /> : "every text"}
        label={dataset.max_tokens ? `at most ${plural(dataset.max_tokens, "token")}` : "no size limit"} />
      <Stage name={place.vocabulary.name} to={place.vocabulary.to} state={at.vocabulary.state} word={at.vocabulary.word} current={here === "vocabulary"}
        value={vocabulary.value} empty={vocabulary.empty} label={vocabulary.label} line={vocabulary.line}
        action={at.vocabulary.state === "next" ? action : undefined} />
      <Stage name={place.corpus.name} to={place.corpus.to} state={at.corpus.state} word={at.corpus.word} current={here === "corpus"}
        value={corpora === null ? "?" : latestTokens !== null ? latestTokens.toLocaleString("en") : "none"} empty={corpora !== null && latestTokens === null}
        label={corpora === null ? "store unreadable" : latestTokens !== null ? `tokens of ${latest!.max_tokens === null ? "every token" : latest!.max_tokens.toLocaleString("en")}${stale.length ? `, ${stale.length} stale` : ""}` : stale.length ? `${stale.length} stale` : "none yet"}
        line={others > 0 && <a href={href(["download", dataset.id])}>+{others} more</a>}
        action={at.vocabulary.state === "done" && role !== "none" && here !== "corpus" ? (step ? action : at.done && <ButtonLink variant="ink" href={at.done.href}>{at.done.label}</ButtonLink>) : undefined} />
    </ol>
  );
}
