// Recipes: the model recipes of the repository checkout the server reads
// (GET /api/recipes, WPT_REPOSITORY), with the checkout's commit, each as its
// stages (snapshot -> recipe -> vocabulary -> corpus, pipeline.tsx) and its
// next step. A recipe is a file of the repository: composed on a snapshot's
// page, committed by a person; nothing here writes one.
//
// #/recipes/NAME is one recipe (GET /api/recipes/NAME): its stages, its data
// section, its measure (from its snapshot's summary), its estimate (from a
// sample) and the corpus of its build key (the last two only once it can be built here) with its downloads (GET
// /api/builds/KEY). Build opens one confirmation that shows what the build
// costs (tokens, time, disk, what the store's cap would delete) and queues it
// as a job of the heavy lane (POST /api/recipes/NAME/build with the estimate
// shown; 409 with the new one when it changed meanwhile), followed to its
// end. ?build=1 opens the confirmation at once. A corpus is a cache: a model
// records the snapshot and the recipe, never the corpus.

import { Button, ButtonLink, Dialog, Question, Skeleton } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, part, post, type Answer, type Job } from "../api";
import { BuildCost, MeasureWords, measureMix, type BuildEstimate, type Measure } from "../composer";
import { MixBar, whole } from "../mix";
import { Facts, Help, PageHead, plural, Problems, Section, takenWords } from "../parts";
import { Chain, pipelineChanged, snapshotHref, snapshotWords, standing, Stage, STAGES, StepButton, useRecipes, type CorpusSummary, type RecipeRow, type Recipes as RecipesAnswer } from "../pipeline";
import { href } from "../route";
import { bytes, tokensShort } from "../sizes";
import type { Page } from "./index";
import { JobLine, JobProgress } from "./jobs";
import { madeWhen } from "./snapshots";

type RecipeShown = RecipeRow & {
  repository: { commit: string | null; dirty: boolean } | null; problems: string[]; measure: Measure | null; estimate: BuildEstimate | null;
};
type BuildFile = { name: string; download: string; bytes: number | null; sha256: string; role: string; tokens: number | null };
type BuildShown = {
  build_key: string; state: "built" | "being built" | "not built" | "stale" | "deleted" | "checking an erasure" | "no store";
  job: Job<unknown> | null; problem?: string; corpus?: CorpusSummary; files?: BuildFile[]; archive?: { name: string; download: string; bytes: number } | null;
};

// The commit of the repository checkout the recipes are read from, with what it means behind its "?".
const commitWords = (r: RecipesAnswer["repository"]) => (r ? <>repository at <span className="ledger">{(r.commit ?? "no commit").slice(0, 12)}</span>{r.dirty && <span className="act"> with changes not committed</span>}
  <Help label="the commit">The recipes shown are the files of the repository checkout this server reads (the setting WPT_REPOSITORY), at this git commit. A model trained from one names this commit, so anyone can read the same recipe again.</Help></> : null);
// The file a model recipe is, in the repository: it is edited there and committed, never here.
// A series is listed as SERIES: (its one corpus, read at its largest size); the server names each file.
export const recipeFile = (name: string, file?: string) => file ?? (name.endsWith(":") ? `data/recipes/series/${name.slice(0, -1)}.json` : `data/recipes/models/${name}.json`);

function RecipesList() {
  const { answer } = useRecipes();
  const rows = answer?.ok ? answer.data.recipes : [];
  const first = rows.find((r) => standing(r).next)?.name;
  return (
    <>
      <PageHead title="Recipes"
        meta={answer?.ok && answer.data.repository ? <span>{commitWords(answer.data.repository)}</span> : undefined}
        chain={<>A model recipe is a file of the repository. Its data section names a <a href={href(["snapshots"])}>snapshot</a>, the kinds and the shares by tag it takes, a size and a <a href={href(["tokenizers"])}>vocabulary</a>. Its build gives a corpus, a cache that the same recipe always gives again. A recipe is edited as its file, <span className="ledger">data/recipes/models/NAME.json</span>, in the repository, and committed there; a new one starts on a snapshot's page. A series is listed as <span className="ledger">SERIES:</span>: its one corpus, which every size of the series reads.</>}
        actions={<ButtonLink variant="quiet" href={href(["snapshots"])}>New recipe from a snapshot</ButtonLink>} />
      {!answer && <Skeleton lines={4} />}
      {answer && !answer.ok && <Problems problems={answer.problems} />}
      {answer?.ok && answer.data.problem && <p className="empty-state">No repository here: {answer.data.problem}</p>}
      {answer?.ok && !answer.data.problem && (rows.length === 0 ? <p className="empty-state">The repository holds no model recipe.</p> : (
        <ul className="plain pipeline-list">
          <li className="pipeline-row pipeline-heads small aside" aria-hidden="true"><span>recipe</span><span>where it stands, and its next step</span></li>
          {rows.map((r) => (
            <li key={r.name} className="pipeline-row">
              <a className="pipeline-name cue" href={href(["recipes", r.name])}>{r.name}</a>
              <Chain recipe={r} at={standing(r)} action={r.name === first ? "primary" : "quiet"} />
            </li>
          ))}
        </ul>
      ))}
    </>
  );
}

// The corpus of a build key and its downloads.
function Corpus({ buildKey, version }: { buildKey: string; version: number }) {
  const [shown, setShown] = useState<Answer<BuildShown> | null>(null);
  useEffect(() => { get<BuildShown>(`/api/builds/${part(buildKey)}`).then(setShown); }, [buildKey, version]);
  if (!shown) return <Skeleton lines={3} />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const b = shown.data;
  const c = b.corpus;
  return (
    <>
      <p className="small">{b.state === "built" ? <span className="cue">Built</span> : <span className={b.state === "stale" || b.state === "no store" ? "not-allowed" : "aside"}>{b.state}</span>}
        {b.problem && <span className="not-allowed">: {b.problem}</span>}
        {c && <> · {madeWhen(c.built_at)} UTC · {takenWords(c.texts, c.conversations, c.tasks)} · {plural(c.total_tokens + (c.items_tokens ?? 0), "token")}{c.code_commit && <> · code <span className="ledger">{c.code_commit.slice(0, 12)}</span></>}</>}
        {c?.stale && <span className="not-allowed"> · stale since the erasures {c.stale.erasures.join(", ")}</span>}</p>
      {b.archive && <p><ButtonLink variant="ink" href={`/api/builds/${part(buildKey)}/${b.archive.name}`} download={b.archive.download}>Download everything ({bytes(b.archive.bytes)})</ButtonLink></p>}
      {b.files && b.files.length > 0 && (
        <div className="table-scroll">
          <table className="ledger-table" aria-label="Its files">
            <thead><tr><th scope="col">file</th><th scope="col">what</th><th scope="col" className="number">size</th><th scope="col" className="number">tokens</th></tr></thead>
            <tbody>
              {b.files.map((f) => (
                <tr key={f.name}>
                  <td><a href={`/api/builds/${part(buildKey)}/files/${part(f.name)}`} download={f.download} title={`sha256 ${f.sha256}`}>{f.download}</a></td>
                  <td>{f.role}</td>
                  <td className="number ledger">{f.bytes === null ? "" : bytes(f.bytes)}</td>
                  <td className="number ledger">{f.tokens === null || f.tokens === undefined ? "" : tokensShort(f.tokens)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="small aside">Build key <span className="ledger">{buildKey}</span>: the same snapshot, data section, vocabulary and build rules always give it.</p>
    </>
  );
}

function OneRecipe({ name, query }: { name: string; query: URLSearchParams }) {
  const [shown, setShown] = useState<Answer<RecipeShown> | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [estimate, setEstimate] = useState<BuildEstimate | null>(null); // the one the confirmation shows
  const [changed, setChanged] = useState(false);
  const [problems, setProblems] = useState<string[]>([]);
  const [started, setStarted] = useState<Job<unknown> | null>(null);
  const [ended, setEnded] = useState<Job<unknown> | null>(null); // the build this page followed, once ended
  const [version, setVersion] = useState(0);
  const load = useCallback(() => { get<RecipeShown>(`/api/recipes/${part(name)}`).then(setShown); setVersion((v) => v + 1); }, [name]);
  useEffect(load, [load]);
  // one recipe's answer gives the corpus of its key in its estimate (built)
  const r = shown?.ok ? { ...shown.data, corpus: shown.data.corpus ?? (shown.data.estimate?.built as CorpusSummary | null | undefined) ?? null } : null;
  const at = r ? standing(r) : null;
  const buildable = r?.state === "ready" && !r.job && !started && !r.estimate?.built?.fresh;
  const openBuild = () => { if (r?.estimate) { setEstimate(r.estimate); setChanged(false); setProblems([]); setConfirming(true); } };
  useEffect(() => { if (query.get("build") === "1" && buildable) openBuild(); }, [query.get("build"), buildable]);
  if (!shown) return <Skeleton lines={6} label="Reading the recipe (its estimate tokenizes a sample)" />;
  if (!shown.ok) return <><PageHead title={name} meta={<a href={href(["recipes"])}>recipes</a>} /><Problems problems={shown.problems} /></>;
  const recipe = r!;
  const d = recipe.data;
  const build = async () => {
    if (!estimate) return;
    const a = await post<Job<unknown>>(`/api/recipes/${part(name)}/build`, { confirmed: true, estimate });
    if (a.ok) { setStarted(a.data); setConfirming(false); return; }
    const fresh = a.status === 409 ? (a.body?.estimate as BuildEstimate | undefined) : undefined;
    if (fresh) { setEstimate(fresh); setChanged(true); }
    setProblems(a.problems);
  };
  const following = started ?? recipe.job ?? null;
  const sum = (d?.shares ?? []).reduce((t, s) => t + s.share, 0) || 1;
  return (
    <>
      <PageHead title={recipe.name}
        meta={<><a href={href(["recipes"])}>recipes</a>{recipe.snapshot && <a href={snapshotHref(recipe.snapshot)}>snapshot {recipe.snapshot.name}</a>}{recipe.repository && <span>{commitWords(recipe.repository)}</span>}</>}
        chain={<>A model recipe of the repository. Its corpus is a cache: a model trained on it records the depot, the snapshot, the recipe and the code, so anyone can check it.</>}
        actions={<>{recipe.state === "no vocabulary" && <span className="small head-why">No vocabulary yet</span>}{buildable && <Button variant="primary" onClick={openBuild}>Build</Button>}{!buildable && at?.next && !at.next.href.startsWith(href(["recipes", recipe.name])) && <StepButton at={at} primary />}</>} />
      {at && (
        <ol className="stages stages-4" aria-label={`Stages of ${recipe.name}`}>
          {STAGES.map((k) => <Stage key={k} name={at.stages[k].name} to={at.stages[k].to} state={at.stages[k].state} word={at.stages[k].state === "done" && k !== "corpus" ? undefined : at.stages[k].word || undefined}
            value={k === "snapshot" ? recipe.snapshot?.name ?? d?.snapshot ?? "" : k === "recipe" ? (d?.total_tokens ? tokensShort(d.total_tokens) : "every token") : k === "vocabulary" ? (recipe.vocabulary ? recipe.vocabulary.size.toLocaleString("en") : d?.vocabulary?.size?.toLocaleString("en") ?? "") : recipe.estimate ? tokensShort(recipe.estimate.tokens) : ""}
            label={k === "snapshot" ? (recipe.snapshot ? plural(recipe.snapshot.texts, "text") : "") : k === "recipe" ? "tokens asked" : k === "vocabulary" ? "tokens in the vocabulary" : recipe.estimate ? "tokens, about" : ""} />)}
        </ol>
      )}
      {recipe.why && recipe.state !== "no vocabulary" && <p className="not-allowed breaks">{recipe.why}</p>}
      {recipe.problems?.length > 0 && <Problems problems={recipe.problems} />}
      {following ? <JobLine key={following.id} job={following} onEnded={(j) => { setStarted(null); setEnded(j); pipelineChanged(); load(); }} />
        : ended && <JobLine key={ended.id} job={ended} />}
      <div className="board">
        <Section title="Data section">
          {d ? <><Facts rows={[
            ["file", <span className="ledger">{recipeFile(recipe.name, recipe.file)}</span>],
            ["snapshot", recipe.snapshot ? <a href={snapshotHref(recipe.snapshot)}>{snapshotWords({ name: d.snapshot, at_change: d.change })}</a> : snapshotWords({ name: d.snapshot, at_change: d.change })],
            ["kinds", d.kinds.length ? d.kinds.join(", ") : "every kind"],
            ["shares", d.shares.length ? d.shares.map((s) => `${s.tag} ${whole(s.share / sum)}`).join(", ") : "none: the plain texts as they come"],
            ["size", d.total_tokens ? plural(d.total_tokens, "token") : "every token"],
            ["vocabulary", recipe.vocabulary ? <a href={href(["tokenizers", recipe.vocabulary.recipe])}>{recipe.vocabulary.recipe} v{recipe.vocabulary.version}, {plural(recipe.vocabulary.size, "token")}</a>
              : recipe.state === "no vocabulary" ? <>not trained yet{d.vocabulary.size ? `, ${plural(d.vocabulary.size, "token")}` : ""}</>
              : <span className="ledger cut" title={d.vocabulary.sha256}>{d.vocabulary.sha256.slice(0, 12)}, {d.vocabulary.size ?? "?"} tokens</span>],
          ]} />
          <p className="small aside">Edited as this file in the repository, then committed: this page reads it again.</p>
          <details className="small data-more">
            <summary className="cue">More</summary>
            <Facts rows={[
              ["depot", <span className="ledger">{d.depot}</span>],
              ["digest", <span className="ledger breaks">{d.digest}</span>],
              ["vocabulary sha256", <span className="ledger breaks">{d.vocabulary.sha256}</span>],
            ]} />
          </details></> : <p className="small aside">None that reads.</p>}
        </Section>
        <Section title="What it takes" help="From its snapshot's summary, which never changes.">
          {recipe.measure ? <><MixBar m={measureMix(recipe.measure)} /><MeasureWords m={recipe.measure} /></> : <p className="small aside">Not known: its snapshot is not here.</p>}
        </Section>
        {recipe.estimate && (
          <Section title="What a build costs">
            <BuildCost estimate={recipe.estimate} />
            {recipe.estimate.warnings.length ? <ul className="small act">{recipe.estimate.warnings.map((w) => <li key={w}>{w}</li>)}</ul> : null}
          </Section>
        )}
        {recipe.build_key && (
          <Section title="Corpus">
            <Corpus buildKey={recipe.build_key} version={version} />
            {following && following.state === "running" && <p className="small"><JobProgress job={following} /></p>}
          </Section>
        )}
      </div>
      <Dialog open={confirming} onClose={() => setConfirming(false)} title={`Build ${recipe.name}?`}>
        {confirming && estimate && (
          <Question question={<>Build the corpus of <strong>{recipe.name}</strong> from the snapshot {recipe.snapshot?.name}?</>} from="before it starts"
            onSubmit={(e) => { e.preventDefault(); build(); }}
            actions={<Button type="submit" variant="primary" disabled={estimate.evicted === null || Boolean(estimate.store)}>Confirm and build</Button>}>
            <BuildCost estimate={estimate} />
            {changed && <p className="small act">The cost changed since it was read: the figures above are the new ones.</p>}
            <p className="small aside">A job of the <a href={href(["jobs"])}>queue</a>'s heavy lane; a restart of the server goes on from where it was.</p>
            <Problems problems={problems} />
          </Question>
        )}
      </Dialog>
    </>
  );
}

export const recipes: Page = {
  path: "recipes", title: "Recipes", group: "data",
  Component: ({ rest, query }) => (rest[0] ? <OneRecipe key={rest[0]} name={rest[0]} query={query} /> : <RecipesList />),
};
