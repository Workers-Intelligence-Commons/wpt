// A model recipe handed over as a file, and the build of its corpus. The
// depot reads no repository: a person drops a recipe's JSON file (a model
// recipe or a series; only its data section counts) on the Build page or on
// a snapshot's page, or the composer hands over the data section it made.
// The browser reads the file at once; the server checks it at once (POST
// /api/recipes/check: its snapshot here with the change and digest it names,
// its vocabulary held, what it takes from the snapshot's summary, the corpus
// of its build key); its estimate is a request of its own (POST
// /api/recipes/estimate), since it tokenizes a sample, a few seconds the
// first time, and the page shows it running with its seconds. Build opens
// one confirmation that shows the cost and queues the build as a job of the
// heavy lane (POST /api/recipes/build with the estimate shown; 409 with the
// new one when it changed meanwhile), followed to its end. The recipe last
// handed over is kept in the browser's tab (sessionStorage), so a reload or
// a way back shows it again. A recipe whose vocabulary the depot does not
// hold yet says so in one line beside "Train a vocabulary", its sha256 under
// "More"; the vocabulary's stage then shows the size asked, not a fact.

import { Button, ButtonLink, Dialog, Question, Skeleton } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { get, part, post, type Answer, type Job } from "./api";
import { BuildCost, MeasureWords, measureMix, type BuildEstimate, type Measure } from "./composer";
import { MixBar, whole } from "./mix";
import { Facts, plural, Problems, Section, takenWords } from "./parts";
import { pipelineChanged, snapshotHref, snapshotWords, standing, Stage, STAGES, type CorpusSummary, type RecipeRow } from "./pipeline";
import { href } from "./route";
import { bytes, tokensShort } from "./sizes";
import { JobLine, JobProgress, useNow } from "./pages/jobs";
import { madeWhen } from "./pages/snapshots";

// A recipe as handed over: its JSON and the name of its file (none for a data section the composer made).
export type Handed = { file: string | null; recipe: Record<string, unknown> };
type Checked = RecipeRow & { measure: Measure | null };
type BuildFile = { name: string; download: string; bytes: number | null; sha256: string; role: string; tokens: number | null };
export type BuildShown = {
  build_key: string; state: "built" | "being built" | "not built" | "stale" | "deleted" | "checking an erasure" | "no store";
  job: Job<unknown> | null; problem?: string; corpus?: CorpusSummary; files?: BuildFile[]; archive?: { name: string; download: string; bytes: number } | null;
};

const MOST_BYTES = 1 << 20; // a recipe is a few kilobytes: a bigger file is no recipe, refused before it is read
const KEPT = "wpt-recipe-handed"; // the recipe last handed over, in the tab's sessionStorage

export function keepHanded(handed: Handed | null) {
  try {
    if (handed) window.sessionStorage.setItem(KEPT, JSON.stringify(handed));
    else window.sessionStorage.removeItem(KEPT);
  } catch { /* storage refused: the recipe is simply not kept */ }
}
export function keptHanded(): Handed | null {
  try {
    const found = JSON.parse(window.sessionStorage.getItem(KEPT) ?? "null");
    return found && typeof found === "object" && found.recipe && typeof found.recipe === "object" ? found as Handed : null;
  } catch { return null; }
}

// A file as a recipe handed over, or what keeps it from being one.
export async function readRecipe(file: File): Promise<{ handed: Handed } | { problem: string }> {
  if (file.size > MOST_BYTES) return { problem: `${file.name} holds ${bytes(file.size)}: a recipe is a JSON file of a few kilobytes.` };
  try {
    const recipe = JSON.parse(await file.text());
    if (!recipe || typeof recipe !== "object" || Array.isArray(recipe)) return { problem: `${file.name} is JSON but not an object: a recipe is an object with a data section.` };
    return { handed: { file: file.name, recipe } };
  } catch (error) {
    return { problem: `${file.name} is not JSON: ${error instanceof Error ? error.message : String(error)}.` };
  }
}

// Where a recipe file is dropped or chosen: a button that opens the file
// picker, and the zone around it taking a file dropped on it. whole: the
// whole window takes a drop while it shows (the Build page).
export function RecipeDrop({ onHanded, whole: wholeWindow = false, primary = true, label = "Choose a recipe file" }: {
  onHanded: (handed: Handed) => void; whole?: boolean; primary?: boolean; label?: string;
}) {
  const picker = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [reading, setReading] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const take = useCallback(async (files: FileList | null | undefined) => {
    const file = files?.[0];
    if (!file) return;
    setProblem(null);
    setReading(file.name);
    const read = await readRecipe(file);
    setReading(null);
    if ("problem" in read) setProblem(read.problem);
    else onHanded(read.handed);
  }, [onHanded]);
  useEffect(() => {
    if (!wholeWindow) return;
    const dragged = (e: globalThis.DragEvent) => { if (e.dataTransfer?.types.includes("Files")) { e.preventDefault(); setOver(true); } };
    const left = (e: globalThis.DragEvent) => { if (!e.relatedTarget) setOver(false); };
    const dropped = (e: globalThis.DragEvent) => { if (e.dataTransfer?.files.length) { e.preventDefault(); setOver(false); take(e.dataTransfer.files); } };
    window.addEventListener("dragover", dragged);
    window.addEventListener("dragleave", left);
    window.addEventListener("drop", dropped);
    return () => { window.removeEventListener("dragover", dragged); window.removeEventListener("dragleave", left); window.removeEventListener("drop", dropped); };
  }, [wholeWindow, take]);
  const dragged = (e: DragEvent) => { e.preventDefault(); setOver(true); };
  const dropped = (e: DragEvent) => { e.preventDefault(); e.stopPropagation(); setOver(false); take(e.dataTransfer.files); };
  return (
    <div className={`recipe-drop${over ? " recipe-drop-over" : ""}`} onDragOver={dragged} onDragLeave={() => setOver(false)} onDrop={dropped}>
      <Button variant={primary ? "primary" : "quiet"} onClick={() => picker.current?.click()}>{label}</Button>
      <span className="small aside">{reading ? `Reading ${reading}…` : over ? "Drop it to read it" : "or drop it here: a model recipe or a series, its JSON file"}</span>
      <input ref={picker} type="file" accept=".json,application/json" hidden onChange={(e) => { take(e.target.files); e.target.value = ""; }} />
      {problem && <Problems problems={[problem]} />}
    </div>
  );
}

// The corpus of a build key and its downloads.
export function Corpus({ buildKey, version }: { buildKey: string; version: number }) {
  const [shown, setShown] = useState<Answer<BuildShown> | null>(null);
  useEffect(() => { get<BuildShown>(`/api/builds/${part(buildKey)}`).then(setShown); }, [buildKey, version]);
  if (!shown) return <Skeleton lines={3} label="Reading the corpus" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const b = shown.data;
  const c = b.corpus;
  return (
    <>
      <p className="small">{b.state === "built" ? <span className="cue">Built</span> : <span className={b.state === "stale" || b.state === "no store" ? "not-allowed" : "aside"}>{b.state}</span>}
        {b.problem && <span className="not-allowed">: {b.problem}</span>}
        {c && <> · {madeWhen(c.built_at)} UTC · {takenWords(c.texts, c.conversations, c.tasks)} · {plural(c.total_tokens + (c.items_tokens ?? 0), "token")}{c.recipe && <> · recipe {c.recipe}</>}{c.code_commit && <> · code <span className="ledger">{c.code_commit.slice(0, 12)}</span></>}</>}
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
      <p className="small aside">Build key <span className="ledger">{buildKey}</span>: the same snapshot, data section, vocabulary and build rules always give it. <a href={href(["build"], { key: buildKey })}>Its own address</a>.</p>
    </>
  );
}

// What a recipe handed over gives here: its stages, its data section, what it takes, what a build costs, its corpus,
// and Build. onClear: the button that puts it away.
export function RecipeBuild({ handed, onClear }: { handed: Handed; onClear?: () => void }) {
  const [checked, setChecked] = useState<Answer<Checked> | null>(null);
  const [estimate, setEstimate] = useState<Answer<BuildEstimate> | null>(null);
  const [estimating, setEstimating] = useState<number | null>(null); // when the estimate was asked, while it runs
  const [confirming, setConfirming] = useState(false);
  const [shownEstimate, setShownEstimate] = useState<BuildEstimate | null>(null); // the one the confirmation shows
  const [changed, setChanged] = useState(false);
  const [sending, setSending] = useState(false);
  const [problems, setProblems] = useState<string[]>([]);
  const [started, setStarted] = useState<Job<unknown> | null>(null);
  const [ended, setEnded] = useState<Job<unknown> | null>(null);
  const [version, setVersion] = useState(0);
  const asked = useRef(0);
  const load = useCallback(() => {
    const at = ++asked.current;
    setChecked(null);
    setEstimate(null);
    post<Checked>("/api/recipes/check", handed).then((a) => {
      if (at !== asked.current) return;
      setChecked(a);
      setVersion((v) => v + 1);
      if (!a.ok || a.data.state !== "ready") return;
      setEstimating(Date.now() / 1000);
      post<BuildEstimate>("/api/recipes/estimate", handed).then((e) => { if (at === asked.current) { setEstimate(e); setEstimating(null); } });
    });
  }, [handed]);
  useEffect(load, [load]);
  const now = useNow(estimating !== null);
  // the confirmation opened before the estimate came: it shows the estimate as soon as it comes
  useEffect(() => { if (confirming && !shownEstimate && estimate?.ok) setShownEstimate(estimate.data); }, [confirming, estimate]);

  const named = handed.file ?? "the composed recipe";
  if (!checked) return <Section title={`Recipe ${named}`}><Skeleton lines={4} label={`Checking ${named}`} /></Section>;
  if (!checked.ok) return (
    <Section title={`Recipe ${named}`} tools={onClear && <Button variant="quiet" onClick={onClear}>Put it away</Button>}>
      <Problems problems={checked.problems} />
    </Section>
  );
  const r = { ...checked.data, corpus: checked.data.corpus ?? null };
  const at = standing(r);
  const d = r.data;
  const following = started ?? r.job ?? null;
  const built = Boolean(r.corpus?.fresh) || Boolean(estimate?.ok && estimate.data.built?.fresh);
  const buildable = r.state === "ready" && !following && !built;
  const openBuild = () => { setShownEstimate(estimate?.ok ? estimate.data : null); setChanged(false); setProblems([]); setConfirming(true); };
  const build = async () => {
    if (!shownEstimate) return;
    setSending(true);
    const a = await post<Job<unknown>>("/api/recipes/build", { ...handed, confirmed: true, estimate: shownEstimate });
    setSending(false);
    if (a.ok) { setStarted(a.data); setConfirming(false); return; }
    const fresh = a.status === 409 ? (a.body?.estimate as BuildEstimate | undefined) : undefined;
    if (fresh) { setShownEstimate(fresh); setChanged(true); }
    setProblems(a.problems);
  };
  const sum = (d?.shares ?? []).reduce((t, s) => t + s.share, 0) || 1;
  const title = r.name ?? named;
  return (
    <Section className="recipe-build" title={<>Recipe {title}{r.kind === "series" && <span className="aside"> (a series: its one corpus, at its largest size)</span>}</>}
      tools={<span className="row tight">{buildable && <Button variant="primary" onClick={openBuild}>Build</Button>}{onClear && <Button variant="quiet" onClick={onClear}>Put it away</Button>}</span>}>
      <ol className="stages stages-4" aria-label={`Stages of ${title}`}>
        {STAGES.map((k) => <Stage key={k} name={at.stages[k].name} to={at.stages[k].to} state={at.stages[k].state} word={at.stages[k].state === "done" && k !== "corpus" ? undefined : at.stages[k].word || undefined}
          value={k === "snapshot" ? r.snapshot?.name ?? d?.snapshot ?? "" : k === "recipe" ? (d?.total_tokens ? tokensShort(d.total_tokens) : "every token") : k === "vocabulary" ? (r.vocabulary ? r.vocabulary.size.toLocaleString("en") : r.state === "no vocabulary" ? "not trained" : d?.vocabulary?.size?.toLocaleString("en") ?? "") : estimate?.ok ? tokensShort(estimate.data.tokens) : ""}
          empty={k === "vocabulary" && !r.vocabulary && r.state === "no vocabulary"}
          label={k === "snapshot" ? (r.snapshot ? plural(r.snapshot.texts, "text") : "") : k === "recipe" ? "tokens asked" : k === "vocabulary" ? (r.vocabulary ? "tokens in the vocabulary" : d?.vocabulary?.size ? `${d.vocabulary.size.toLocaleString("en")} tokens asked` : "") : estimate?.ok ? "tokens, about" : ""} />)}
      </ol>
      {r.state === "no vocabulary"
        ? <p className="row tight no-vocabulary"><span className="not-allowed">No vocabulary trained yet for this recipe.</span><ButtonLink variant="primary" href={href(["tokenizers"])}>Train a vocabulary</ButtonLink></p>
        : r.why && <p className="not-allowed breaks">{r.why}</p>}
      {following ? <JobLine key={following.id} job={following} onEnded={(j) => { setStarted(null); setEnded(j); pipelineChanged(); load(); }} />
        : ended && <JobLine key={ended.id} job={ended} />}
      <div className="board">
        <Section title="Data section">
          {d ? <><Facts rows={[
            ["file", handed.file ? <span className="ledger">{handed.file}</span> : "composed on a snapshot's page"],
            ["snapshot", r.snapshot ? <a href={snapshotHref(r.snapshot)}>{snapshotWords({ name: d.snapshot, at_change: d.change })}</a> : d.snapshot ? snapshotWords({ name: d.snapshot, at_change: d.change }) : null],
            ["kinds", d.kinds?.length ? d.kinds.join(", ") : "every kind"],
            ["shares", d.shares?.length ? d.shares.map((s) => `${s.tag} ${whole(s.share / sum)}`).join(", ") : "none: the plain texts as they come"],
            ["size", d.total_tokens ? plural(d.total_tokens, "token") : "every token"],
            ["vocabulary", r.vocabulary ? <a href={href(["tokenizers", r.vocabulary.recipe])}>{r.vocabulary.recipe} v{r.vocabulary.version}, {plural(r.vocabulary.size, "token")}</a>
              : r.state === "no vocabulary" ? <>not trained yet{d.vocabulary?.size ? `, ${plural(d.vocabulary.size, "token")}` : ""}</>
              : d.vocabulary ? <span className="ledger cut" title={d.vocabulary.sha256}>{d.vocabulary.sha256.slice(0, 12)}, {d.vocabulary.size ?? "?"} tokens</span> : null],
          ]} />
          <details className="small data-more">
            <summary className="cue">More</summary>
            <Facts rows={[
              ["depot", <span className="ledger">{d.depot}</span>],
              ["digest", d.digest ? <span className="ledger breaks">{d.digest}</span> : null],
              ["vocabulary sha256", d.vocabulary ? <span className="ledger breaks">{d.vocabulary.sha256}</span> : null],
            ]} />
          </details></> : <p className="small aside">None that reads.</p>}
        </Section>
        <Section title="What it takes" help="From its snapshot's summary, which never changes.">
          {r.measure ? <><MixBar m={measureMix(r.measure)} /><MeasureWords m={r.measure} /></> : <p className="small aside">Not known: its snapshot is not here.</p>}
        </Section>
        {r.state === "ready" && (
          <Section title="What a build costs">
            {estimating !== null && <p className="small" role="status" aria-live="polite"><span className="running-dot" aria-hidden="true" /> Working it out: tokenizing a sample, {Math.max(0, Math.round(now - estimating))} s so far (a few seconds the first time, then kept).</p>}
            {estimate && !estimate.ok && <Problems problems={estimate.problems} />}
            {estimate?.ok && <><BuildCost estimate={estimate.data} />{estimate.data.warnings.length ? <ul className="small act">{estimate.data.warnings.map((w) => <li key={w}>{w}</li>)}</ul> : null}</>}
          </Section>
        )}
        {r.build_key && (
          <Section title="Corpus">
            <Corpus buildKey={r.build_key} version={version} />
            {following && following.state === "running" && <p className="small"><JobProgress job={following} /></p>}
          </Section>
        )}
      </div>
      <Dialog open={confirming} onClose={() => setConfirming(false)} title={`Build ${title}?`}>
        {confirming && (
          <Question question={<>Build the corpus of <strong>{title}</strong> from the snapshot {r.snapshot?.name}?</>} from="before it starts"
            onSubmit={(e) => { e.preventDefault(); build(); }}
            actions={<Button type="submit" variant="primary" disabled={!shownEstimate || sending || shownEstimate.evicted === null || Boolean(shownEstimate.store)}>{sending ? "Queuing…" : "Confirm and build"}</Button>}>
            {shownEstimate ? <BuildCost estimate={shownEstimate} />
              : estimate && !estimate.ok ? <Problems problems={estimate.problems} />
              : <p className="small" role="status"><span className="running-dot" aria-hidden="true" /> Working out its cost: tokenizing a sample{estimating !== null ? `, ${Math.max(0, Math.round(now - estimating))} s so far` : ""}. The button turns on once it is shown.</p>}
            {changed && <p className="small act">The cost changed since it was read: the figures above are the new ones.</p>}
            <p className="small aside">A job of the <a href={href(["jobs"])}>queue</a>'s heavy lane; a restart of the server goes on from where it was.</p>
            <Problems problems={problems} />
          </Question>
        )}
      </Dialog>
    </Section>
  );
}
