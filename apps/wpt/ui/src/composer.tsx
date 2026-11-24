// The recipe composer of a snapshot's page: a recipe's data section made of
// clicks, nothing typed. Kinds as chips from the snapshot's summary (none
// picked: every kind), shares of the plain text as tag chips from the
// summary, each picked tag with a slider (the shares are parts of their sum),
// the size as preset buttons plus a slider, the vocabulary picked in the list
// of those the depot holds (GET /api/vocabularies). Each change asks the
// server, a moment after the last click, what the section gives (POST
// /api/recipes/compose, which writes nothing, at once: its measure from the
// summary), then its estimate from a sample (POST /api/recipes/estimate:
// tokens, disk, time, what the store's cap would delete), shown running
// until it comes. "Download recipe" saves the data section as a JSON file
// and "Copy JSON" copies it: the recipe is the person's file, beside its
// model part wherever they keep it. "Build it" hands it to the Build page.

import { Button, ButtonLink, Picker, Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useMemo, useRef, useState } from "react";
import { post, type Answer } from "./api";
import { keepHanded } from "./recipe-build";
import { MixBar, whole, type Mixed } from "./mix";
import { Facts, Help, plural, Problems, Section } from "./parts";
import { useRecipesAndVocabularies, vocabularyName, type DataSection, type SnapshotShown } from "./pipeline";
import { href } from "./route";
import { bytes, duration, tokensShort } from "./sizes";

// What POST /api/recipes/check and /api/recipes/compose give of what a recipe takes, and /api/recipes/estimate its cost.
export type Measure = {
  shares: { tag: string | null; share: number; texts: number; bytes: number }[]; plain: { texts: number; bytes: number };
  items: { texts: number; bytes: number }; no_share: number; other: number; kinds: Record<string, { texts: number; bytes: number }>;
};
export type BuildEstimate = {
  build_key: string; tokens: number; plain_tokens: number; item_tokens: number; item_share: number;
  share_tokens: { tag: string; tokens: number; part: number }[] | null; texts: number; plain_texts: number; item_texts: number;
  bytes: number; seconds: number | null; warnings: string[]; cap: number | null; taken: number | null;
  evicted: { corpus: string; bytes: number }[] | null; fits: boolean; built: { id: string; fresh: boolean } | null; store: string | null;
  sample: { texts: number; tokenized: number; bytes: number; seconds: number };
};
type Composed = { recipe: { data: DataSection }; build_key: string; vocabulary: { sha256: string; size: number }; measure: Measure };

export const PRESETS = [1e6, 1e7, 1e8, 1e9, 1e10];
const LOW = 5, HIGH = 11; // the slider's ends, as powers of ten of tokens
const STEPS = 60; // the slider's positions between them
const WAIT = 150; // milliseconds after the last click before the server is asked: a slider's moves come as one
const DEFAULT_SHARE = 50;

const toSlider = (tokens: number) => Math.round(((Math.log10(tokens) - LOW) / (HIGH - LOW)) * STEPS);
// a slider position as tokens, rounded to two significant digits
const fromSlider = (at: number) => { const n = 10 ** (LOW + (at / STEPS) * (HIGH - LOW)); const p = 10 ** Math.max(0, Math.floor(Math.log10(n)) - 1); return Math.round(n / p) * p; };

// The tags plain texts carry, from the summary's plain tag sets, the most texts first.
function plainTags(s: SnapshotShown): [string, number][] {
  const found = new Map<string, number>();
  for (const [set, sized] of Object.entries(s.summary.plain_tagsets))
    for (const tag of set.split("\u0000").filter(Boolean)) found.set(tag, (found.get(tag) ?? 0) + sized[0]);
  return [...found.entries()].sort((a, b) => b[1] - a[1]);
}

// The cost of a build, as facts: what a confirmation covers.
export function BuildCost({ estimate }: { estimate: BuildEstimate }) {
  const evicted = estimate.evicted;
  return (
    <Facts rows={[
      ["tokens", <>about {plural(estimate.tokens, "token")}{estimate.item_tokens > 0 && <span className="aside"> ({tokensShort(estimate.plain_tokens)} of plain text, {tokensShort(estimate.item_tokens)} of conversations and tasks)</span>}</>],
      ["time", estimate.seconds === null ? "not known" : estimate.seconds < 60 ? duration(estimate.seconds) : `about ${duration(estimate.seconds)}`],
      ["disk", <>about {bytes(estimate.bytes)} in the store{estimate.cap !== null && <span className="aside"> (cap {bytes(estimate.cap)}{estimate.taken !== null ? `, ${bytes(estimate.taken)} taken` : ""})</span>}</>],
      ["store", estimate.store ? <span className="not-allowed">{estimate.store}</span>
        : evicted === null ? <span className="not-allowed">does not fit under the cap, even once every other corpus it may delete is deleted</span>
        : evicted.length ? <span className="act">deletes {evicted.map((e) => `${e.corpus} (${bytes(e.bytes)})`).join(", ")} to make room</span> : "fits"],
      ["built", estimate.built ? (estimate.built.fresh ? "built already: download it" : "built before, stale since") : null],
      ["estimate", <span className="aside">from a sample of {plural(estimate.sample.tokenized, "text")} ({bytes(estimate.sample.bytes)}) tokenized in {estimate.sample.seconds.toFixed(1)} s</span>],
    ]} />
  );
}

// A measure as a mix: the items' part of the corpus and each share's part of the plain text, by bytes,
// each with how many texts and bytes it holds.
export function measureMix(m: Measure): Mixed {
  const all = m.plain.bytes + m.items.bytes || 1;
  const plain = m.shares.reduce((t, s) => t + s.bytes, 0) || 1;
  return {
    items: m.items.bytes > 0 ? { part: "conversations and tasks", value: m.items.bytes / all, expected: true, detail: `${plural(m.items.texts, "text")}, ${bytes(m.items.bytes)}` } : null,
    shares: m.shares.map((s) => ({ part: s.tag ?? "every plain text", value: s.bytes / plain, detail: `${plural(s.texts, "text")}, ${bytes(s.bytes)}` })),
  };
}

// What a measure leaves out, in words under its bar (the bar's legend says what it takes).
export function MeasureWords({ m }: { m: Measure }) {
  if (m.no_share <= 0 && m.other <= 0) return null;
  return (
    <ul className="plain small">
      {m.no_share > 0 && <li className="aside">{m.no_share === 1 ? "1 plain text is" : `${m.no_share.toLocaleString("en")} plain texts are`} in no share</li>}
      {m.other > 0 && <li className="aside">{m.other === 1 ? "1 text is" : `${m.other.toLocaleString("en")} texts are`} of a kind it does not take</li>}
    </ul>
  );
}

export function Composer({ snapshot, vocabulary: givenVocabulary, onClose }: { snapshot: SnapshotShown; vocabulary: string | null; onClose: () => void }) {
  const kinds = Object.entries(snapshot.summary.kinds).sort((a, b) => b[1][0] - a[1][0]);
  const tags = useMemo(() => plainTags(snapshot), [snapshot]);
  const { trained } = useRecipesAndVocabularies();
  const [picked, setPicked] = useState<string[]>([]); // kinds; none: every kind
  const [shares, setShares] = useState<{ tag: string; share: number }[]>([]);
  const [tokens, setTokens] = useState(1e8);
  const [vocabulary, setVocabulary] = useState<string | null>(givenVocabulary);
  const [composed, setComposed] = useState<Answer<Composed> | null>(null);
  const [estimate, setEstimate] = useState<Answer<BuildEstimate> | null>(null);
  const [asking, setAsking] = useState(false);
  const [estimating, setEstimating] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const asked = useRef(0);
  // the latest vocabulary first, picked when none is
  const vocabularies = trained?.ok ? trained.data : [];
  useEffect(() => { if (!vocabulary && vocabularies.length) setVocabulary(vocabularies[0].sha256); }, [vocabularies.length]);

  const body = vocabulary ? { snapshot: snapshot.id, kinds: picked, shares: shares.filter((s) => s.share > 0).map((s) => ({ tag: s.tag, share: s.share / 100 })), total_tokens: tokens, vocabulary } : null;
  const key = JSON.stringify(body);
  useEffect(() => {
    if (!body) return;
    const at = ++asked.current;
    setAsking(true);
    setEstimating(true);
    const timer = window.setTimeout(() => {
      post<Composed>("/api/recipes/compose", body).then((a) => {
        if (at !== asked.current) return;
        setComposed(a);
        setAsking(false);
        if (!a.ok) { setEstimating(false); return; }
        post<BuildEstimate>("/api/recipes/estimate", { recipe: a.data.recipe }).then((e) => { if (at === asked.current) { setEstimate(e); setEstimating(false); } });
      });
    }, WAIT);
    return () => window.clearTimeout(timer);
  }, [key]);

  const toggleKind = (k: string) => setPicked((p) => (p.includes(k) ? p.filter((x) => x !== k) : [...p, k]));
  const toggleTag = (t: string) => setShares((s) => (s.some((x) => x.tag === t) ? s.filter((x) => x.tag !== t) : [...s, { tag: t, share: DEFAULT_SHARE }]));
  const sum = shares.reduce((t, s) => t + s.share, 0) || 1;
  const json = composed?.ok ? JSON.stringify({ data: composed.data.recipe.data }, null, 2) : null;
  const fileName = `${snapshot.name}-${tokensShort(tokens)}${shares.length ? `-${shares.map((s) => s.tag).join("-")}` : ""}.json`.replace(/[^\w.-]+/g, "-");
  const download = () => {
    if (!json) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([json + "\n"], { type: "application/json" }));
    link.download = fileName;
    link.click();
    URL.revokeObjectURL(link.href);
  };
  const buildIt = () => { if (composed?.ok) { keepHanded({ file: fileName, recipe: { data: composed.data.recipe.data } }); window.location.hash = href(["build"]); } };
  const copy = async () => { if (!json) return; try { await navigator.clipboard.writeText(json); setCopied("Copied"); } catch { setCopied("Could not copy: the browser refused"); } };
  const options = vocabularies.map((v) => ({ value: v.sha256, label: `${vocabularyName(v)}`, group: v.name ?? v.recipe, slug: v.sha256.slice(0, 12) }));

  return (
    <Section className="composer" title={`New recipe from ${snapshot.name}`} tools={<Button variant="quiet" onClick={onClose}>Close</Button>}
      help="A recipe's data section, made of clicks. Nothing is written: build it here, or download it or copy it into a recipe file beside its model part.">
      <div className="composer-grid">
        <div className="composer-controls">
          <fieldset className="chips-field">
            <legend className="cue">Kinds</legend>
            <div className="chips">
              <button type="button" className="chip-button" aria-pressed={picked.length === 0} onClick={() => setPicked([])}>every kind</button>
              {kinds.map(([k, s]) => <button key={k} type="button" className="chip-button" aria-pressed={picked.includes(k)} onClick={() => toggleKind(k)}>{k} <span className="aside">{s[0].toLocaleString("en")}</span></button>)}
            </div>
          </fieldset>
          <fieldset className="chips-field">
            <legend className="cue">Shares of the plain text, by tag<Help label="shares">Each tag picked takes its part of the plain text, its slider against the sum of all; a text with several picked tags counts for the first. None picked: the plain texts as they come. Conversations and tasks take their natural part beside them.</Help></legend>
            {tags.length === 0 ? <p className="small aside">No plain text has a tag.</p> : (
              <div className="chips">
                {tags.map(([t, n]) => <button key={t} type="button" className="chip-button" aria-pressed={shares.some((s) => s.tag === t)} onClick={() => toggleTag(t)}>{t} <span className="aside">{n.toLocaleString("en")}</span></button>)}
              </div>
            )}
            {shares.map((s) => (
              <label key={s.tag} className="share-slider">
                <span className="cue">{s.tag}</span>
                <input type="range" min={0} max={100} step={5} value={s.share} aria-label={`share of ${s.tag}`}
                  onChange={(e) => setShares((all) => all.map((x) => (x.tag === s.tag ? { ...x, share: Number(e.target.value) } : x)))} />
                <span className="number ledger">{whole(s.share / sum)}</span>
              </label>
            ))}
          </fieldset>
          <fieldset className="chips-field">
            <legend className="cue">Size: {plural(tokens, "token")}</legend>
            <div className="chips">
              {PRESETS.map((p) => <button key={p} type="button" className="chip-button" aria-pressed={tokens === p} onClick={() => setTokens(p)}>{tokensShort(p)}</button>)}
            </div>
            <input className="size-slider" type="range" min={0} max={STEPS} value={toSlider(tokens)} aria-label="size in tokens" onChange={(e) => setTokens(fromSlider(Number(e.target.value)))} />
          </fieldset>
          <fieldset className="chips-field">
            <legend className="cue">Vocabulary</legend>
            {!trained && <Skeleton lines={1} />}
            {trained && !trained.ok && <Problems problems={trained.problems} />}
            {trained?.ok && (vocabularies.length === 0
              ? <p className="small">The depot holds no vocabulary yet. <ButtonLink variant="quiet" href={href(["tokenizers"], { new: "1", snapshot: snapshot.id })}>Train one on this snapshot</ButtonLink></p>
              : <Picker label="Vocabulary" options={options} value={vocabulary} onChange={setVocabulary} />)}
          </fieldset>
        </div>
        <div className="composer-result" aria-live="polite" aria-busy={asking}>
          {!body && <p className="small aside">Pick a vocabulary to measure it.</p>}
          {body && !composed && <Skeleton lines={5} label="Measuring" />}
          {composed && !composed.ok && <Problems problems={composed.problems} />}
          {composed?.ok && (
            <>
              <h3 className="aside">What it takes{asking && " (measuring again)"}</h3>
              <MixBar m={measureMix(composed.data.measure)} />
              <MeasureWords m={composed.data.measure} />
              <h3 className="aside">What the build costs{estimating && estimate && " (working it out again)"}</h3>
              {estimating && !estimate && <p className="small" role="status"><span className="running-dot" aria-hidden="true" /> Working it out: tokenizing a sample, a few seconds the first time.</p>}
              {estimate && !estimate.ok && <Problems problems={estimate.problems} />}
              {estimate?.ok && <>
                <BuildCost estimate={estimate.data} />
                {estimate.data.tokens < tokens * 0.99 && <p className="small act">The snapshot gives less than the {tokensShort(tokens)} tokens asked: the build takes all it has.</p>}
                {estimate.data.warnings.length > 0 && <ul className="small act">{estimate.data.warnings.map((w) => <li key={w}>{w}</li>)}</ul>}
              </>}
              <div className="row tight">
                <Button variant="primary" onClick={buildIt}>Build it</Button>
                <Button variant="ink" onClick={download}>Download recipe</Button>
                <Button variant="quiet" onClick={copy}>Copy JSON</Button>
                {copied && <span className="small aside" role="status">{copied}</span>}
              </div>
              <details className="fold small">
                <summary className="cue">The JSON</summary>
                <pre className="recipe-json">{json}</pre>
              </details>
            </>
          )}
        </div>
      </div>
    </Section>
  );
}
