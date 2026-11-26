// A plain text read translated, in place of its editor on the Texts page (text-panel.tsx): read only, made in this
// browser by Mozilla's models (engine.ts), and never written anywhere, so it cannot reach the depot. The text's
// language is what Firefox's detector finds when it is sure (detect.ts), else what a tag names, else the person's
// choice; a choice is kept for the text while the tab is open, and the language to read it in is kept in this
// browser. Its first PART characters are translated a line at a time, each batch shown as it comes, and the next PART
// on a click.

import { Button, Select, Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Answer } from "../api";
import { detect, type Detected } from "./detect";
import { listPairs, translate } from "./engine";
import { languageName, languages, route, source, type Pair } from "./languages";

const PART = 20000; // characters translated before asking to go on
const BATCH = 2000; // characters of lines sent to the engine at once, so the first lines show early
const TARGET = "wpt-translation-target"; // where this browser keeps the language to read in

let listed: Promise<Answer<{ pairs: Pair[] }>> | null = null;
const chosen = new Map<string, string>(); // the language chosen for a text, by its id

export function Translated({ id, text, tags }: { id: string; text: string; tags: string[] }) {
  const [answer, setAnswer] = useState<Answer<{ pairs: Pair[] }> | null>(null);
  useEffect(() => {
    if (!listed) listed = listPairs().then((a) => { if (!a.ok) listed = null; return a; });
    let live = true;
    void listed.then((a) => { if (live) setAnswer(a); });
    return () => { live = false; };
  }, []);
  // what the detector found in the text; null when it could not run, which leaves the language to the tags
  const [detected, setDetected] = useState<{ text: string; found: Detected | null } | null>(null);
  useEffect(() => {
    let live = true;
    detect(text).then((found) => found, () => null).then((found) => { if (live) setDetected({ text, found }); });
    return () => { live = false; };
  }, [text]);
  if (!answer || !detected) return <div className="translated"><Skeleton lines={6} label="Reading the list of translation models" /></div>;
  if (!answer.ok) return <div className="translated"><p className="act" role="alert">No translation: {answer.problems.join(" ")}</p></div>;
  return <WithModels id={id} text={text} tags={tags} detected={detected.found} pairs={answer.data.pairs} />;
}

function WithModels({ id, text, tags, detected, pairs }: {
  id: string; text: string; tags: string[]; detected: Detected | null; pairs: Pair[];
}) {
  const known = useMemo(() => languages(pairs), [pairs]);
  const found = useMemo(() => source(detected, tags, known), [detected, tags, known]);
  const [from, setFrom] = useState(() => chosen.get(id) ?? (found && known.includes(found.language) ? found.language : ""));
  const how = chosen.has(id) ? "chosen" : from && found?.language === from ? (found.how === "detected" ? "detected" : "from its tags") : null;
  const [to, setTo] = useState(() => {
    const kept = readTarget();
    const browser = navigator.language.split("-")[0];
    return [kept, browser, "en"].find((code) => code && known.includes(code))!;
  });
  const [upTo, setUpTo] = useState(PART); // characters of the text to translate
  const [state, setState] = useState<string | null>(null); // what the engine is doing, or why it stopped
  const [running, setRunning] = useState(false);
  const models = from ? route(pairs, from, to) : null;
  const lines = useMemo(() => text.split("\n"), [text]);
  const size = models?.reduce((sum, pair) => sum + Object.values(pair.files).reduce((s, f) => s + f.size, 0), 0) ?? 0;
  const key = models?.map((pair) => `${pair.from}|${pair.to}`).join(" ") ?? "";
  // the lines translated so far, for these models and this text; a run goes on from there
  const progress = useRef({ key, lines, out: [] as string[] });
  const [done, setDone] = useState<string[]>([]);

  useEffect(() => {
    if (progress.current.key !== key || progress.current.lines !== lines) {
      progress.current = { key, lines, out: [] };
      setDone([]);
      setUpTo(PART);
    }
    if (!models) return;
    const kept = progress.current;
    let live = true;
    setRunning(true);
    (async () => {
      let characters = lines.slice(0, kept.out.length).reduce((n, l) => n + l.length + 1, 0);
      if (kept.out.length === 0) setState(`Loading the models (${Math.round(size / 1e6)} MB the first time)…`);
      while (live && kept.out.length < lines.length && characters < upTo) {
        const start = kept.out.length;
        let end = start, batch = 0;
        while (end < lines.length && (end === start || batch + lines[end].length <= BATCH)) batch += lines[end++].length + 1;
        const some = lines.slice(start, end);
        const asked = some.map((line, i) => [line, i] as const).filter(([line]) => line.trim());
        const answers = await translate(models, asked.map(([line]) => line));
        if (!live) return;
        const out = some.map(() => "");
        asked.forEach(([, i], k) => { out[i] = answers[k]; });
        kept.out.push(...out);
        characters += batch;
        setDone([...kept.out]);
        setState(null);
      }
    })().catch((error) => { if (live) setState(`Stopped: ${error instanceof Error ? error.message : String(error)}`); })
      .finally(() => { if (live) setRunning(false); });
    return () => { live = false; };
  }, [key, lines, upTo]); // models and size follow key

  const left = lines.length - done.length;
  return (
    <div className="translated">
      <div className="translated-line small">
        <Select label={how ? `From (${how})` : "From"} value={from} onChange={(e) => { setFrom(e.target.value); chosen.set(id, e.target.value); }}>
          {!from && <option value="">Choose the text's language</option>}
          {known.map((code) => <option key={code} value={code}>{languageName(code)}</option>)}
        </Select>
        <Select label="Into" value={to} onChange={(e) => { setTo(e.target.value); keepTarget(e.target.value); }}>
          {known.filter((code) => code === to || !from || route(pairs, from, code)).map((code) => <option key={code} value={code}>{languageName(code)}</option>)}
        </Select>
        <span className="aside" role="status">
          {!from ? (found ? `Detected ${languageName(found.language)}, which no model translates: choose another language if that is wrong.`
              : "Neither the detector nor a tag is sure of its language: choose it.")
            : !models ? (from === to ? "It is in that language already." : `No model goes from ${languageName(from)} to ${languageName(to)}.`)
            : state ?? (running ? "Translating…" : left > 0 ? `Translated ${done.length.toLocaleString("en")} of ${lines.length.toLocaleString("en")} lines.` : "Translated whole.")}
        </span>
      </div>
      <div className="viewer content-scroll" lang={to} translate="no" aria-label="The text translated, read only">
        <p className="text-page">{done.join("\n")}</p>
        {models && left > 0 && !state && !running && (
          <Button variant="quiet" onClick={() => setUpTo((n) => n + PART)}>Translate the next {PART.toLocaleString("en")} characters</Button>
        )}
      </div>
    </div>
  );
}

function readTarget(): string | null {
  try { return window.localStorage.getItem(TARGET); } catch { return null; }
}
function keepTarget(code: string) {
  try { window.localStorage.setItem(TARGET, code); } catch { /* not kept: chosen again next time */ }
}
