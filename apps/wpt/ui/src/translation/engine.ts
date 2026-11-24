// The translation engine: Bergamot (@browsermt/bergamot-translator, MPL-2.0) in one Web Worker of this page, started
// on the first translation, with Mozilla's models read from GET /api/translations/files/SHA256. Everything runs on
// this machine: no text leaves the browser. The worker loads its glue and WebAssembly by name beside its own address,
// so its files are served as they are, from bergamot/ (vite.config.ts), and spoken to directly through its messages
// ({id, name, args} answered by {id, result} or {id, error}) rather than through the package's translator.js, which
// expects them beside itself.

import { get } from "../api";
import type { Pair } from "./languages";

export const listPairs = () => get<{ pairs: Pair[] }>("/api/translations");

// models loaded in the worker at most; each takes tens of MB of its memory, so the one used longest ago is freed
const KEPT_MODELS = 4;

type Call = (name: string, ...args: unknown[]) => Promise<any>;
let engine: Promise<Call> | null = null;
const loaded = new Map<string, Promise<void>>(); // by "from|to", in the order last used

function start(): Promise<Call> {
  const worker = new Worker(new URL("bergamot/translator-worker.js", document.baseURI));
  const pending = new Map<number, { accept: (v: unknown) => void; reject: (e: Error) => void }>();
  let serial = 0;
  worker.addEventListener("message", ({ data: { id, result, error } }) => {
    const waiting = pending.get(id);
    if (!waiting) return;
    pending.delete(id);
    if (error !== undefined) waiting.reject(new Error(error.message));
    else waiting.accept(result);
  });
  worker.addEventListener("error", (event) => {
    const error = new Error(event.message || "the translation engine stopped");
    for (const waiting of pending.values()) waiting.reject(error);
    pending.clear();
    engine = null;
    loaded.clear();
  });
  const call: Call = (name, ...args) => new Promise((accept, reject) => {
    const id = ++serial;
    pending.set(id, { accept, reject });
    worker.postMessage({ id, name, args });
  });
  return call("initialize", { cacheSize: 0 }).then(() => call);
}

async function file(sha256: string): Promise<ArrayBuffer> {
  const answer = await fetch(`/api/translations/files/${sha256}`);
  if (!answer.ok) {
    const problems = await answer.json().then((b) => b.problems as string[]).catch(() => [`HTTP ${answer.status}`]);
    throw new Error(problems.join(" "));
  }
  return answer.arrayBuffer();
}

function load(call: Call, pair: Pair): Promise<void> {
  const key = `${pair.from}|${pair.to}`;
  const known = loaded.get(key);
  if (known) {
    loaded.delete(key);
    loaded.set(key, known);
    return known;
  }
  const f = pair.files;
  const loading = Promise.all([f.model, f.lex, ...(f.vocab ? [f.vocab] : [f.srcvocab, f.trgvocab])].map((one) => file(one.sha256)))
    .then(([model, shortlist, ...vocabs]) => call("loadTranslationModel", { from: pair.from, to: pair.to },
      { model, shortlist, vocabs, qualityModel: null, config: {} }))
    .catch((error) => { loaded.delete(key); throw error; });
  loaded.set(key, loading);
  while (loaded.size > KEPT_MODELS) {
    const [oldest] = loaded.keys();
    loaded.delete(oldest);
    const [from, to] = oldest.split("|");
    void call("freeTranslationModel", { from, to });
  }
  return loading;
}

// The texts translated through the models given (languages.ts's route), in order.
export async function translate(models: Pair[], texts: string[]): Promise<string[]> {
  if (!engine) engine = start().catch((error) => { engine = null; throw error; });
  const call = await engine;
  await Promise.all(models.map((pair) => load(call, pair)));
  const answers: { target: { text: string } }[] = await call("translate", {
    models: models.map(({ from, to }) => ({ from, to })),
    texts: texts.map((text) => ({ text, html: false, qualityScores: false })),
  });
  return answers.map((one) => one.target.text);
}
