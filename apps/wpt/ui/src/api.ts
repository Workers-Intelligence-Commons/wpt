// Calls to the wpt depot web server (its routes are described at /api/openapi.json).
// Every answer is JSON; a refusal carries {problems: string[]} with the depot's
// own messages: 400 a query or remote that is not one, 404 not found, 409 a
// job of the same snapshot or recipe queued or running already, or an
// estimate that changed since it was shown (with the new one), 422 refused by
// the depot's rules, 429 as many text exports running as the server sends at
// once, 503 missing on this machine. A proxy in front of the server answers
// 429 to a request past its limits too. A refusal that is not JSON keeps its status, with a message
// naming it. A request the server never answers (stopped, connection
// closed), or a success that is not JSON, comes back as status 0 with the
// error's message, so every page shows it.

// A refusal keeps the rest of what the server said in body (a 409 over a
// changed estimate gives the new one there).
export type Answer<T> = { ok: true; data: T } | { ok: false; status: number; problems: string[]; body?: Record<string, unknown> };

async function call<T>(method: string, path: string, body?: unknown): Promise<Answer<T>> {
  try {
    const response = await fetch(path, {
      method,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = response.ok ? await response.json() : await response.json().catch(() => ({}));
    return response.ok ? { ok: true, data } : { ok: false, status: response.status, problems: data.problems ?? [`the server answered ${response.status}`], body: data };
  } catch (error) {
    return { ok: false, status: 0, problems: [`no answer from the depot's server at ${path}: ${error instanceof Error ? error.message : String(error)}`] };
  }
}

export const get = <T,>(path: string) => call<T>("GET", path);
export const post = <T,>(path: string, body: unknown) => call<T>("POST", path, body);

// A write over a selection of texts, a vocabulary training, a snapshot and a
// corpus build run as jobs of the server's queue (GET /api/jobs/ID), in two
// lanes that each run one job at a time: write (the writes over a selection)
// and heavy (snapshot, vocabulary, corpus), so a long build never holds up a
// write. A job is queued, running, then done with its answer, or refused,
// failed or cancelled with its problem and status; stages, each stage a write
// started, with its time in seconds; progress, what a training, a snapshot
// or a build says it did so far; key, the snapshot's name or the corpus's
// build key; recipe, the tokenizer or model recipe it works for.
export type JobState = "queued" | "running" | "done" | "refused" | "failed" | "cancelled";
export type JobKind = "write" | "vocabulary" | "corpus" | "snapshot";
export type Lane = "write" | "heavy";
export type Job<T> = {
  id: string; kind: JobKind; lane: Lane; title: string; state: JobState; stages: { stage: string; at: number }[];
  progress: Record<string, unknown> | null; key: string | null; recipe: string | null;
  target: Record<string, unknown> | null; restarts?: number;
  created: number; started: number | null; ended: number | null; answer: T | null; problem: string | null; status: number | null;
};
export const waiting = (state: JobState) => state === "queued" || state === "running";

// milliseconds before each reading of a job not ended: soon at first, since most writes over a selection end
// within a second, then every POLL
const POLL = 1000;
const FIRST_READINGS = [100, 150, 250, 500];

// Send a write over a selection and follow its job to its end, calling
// progress with each reading while it waits or runs: its answer, or its
// refusal as any other refusal.
export async function write<T>(path: string, body: unknown, progress?: (job: Job<T>) => void): Promise<Answer<T>> {
  let a = await post<Job<T>>(path, body);
  for (let reading = 0; a.ok && waiting(a.data.state); reading++) {
    progress?.(a.data);
    await new Promise((done) => window.setTimeout(done, FIRST_READINGS[reading] ?? POLL));
    a = await get<Job<T>>(`/api/jobs/${encodeURIComponent(a.data.id)}`);
  }
  if (!a.ok) return a;
  if (a.data.state !== "done") return { ok: false, status: a.data.status ?? 500, problems: [a.data.problem ?? "the write failed"] };
  return { ok: true, data: a.data.answer as T };
}

export const put = <T,>(path: string, body: unknown) => call<T>("PUT", path, body);

// One part of a route: a name stays one part ("/" becomes %2F).
export const part = encodeURIComponent;
