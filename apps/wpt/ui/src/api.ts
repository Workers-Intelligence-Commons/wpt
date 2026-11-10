// Calls to the wpt depot web server (its routes are described at /api/openapi.json).
// Every answer is JSON; a refusal carries {problems: string[]} with the depot's
// own messages: 400 a query or remote that is not one, 404 not found, 409 an
// address that holds a depot, 422 refused by the depot's rules, 503 missing on
// this machine. A request the server never
// answers (stopped, connection closed) or an answer that is not JSON comes
// back as status 0 with the error's message, so every page shows it.

export type Answer<T> = { ok: true; data: T } | { ok: false; status: number; problems: string[] };

async function call<T>(method: string, path: string, body?: unknown): Promise<Answer<T>> {
  try {
    const response = await fetch(path, {
      method,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await response.json();
    return response.ok ? { ok: true, data } : { ok: false, status: response.status, problems: data.problems ?? [`the server answered ${response.status}`] };
  } catch (error) {
    return { ok: false, status: 0, problems: [`no answer from the depot's server at ${path}: ${error instanceof Error ? error.message : String(error)}`] };
  }
}

export const get = <T,>(path: string) => call<T>("GET", path);
export const post = <T,>(path: string, body: unknown) => call<T>("POST", path, body);

// One part of a route: a dataset's name stays one part ("/" becomes %2F).
export const part = encodeURIComponent;
