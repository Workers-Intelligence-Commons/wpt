// Erasure requests: every one, kept for good (wpt depot erasures), one
// decision on several open ones together (wpt depot decide --erasure), the
// page's main action, asking for an erasure (a text found by its title or
// id, then the form the workspace's text shares, erase-form.tsx), and the view of
// what one request asks to erase (#/erasures?request=ID, erasure-view.tsx).
// A request names passages by their place, never by their content. Once a
// decision admits a request, the depot carries it out at once, which cannot
// be undone. A request that belongs to the upstream this depot follows, or
// names a text of it, says so and has no tick, and a text of the upstream takes no request here: only
// the upstream asks for and decides on those, and a pull carries them out.

import { Button, Dialog, Skeleton, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { get, post, type Answer } from "../api";
import { EraseForm, type Erasable } from "../erase-form";
import { ErasureView } from "../erasure-view";
import { search, EVERY, type Row } from "../filter";
import { codePoints } from "../reader";
import { Confirm, Cut, optional, Outcome, PageHead, plural, Problems, required, RequiredNote, StatusBadge, type Declaration, type Status } from "../parts";
import { href } from "../route";
import type { Page } from "./index";

type Needle = { text: string; field: string | null; start: number | null; end: number | null };
// upstream: it belongs to the upstream this depot follows or names a text of it, which the upstream alone decides
type Erasure = { id: string; reason: string; decisions: string[]; needles: Needle[]; erased: { length: number }[] | null; status: Status; upstream: boolean };
type Decided = { carried_out: Record<string, boolean> };
type Choice = "admitted" | "refused";

function place(n: Needle): string {
  if (n.start === null) return "the whole text";
  return `${n.start} to ${n.end}${n.field ? ` of ${n.field}` : ""}`;
}

function Erasures({ query }: { query: URLSearchParams }) {
  const shownId = query.get("request");
  const [asking, setAsking] = useState(false);
  const [said, setSaid] = useState<ReactNode>(null);
  const [rows, setRows] = useState<Answer<Erasure[]> | null>(null);
  const [ticked, setTicked] = useState<string[]>([]);
  const [deciding, setDeciding] = useState<Choice | null>(null);
  const [answer, setAnswer] = useState<{ status: Choice; answer: Answer<Decided> } | null>(null);
  const load = useCallback(() => { get<Erasure[]>("/api/erasures").then(setRows); }, []);
  useEffect(load, [load]);
  const open = rows?.ok ? rows.data.filter((e) => e.status === "under review" && !e.upstream) : []; // those this depot decides
  const flip = (ids: string[], on: boolean) => setTicked(on ? [...new Set([...ticked, ...ids])] : ticked.filter((t) => !ids.includes(t)));
  const done = (status: Choice, a: Answer<Decided>) => {
    setAnswer({ status, answer: a });
    if (a.ok) { setDeciding(null); setTicked([]); load(); }
  };
  return (
    <>
      <PageHead title="Erasure requests"
        chain={<>A request names passages of <a href={href(["texts"])}>texts</a> by their place, asked here or beside a text in the workspace; admitted, it cuts them, and every corpus holding them goes stale. Every request is kept.</>}
        actions={<Button variant={ticked.length ? "quiet" : "primary"} onClick={() => { setSaid(null); setAsking(true); }}>Ask for an erasure</Button>} />
      {said && <p className="done cue" role="status">{said}</p>}
      {answer && !deciding && <Outcome answer={answer.answer} done={(d) => {
        const ids = Object.keys(d.carried_out);
        const carried = ids.filter((id) => d.carried_out[id]).length;
        return `${plural(ids.length, "request")} ${answer.status}${carried ? `; ${carried} carried out` : ""}.`;
      }} />}
      {!rows && <Skeleton lines={3} label="Reading the requests" />}
      {rows && !rows.ok && <Problems problems={rows.problems} />}
      {rows?.ok && (rows.data.length === 0 ? <p>No erasure request.</p> : (
        <div className="table-scroll">
          <table className="ledger-table" aria-label="The erasure requests">
            <thead>
              <tr>
                <th scope="col" className="tick-cell">
                  <input type="checkbox" className="wic-check-box wic-focus" aria-label="Tick every open request" disabled={open.length === 0}
                    checked={open.length > 0 && open.every((e) => ticked.includes(e.id))} onChange={(e) => flip(open.map((o) => o.id), e.target.checked)} />
                </th>
                <th scope="col">request</th><th scope="col">status</th><th scope="col">reason</th><th scope="col">passages (characters)</th><th scope="col">carried out</th>
              </tr>
            </thead>
            <tbody>
              {rows.data.map((e) => (
                <tr key={e.id} className={ticked.includes(e.id) ? "ticked" : undefined}>
                  <td className="tick-cell">
                    {e.status === "under review" && !e.upstream && <input type="checkbox" className="wic-check-box wic-focus" aria-label={`Tick ${e.id}`} checked={ticked.includes(e.id)} onChange={(x) => flip([e.id], x.target.checked)} />}
                  </td>
                  <td className="ledger nowrap"><a href={href(["erasures"], { request: e.id })} title={`What ${e.id} asks to erase`}>{e.id}</a>{e.upstream && <span className="aside"> the upstream's</span>}</td>
                  <td className="nowrap"><StatusBadge status={e.status} /></td>
                  <td className="wide-cell"><Cut text={e.reason} /></td>
                  <td>
                    <span className="cut" title={e.needles.map((n) => `${n.text}: ${place(n)}`).join("; ")}>
                      {e.needles.map((n, i) => <span key={i}>{i > 0 && "; "}<a href={href(["texts", n.text])}>{n.text}</a> {place(n)}</span>)}
                    </span>
                  </td>
                  <td className="nowrap">{e.erased ? plural(e.erased.length, "passage") : <span className="aside">no</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      {ticked.length > 0 && (
        <div className="selection-bar" role="region" aria-label="The ticked requests">
          <strong className="cue">{plural(ticked.length, "request")} ticked</strong>
          <Button variant="primary" onClick={() => { setAnswer(null); setDeciding("admitted"); }}>Admit</Button>
          <Button variant="quiet" onClick={() => { setAnswer(null); setDeciding("refused"); }}>Refuse</Button>
          <Button variant="quiet" onClick={() => setTicked([])}>Untick all</Button>
        </div>
      )}
      <Dialog open={asking} onClose={() => setAsking(false)} title="Ask for an erasure">
        {asking && <div className="wide"><AskErasure onDone={(words) => { setAsking(false); setSaid(words); load(); }} /></div>}
      </Dialog>
      <Dialog open={shownId !== null} onClose={() => { window.location.hash = href(["erasures"]); }} title={`What ${shownId ?? "the request"} asks to erase`}>
        {shownId !== null && <div className="wide"><ErasureView id={shownId} /></div>}
      </Dialog>
      <Dialog open={deciding !== null} onClose={() => setDeciding(null)} title={`${deciding === "admitted" ? "Admit" : "Refuse"} ${plural(ticked.length, "erasure request")}`}>
        {deciding && rows?.ok && (
          <DecideRequests status={deciding} requests={rows.data.filter((e) => ticked.includes(e.id))} onDone={(a) => done(deciding, a)} />
        )}
        {deciding && answer && !answer.answer.ok && <Problems problems={answer.answer.problems} />}
      </Dialog>
    </>
  );
}

// SQL text of a string, for a condition over the texts table.
const quoted = (value: string) => `'${value.replace(/'/g, "''")}'`;

// The first step of a request asked from this page: the text, found by a
// part of its title or id, or named by its exact id; then the form that
// asks, on that text.
function AskErasure({ onDone }: { onDone: (words: ReactNode) => void }) {
  const [words, setWords] = useState("");
  const [found, setFound] = useState<Answer<{ total: number; rows: Row[] }> | null>(null);
  const [chosen, setChosen] = useState<Answer<{ text: Erasable & { title: string | null }; upstream: boolean }> | null>(null);
  useEffect(() => {
    const typed = words.trim();
    if (!typed) { setFound(null); return; }
    let live = true;
    const wait = window.setTimeout(() => {
      const where = `contains(lower(coalesce(title, '')), lower(${quoted(typed)})) OR contains(id, ${quoted(typed)})`;
      search({ ...EVERY, where }, 0, 8).then((a) => live && setFound(a));
    }, 250);
    return () => { live = false; window.clearTimeout(wait); };
  }, [words]);
  const open = async (id: string) => setChosen(await get<{ text: Erasable & { title: string | null }; upstream: boolean }>(`/api/texts/${encodeURIComponent(id)}`));
  if (chosen?.ok) {
    const row = chosen.data.text;
    return (
      <>
        <div className="chosen-text row tight">
          <span>On <a href={href(["texts", row.id])} target="_blank" rel="noreferrer">{row.title ?? (row.kind ? `an item of kind ${row.kind}` : "an untitled text")}</a> <span className="ledger aside">{row.id}</span>{row.kind === null && <span className="aside">, {plural(codePoints(row.text ?? ""), "character")}</span>}</span>
          <Button variant="quiet" onClick={() => setChosen(null)}>Choose another text</Button>
        </div>
        {chosen.data.upstream
          ? <p className="small">It belongs to the upstream this depot follows: ask the upstream to erase it, and a pull carries out here what the upstream erased.</p>
          : <EraseForm key={row.id} row={row} onDone={onDone} />}
      </>
    );
  }
  const typed = words.trim();
  return (
    <div className="action-form form-narrow">
      <RequiredNote />
      <div className="entry-row">
        <TextInput label={required("The text: a part of its title or id")} required value={words} onChange={(e) => setWords(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && typed) open(typed); }} />
        <Button variant="quiet" disabled={!typed} onClick={() => open(typed)}>Open this id</Button>
      </div>
      <p className="form-note small">Matching texts are listed as you type; Enter, or "Open this id", opens the text with exactly this id.</p>
      {chosen && !chosen.ok && <Problems problems={chosen.problems} />}
      {found && !found.ok && <Problems problems={found.problems} />}
      {found?.ok && (found.data.total === 0 ? <p className="small">No text's title or id holds these words.</p> : (
        <>
          <p className="small aside">{plural(found.data.total, "text")} match{found.data.total === 1 ? "es" : ""}{found.data.total > found.data.rows.length ? `; the first ${found.data.rows.length} by id` : ""}.</p>
          <ul className="plain picks">
            {found.data.rows.map((r) => (
              <li key={r.id}>
                <button type="button" className="pick wic-focus" onClick={() => open(r.id)}>
                  <span className="cut">{r.title ?? (r.kind ? `an item of kind ${r.kind}` : "untitled text")}</span>
                  <span className="ledger aside">{r.id}</span>
                  <span className="aside">{r.kind ?? plural(r.length, "character")}</span>
                </button>
              </li>
            ))}
          </ul>
        </>
      ))}
    </div>
  );
}

function DecideRequests({ status, requests, onDone }: { status: Choice; requests: Erasure[]; onDone: (a: Answer<Decided>) => void }) {
  const [reason, setReason] = useState("");
  const send = async (declaration: Declaration) => onDone(await post<Decided>("/api/erasures/decisions", {
    erasures: requests.map((e) => e.id), status, ...(reason.trim() ? { reason: reason.trim() } : {}), declaration,
  }));
  return (
    <div className="action-form">
      <ul className="plain small form-part">
        {requests.map((e) => <li key={e.id}><span className="ledger">{e.id}</span> <Cut text={e.reason} /> <span className="aside">{plural(e.needles.length, "passage")}</span></li>)}
      </ul>
      <TextInput label={optional("Reason")} hint="Public, with the decision." value={reason} onChange={(e) => setReason(e.target.value)} />
      <Confirm label={status === "admitted" ? "Confirm and admit" : "Confirm and refuse"} onConfirm={send}
        note={status === "admitted"
          ? <><strong>This cannot be undone.</strong> Admitting carries each request out at once: every place where its passages appear in their texts is erased, and every older state of the depot is dropped.</>
          : <>Refuses the {plural(requests.length, "request")}; their texts stay as they are.</>} />
    </div>
  );
}

export const erasures: Page = { path: "erasures", title: "Erasures", group: "data", Component: ({ query }) => <Erasures query={query} /> };
