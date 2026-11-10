// The texts of a decision's review that the decision cannot cover, set
// apart from the rest with the depot's reason for each (POST
// /api/texts/blocked, the same check a decision applies): the text belongs
// to the upstream this depot follows (any decision), or none of its
// licenses is allowed (an admission). They are never part of the decision.
// From the list a person opens a text in a new tab, keeping the review; a
// text of this depot's own with no allowed license can be ticked, then
// refused or given other licenses (the review then checks it again). A text
// of the upstream has no tick: only the upstream decides on it.

import { Button } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { post, type Answer } from "./api";
import { DecideForm, RelicenseForm, relicensed, type Decided, type Relicensed } from "./decide-forms";
import { EVERY, search, type Found } from "./filter";
import { Cut, Outcome, Pager, plural, Problems, StatusBadge } from "./parts";
import { href } from "./route";

export type Blocked = { id: string; reasons: string[] };
export const NO_ALLOWED_LICENSE = "no allowed license";
export const BELONGS_TO_UPSTREAM = "belongs to the upstream";
export type Deciding = "admitted" | "refused";

// The texts among ids that a decision of that status cannot cover, by the depot's check.
export const blocked = (ids: string[], status: Deciding) => post<{ texts: Blocked[] }>("/api/texts/blocked", { texts: ids, status });

type Acting = { act: "refuse" } | { act: "relicense" };

// found: the blocked texts; onDecided: the ids a decision made here took,
// after it was recorded; onRelicensed: after their licenses changed; both
// get what happened in words, for the review to say when no blocked text is
// left to show it here.
export function BlockedTexts({ found, status, onDecided, onRelicensed }: {
  found: Blocked[]; status: Deciding; onDecided: (ids: string[], words: string) => void; onRelicensed: (words: string) => void;
}) {
  const [wanted, setOffset] = useState(0);
  const [limit, setLimit] = useState(25);
  const [page, setPage] = useState<Answer<Found> | null>(null);
  const [ticked, setTicked] = useState<Set<string>>(new Set());
  const [acting, setActing] = useState<Acting | null>(null);
  const [said, setSaid] = useState<Answer<unknown> | null>(null);
  const [words, setWords] = useState("");
  // A refusal made here shortens found, so the page asked for may now lie
  // past the end: the last page that still holds texts is shown instead.
  const offset = Math.min(wanted, Math.max(0, (Math.ceil(found.length / limit) - 1) * limit));
  const slice = found.slice(offset, offset + limit);
  const sliceKey = JSON.stringify(slice.map((b) => b.id));
  useEffect(() => {
    const shown: string[] = JSON.parse(sliceKey);
    if (shown.length) search({ ...EVERY, ids: shown }, 0, shown.length).then(setPage);
  }, [sliceKey]);
  const rows = new Map(page?.ok ? page.data.rows.map((r) => [r.id, r]) : []);
  // only a text of this depot's own can be refused or given other licenses here
  const every = found.filter((b) => !b.reasons.includes(BELONGS_TO_UPSTREAM)).map((b) => b.id);
  const theirs = found.length - every.length;
  const decision = status === "admitted" ? "admission" : "refusal";
  const chosen = every.filter((id) => ticked.has(id));
  const flip = (id: string, on: boolean) => {
    const next = new Set(ticked);
    if (on) next.add(id); else next.delete(id);
    setTicked(next);
  };
  const decided = (a: Answer<Decided>) => {
    setSaid(a);
    const w = a.ok ? `Decision ${a.data.decision}: refused, on ${plural(a.data.texts.length, "text")}.` : "";
    setWords(w);
    if (a.ok) { setActing(null); setTicked(new Set()); onDecided(a.data.texts, w); }
  };
  const changed = (a: Answer<Relicensed>, licenses: string[]) => {
    setSaid(a);
    const w = a.ok ? relicensed(a.data, licenses) : "";
    setWords(w);
    if (a.ok) { setActing(null); setTicked(new Set()); onRelicensed(w); }
  };
  return (
    <section className="blocked" aria-labelledby="blocked-title">
      <h3 id="blocked-title" className="cue">{plural(found.length, "text")} cannot be {status}</h3>
      <p className="small">
        They are left out of the {decision}, since the depot refuses the whole {decision} when it names one of them.
        {theirs > 0 && <> {plural(theirs, "text")} {theirs === 1 ? "belongs" : "belong"} to the upstream this depot follows: only the upstream decides on {theirs === 1 ? "it" : "them"}, and a pull brings its decisions here.</>}
        {every.length > 0 && <> {plural(every.length, "text")} {every.length === 1 ? "names" : "name"} no allowed license; such a text can get other licenses, its decisions then start over, and with an allowed license it joins the admission.</>}
      </p>
      <div className="table-scroll">
        <table className="ledger-table" aria-label="The texts that cannot be admitted">
          <thead>
            <tr>
              <th scope="col" className="tick-cell"><span className="visually-hidden">ticked</span></th>
              <th scope="col">id</th><th scope="col">title</th><th scope="col">status</th><th scope="col">licenses</th><th scope="col">why</th>
            </tr>
          </thead>
          <tbody>
            {slice.map((b) => {
              const r = rows.get(b.id);
              return (
                <tr key={b.id} className={ticked.has(b.id) ? "ticked" : undefined}>
                  <td className="tick-cell">
                    {!b.reasons.includes(BELONGS_TO_UPSTREAM) && <input type="checkbox" className="wic-check-box wic-focus" checked={ticked.has(b.id)}
                      aria-label={`Tick ${b.id}`} onChange={(e) => flip(b.id, e.target.checked)} />}
                  </td>
                  <td className="ledger nowrap">{b.id}</td>
                  <td className="wide-cell">
                    <a href={href(["texts", b.id])} target="_blank" rel="noreferrer" className="cut" title={`${r?.title ?? b.id} (opens in a new tab)`}>{r?.title ?? (r?.kind ? `an item of kind ${r.kind}` : "untitled text")}</a>
                  </td>
                  <td className="nowrap">{r && <StatusBadge status={r.status} />}</td>
                  <td className="ledger">{r && <Cut text={r.licenses.join(", ")} />}</td>
                  <td className="why">
                    {b.reasons.includes(BELONGS_TO_UPSTREAM) ? <span className="aside">belongs to the upstream</span>
                      : b.reasons.includes(NO_ALLOWED_LICENSE) && <span className="not-allowed">no allowed license</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {page && !page.ok && <Problems problems={page.problems} />}
      <Pager offset={offset} limit={limit} total={found.length} onOffset={setOffset} onLimit={(n) => { setLimit(n); setOffset(0); }} label="Pages of the texts that cannot be admitted" />
      {every.length > 0 && <div className="row tight blocked-tools">
        {chosen.length < every.length
          ? <Button variant="quiet" onClick={() => setTicked(new Set(every))}>Tick all {plural(every.length, "text")}</Button>
          : <Button variant="quiet" onClick={() => setTicked(new Set())}>Untick all</Button>}
        <Button variant="ink" disabled={chosen.length === 0} onClick={() => { setSaid(null); setActing({ act: "refuse" }); }}>Refuse {plural(chosen.length, "ticked text")}</Button>
        <Button variant="quiet" disabled={chosen.length === 0} onClick={() => { setSaid(null); setActing({ act: "relicense" }); }}>Change their licenses</Button>
      </div>}
      {!acting && <Outcome answer={said} done={() => words} />}
      {acting?.act === "refuse" && (
        <div className="blocked-form">
          <DecideForm status="refused" chosen={chosen} onDone={decided} variant="ink" />
          <Button variant="quiet" onClick={() => setActing(null)}>Cancel the refusal</Button>
        </div>
      )}
      {acting?.act === "relicense" && (
        <div className="blocked-form">
          <RelicenseForm chosen={chosen} subject={`the ${plural(chosen.length, "ticked text")} with no allowed license`} onDone={changed} variant="ink" />
          <Button variant="quiet" onClick={() => setActing(null)}>Cancel the license change</Button>
        </div>
      )}
      {acting && said && !said.ok && <Problems problems={said.problems} />}
    </section>
  );
}
