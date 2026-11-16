// The texts of a decision's review that the decision cannot cover, set
// apart from the rest with the depot's reason for each (POST
// /api/texts/blocked, the same check a decision applies, a page at a time
// over the selection): the text belongs to the upstream this depot follows
// (any decision), or none of its licenses is allowed (an admission). The
// decision leaves them out. From the list a person opens a text in a new
// tab, keeping the review; a text of this depot's own with no allowed
// license can be ticked, then refused or given other licenses (the review
// then checks again). A text of the upstream has no tick: only the upstream
// decides on it.

import { Button } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { post, type Answer } from "./api";
import { DecideForm, RelicenseForm, relicensed, type Decided, type Relicensed } from "./decide-forms";
import { EVERY, named, search, sent, type Found, type Place, type Selection } from "./filter";
import { Cut, Outcome, PlacePager, plural, Problems, StatusBadge, textName } from "./parts";
import { href } from "./route";

export type Blocked = { id: string; reasons: string[] };
export const NO_ALLOWED_LICENSE = "no allowed license";
export const BELONGS_TO_UPSTREAM = "belongs to the upstream";
// a decision's status: "under review" puts texts back under review, undoing their admission or refusal
export type Deciding = "admitted" | "refused" | "under review";
type Page = { rows: Blocked[]; next: Place | null };

const LIMIT = 25; // blocked texts a page shows

// How many texts of a count a decision of that status cannot cover (counts.tsx's blocked).
export const blockedCount = (blocked: Record<string, number>, status: Deciding) =>
  (blocked[BELONGS_TO_UPSTREAM] ?? 0) + (status === "admitted" ? blocked[NO_ALLOWED_LICENSE] ?? 0 : 0);

type Acting = { act: "refuse" } | { act: "relicense" };

// selection: the review's; total: how many of its texts are blocked, as
// counted; version: raised after a change, so the list is read again;
// onChanged: after a refusal or a license change made here, with what it did
// in words.
export function BlockedTexts({ selection, status, total, version, onChanged }: {
  selection: Selection; status: Deciding; total: number; version: number; onChanged: (words: string) => void;
}) {
  const [places, setPlaces] = useState<(Place | null)[]>([null]); // where each page read so far starts after
  const [page, setPage] = useState<Answer<Page> | null>(null);
  const [rows, setRows] = useState<Answer<Found> | null>(null);
  const [ticked, setTicked] = useState<Set<string>>(new Set());
  const [acting, setActing] = useState<Acting | null>(null);
  const [said, setSaid] = useState<Answer<unknown> | null>(null);
  const [words, setWords] = useState("");
  const after = places[places.length - 1];
  const key = JSON.stringify([sent(selection), status, after, version]);
  useEffect(() => {
    const [s, st, a] = JSON.parse(key);
    post<Page>("/api/texts/blocked", { selection: s, status: st, limit: LIMIT, ...(a ? { after: a } : {}) }).then(setPage);
  }, [key]);
  const found = page?.ok ? page.data.rows : [];
  const shownKey = JSON.stringify(found.map((b) => b.id));
  useEffect(() => {
    const shown: string[] = JSON.parse(shownKey);
    if (shown.length) search({ ...EVERY, ids: shown }, {}, shown.length).then(setRows);
  }, [shownKey]);
  const credits = new Map(rows?.ok ? rows.data.rows.map((r) => [r.id, r]) : []);
  // only a text of this depot's own can be refused or given other licenses here
  const own = found.filter((b) => !b.reasons.includes(BELONGS_TO_UPSTREAM)).map((b) => b.id);
  const chosen = own.filter((id) => ticked.has(id));
  const decision = status === "admitted" ? "admission" : status === "refused" ? "refusal" : "putting back under review";
  const flip = (id: string, on: boolean) => {
    const next = new Set(ticked);
    if (on) next.add(id); else next.delete(id);
    setTicked(next);
  };
  const done = (a: Answer<unknown>, w: string) => {
    setSaid(a);
    setWords(w);
    if (a.ok) { setActing(null); setTicked(new Set()); setPlaces([null]); onChanged(w); }
  };
  const decided = (a: Answer<Decided>) => done(a, a.ok ? `Decision ${a.data.decision}: refused, on ${plural(a.data.texts, "text")}.` : "");
  const changed = (a: Answer<Relicensed>, licenses: string[]) => done(a, a.ok ? relicensed(a.data, licenses) : "");
  return (
    <section className="blocked" aria-labelledby="blocked-title">
      <h3 id="blocked-title" className="cue">{plural(total, "text")} cannot be {status === "under review" ? "put back under review" : status}</h3>
      <p className="small">
        The {decision} leaves them out: only the upstream this depot follows decides on its own texts, and a text that
        names no allowed license can get other licenses, its decisions then starting over; with an allowed license it joins the admission.
      </p>
      {page && !page.ok && <Problems problems={page.problems} />}
      <div className="table-scroll">
        <table className="ledger-table" aria-label="The texts that cannot be admitted">
          <thead>
            <tr>
              <th scope="col" className="tick-cell"><span className="visually-hidden">ticked</span></th>
              <th scope="col">id</th><th scope="col">title</th><th scope="col">status</th><th scope="col">licenses</th><th scope="col">why</th>
            </tr>
          </thead>
          <tbody>
            {found.map((b) => {
              const r = credits.get(b.id);
              return (
                <tr key={b.id} className={ticked.has(b.id) ? "ticked" : undefined}>
                  <td className="tick-cell">
                    {!b.reasons.includes(BELONGS_TO_UPSTREAM) && <input type="checkbox" className="wic-check-box wic-focus" checked={ticked.has(b.id)}
                      aria-label={`Tick ${b.id}`} onChange={(e) => flip(b.id, e.target.checked)} />}
                  </td>
                  <td className="ledger nowrap">{b.id}</td>
                  <td className="wide-cell">
                    <a href={href(["texts", b.id])} target="_blank" rel="noreferrer" className="cut" title={`${r?.title ?? b.id} (opens in a new tab)`}>{r ? textName(r) : b.id}</a>
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
      <PlacePager label="Pages of the texts that cannot be admitted" limit={LIMIT}
        onFirst={() => setPlaces([null])} onPrevious={places.length > 1 ? () => setPlaces(places.slice(0, -1)) : null}
        onNext={page?.ok && page.data.next ? () => setPlaces([...places, page.data.next]) : null} />
      {own.length > 0 && <div className="row tight blocked-tools">
        {chosen.length < own.length
          ? <Button variant="quiet" onClick={() => setTicked(new Set(own))}>Tick the {plural(own.length, "text")} of this page</Button>
          : <Button variant="quiet" onClick={() => setTicked(new Set())}>Untick all</Button>}
        <Button variant="ink" disabled={chosen.length === 0} onClick={() => { setSaid(null); setActing({ act: "refuse" }); }}>Refuse {plural(chosen.length, "ticked text")}</Button>
        <Button variant="quiet" disabled={chosen.length === 0} onClick={() => { setSaid(null); setActing({ act: "relicense" }); }}>Change their licenses</Button>
      </div>}
      {!acting && <Outcome answer={said} done={() => words} />}
      {acting?.act === "refuse" && (
        <div className="blocked-form">
          <DecideForm status="refused" selection={named(chosen)} size={chosen.length} onDone={decided} variant="ink" />
          <Button variant="quiet" onClick={() => setActing(null)}>Cancel the refusal</Button>
        </div>
      )}
      {acting?.act === "relicense" && (
        <div className="blocked-form">
          <RelicenseForm selection={named(chosen)} subject={`the ${plural(chosen.length, "ticked text")} with no allowed license`} onDone={changed} variant="ink" />
          <Button variant="quiet" onClick={() => setActing(null)}>Cancel the license change</Button>
        </div>
      )}
      {acting && said && !said.ok && <Problems problems={said.problems} />}
    </section>
  );
}
