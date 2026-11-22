// Decisions: #/decisions/ID is one decision's view (GET /api/decisions/ID):
// its status, the declared name and email, its time and its reason, and what
// it concerns, each linked: the texts it is on and the erasure requests it
// decides. A decision that put texts back under review shows as a
// correction of the decisions before it. A decision on no text and no
// erasure request (a problem reported on a library before snapshots took the
// place of libraries, whose words stay in the depot's file unread, or one
// whose texts were all erased since) shows read only, as what it is. #/decisions lists the
// latest ones. The page is not in the navigation: lists link to it.

import { Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, part, type Answer } from "../api";
import { EVERY } from "../filter";
import { Cut, Facts, PageHead, plural, Problems, Section, StatusBadge, type Status } from "../parts";
import { href } from "../route";
import { MatchedTexts } from "../table";
import type { Page } from "./index";

// put_back: it put texts back under review, undoing their admission or refusal
export type DecisionRow = { id: string; status: Status; name: string; email: string; made_at: string; reason: string | null; put_back: boolean };

// A correction's mark beside its status.
export const Correction = ({ d }: { d: { put_back: boolean } }) => (d.put_back ? <span className="aside" title="It undid the admission or refusal of its texts: the decisions before it no longer count">correction: put back under review</span> : null);
export type DecisionShown = { decision: DecisionRow; texts: number; erasures: string[] };

// A decision on nothing the depot holds now: an old problem report, or one whose texts were erased since.
const onNothing = ({ texts, erasures }: { texts: number; erasures: number | string[] }) =>
  texts === 0 && (typeof erasures === "number" ? erasures === 0 : erasures.length === 0);

export const when = (t: string) => t.replace("T", " ").slice(0, 16);

// A decision in words, for its title: what it did and to how much, and who decided ("Admitted 20 texts, by Ada Lovelace").
function headline(shown: DecisionShown): string {
  const { decision: d, texts, erasures } = shown;
  const verb = d.status === "admitted" ? "Admitted" : d.status === "refused" ? "Refused" : d.put_back ? "Put back under review" : "Put under review";
  const what = onNothing(shown) ? "A decision on no text the depot holds" : erasures.length ? `${verb} ${plural(erasures.length, "erasure request")}` : `${verb} ${plural(texts, "text")}`;
  return `${what}, by ${d.name}`;
}

function OneDecision({ id }: { id: string }) {
  const [found, setFound] = useState<Answer<DecisionShown> | null>(null);
  useEffect(() => { get<DecisionShown>(`/api/decisions/${part(id)}`).then(setFound); }, [id]);
  if (!found) return <Skeleton lines={6} label="Reading the decision" />;
  if (!found.ok) return <Problems problems={found.problems} />;
  const { decision: d, texts, erasures } = found.data;
  const nothing = onNothing(found.data);
  return (
    <>
      <PageHead title={headline(found.data)}
        meta={<><StatusBadge status={d.status} /><Correction d={d} /><time dateTime={d.made_at} className="ledger">{when(d.made_at)} UTC</time><span className="ledger aside" title="The decision's id, which the command line and the other pages name it by">{d.id}</span></>}
        chain={<>A decision records who decided, under their declaration, on <a href={href(["texts"])}>texts</a> or an <a href={href(["erasures"])}>erasure request</a>. Its name and email are public and nobody verifies them. A decision never changes once made.</>} />
      <Facts rows={[
        ["by", d.name],
        ["email", <span className="ledger">{d.email}</span>],
        ["when (UTC)", <span className="ledger">{when(d.made_at)}</span>],
        ["reason", d.reason ?? <span className="aside">none</span>],
      ]} />
      {nothing && (
        <Section title="What it concerns">
          <p className="small">It is on no text the depot holds and decides no erasure request. Either it records a problem reported on a library before snapshots took the place of libraries (its words stay in the depot's file, unread by this page), or every text it was on has been erased since. It is kept as it was made, read only, and changes nothing.</p>
        </Section>
      )}
      {erasures.length > 0 && (
        <Section title="Erasure requests it decides">
          <ul className="plain">
            {erasures.map((e) => <li key={e}><a className="ledger" href={href(["erasures"], { request: e })}>{e}</a></li>)}
          </ul>
        </Section>
      )}
      {texts > 0 && (
        <Section title={`Texts it is on: ${texts.toLocaleString("en")}`}
          tools={texts > 0 ? <a className="small" href={href(["texts"], { decision: d.id })}>Open them in the texts</a> : undefined}>
          <MatchedTexts filter={{ ...EVERY, decision: d.id }} label="The texts it is on"
            empty="It is on no text the depot still holds." />
        </Section>
      )}
    </>
  );
}

export type Latest = DecisionRow & { texts: number; erasures: number };

function LatestDecisions() {
  const [latest, setLatest] = useState<Answer<Latest[]> | null>(null);
  useEffect(() => { get<Latest[]>("/api/decisions/latest").then(setLatest); }, []);
  return (
    <>
      <PageHead title="Latest decisions" chain={<>The ten latest decisions on <a href={href(["texts"])}>texts</a>, <a href={href(["erasures"])}>erasure requests</a>.</>} />
      {!latest && <Skeleton lines={4} />}
      {latest && !latest.ok && <Problems problems={latest.problems} />}
      {latest?.ok && (latest.data.length === 0 ? <p>No decision yet.</p> : <DecisionTable rows={latest.data} />)}
    </>
  );
}

// Decisions as a table, each linked to its view.
// A column every row says the same in (the decider, the minute) is said
// once above the table instead; the reasons only when one row has one.
export function DecisionTable({ rows }: { rows: Latest[] }) {
  const same = (key: (d: Latest) => string) => rows.length > 1 && rows.every((d) => key(d) === key(rows[0]));
  const oneName = same((d) => d.name);
  const oneTime = same((d) => when(d.made_at));
  const reasons = rows.some((d) => d.reason);
  return (
    <div className="table-scroll cards">
      <table className="ledger-table" aria-label="The latest decisions">
        {(oneName || oneTime) && <caption className="table-caption small aside">{[oneName ? `by ${rows[0].name}` : null, oneTime ? `${when(rows[0].made_at)} UTC` : null].filter(Boolean).join(" · ")}</caption>}
        <thead><tr><th scope="col">what was decided</th><th scope="col">status</th>{!oneName && <th scope="col">by</th>}{!oneTime && <th scope="col">when (UTC)</th>}{reasons && <th scope="col">reason</th>}</tr></thead>
        <tbody>
          {rows.map((d) => (
            <tr key={d.id}>
              <td className="nowrap" data-label="decided"><a href={href(["decisions", d.id])} title={`decision ${d.id}`}>{onNothing(d) ? <span className="aside">no text now</span> : d.erasures ? plural(d.erasures, "erasure request") : plural(d.texts, "text")}</a></td>
              <td className="nowrap" data-label="status"><StatusBadge status={d.status} /> <Correction d={d} /></td>
              {!oneName && <td data-label="by"><Cut text={d.name} /></td>}
              {!oneTime && <td className="ledger nowrap" data-label="when (UTC)">{when(d.made_at)}</td>}
              {reasons && <td className="wide-cell" data-label="reason">{d.reason ? <Cut text={d.reason} /> : <span className="aside" aria-label="none">—</span>}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export const decisions: Page = {
  path: "decisions", title: "Decisions", group: null, under: "texts",
  Component: ({ rest }) => (rest[0] ? <OneDecision key={rest[0]} id={rest[0]} /> : <LatestDecisions />),
};
