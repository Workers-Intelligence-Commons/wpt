// Decisions: #/decisions/ID is one decision's view (GET /api/decisions/ID):
// its status, the declared name and email, its time and its reason, and what
// it concerns, each linked: the texts it is on, the erasure requests it
// decides, or the problem it records with its library. A decision that put
// texts back under review shows as a correction of the decisions before it.
// #/decisions lists the latest ones. The page is not in the navigation: lists link to it.

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
export type DecisionShown = {
  decision: DecisionRow; texts: number; erasures: string[];
  problem: { library: string; texts: string[]; problem: string } | null;
};

export const when = (t: string) => t.replace("T", " ").slice(0, 16);

// A decision in words, for its title: what it did and to how much, and who decided ("Admitted 20 texts, by Ada Lovelace").
function headline({ decision: d, texts, erasures, problem }: DecisionShown): string {
  const verb = d.status === "admitted" ? "Admitted" : d.status === "refused" ? "Refused" : d.put_back ? "Put back under review" : "Put under review";
  const what = problem ? "Reported a problem" : erasures.length ? `${verb} ${plural(erasures.length, "erasure request")}` : `${verb} ${plural(texts, "text")}`;
  return `${what}, by ${d.name}`;
}

function OneDecision({ id }: { id: string }) {
  const [found, setFound] = useState<Answer<DecisionShown> | null>(null);
  useEffect(() => { get<DecisionShown>(`/api/decisions/${part(id)}`).then(setFound); }, [id]);
  if (!found) return <Skeleton lines={6} label="Reading the decision" />;
  if (!found.ok) return <Problems problems={found.problems} />;
  const { decision: d, texts, erasures, problem } = found.data;
  return (
    <>
      <PageHead title={headline(found.data)}
        meta={<><StatusBadge status={d.status} /><Correction d={d} /><time dateTime={d.made_at} className="ledger">{when(d.made_at)} UTC</time><span className="ledger aside" title="The decision's id, which the command line and the other pages name it by">{d.id}</span></>}
        chain={<>A decision records who decided, under their declaration, on <a href={href(["texts"])}>texts</a>, an <a href={href(["erasures"])}>erasure request</a> or a problem reported on a <a href={href(["libraries"])}>library</a>. Its name and email are public and nobody verifies them.</>} />
      <Facts rows={[
        ["by", d.name],
        ["email", <span className="ledger">{d.email}</span>],
        ["when (UTC)", <span className="ledger">{when(d.made_at)}</span>],
        ["reason", d.reason ?? <span className="aside">none</span>],
      ]} />
      {problem && (
        <Section title="The problem it records">
          <p><a href={href(["libraries", problem.library, "problems", d.id])}>{problem.problem}</a></p>
          <p className="small">Reported on the library <a href={href(["libraries", problem.library])}>{problem.library}</a>, {problem.texts.length ? plural(problem.texts.length, "text") : "as a whole"}.</p>
        </Section>
      )}
      {erasures.length > 0 && (
        <Section title="Erasure requests it decides">
          <ul className="plain">
            {erasures.map((e) => <li key={e}><a className="ledger" href={href(["erasures"], { request: e })}>{e}</a></li>)}
          </ul>
        </Section>
      )}
      {(texts > 0 || (!problem && erasures.length === 0)) && (
        <Section title={`Texts it is on: ${texts.toLocaleString("en")}`}
          tools={texts > 0 ? <a className="small" href={href(["texts"], { decision: d.id })}>Open them in the texts</a> : undefined}>
          <MatchedTexts filter={{ ...EVERY, decision: d.id }} label="The texts it is on"
            empty="It is on no text the depot still holds." />
        </Section>
      )}
    </>
  );
}

export type Latest = DecisionRow & { texts: number; erasures: number; library: string | null };

function LatestDecisions() {
  const [latest, setLatest] = useState<Answer<Latest[]> | null>(null);
  useEffect(() => { get<Latest[]>("/api/decisions/latest").then(setLatest); }, []);
  return (
    <>
      <PageHead title="Latest decisions" chain={<>The ten latest decisions on <a href={href(["texts"])}>texts</a>, <a href={href(["erasures"])}>erasure requests</a> and problems reported on <a href={href(["libraries"])}>libraries</a>.</>} />
      {!latest && <Skeleton lines={4} />}
      {latest && !latest.ok && <Problems problems={latest.problems} />}
      {latest?.ok && (latest.data.length === 0 ? <p>No decision yet.</p> : <DecisionTable rows={latest.data} />)}
    </>
  );
}

// Decisions as a table, each linked to its view.
export function DecisionTable({ rows, libraryName }: { rows: Latest[]; libraryName?: (id: string) => string }) {
  return (
    <div className="table-scroll">
      <table className="ledger-table" aria-label="The latest decisions">
        <thead><tr><th scope="col">what was decided</th><th scope="col">status</th><th scope="col">by</th><th scope="col">when (UTC)</th><th scope="col">reason</th></tr></thead>
        <tbody>
          {rows.map((d) => (
            <tr key={d.id}>
              <td className="nowrap"><a href={href(["decisions", d.id])} title={`decision ${d.id}`}>{d.library ? <>a problem of {libraryName?.(d.library) ?? d.library}</> : d.erasures ? plural(d.erasures, "erasure request") : plural(d.texts, "text")}</a></td>
              <td className="nowrap"><StatusBadge status={d.status} /> <Correction d={d} /></td>
              <td><Cut text={d.name} /></td>
              <td className="ledger nowrap">{when(d.made_at)}</td>
              <td className="wide-cell">{d.reason ? <Cut text={d.reason} /> : <span className="aside">none</span>}</td>
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
