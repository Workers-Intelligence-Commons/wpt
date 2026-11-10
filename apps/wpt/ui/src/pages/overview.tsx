// What needs the maintainer: the chain of the depot (texts, libraries,
// datasets) with their counts, the texts under review counted by tag and by
// license, each tag and license linked to its view and each count to those
// texts, the latest decisions, the open erasure requests and the problems
// reported on libraries, each linked to its view.

import { Notice, Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, type Answer } from "../api";
import { declared, useDeclarant } from "../declarant";
import { EVERY, search, type Found } from "../filter";
import { Cut, LibraryLink, PageHead, plural, Problems, Section, useDatasets, useLibraries, type Status } from "../parts";
import { href } from "../route";
import { DecisionTable, type Latest } from "./decisions";
import type { Page } from "./index";

type Needle = { text: string };
type Erasure = { id: string; reason: string; status: Status; needles: Needle[] };

const REVIEW = "under review";

// The problem a later decision records first: a depot numbers its decisions
// in the order it makes them, so the larger number after the last hyphen is
// the later report; equal numbers fall back to the depot's name.
function newestFirst(a: string, b: string): number {
  const split = (id: string) => { const m = /^(.*)-(\d+)$/.exec(id); return m ? [m[1], Number(m[2])] as const : [id, -1] as const; };
  const [an, ax] = split(a);
  const [bn, bx] = split(b);
  return bx - ax || bn.localeCompare(an);
}

function Overview() {
  const who = useDeclarant();
  const { libraries, loaded, problems: libraryProblems } = useLibraries();
  const { datasets } = useDatasets();
  const [all, setAll] = useState<Answer<Found> | null>(null);
  const [waiting, setWaiting] = useState<Answer<Found> | null>(null);
  const [latest, setLatest] = useState<Answer<Latest[]> | null>(null);
  const [erasures, setErasures] = useState<Answer<Erasure[]> | null>(null);
  useEffect(() => {
    search(EVERY, 0, 1).then(setAll);
    search({ ...EVERY, statuses: [REVIEW] }, 0, 1).then(setWaiting);
    get<Latest[]>("/api/decisions/latest").then(setLatest);
    get<Erasure[]>("/api/erasures").then(setErasures);
  }, []);
  const open = erasures?.ok ? erasures.data.filter((e) => e.status === REVIEW) : [];
  const reported = libraries.flatMap((l) => (l.problems ?? []).map((p) => ({ library: l, problem: p })))
    .sort((a, b) => newestFirst(a.problem.decision, b.problem.decision));
  const byName = new Map(libraries.map((l) => [l.id, l]));
  const count = (n: number | undefined, one: string, many?: string) => (n === undefined ? one : plural(n, one, many));

  return (
    <>
      <PageHead title="Overview" chain="What waits for a decision in this depot, and where it goes next." />
      {!declared(who) && (
        <Notice title="who decides">No legal name and email are set in this browser, so no decision can be sent. <a href={href(["settings"])}>Set them in the settings</a>.</Notice>
      )}
      <nav aria-label="The chain of the depot">
        <ol className="chain-strip small">
          <li>
            <a className="chain-step" href={href(["texts"])}>
              <span className="cue">{count(all?.ok ? all.data.total : undefined, "text")}</span>
              <span className="quiet">added under review, then admitted</span>
            </a>
          </li>
          <li>
            <a className="chain-step" href={href(["libraries"])}>
              <span className="cue">{count(loaded ? libraries.length : undefined, "library", "libraries")}</span>
              <span className="quiet">hold admitted texts</span>
            </a>
          </li>
          <li>
            <a className="chain-step" href={href(["datasets"])}>
              <span className="cue">{count(datasets?.ok ? datasets.data.length : undefined, "dataset")}</span>
              <span className="quiet">read libraries, build a corpus</span>
            </a>
          </li>
        </ol>
      </nav>
      <div className="board">
        <Section className="tall" title={<>Under review{waiting?.ok && <>: <a href={href(["texts"], { status: REVIEW })}>{plural(waiting.data.total, "text")}</a></>}</>}>
          {!waiting && <Skeleton lines={4} />}
          {waiting && !waiting.ok && <Problems problems={waiting.problems} />}
          {waiting?.ok && waiting.data.total === 0 && <p className="small">Nothing waits for a decision.</p>}
          {waiting?.ok && waiting.data.total > 0 && (
            <div className="two-lists">
              <Counted title="by tag" rows={waiting.data.tags.map((t) => ({ key: t.tag, n: t.texts, view: href(["tags", t.tag]), link: href(["texts"], { status: REVIEW, tag: t.tag }) }))} />
              <Counted title="by license" ledger rows={waiting.data.licenses.map((l) => ({ key: l.license, n: l.texts, view: href(["licenses", l.license]), link: href(["texts"], { status: REVIEW, license: l.license }) }))} />
            </div>
          )}
        </Section>

        <Section title={<>Open erasure requests{erasures?.ok && `: ${open.length}`}</>} tools={<a className="small" href={href(["erasures"])}>Decide on the requests</a>}>
          {!erasures && <Skeleton lines={2} />}
          {erasures && !erasures.ok && <Problems problems={erasures.problems} />}
          {erasures?.ok && (open.length === 0 ? <p className="small">No request is open.</p> : (
            <ul className="plain small">
              {open.slice(0, 8).map((e) => <li key={e.id}><a className="ledger" href={href(["erasures"], { request: e.id })} title={`What ${e.id} asks to erase`}>{e.id}</a> <Cut text={e.reason} /> <span className="aside">{plural(e.needles.length, "passage")}</span></li>)}
            </ul>
          ))}
        </Section>

        <Section title={<>Problems reported on libraries{loaded && libraryProblems.length === 0 && `: ${reported.length}`}</>} tools={<a className="small" href={href(["libraries"])}>Decide in the libraries</a>}>
          {!loaded && <Skeleton lines={2} />}
          {loaded && libraryProblems.length > 0 && <Problems problems={libraryProblems} />}
          {loaded && libraryProblems.length === 0 && (reported.length === 0 ? <p className="small">No problem is reported.</p> : (
            <ul className="plain small">
              {reported.slice(0, 8).map(({ library, problem }) => (
                <li key={problem.decision}><span><LibraryLink library={library} />:</span> <a href={href(["libraries", library.id, "problems", problem.decision])} className="cut" title={problem.problem}>{problem.problem}</a> <span className="aside">{problem.texts.length ? plural(problem.texts.length, "text") : "the whole library"}</span></li>
              ))}
              {reported.length > 8 && <li className="aside"><a href={href(["libraries"])}>{reported.length - 8} more in the libraries</a></li>}
            </ul>
          ))}
        </Section>

        <Section className="span-2" title="Latest decisions">
          {!latest && <Skeleton lines={4} />}
          {latest && !latest.ok && <Problems problems={latest.problems} />}
          {latest?.ok && (latest.data.length === 0 ? <p className="small">No decision yet.</p>
            : <DecisionTable rows={latest.data} libraryName={(id) => byName.get(id)?.name ?? id} />)}
        </Section>
      </div>
    </>
  );
}

// Counts, each name linked to its view and each count to the texts it counts.
function Counted({ title, rows, ledger }: { title: string; rows: { key: string; n: number; view: string; link: string }[]; ledger?: boolean }) {
  return (
    <div>
      <h3 className="aside">{title}</h3>
      <ul className="plain counted small">
        {rows.slice(0, 15).map((r) => (
          <li key={r.key}><a href={r.view} className={ledger ? "ledger cut" : "cut"} title={r.key}>{r.key}</a><a href={r.link} className="number" title={`The ${plural(r.n, "text")} under review`}>{r.n.toLocaleString("en")}</a></li>
        ))}
        {rows.length > 15 && <li className="aside">{rows.length - 15} more</li>}
      </ul>
    </div>
  );
}

export const overview: Page = { path: "", title: "Overview", group: null, Component: Overview };
