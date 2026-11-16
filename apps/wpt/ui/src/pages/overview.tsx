// What needs the maintainer: the chain of the depot (texts, libraries,
// datasets, tokenizer recipes, jobs waiting or running, with their counts,
// then the corpora, which the box leads to without counting them: they are
// listed per dataset, from the store), the number of texts under review (the depot's
// own counts, GET /api/counts), those texts counted by tag and by license (a
// count made in steps, which grows as it goes, counts.tsx), each tag and
// license linked to its view and each count to those texts, the latest decisions, the open erasure requests and the problems
// reported on libraries, each linked to its view.

import { Notice, Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, waiting as queued, type Answer } from "../api";
import { CountingNote, matched, useCounts } from "../counts";
import { declared, useDeclarant } from "../declarant";
import { EVERY, type TextFilter } from "../filter";
import { Cut, LibraryLink, PageHead, plural, Problems, Section, useDatasets, useLibraries, type Status } from "../parts";
import { href } from "../route";
import { DecisionTable, type Latest } from "./decisions";
import type { Page } from "./index";
import { MARK_SECONDS, useJobs } from "./jobs";

type Needle = { text: string };
type Erasure = { id: string; reason: string; status: Status; needles: Needle[] };
type DepotCounts = { texts: number; statuses: Record<Status, number> };

const REVIEW = "under review";
const WAITING: TextFilter = { ...EVERY, statuses: [REVIEW] };

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
  const waiting = useCounts(WAITING); // by tag and by license
  const [counts, setCounts] = useState<Answer<DepotCounts> | null>(null);
  const [latest, setLatest] = useState<Answer<Latest[]> | null>(null);
  const [erasures, setErasures] = useState<Answer<Erasure[]> | null>(null);
  const [recipes, setRecipes] = useState<Answer<{ name: string; trained?: number | boolean }[]> | null>(null);
  const only = datasets?.ok && datasets.data.length === 1 ? datasets.data[0] : null; // the corpora box leads to its corpora
  const jobs = useJobs(MARK_SECONDS);
  const known = counts?.ok ? counts.data : null;
  useEffect(() => {
    get<DepotCounts>("/api/counts").then(setCounts);
    get<Latest[]>("/api/decisions/latest").then(setLatest);
    get<Erasure[]>("/api/erasures").then(setErasures);
    get<{ name: string; trained?: number | boolean }[]>("/api/tokenizers").then(setRecipes);
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
              <span className="cue">{count(known?.texts, "text")}</span>
              <span className="quiet">{known ? `${known.statuses.admitted.toLocaleString("en")} admitted, ${known.statuses[REVIEW].toLocaleString("en")} under review, ${known.statuses.refused.toLocaleString("en")} refused` : "added, then admitted"}</span>
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
              <span className="quiet">read libraries</span>
            </a>
          </li>
          <li>
            <a className="chain-step" href={href(["tokenizers"])}>
              <span className="cue">{count(recipes?.ok ? recipes.data.length : undefined, "tokenizer recipe")}</span>
              <span className="quiet">{recipes?.ok ? `${recipes.data.filter((r) => r.trained).length.toLocaleString("en")} with a vocabulary` : "train vocabularies"}</span>
            </a>
          </li>
          <li>
            <a className="chain-step" href={href(["jobs"])}>
              <span className="cue">{jobs?.ok ? plural(jobs.data.filter((j) => queued(j.state)).length, "job") : "jobs"}</span>
              <span className="quiet">waiting or running</span>
            </a>
          </li>
          <li>
            <a className="chain-step" href={only ? href(["download", only.id]) : href(["datasets"])}>
              <span className="cue">corpora</span>
              <span className="quiet">{only ? `built from ${only.name}` : "built from each dataset"}</span>
            </a>
          </li>
        </ol>
      </nav>
      <div className="board">
        <Section className="tall" title={<>Under review{known && <>: <a href={href(["texts"], { status: REVIEW })}>{plural(known.statuses[REVIEW], "text")}</a></>}</>}>
          {counts && !counts.ok && <Problems problems={counts.problems} />}
          {waiting && !waiting.done && <p className="small"><CountingNote counting={waiting} /></p>}
          {waiting?.counted === 0 && !waiting.done && <Skeleton lines={4} />}
          {waiting?.problems && <Problems problems={waiting.problems} />}
          {known?.statuses[REVIEW] === 0 && <p className="small">Nothing waits for a decision.</p>}
          {waiting && matched(waiting) > 0 && (
            <div className="two-lists">
              <Counted title="by tag" more={href(["tags"])} rows={waiting.tags.map((t) => ({ key: t.tag, n: t.texts, view: href(["tags", t.tag]), link: href(["texts"], { status: REVIEW, tag: t.tag }) }))} />
              <Counted title="by license" ledger more={href(["licenses"])} rows={waiting.licenses.map((l) => ({ key: l.license, n: l.texts, view: href(["licenses", l.license]), link: href(["texts"], { status: REVIEW, license: l.license }) }))} />
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

// Counts, each name linked to its view and each count to the texts it counts;
// past the first ones, "N more" links to the page that lists them all (more).
function Counted({ title, rows, ledger, more }: { title: string; rows: { key: string; n: number; view: string; link: string }[]; ledger?: boolean; more: string }) {
  return (
    <div>
      <h3 className="aside">{title}</h3>
      <ul className="plain counted small">
        {rows.slice(0, 15).map((r) => (
          <li key={r.key}><a href={r.view} className={ledger ? "ledger cut" : "cut"} title={r.key}>{r.key}</a><a href={r.link} className="number" title={`The ${plural(r.n, "text")} under review`}>{r.n.toLocaleString("en")}</a></li>
        ))}
        {rows.length > 15 && <li className="aside"><a href={more}>{rows.length - 15} more</a></li>}
      </ul>
    </div>
  );
}

export const overview: Page = { path: "", title: "Overview", group: null, Component: Overview };
