// What needs the maintainer: the texts admitted and the jobs waiting or
// running, each dataset's stages in one line with its next step
// (pipeline.tsx) and the start of a new one, the next step of
// the first dataset that has one the page's main button, the number of texts
// under review (the depot's own counts, GET /api/counts), those texts
// counted by tag and by license (a count made in steps, which grows as it
// goes, counts.tsx), each tag and license linked to its view and each count
// to those texts, the latest decisions, the open erasure requests and the
// problems reported on libraries, each linked to its view (one line when
// there is none of either: then two zero counts beside the others).

import { ButtonLink, Notice, Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, waiting as queued, type Answer } from "../api";
import { CountingNote, matched, useCounts } from "../counts";
import { declared, useDeclarant } from "../declarant";
import { EVERY, type TextFilter } from "../filter";
import { Cut, LibraryLink, PageHead, plural, Problems, Section, settingsHref, Stat, useDatasets, useLibraries, type Status } from "../parts";
import { href } from "../route";
import { Chain, standing, useCorpora, useRecipesAndVocabularies } from "../pipeline";
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
  const { recipes, trained } = useRecipesAndVocabularies();
  const corpora = useCorpora(datasets?.ok ? datasets.data.map((d) => d.id) : []);
  // the one main button of the page: the next step of the first dataset that has one
  const firstNext = datasets?.ok ? datasets.data.find((d) => standing(d, recipes?.ok ? recipes.data : null, trained?.ok ? trained.data : null, corpora.get(d.id)).next)?.id : undefined;
  const jobs = useJobs(MARK_SECONDS);
  const known = counts?.ok ? counts.data : null;
  useEffect(() => {
    get<DepotCounts>("/api/counts").then(setCounts);
    get<Latest[]>("/api/decisions/latest").then(setLatest);
    get<Erasure[]>("/api/erasures").then(setErasures);
  }, []);
  const open = erasures?.ok ? erasures.data.filter((e) => e.status === REVIEW) : [];
  const reported = libraries.flatMap((l) => (l.problems ?? []).map((p) => ({ library: l, problem: p })))
    .sort((a, b) => newestFirst(a.problem.decision, b.problem.decision));
  const byName = new Map(libraries.map((l) => [l.id, l]));
  // nothing open and nothing reported: one line in place of two empty blocks
  const quiet = erasures?.ok && open.length === 0 && loaded && libraryProblems.length === 0 && reported.length === 0;

  return (
    <>
      <PageHead title="Overview" chain="What waits for a decision in this depot, and the chain it goes through: texts, libraries, datasets, tokenizer recipes, jobs, corpora." />
      {!declared(who) && (
        <Notice title="who decides">No name and email set. <a href={settingsHref()}>Set them</a> to decide.</Notice>
      )}
      <div className="stats">
        <Stat href={href(["texts"], { status: "admitted" })} value={known ? known.statuses.admitted.toLocaleString("en") : "…"} label={known ? `admitted of ${plural(known.texts, "text")}` : "admitted"} />
        <Stat href={href(["jobs"])} value={jobs?.ok ? jobs.data.filter((j) => queued(j.state)).length.toLocaleString("en") : "…"} label="jobs open" />
        {quiet && <Stat href={href(["erasures"])} value="0" label="erasures open" />}
        {quiet && <Stat href={href(["libraries"])} value="0" label="library problems" />}
      </div>
      {datasets?.ok && datasets.data.length > 0 && (
        <Section title={<a href={href(["datasets"])}>Datasets</a>} tools={<ButtonLink variant="quiet" href={href(["datasets"], { new: "1" })}>New dataset</ButtonLink>}>
          <ul className="plain pipeline-list">
            {datasets.data.map((d) => {
              const at = standing(d, recipes?.ok ? recipes.data : null, trained?.ok ? trained.data : null, corpora.get(d.id));
              return (
                <li key={d.id} className="pipeline-row">
                  <a className="pipeline-name cue" href={href(["datasets", d.id])}>{d.name}</a>
                  <Chain dataset={d} at={at} action={d.id === firstNext ? "primary" : "quiet"} />
                </li>
              );
            })}
          </ul>
        </Section>
      )}
      <div className="board">
        <Section className={quiet ? "span-2" : "tall"} title={<>Under review{known && <>: <a href={href(["texts"], { status: REVIEW })}>{plural(known.statuses[REVIEW], "text")}</a></>}</>}>
          {counts && !counts.ok && <Problems problems={counts.problems} />}
          {waiting && !waiting.done && <p className="small"><CountingNote counting={waiting} /></p>}
          {waiting?.counted === 0 && !waiting.done && <Skeleton lines={4} />}
          {waiting?.problems && <Problems problems={waiting.problems} />}
          {known?.statuses[REVIEW] === 0 && <p className="small aside">Nothing waits.</p>}
          {waiting && matched(waiting) > 0 && (
            <div className="two-lists">
              <Counted title="by tag" more={href(["tags"])} rows={waiting.tags.map((t) => ({ key: t.tag, n: t.texts, view: href(["tags", t.tag]), link: href(["texts"], { status: REVIEW, tag: t.tag }) }))} />
              <Counted title="by license" ledger more={href(["licenses"])} rows={waiting.licenses.map((l) => ({ key: l.license, n: l.texts, view: href(["licenses", l.license]), link: href(["texts"], { status: REVIEW, license: l.license }) }))} />
            </div>
          )}
        </Section>

        {!quiet && <>
        <Section title={<>Open erasure requests{erasures?.ok && `: ${open.length}`}</>} tools={<a className="small" href={href(["erasures"])}>Erasures</a>}>
            {!erasures && <Skeleton lines={2} />}
            {erasures && !erasures.ok && <Problems problems={erasures.problems} />}
            {erasures?.ok && (open.length === 0 ? <p className="small aside">None.</p> : (
              <ul className="plain small">
                {open.slice(0, 8).map((e) => <li key={e.id}><a className="ledger" href={href(["erasures"], { request: e.id })} title={`What ${e.id} asks to erase`}>{e.id}</a> <Cut text={e.reason} /> <span className="aside">{plural(e.needles.length, "passage")}</span></li>)}
              </ul>
            ))}
          </Section>
  
          <Section title={<>Problems on libraries{loaded && libraryProblems.length === 0 && `: ${reported.length}`}</>} tools={<a className="small" href={href(["libraries"])}>Libraries</a>}>
            {!loaded && <Skeleton lines={2} />}
            {loaded && libraryProblems.length > 0 && <Problems problems={libraryProblems} />}
            {loaded && libraryProblems.length === 0 && (reported.length === 0 ? <p className="small aside">None.</p> : (
              <ul className="plain small">
                {reported.slice(0, 8).map(({ library, problem }) => (
                  <li key={problem.decision}><span><LibraryLink library={library} />:</span> <a href={href(["libraries", library.id, "problems", problem.decision])} className="cut" title={problem.problem}>{problem.problem}</a> <span className="aside">{problem.texts.length ? plural(problem.texts.length, "text") : "the whole library"}</span></li>
                ))}
                {reported.length > 8 && <li className="aside"><a href={href(["libraries"])}>{reported.length - 8} more in the libraries</a></li>}
              </ul>
            ))}
          </Section>
        </>}
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

const SHOWN = 4; // the counts a list shows before "N more"

// Counts, each name linked to its view and each count to the texts it counts;
// past the first ones, "N more" links to the page that lists them all (more).
function Counted({ title, rows, ledger, more }: { title: string; rows: { key: string; n: number; view: string; link: string }[]; ledger?: boolean; more: string }) {
  return (
    <div>
      <h3 className="aside">{title}</h3>
      <ul className="plain counted small">
        {rows.slice(0, SHOWN).map((r) => (
          <li key={r.key}><a href={r.view} className={ledger ? "ledger cut" : "cut"} title={r.key}>{r.key}</a><a href={r.link} className="number" title={`The ${plural(r.n, "text")} under review`}>{r.n.toLocaleString("en")}</a></li>
        ))}
        {rows.length > SHOWN && <li className="aside"><a href={more}>{rows.length - SHOWN} more</a></li>}
      </ul>
    </div>
  );
}

export const overview: Page = { path: "", title: "Overview", group: null, Component: Overview };
