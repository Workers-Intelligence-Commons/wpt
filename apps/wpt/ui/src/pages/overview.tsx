// What needs the maintainer: the texts admitted, the jobs open (by lane),
// the latest snapshots with "Make a snapshot" and "Build" (a recipe file
// handed over on the Build page), the jobs queued or running (only while there are some),
// the number of texts under review (the depot's own counts, GET
// /api/counts), those texts counted by tag and by license (a count made in
// steps, which grows as it goes, counts.tsx), each tag and license linked to
// its view and each count to those texts, the latest decisions and the open
// erasure requests, each linked to its view.

import { ButtonLink, Notice, Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, waiting as queued, type Answer } from "../api";
import { CountingNote, matched, useCounts, BY_TAG_AND_LICENSE } from "../counts";
import { declared, useDeclarant } from "../declarant";
import { EVERY, type TextFilter } from "../filter";
import { Cut, PageHead, plural, Problems, Section, settingsHref, Stat, type Status } from "../parts";
import { href } from "../route";
import { snapshotHref, useSnapshots } from "../pipeline";
import { bytes } from "../sizes";
import { DecisionTable, type Latest } from "./decisions";
import type { Page } from "./index";
import { capital, JobProgress, JobState, laneOf, MARK_SECONDS, useJobs, useNow } from "./jobs";
import { madeWhen } from "./snapshots";

type Needle = { text: string };
type Erasure = { id: string; reason: string; status: Status; needles: Needle[] };
type DepotCounts = { texts: number; statuses: Record<Status, number> };

const REVIEW = "under review";
const WAITING: TextFilter = { ...EVERY, statuses: [REVIEW] };

const LATEST = 3; // the snapshots the overview shows, the latest first

function Overview() {
  const who = useDeclarant();
  const waiting = useCounts(WAITING, 0, BY_TAG_AND_LICENSE); // by tag and by license
  const [counts, setCounts] = useState<Answer<DepotCounts> | null>(null);
  const [latest, setLatest] = useState<Answer<Latest[]> | null>(null);
  const [erasures, setErasures] = useState<Answer<Erasure[]> | null>(null);
  const { answer: snapshots } = useSnapshots();
  const jobs = useJobs(MARK_SECONDS);
  const open = jobs?.ok ? jobs.data.filter((j) => queued(j.state)) : [];
  const now = useNow(open.length > 0);
  const known = counts?.ok ? counts.data : null;
  useEffect(() => {
    get<DepotCounts>("/api/counts").then(setCounts);
    get<Latest[]>("/api/decisions/latest").then(setLatest);
    get<Erasure[]>("/api/erasures").then(setErasures);
  }, []);
  const openErasures = erasures?.ok ? erasures.data.filter((e) => e.status === REVIEW) : [];
  const shownSnapshots = snapshots?.ok ? [...snapshots.data].reverse().slice(0, LATEST) : [];

  return (
    <>
      <PageHead title="Overview" chain="What waits for a decision in this depot, and the chain a model's data goes through: texts, snapshots, recipes, vocabularies, corpora." />
      {!declared(who) && (
        <Notice title="who decides">No name and email set. <a href={settingsHref()}>Set them</a> to decide.</Notice>
      )}
      <div className="stats stats-row">
        <Stat href={href(["texts"], { status: "admitted" })} value={known ? known.statuses.admitted.toLocaleString("en") : "…"} label={known ? `admitted of ${plural(known.texts, "text")}` : "admitted"} />
        <Stat href={href(["jobs"])} value={jobs?.ok ? open.length.toLocaleString("en") : "…"} label={jobs?.ok && open.length > 0 ? `jobs open: ${open.filter((j) => laneOf(j) === "heavy").length} heavy, ${open.filter((j) => laneOf(j) === "write").length} writes` : "jobs open"} />
        {erasures?.ok && openErasures.length === 0 && <Stat href={href(["erasures"])} value="0" label="erasures open" />}
      </div>
      <div className="board">
        <Section className={open.length === 0 ? "span-2" : undefined} title={<a href={href(["snapshots"])}>Latest snapshots</a>} tools={<><ButtonLink variant={snapshots?.ok && snapshots.data.length === 0 ? "primary" : "quiet"} href={href(["snapshots"], { new: "1" })}>Make a snapshot</ButtonLink>{snapshots?.ok && snapshots.data.length > 0 && <ButtonLink variant="quiet" href={href(["build"])}>Build a recipe</ButtonLink>}</>}>
          {!snapshots && <Skeleton lines={2} />}
          {snapshots && !snapshots.ok && <Problems problems={snapshots.problems} />}
          {snapshots?.ok && (shownSnapshots.length === 0 ? <p className="small aside">None yet.</p> : (
            <ul className="plain small">
              {shownSnapshots.map((s) => <li key={s.id}><a href={snapshotHref(s)} className="cue">{s.name}</a> <span className="aside">change {s.at_change.toLocaleString("en")} · {plural(s.texts, "text")} · {bytes(s.bytes)} · {madeWhen(s.made_at)} UTC</span></li>)}
              {snapshots.data.length > LATEST && <li className="aside"><a href={href(["snapshots"])}>{snapshots.data.length - LATEST} more</a></li>}
            </ul>
          ))}
        </Section>
        {jobs && !jobs.ok && <Section title="Jobs open"><Problems problems={jobs.problems} /></Section>}
        {open.length > 0 && (
          <Section title={<>Jobs open: {open.length}</>} tools={<a className="small" href={href(["jobs"])}>Jobs</a>}>
            <ul className="plain small">
              {open.slice(0, 6).map((j) => <li key={j.id}><span className="cue">{capital(j.title)}</span>: <JobState job={j} now={now} />{j.state === "running" && <>: <JobProgress job={j} /></>}</li>)}
            </ul>
          </Section>
        )}
      </div>
      <div className="board">
        <Section className={erasures?.ok && openErasures.length === 0 ? "span-2" : "tall"} title={<>Under review{known && <>: <a href={href(["texts"], { status: REVIEW })}>{plural(known.statuses[REVIEW], "text")}</a></>}</>}>
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

        {!(erasures?.ok && openErasures.length === 0) && (
          <Section title={<>Open erasure requests{erasures?.ok && `: ${openErasures.length}`}</>} tools={<a className="small" href={href(["erasures"])}>Erasures</a>}>
            {!erasures && <Skeleton lines={2} />}
            {erasures && !erasures.ok && <Problems problems={erasures.problems} />}
            {erasures?.ok && (
              <ul className="plain small">
                {openErasures.slice(0, 8).map((e) => <li key={e.id}><a className="ledger" href={href(["erasures"], { request: e.id })} title={`What ${e.id} asks to erase`}>{e.id}</a> <Cut text={e.reason} /> <span className="aside">{plural(e.needles.length, "passage")}</span></li>)}
              </ul>
            )}
          </Section>
        )}
        <Section className="span-2" title="Latest decisions">
          {!latest && <Skeleton lines={4} />}
          {latest && !latest.ok && <Problems problems={latest.problems} />}
          {latest?.ok && (latest.data.length === 0 ? <p className="small">No decision yet.</p>
            : <DecisionTable rows={latest.data} />)}
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
