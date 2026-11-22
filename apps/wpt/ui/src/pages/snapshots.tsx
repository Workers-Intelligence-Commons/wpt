// Snapshots: the depot's snapshots (GET /api/snapshots), each a name on a
// change number with the set of texts admitted then, which a recipe names
// and a build reads, every text at its revision as of that change, so a
// corpus built from it is the same next year. "Make a snapshot" is one
// confirmation: the next free name filled in (GET /api/snapshots/new: the
// year and month, then with b, c, ...), what it costs shown before the click
// (texts, time, disk, the change it reads), a note optional; the snapshot is
// made as a job of the heavy lane (POST /api/snapshots with the estimate
// shown; 409 with the new one when it changed meanwhile), followed to its
// end, then its page opens.
//
// #/snapshots/ID is one snapshot (GET /api/snapshots/ID): its facts and
// digest, its texts by kind and by tag (from its summary, which never
// changes), what erasures carried out since did to it, the repository's
// recipes that name it, its sets for anyone to check the digest, and the
// recipe composer (composer.tsx; ?compose=1 opens it, &vocabulary=SHA256
// picks that vocabulary).

import { Button, Dialog, Question, Skeleton, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, part, post, type Answer, type Job } from "../api";
import { Composer } from "../composer";
import { whole } from "../mix";
import { Facts, Help, optional, PageHead, plural, Problems, Section } from "../parts";
import { pipelineChanged, snapshotHref, useSnapshots, type Sized, type SnapshotEstimate, type SnapshotOffer, type SnapshotShown } from "../pipeline";
import { href } from "../route";
import { bytes, duration } from "../sizes";
import type { Page } from "./index";
import { JobLine, useJobs } from "./jobs";

export const madeWhen = (at: string) => at.replace("T", " ").slice(0, 16);
// What a snapshot was made from, in words: nothing for one made from the texts admitted then.
const madeFrom = (origin: SnapshotShown["origin"]) => {
  const name = typeof origin.name === "string" ? origin.name : null;
  if (origin.kind === "library") return `the locked library ${name ?? origin.library}`;
  if (origin.kind === "dataset") return `revision ${origin.revision} of the dataset ${name ?? origin.dataset}`;
  return origin.kind === "made" ? null : origin.kind.replace(/_/g, " ");
};

// The cost of making a snapshot now, as facts: what the confirmation covers.
export function SnapshotCost({ estimate }: { estimate: SnapshotEstimate }) {
  return (
    <Facts rows={[
      ["texts", <>{plural(estimate.texts, "admitted text")} now</>],
      ["time", <>{estimate.seconds < 60 ? `about ${Math.max(1, Math.round(estimate.seconds))} s` : `about ${duration(estimate.seconds)}`}<Help label="the estimate">{estimate.basis}.</Help></>],
      ["disk", <>at most {bytes(estimate.bytes)} in the depot's file</>],
      ["reads", <>{estimate.at}</>],
    ]} />
  );
}

// "Make a snapshot": the offer read when opened, one button that confirms it.
// onMade: the snapshot's id, once its job ended well.
function MakeSnapshot({ open, onClose, onMade }: { open: boolean; onClose: () => void; onMade: (id: string) => void }) {
  const [offer, setOffer] = useState<Answer<SnapshotOffer> | null>(null);
  const [name, setName] = useState("");
  const [note, setNote] = useState("");
  const [changed, setChanged] = useState(false); // the estimate shown is the one a 409 gave
  const [problems, setProblems] = useState<string[]>([]);
  const [job, setJob] = useState<Job<unknown> | null>(null);
  useEffect(() => {
    if (!open) return;
    setOffer(null); setJob(null); setProblems([]); setChanged(false); setNote("");
    get<SnapshotOffer>("/api/snapshots/new").then((a) => { setOffer(a); if (a.ok) setName(a.data.name); });
  }, [open]);
  const estimate = offer?.ok ? offer.data.estimate : null;
  const make = async () => {
    if (!estimate) return;
    const a = await post<Job<unknown>>("/api/snapshots", { name: name.trim(), note: note.trim() || null, confirmed: true, estimate });
    if (a.ok) { setJob(a.data); setProblems([]); return; }
    const fresh = a.status === 409 ? (a.body?.estimate as SnapshotEstimate | undefined) : undefined;
    if (fresh && offer?.ok) { setOffer({ ok: true, data: { ...offer.data, estimate: fresh } }); setChanged(true); }
    setProblems(a.problems);
  };
  const ended = useCallback((j: Job<unknown>) => {
    const id = (j.answer as { id?: string } | null)?.id;
    if (j.state === "done" && id) onMade(id);
  }, [onMade]);
  return (
    <Dialog open={open} onClose={onClose} title="Make a snapshot">
      {open && !offer && <Skeleton lines={4} label="Reading what it costs" />}
      {offer && !offer.ok && <Problems problems={offer.problems} />}
      {job ? <JobLine job={job} onEnded={ended} /> : estimate && offer?.ok && (
        <Question question={<>Make the snapshot <strong className="ledger">{name.trim() || "…"}</strong> of the texts admitted now?</>}
          from="before it starts"
          onSubmit={(e) => { e.preventDefault(); make(); }}
          actions={<Button type="submit" variant="primary" disabled={!name.trim()}>Make {name.trim() || "the snapshot"}</Button>}>
          <SnapshotCost estimate={estimate} />
          {changed && <p className="small act">The cost changed since it was read: the figures above are the new ones.</p>}
          {offer.data.being_made.length > 0 && <p className="small aside">Being made now: {offer.data.being_made.join(", ")}.</p>}
          <div className="fields">
            <TextInput label="Name" hint="Filled in with the next free one; kept for good." value={name} onChange={(e) => setName(e.target.value)} />
            <TextInput label={optional("Note")} value={note} onChange={(e) => setNote(e.target.value)} />
          </div>
          <p className="small aside">A job of the <a href={href(["jobs"])}>queue</a>; it never changes a text, and a snapshot is never deleted.</p>
          <Problems problems={problems} />
        </Question>
      )}
    </Dialog>
  );
}

function Snapshots({ query }: { query: URLSearchParams }) {
  const { answer: rows, reload } = useSnapshots();
  const [making, setMaking] = useState(query.get("new") === "1");
  const jobs = useJobs();
  const open = jobs?.ok ? jobs.data.filter((j) => j.kind === "snapshot" && (j.state === "queued" || j.state === "running")) : [];
  const owners = new Set(rows?.ok ? rows.data.map((r) => r.owner) : []);
  const made = useCallback((id: string) => { pipelineChanged(); reload(); window.location.hash = snapshotHref({ id }); }, [reload]);
  return (
    <>
      <PageHead title="Snapshots"
        chain={<>A snapshot is a name on a change of the depot with the texts admitted then. A <a href={href(["recipes"])}>recipe</a> names one, and every build of it reads each text as it was at that change, so a model trained on it can be checked later. Daily work on the texts never changes a snapshot.</>}
        actions={<Button variant={rows?.ok && rows.data.length === 0 ? "primary" : "quiet"} onClick={() => setMaking(true)}>Make a snapshot</Button>} />
      {open.map((j) => <JobLine key={j.id} job={j} onEnded={() => { pipelineChanged(); reload(); }} />)}
      {!rows && <Skeleton lines={4} />}
      {rows && !rows.ok && <Problems problems={rows.problems} />}
      {rows?.ok && (rows.data.length === 0 ? <p className="empty-state">No snapshot yet. Make one to compose a recipe.</p> : (
        <div className="table-scroll cards">
          <table className="ledger-table" aria-label="The snapshots">
            <thead><tr><th scope="col">name</th>{owners.size > 1 && <th scope="col">depot</th>}<th scope="col" className="number">change</th><th scope="col" className="number">texts</th><th scope="col" className="number">size</th><th scope="col">made</th><th scope="col">note</th></tr></thead>
            <tbody>
              {[...rows.data].reverse().map((r) => (
                <tr key={r.id}>
                  <td className="nowrap" data-label="name"><a href={snapshotHref(r)}>{r.name}</a></td>
                  {owners.size > 1 && <td data-label="depot" className="ledger">{r.owner}</td>}
                  <td className="number ledger" data-label="change">{r.at_change.toLocaleString("en")}</td>
                  <td className="number ledger" data-label="texts">{r.texts.toLocaleString("en")}</td>
                  <td className="number ledger" data-label="size">{bytes(r.bytes)}</td>
                  <td className="nowrap" data-label="made">{madeWhen(r.made_at)} UTC</td>
                  <td className="wrap-cell small" data-label="note">{r.note ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <MakeSnapshot open={making} onClose={() => setMaking(false)} onMade={made} />
    </>
  );
}

// A summary's parts as a table: texts and bytes of each, the largest first, each linked when link gives a place;
// share: each one's part of the bytes too (for parts that never overlap, such as kinds).
function Counted({ parts, label, link, share = false }: { parts: Record<string, Sized>; label: string; link?: (name: string) => string; share?: boolean }) {
  const rows = Object.entries(parts).sort((a, b) => b[1][0] - a[1][0]);
  const sum = rows.reduce((t, [, s]) => t + s[2], 0) || 1;
  return (
    <table className="ledger-table compact" aria-label={label}>
      <thead><tr><th scope="col">{label}</th><th scope="col" className="number">texts</th><th scope="col" className="number">size</th>{share && <th scope="col" className="number">share</th>}</tr></thead>
      <tbody>{rows.map(([name, s]) => <tr key={name}><td>{link ? <a href={link(name)}>{name}</a> : name}</td><td className="number ledger">{s[0].toLocaleString("en")}</td><td className="number ledger">{bytes(s[2])}</td>{share && <td className="number ledger">{whole(s[2] / sum)}</td>}</tr>)}</tbody>
    </table>
  );
}

function OneSnapshot({ id, query }: { id: string; query: URLSearchParams }) {
  const [shown, setShown] = useState<Answer<SnapshotShown> | null>(null);
  const [composing, setComposing] = useState(query.get("compose") === "1");
  const [copied, setCopied] = useState(false);
  useEffect(() => { get<SnapshotShown>(`/api/snapshots/${part(id)}`).then(setShown); }, [id]);
  if (!shown) return <Skeleton lines={6} label="Reading the snapshot" />;
  if (!shown.ok) return <><PageHead title={id} meta={<a href={href(["snapshots"])}>snapshots</a>} /><Problems problems={shown.problems} /></>;
  const s = shown.data;
  const copy = async () => { try { await navigator.clipboard.writeText(s.digest); setCopied(true); } catch { setCopied(false); } };
  return (
    <>
      <PageHead title={s.name}
        meta={<><a href={href(["snapshots"])}>snapshots</a><span>change {s.at_change.toLocaleString("en")}</span><span>{plural(s.texts, "text")}</span><span>{bytes(s.bytes)}</span><span className="ledger">{s.owner}</span></>}
        chain={<>The texts admitted at change {s.at_change.toLocaleString("en")} of {s.owner}, each read as it was then. It never changes; an erasure carried out later still cuts what it names.</>}
        actions={!composing && <Button variant="primary" onClick={() => setComposing(true)}>New recipe from this snapshot</Button>} />
      <Facts rows={[
        ["made", `${madeWhen(s.made_at)} UTC`],
        ["made from", madeFrom(s.origin)],
        ["note", s.note],
        ["digest", <><span className="ledger breaks">{s.digest}</span> <button type="button" className="link-button small" onClick={copy}>{copied ? "Copied" : "Copy"}</button>
          <Help label="the digest">A sha256 of the text numbers it holds and the change it reads them at. Anyone can work it out again from its sets, without a copy of the depot.</Help></>],
        ["sets", <a href={`/api/snapshots/${part(s.id)}/sets`} download={`${s.name}-sets.json`}>Download its sets</a>],
      ]} />
      {composing && <Composer snapshot={s} vocabulary={query.get("vocabulary")} onClose={() => setComposing(false)} />}
      <div className="board">
        <Section title="By kind">
          <Counted parts={s.summary.kinds} label="kind" share link={(k) => href(["texts"], { kind: k })} />
        </Section>
        <Section title="By tag" help="A text with several tags counts once for each.">
          {Object.keys(s.summary.tags).length === 0 ? <p className="small aside">No text has a tag.</p>
            : <Counted parts={s.summary.tags} label="tag" link={(t) => href(["tags", t])} />}
        </Section>
        <Section title={<>Erased since{s.erased_since.texts ? `: ${plural(s.erased_since.texts, "text")}` : ""}</>}>
          {s.erased_since.erasures.length === 0 ? <p className="small aside">No erasure carried out since it was made.</p> : (
            <p className="small">{plural(s.erased_since.texts, "text")} of it cut or removed by {s.erased_since.erasures.map((e, i) => <span key={e}>{i ? ", " : ""}<a className="ledger" href={href(["erasures"], { request: e })}>{e}</a></span>)}. A build leaves out what they erased.</p>
          )}
        </Section>
        <Section title={<>Recipes naming it{s.recipes.length ? `: ${s.recipes.length}` : ""}</>}>
          {s.recipes.length === 0 ? <p className="small aside">No recipe of the repository names it yet.</p> : (
            <ul className="plain linked">
              {s.recipes.map((r) => <li key={r.name}><a href={href(["recipes", r.name])}>{r.name}</a>{!r.matches && <span className="not-allowed"> its change or digest differs</span>}</li>)}
            </ul>
          )}
        </Section>
      </div>
    </>
  );
}

export const snapshots: Page = {
  path: "snapshots", title: "Snapshots", group: "data",
  Component: ({ rest, query }) => (rest[0] ? <OneSnapshot key={rest[0]} id={rest[0]} query={query} /> : <Snapshots query={query} />),
};
