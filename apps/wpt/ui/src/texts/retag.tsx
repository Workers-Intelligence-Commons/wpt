// Adding tags to, or removing tags from, the ticked texts of the Texts page
// (POST /api/texts/retag). Clicks first: the depot's tags show as chips, one
// click picks or drops one; only a new tag is typed. For a removal the chips
// are the tags the ticked texts have, once counted (counts.tsx), the most
// held first. Each time the tags picked change, the cost is read again (POST
// /api/texts/retag/estimate: how many texts change, the rows written, what
// the depot's file grows by, the time), and the confirm button stays off
// until that cost is shown for exactly the tags picked. The confirm sends the
// estimate it showed; when the server finds the cost grew meanwhile (409), the
// new one takes its place and the button asks again. The declaration is the
// one this browser keeps (declarant.ts), named beside the button.

import { Button, TextInput } from "@workers-intelligence-commons/components";
import { useEffect, useState, type ReactNode } from "react";
import { get, post, type Answer } from "../api";
import { shownCounts, useCounts, BY_TAG } from "../counts";
import { leftOut, useWrite, WriteProgress, type Left } from "../decide-forms";
import { sent, type Selection } from "../filter";
import { Confirm, Help, plural, Problems, type Declaration } from "../parts";
import { href } from "../route";
import { bytes } from "../sizes";

export type RetagEstimate = {
  texts: number; changed: number; upstream?: number; written: number; bytes: number; change_bytes?: number; seconds: number;
  rows: Record<string, number>;
};
type Retagged = { decision: string | null; changed: number; left_out: number; left: Left };
type Tag = { tag: string; texts: number };

const PAUSE = 400; // milliseconds after the last click before the cost is read
const CHIPS = 40; // tags shown as chips before "more"
// what a tag may be: what the depot accepts as a name (no blank, no space at either end)
const tagOf = (typed: string) => typed.trim();

const seconds = (s: number) => (s < 1 ? "under a second" : s < 90 ? `about ${Math.round(s)} s` : `about ${Math.round(s / 60)} minutes`);

// The cost of a change of tags, in words.
export function RetagCost({ estimate, removing }: { estimate: RetagEstimate; removing: boolean }) {
  const unchanged = estimate.texts - estimate.changed;
  return (
    <div className="retag-cost small" role="status">
      <p>
        <strong>{estimate.changed === 0 ? "Nothing to change" : `Changes ${plural(estimate.changed, "text")}`}</strong>
        {unchanged > 0 && <> of the {plural(estimate.texts, "ticked text")}; {unchanged.toLocaleString("en")} {removing ? "have none of these tags" : "have them already"} and stay as they are</>}.
      </p>
      {estimate.changed > 0 && (
        <p>
          Writes {plural(estimate.written, "row")}, about {bytes(estimate.bytes)} added to the depot's file, {seconds(estimate.seconds)}.
          <Help label="the cost">A small row for each tag a text gains or loses, never a new revision of the text, all under one decision; the depot's counts and tag listings follow. A snapshot made before reads each text's tags as they were when it was made, so it never changes.</Help>
        </p>
      )}
      {(estimate.upstream ?? 0) > 0 && <Problems problems={[`${plural(estimate.upstream!, "ticked text")} belong to the upstream this depot follows, which alone changes them: untick them to go on.`]} />}
    </div>
  );
}

export function RetagForm({ selection, size, removing, onDone }: {
  selection: Selection; size: number | null; removing: boolean; onDone: (a: Answer<unknown>, words: ReactNode) => void;
}) {
  const [tags, setTags] = useState<Tag[] | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [typed, setTyped] = useState("");
  const [more, setMore] = useState(false);
  const [cost, setCost] = useState<{ key: string; answer: Answer<RetagEstimate> } | null>(null);
  const [grew, setGrew] = useState(false); // the estimate shown is the one a 409 gave
  const { send, job, busy } = useWrite();
  useEffect(() => { get<Tag[]>("/api/tags").then((a) => setTags(a.ok ? a.data : [])); }, []);
  // for a removal, the tags the ticked texts have, the most held first, once counted
  const counting = useCounts(removing ? selection : null, 0, BY_TAG);
  const held = removing ? shownCounts(counting)?.tags ?? null : null;
  const offered: Tag[] = (held ?? tags ?? []).slice().sort((a, b) => b.texts - a.texts || a.tag.localeCompare(b.tag));
  const shown = more ? offered : offered.slice(0, CHIPS);
  const extra = picked.filter((t) => !offered.some((o) => o.tag === t)); // new tags typed, shown as chips too
  const toggle = (tag: string) => { setGrew(false); setPicked(picked.includes(tag) ? picked.filter((t) => t !== tag) : [...picked, tag]); };
  const addTyped = () => {
    const tag = tagOf(typed);
    if (tag && !picked.includes(tag)) { setGrew(false); setPicked([...picked, tag]); }
    setTyped("");
  };

  const selectionKey = JSON.stringify(sent(selection));
  const key = JSON.stringify([selectionKey, removing, [...picked].sort()]);
  useEffect(() => {
    if (!picked.length) return;
    let live = true;
    const timer = window.setTimeout(() => {
      post<RetagEstimate>("/api/texts/retag/estimate", { selection: sent(selection), [removing ? "remove" : "add"]: picked })
        .then((answer) => live && setCost({ key, answer }));
    }, PAUSE);
    return () => { live = false; window.clearTimeout(timer); };
  }, [key]);
  const answer = picked.length && cost?.key === key ? cost.answer : null;
  const estimate = answer?.ok ? answer.data : null;
  const ready = estimate !== null && estimate.changed > 0 && !(estimate.upstream ?? 0) && size !== 0 && !busy;

  const go = async (declaration: Declaration) => {
    if (!estimate) return;
    const a = await send<Retagged>("/api/texts/retag", { selection: sent(selection), [removing ? "remove" : "add"]: picked, declaration, confirmed: true, estimate });
    const fresh = !a.ok && a.status === 409 ? (a.body?.estimate as RetagEstimate | undefined) : undefined;
    if (fresh) { setCost({ key, answer: { ok: true, data: fresh } }); setGrew(true); return; }
    const names = picked.join(", ");
    onDone(a, a.ok ? a.data.decision === null
      ? <>Nothing changed: every text {removing ? `had none of ${names}` : `had ${names} already`}.{leftOut(a.data.left)}</>
      : <>{removing ? `Removed ${names} from` : `Added ${names} to`} {plural(a.data.changed, "text")}{a.data.left_out ? `; ${a.data.left_out.toLocaleString("en")} unchanged` : ""}.{leftOut(a.data.left)}{" "}
        <a href={href(["decisions", a.data.decision])}>See the decision</a></> : "");
  };

  return (
    <div className="action-form">
      <fieldset className="chips-field">
        <legend className="small">{removing ? "Tags to remove: click each" : "Tags to add: click each, or type a new one"}</legend>
        {tags === null || (removing && held === null && !counting?.problems) ? <p className="small aside">{removing ? "Counting the tags of the ticked texts" : "Reading the tags"}</p> : (
          <div className="chips" role="group" aria-label={removing ? "Tags the ticked texts have" : "Tags of the depot"}>
            {offered.length === 0 && extra.length === 0 && <span className="small aside">{removing ? "The ticked texts have no tag." : "No tag yet: type the first one."}</span>}
            {shown.map((t) => (
              <button key={t.tag} type="button" className="chip-button wic-focus" aria-pressed={picked.includes(t.tag)} onClick={() => toggle(t.tag)}
                title={removing ? `${plural(t.texts, "ticked text")} have it` : `${plural(t.texts, "text")} of the depot have it`}>
                {t.tag} <span className="aside">{t.texts.toLocaleString("en")}</span>
              </button>
            ))}
            {extra.map((t) => (
              <button key={t} type="button" className="chip-button wic-focus" aria-pressed={true} onClick={() => toggle(t)} title="A new tag: click to drop it">
                {t} <span className="aside">new</span>
              </button>
            ))}
            {offered.length > CHIPS && <Button variant="quiet" onClick={() => setMore(!more)}>{more ? "Fewer" : `${(offered.length - CHIPS).toLocaleString("en")} more`}</Button>}
          </div>
        )}
      </fieldset>
      {!removing && (
        <form className="row tight new-tag" onSubmit={(e) => { e.preventDefault(); addTyped(); }}>
          <TextInput label="A new tag" value={typed} onChange={(e) => setTyped(e.target.value)} hint="Only for a tag no text has yet." />
          <Button type="submit" variant="quiet" disabled={!tagOf(typed)}>Pick it</Button>
        </form>
      )}
      {picked.length === 0 ? <p className="small aside">Pick a tag to see what the change costs.</p>
        : !answer ? <p className="small aside" role="status">Reading the cost</p>
        : !answer.ok ? <Problems problems={answer.problems} />
        : <RetagCost estimate={answer.data} removing={removing} />}
      {grew && <p className="small act">The cost grew since it was shown: the figures above are the new ones. Confirm again to go on.</p>}
      <Confirm label={removing ? "Confirm and remove the tags" : "Confirm and add the tags"} disabled={!ready} onConfirm={go}
        note={<>{removing ? "Removes" : "Adds"} {picked.length ? picked.join(", ") : "the tags picked"} {removing ? "from" : "to"} {size === null ? "the ticked texts" : `the ${plural(size, "ticked text")}`}; they keep their status. A text changed since this page was read is left out.</>} />
      <WriteProgress job={job} />
    </div>
  );
}
