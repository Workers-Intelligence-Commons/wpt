// The review before an operation on texts: a selection (filter.tsx), the
// texts a filter matches at the change its page was read at, summed up by
// status, tag and license as the operation takes them (a count, counts.tsx),
// then the paged list of the texts the filter matches, each ticked, so that
// unticking one leaves it out. The operation sends the selection with the
// texts unticked here, never a list of the ids it reaches, and leaves out
// every text changed after the page was read. The review of a decision also
// sets apart the texts the decision cannot cover (blocked.tsx), which it
// then leaves out; after a refusal or a license change made there, the
// summary and the list are read again.

import { Button, Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState, type ReactNode } from "react";
import { type Answer } from "./api";
import { BlockedTexts, blockedCount, type Deciding } from "./blocked";
import { CountingNote, matched, useCounts } from "./counts";
import { search, type Found, type Place, type Selection } from "./filter";
import { PlacePager, Problems, plural, STATUSES, StatusBadge, type Library } from "./parts";
import { TextTable } from "./table";

const SHOWN = 12; // tags and licenses named in the summary
const LIMIT = 25; // texts of a page of the list

// decision: the review of a decision of that status; children: the form
// that sends the operation, given the selection with the texts unticked here,
// how many texts it takes once counted (blocked ones aside) and, for a
// decision, how many of them have its status already (null until counted, or
// once a text is unticked here, whose status the count does not know); onChange:
// after a refusal or a license change made from the blocked texts.
export function Review({ selection, libraries, decision, onChange, children }: {
  selection: Selection; libraries: Library[]; decision?: Deciding; onChange?: () => void;
  children: (chosen: Selection, size: number | null, already: number | null) => ReactNode;
}) {
  const [left, setLeft] = useState<Set<string>>(new Set(selection.unticked ?? []));
  const [changes, setChanges] = useState(0); // refusals and license changes made inside the review
  const [said, setSaid] = useState("");
  const [places, setPlaces] = useState<(Place | null)[]>([null]);
  const [page, setPage] = useState<Answer<Found> | null>(null);
  const data = useCounts(selection, changes); // the texts as the selection first took them, counted again after a change
  const after = places[places.length - 1];
  const pageKey = JSON.stringify([selection.filter, after, changes]);
  useEffect(() => {
    const [f, a] = JSON.parse(pageKey);
    search(f, a ? { after: a } : {}, LIMIT).then(setPage);
  }, [pageKey]);
  const changed = (words: string) => { setSaid(words); setPlaces([null]); setChanges((n) => n + 1); onChange?.(); };
  const counted = data?.done && !data.problems ? data : null;
  const apart = counted && decision ? blockedCount(counted.blocked, decision) : 0;
  const newly = [...left].filter((id) => !(selection.unticked ?? []).includes(id)).length; // unticked here, after the count
  const size = counted ? Math.max(0, matched(counted) - apart - newly) : null;
  const chosen: Selection = { ...selection, unticked: [...left] };
  const already = counted && decision && newly === 0 ? Math.min(counted.statuses[decision], size ?? 0) : null;
  const tick = (some: string[], on: boolean) => {
    const next = new Set(left);
    some.forEach((id) => (on ? next.delete(id) : next.add(id)));
    setLeft(next);
  };
  if (!page) return <Skeleton lines={4} label="Reading the texts it reaches" />;
  if (!page.ok) return <Problems problems={page.problems} />;
  const rows = page.data.rows;
  return (
    <div className="review">
      {data?.problems && <Problems problems={data.problems} />}
      {data && !data.done && <p className="small"><CountingNote counting={data} /></p>}
      {counted && <div className="reach-summary small">
        <section aria-label="By status">
          <h3 className="cue">By status</h3>
          <ul className="plain">
            {STATUSES.filter((s) => counted.statuses[s] > 0).map((s) => <li key={s}><StatusBadge status={s} /> {counted.statuses[s].toLocaleString("en")}</li>)}
            {counted.locked > 0 && <li><span className="aside">in a locked library</span> {counted.locked.toLocaleString("en")}</li>}
          </ul>
        </section>
        <section aria-label="By tag">
          <h3 className="cue">By tag</h3>
          <ul className="plain">
            {counted.tags.slice(0, SHOWN).map((t) => <li key={t.tag}><span className="cut" title={t.tag}>{t.tag}</span> <span className="aside">{t.texts.toLocaleString("en")}</span></li>)}
            {counted.tags.length > SHOWN && <li className="aside">{counted.tags.length - SHOWN} more tags</li>}
            {counted.tags.length === 0 && <li className="aside">no tag</li>}
          </ul>
        </section>
        <section aria-label="By license">
          <h3 className="cue">By license</h3>
          <ul className="plain">
            {counted.licenses.slice(0, SHOWN).map((l) => <li key={l.license}><span className="ledger cut" title={l.license}>{l.license}</span> <span className="aside">{l.texts.toLocaleString("en")}</span></li>)}
            {counted.licenses.length > SHOWN && <li className="aside">{counted.licenses.length - SHOWN} more licenses</li>}
          </ul>
        </section>
      </div>}
      {decision && apart > 0 ? (
        <BlockedTexts selection={chosen} status={decision} total={apart} version={changes} onChanged={changed} />
      ) : said && <p className="done cue" role="status">{said}</p>}
      {counted && matched(counted) === 0 ? <p>It reaches no text.</p> : <>
        <p className="row tight">
          <strong>{size === null ? "Counting the ticked texts" : `${plural(size, apart ? "other text" : "text")} ticked`}.</strong>
          <span className="aside">Untick a text to leave it out.</span>
          {left.size > 0 && <Button variant="quiet" onClick={() => setLeft(new Set())}>Tick every one again</Button>}
        </p>
        <TextTable rows={rows} libraries={libraries} label="The texts it reaches"
          ticked={new Set(rows.filter((r) => !left.has(r.id)).map((r) => r.id))} onTick={tick} />
        <PlacePager label="Pages of the texts it reaches" limit={LIMIT}
          onFirst={() => setPlaces([null])} onPrevious={places.length > 1 ? () => setPlaces(places.slice(0, -1)) : null}
          onNext={page.data.next ? () => setPlaces([...places, page.data.next]) : null} />
      </>}
      {children(chosen, size, already)}
    </div>
  );
}
