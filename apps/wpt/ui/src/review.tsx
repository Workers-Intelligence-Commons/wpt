// The review before an operation on texts: the ids of every text a filter
// matches, read once when it opens, a summary of them by status, tag and
// license, then the paged list of them, each ticked, so that unticking one
// leaves it out. The operation then names exactly the ticked ids: a text
// that comes to match the filter after the review is never reached. The
// review of a decision first asks the depot which of those texts the
// decision cannot cover, and sets them apart (blocked.tsx): they are never ticked,
// and a text decided from there leaves the review. After a refusal or a
// license change made there, the summary is read again for the texts the
// review still reaches, and the check is made again: a text that now has
// an allowed license joins the others, ticked.

import { Button, Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState, type ReactNode } from "react";
import { type Answer } from "./api";
import { blocked, BlockedTexts, type Blocked, type Deciding } from "./blocked";
import { BY_ID, EVERY, search, type Found, type TextFilter } from "./filter";
import { Pager, Problems, plural, STATUSES, StatusBadge, type Library } from "./parts";
import { TextTable } from "./table";

export type Untick = { label: string; part: Partial<TextFilter> }; // a button that unticks the texts matching part

const SHOWN = 12; // tags and licenses named in the summary

// decision: the review of a decision of that status; onChange: after a
// decision or a license change made from its texts that it cannot cover.
export function Review({ filter, libraries, unticks = [], decision, onChange, children }: {
  filter: TextFilter; libraries: Library[]; unticks?: Untick[]; decision?: Deciding; onChange?: () => void;
  children: (chosen: string[], found: Found) => ReactNode;
}) {
  const [found, setFound] = useState<Answer<Found> | null>(null);
  const [checked, setChecked] = useState<Answer<{ texts: Blocked[] }> | null>(null);
  const [gone, setGone] = useState<Set<string>>(new Set()); // texts decided from the review, which leave it
  const [changes, setChanges] = useState(0); // refusals and license changes made inside the review
  const [summary, setSummary] = useState<Answer<Found> | null>(null); // read again after each of them
  const [left, setLeft] = useState<Set<string>>(new Set());
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(25);
  const [page, setPage] = useState<Answer<Found> | null>(null);
  const filterKey = JSON.stringify(filter);
  useEffect(() => { search(JSON.parse(filterKey), 0, 1, BY_ID, true).then(setFound); }, [filterKey]);
  const reached = found?.ok ? (found.data.ids ?? []).filter((id) => !gone.has(id)) : [];
  const reachedKey = decision && found ? JSON.stringify(reached) : "";
  useEffect(() => {
    if (!reachedKey) return;
    const all: string[] = JSON.parse(reachedKey);
    if (all.length) blocked(all, decision!).then(setChecked); else setChecked({ ok: true, data: { texts: [] } });
  }, [reachedKey, changes, decision]);
  useEffect(() => {
    if (!changes || !reachedKey) return;
    search({ ...EVERY, ids: JSON.parse(reachedKey) }, 0, 1).then(setSummary);
  }, [reachedKey, changes]);
  const [said, setSaid] = useState(""); // what the latest of them did, shown here once no blocked text is left
  const changed = (words: string) => { setSaid(words); setChanges((n) => n + 1); onChange?.(); };
  const apart = checked?.ok ? checked.data.texts : [];
  const blockedIds = new Set(apart.map((b) => b.id));
  const ids = reached.filter((id) => !blockedIds.has(id));
  const slice = ids.slice(offset, offset + limit);
  const sliceKey = JSON.stringify(slice);
  useEffect(() => {
    const shown: string[] = JSON.parse(sliceKey);
    if (shown.length) search({ ...EVERY, ids: shown }, 0, shown.length).then(setPage);
  }, [sliceKey]);
  if (!found) return <Skeleton lines={4} label="Reading the texts it reaches" />;
  if (!found.ok) return <Problems problems={found.problems} />;
  if (decision && !checked) return <Skeleton lines={4} label={`Checking which texts can be ${decision}`} />;
  if (checked && !checked.ok) return <Problems problems={checked.problems} />;
  if (found.data.total === 0) return <p>It reaches no text.</p>;
  const data = summary?.ok ? summary.data : found.data;
  const chosen = ids.filter((id) => !left.has(id));
  const tick = (some: string[], on: boolean) => {
    const next = new Set(left);
    some.forEach((id) => (on ? next.delete(id) : next.add(id)));
    setLeft(next);
  };
  const untick = async (part: Partial<TextFilter>) => {
    const a = await search({ ...EVERY, ...part, ids: chosen }, 0, 1, BY_ID, true);
    if (a.ok) tick(a.data.ids ?? [], false);
  };
  return (
    <div className="review">
      {summary && !summary.ok && <Problems problems={summary.problems} />}
      <div className="reach-summary small">
        <section aria-label="By status">
          <h3 className="cue">By status</h3>
          <ul className="plain">
            {STATUSES.filter((s) => data.statuses[s] > 0).map((s) => <li key={s}><StatusBadge status={s} /> {data.statuses[s].toLocaleString("en")}</li>)}
            {data.locked > 0 && <li><span className="aside">in a locked library</span> {data.locked.toLocaleString("en")}</li>}
          </ul>
        </section>
        <section aria-label="By tag">
          <h3 className="cue">By tag</h3>
          <ul className="plain">
            {data.tags.slice(0, SHOWN).map((t) => <li key={t.tag}><span className="cut" title={t.tag}>{t.tag}</span> <span className="aside">{t.texts.toLocaleString("en")}</span></li>)}
            {data.tags.length > SHOWN && <li className="aside">{data.tags.length - SHOWN} more tags</li>}
            {data.tags.length === 0 && <li className="aside">no tag</li>}
          </ul>
        </section>
        <section aria-label="By license">
          <h3 className="cue">By license</h3>
          <ul className="plain">
            {data.licenses.slice(0, SHOWN).map((l) => <li key={l.license}><span className="ledger cut" title={l.license}>{l.license}</span> <span className="aside">{l.texts.toLocaleString("en")}</span></li>)}
            {data.licenses.length > SHOWN && <li className="aside">{data.licenses.length - SHOWN} more licenses</li>}
          </ul>
        </section>
      </div>
      {apart.length > 0 ? (
        <BlockedTexts found={apart} status={decision!} onRelicensed={changed}
          onDecided={(decided, words) => { setGone(new Set([...gone, ...decided])); changed(words); }} />
      ) : said && <p className="done cue" role="status">{said}</p>}
      {ids.length === 0 ? <p>No other text: the {decision === "refused" ? "refusal has nothing to refuse" : "admission has nothing to admit"}.</p> : <>
      <p className="row tight">
        <strong>{chosen.length.toLocaleString("en")} of {plural(ids.length, apart.length || gone.size ? "other text" : "text")} ticked.</strong>
        <span className="aside">Untick a text to leave it out.</span>
        {left.size > 0 && <Button variant="quiet" onClick={() => setLeft(new Set())}>Tick every one again</Button>}
        {unticks.map((u) => <Button key={u.label} variant="quiet" disabled={chosen.length === 0} onClick={() => untick(u.part)}>{u.label}</Button>)}
      </p>
      {page && !page.ok && <Problems problems={page.problems} />}
      {page?.ok && (
        <TextTable rows={page.data.rows} libraries={libraries} label="The texts it reaches" ticked={new Set(slice.filter((id) => !left.has(id)))} onTick={tick} />
      )}
      <Pager offset={offset} limit={limit} total={ids.length} onOffset={setOffset} onLimit={(n) => { setLimit(n); setOffset(0); }} label="Pages of the texts it reaches" />
      </>}
      {children(chosen, data)}
    </div>
  );
}
