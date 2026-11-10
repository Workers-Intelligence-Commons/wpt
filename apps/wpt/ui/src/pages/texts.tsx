// The depot's texts and items: a filter bar that applies as it changes, a
// sortable paged table, and ticks. Ticking rows brings up a bar at the
// bottom of the screen whose actions (admit, refuse, put in a library) open
// the review of the ticked texts; the operation then names exactly the
// texts left ticked there. #/texts/ID is one text's page (text.tsx).

import { Button, Dialog, Select, Skeleton } from "@workers-intelligence-commons/components";
import { useCallback, useState } from "react";
import { post, type Answer } from "../api";
import { BY_ID, EVERY, FilterBar, fromQuery, search, toQuery, useSearch, type TextFilter } from "../filter";
import { Outcome, PageHead, Pager, plural, Problems, required, RequiredNote, useLibraries, type Library } from "../parts";
import { DecideForm } from "../decide-forms";
import { Review } from "../review";
import { href } from "../route";
import { TextTable } from "../table";
import type { Page } from "./index";
import { Text } from "./text";

type Action = "admitted" | "refused" | "put";
type Done = { decision?: string; texts?: string[]; joined?: string[]; left_out?: string[] };

function Texts({ query }: { query: URLSearchParams }) {
  const filter = fromQuery(query);
  const { libraries, reload: reloadLibraries } = useLibraries();
  const { answer, pending, offset, setOffset, limit, setLimit, order, setOrder, reload } = useSearch(filter);
  const [ticked, setTicked] = useState<Set<string>>(new Set());
  const [action, setAction] = useState<Action | null>(null);
  const [done, setDone] = useState<{ action: Action; answer: Answer<Done> } | null>(null);
  const [tickingAll, setTickingAll] = useState(false);

  const apply = useCallback((f: TextFilter) => {
    setTicked(new Set());
    window.location.hash = href(["texts"], toQuery(f));
  }, []);
  const tick = (ids: string[], on: boolean) => {
    const next = new Set(ticked);
    ids.forEach((id) => (on ? next.add(id) : next.delete(id)));
    setTicked(next);
  };
  const tickAll = async () => {
    setTickingAll(true);
    const a = await search(filter, 0, 1, BY_ID, true);
    setTickingAll(false);
    if (a.ok) setTicked(new Set(a.data.ids));
  };
  const finished = (act: Action, a: Answer<Done>) => {
    setDone({ action: act, answer: a });
    if (a.ok) {
      setAction(null);
      setTicked(new Set());
      reload();
      reloadLibraries();
    }
  };
  const total = answer?.ok ? answer.data.total : 0;
  const filtered = JSON.stringify(filter) !== JSON.stringify(EVERY);

  return (
    <>
      <PageHead title="Texts"
        meta={answer?.ok && (
          <span>
            {plural(answer.data.statuses["under review"], "text")} under review, {answer.data.statuses.admitted.toLocaleString("en")} admitted, {answer.data.statuses.refused.toLocaleString("en")} refused
            {answer.data.locked > 0 && <>; {answer.data.locked.toLocaleString("en")} in a locked library</>}
          </span>
        )}
        chain={<>Texts are <a href={href(["add"])}>added</a> under review, then admitted or refused; admitted texts join <a href={href(["libraries"])}>libraries</a>, and each names its <a href={href(["licenses"])}>licenses</a>. Tick texts to act on them.</>} />
      <FilterBar value={filter} onApply={apply} libraries={libraries} />
      {done && !action && <Outcome answer={done.answer} done={(d) => said(done.action, d)} />}
      {!answer && <Skeleton lines={8} label="Reading the texts" />}
      {answer && !answer.ok && <Problems problems={answer.problems} />}
      {answer?.ok && total === 0 && <p>No text matches the filter.</p>}
      {answer?.ok && total > 0 && (
        <>
          <div className="table-tools small">
            <span>{plural(total, "text")} {filtered ? "match the filter" : "in the depot"}</span>
            {ticked.size < total && total > answer.data.rows.length && (
              <Button variant="quiet" disabled={tickingAll} onClick={tickAll}>{tickingAll ? "Ticking" : `Tick all ${total.toLocaleString("en")}`}</Button>
            )}
          </div>
          <TextTable rows={answer.data.rows} libraries={libraries} ticked={ticked} onTick={tick} busy={pending} order={order} onOrder={setOrder} label="The texts the filter matches" />
          <Pager offset={offset} limit={limit} total={total} onOffset={setOffset} onLimit={setLimit} label="Pages of the texts" />
        </>
      )}
      {ticked.size > 0 && (
        <div className="selection-bar" role="region" aria-label="The ticked texts">
          <strong className="cue">{plural(ticked.size, "text")} ticked</strong>
          <Button variant="primary" onClick={() => { setDone(null); setAction("admitted"); }}>Admit</Button>
          <Button variant="quiet" onClick={() => { setDone(null); setAction("refused"); }}>Refuse</Button>
          <Button variant="quiet" onClick={() => { setDone(null); setAction("put"); }}>Put in a library</Button>
          <Button variant="quiet" onClick={() => setTicked(new Set())}>Untick all</Button>
        </div>
      )}
      <Dialog open={action !== null} onClose={() => setAction(null)} title={action === "put" ? "Put the ticked texts in a library" : `${action === "admitted" ? "Admit" : "Refuse"} the ticked texts`}>
        {action && (
          <div className="wide">
            <Review filter={{ ...EVERY, ids: [...ticked] }} libraries={libraries} decision={action === "put" ? undefined : action}
              onChange={() => { reload(); reloadLibraries(); }}
              unticks={action === "put" ? [{ label: "Untick those not admitted", part: { statuses: ["under review", "refused"] } }]
                : []}>
              {(chosen) => (action === "put"
                ? <PutForm chosen={chosen} libraries={libraries} onDone={(a) => finished("put", a)} />
                : <DecideForm status={action} chosen={chosen} onDone={(a) => finished(action, a)} />)}
            </Review>
            {done && !done.answer.ok && <Outcome answer={done.answer} done={() => ""} />}
          </div>
        )}
      </Dialog>
    </>
  );
}

function said(action: Action, d: Done): string {
  return d.decision
    ? `Decision ${d.decision}: ${action}, on ${plural(d.texts!.length, "text")}.`
    : `${plural(d.joined!.length, "text")} joined the library; ${d.left_out!.length.toLocaleString("en")} left out, not admitted.`;
}

function PutForm({ chosen, libraries, onDone }: { chosen: string[]; libraries: Library[]; onDone: (a: Answer<Done>) => void }) {
  const open = libraries.filter((l) => !l.locked_at && !l.upstream);
  const [library, setLibrary] = useState(open[0]?.id ?? "");
  const put = async () => onDone(await post<Done>(`/api/libraries/${encodeURIComponent(library)}/texts`, { texts: chosen }));
  return (
    <div className="action-form">
      <RequiredNote />
      <div className="fields">
      <Select label={required("Library")} required value={library} onChange={(e) => setLibrary(e.target.value)} hint="Only this depot's own open libraries take texts; only admitted texts join.">
        {open.length === 0 && <option value="">No open library</option>}
        {open.map((l) => <option key={l.id} value={l.id}>{l.name} ({l.id})</option>)}
      </Select>
      </div>
      <Button variant="primary" disabled={!library || chosen.length === 0} onClick={put}>Put {plural(chosen.length, "text")} in it</Button>
    </div>
  );
}

export const texts: Page = {
  path: "texts", title: "Texts", group: "data",
  Component: ({ rest, query }) => (rest[0] ? <Text id={rest[0]} /> : <Texts query={query} />),
};
