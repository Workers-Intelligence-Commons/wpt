// What the workspace does with the texts ticked in its playlist: a bar at
// the bottom of the screen whose actions (admit, refuse, put in a library,
// and, when one library is open, take them out of it, report a problem on
// them or fork it without them) open the review of the ticked texts
// (review.tsx). The ticked texts are a selection (filter.tsx): those ticked
// one by one, or every text the filter matches less those unticked; the
// operation sends it, runs as a job on the server and shows its stage while
// it runs. Taking out leaves the texts admitted.

import { Button, Dialog, Select } from "@workers-intelligence-commons/components";
import { useState } from "react";
import { type Answer } from "../api";
import { DecideForm, JobProgress, leftOut, ReportForm, useWrite, type Left } from "../decide-forms";
import { sent, type Selection } from "../filter";
import { ForkForm } from "../pages/libraries";
import { Outcome, plural, required, RequiredNote, type Library } from "../parts";
import { Review } from "../review";

type Action = "admitted" | "refused" | "put" | "take-out" | "report" | "fork";
type Put = { joined: number; left_out: number; left: Left };
type TakenOut = { taken_out: number; not_held: number; left: Left };

const TITLES: Record<Action, (library: string) => string> = {
  admitted: () => "Admit the ticked texts", refused: () => "Refuse the ticked texts", put: () => "Put the ticked texts in a library",
  "take-out": (l) => `Take the ticked texts out of ${l}`, report: (l) => `Report a problem with texts of ${l}`, fork: (l) => `Fork ${l} without these texts`,
};

// selection: the ticked texts, null when none is; count: how many, when
// known; library: the one library open in the workspace, if any; onDone:
// after a change, with what it did in words.
export function Bulk({ selection, count, libraries, library, into, onUntick, onDone }: {
  selection: Selection | null; count: number | null; libraries: Library[]; library: Library | null; into: string | null;
  onUntick: () => void; onDone: (words: string) => void;
}) {
  const [action, setAction] = useState<Action | null>(null);
  const [failed, setFailed] = useState<Answer<unknown> | null>(null);
  const finished = (a: Answer<unknown>, words: string) => {
    setFailed(a.ok ? null : a);
    if (a.ok) { setAction(null); onDone(words); }
  };
  const start = (a: Action) => () => { setFailed(null); setAction(a); };
  const own = library && !library.upstream ? library : null;
  const takes = own && !own.locked_at ? own : null;
  if (selection === null) return null;
  return (
    <>
      <div className="selection-bar" role="region" aria-label="The ticked texts">
        <strong className="cue">{count === null ? "Every text ticked" : `${plural(count, "text")} ticked`}</strong>
        <Button variant="primary" onClick={start("admitted")}>Admit</Button>
        <Button variant="quiet" onClick={start("refused")}>Refuse</Button>
        <Button variant="quiet" onClick={start("put")}>Put in a library</Button>
        {takes && <Button variant="quiet" onClick={start("take-out")}>Take out of {takes.name}</Button>}
        {own && <Button variant="quiet" onClick={start("report")}>Report a problem on them</Button>}
        {library && <Button variant="quiet" onClick={start("fork")}>Fork {library.name} without them</Button>}
        <Button variant="quiet" onClick={onUntick}>Untick all</Button>
      </div>
      <Dialog open={action !== null} onClose={() => setAction(null)} title={action ? TITLES[action](library?.name ?? "") : ""}>
        {action && (
          <div className="wide">
            <Review selection={selection} libraries={libraries} decision={action === "admitted" || action === "refused" ? action : undefined}>
              {(chosen, size) => action === "admitted" || action === "refused"
                ? <DecideForm status={action} selection={chosen} size={size} leaveBlocked
                    onDone={(a) => finished(a, a.ok ? `Decision ${a.data.decision}: ${action}, on ${plural(a.data.texts, "text")}.${leftOut(a.data.left)}` : "")} />
                : action === "put" ? <PutForm selection={chosen} size={size} libraries={libraries} into={into} onDone={finished} />
                : action === "take-out" ? <TakeOutForm selection={chosen} size={size} library={takes!} onDone={finished} />
                : action === "report" ? <ReportForm library={own!} selection={chosen} size={size} onDone={finished} />
                : <ForkForm library={library!} without={chosen} onDone={finished} />}
            </Review>
            {failed && <Outcome answer={failed} done={() => ""} />}
          </div>
        )}
      </Dialog>
    </>
  );
}

function PutForm({ selection, size, libraries, into, onDone }: {
  selection: Selection; size: number | null; libraries: Library[]; into: string | null; onDone: (a: Answer<unknown>, words: string) => void;
}) {
  const open = libraries.filter((l) => !l.locked_at && !l.upstream);
  const [library, setLibrary] = useState(open.find((l) => l.id === into)?.id ?? open[0]?.id ?? "");
  const { send, job, busy } = useWrite();
  const put = async () => {
    const a = await send<Put>(`/api/libraries/${encodeURIComponent(library)}/texts`, { selection: sent(selection) });
    onDone(a, a.ok ? `${plural(a.data.joined, "text")} joined the library; ${a.data.left_out.toLocaleString("en")} left out, not admitted.${leftOut(a.data.left)}` : "");
  };
  return (
    <div className="action-form">
      <RequiredNote />
      <div className="fields">
        <Select label={required("Library")} required value={library} onChange={(e) => setLibrary(e.target.value)} hint="Only this depot's own open libraries take texts; only admitted texts join.">
          {open.length === 0 && <option value="">No open library</option>}
          {open.map((l) => <option key={l.id} value={l.id}>{l.name} ({l.id})</option>)}
        </Select>
      </div>
      <Button variant="primary" disabled={!library || size === 0 || busy} onClick={put}>Put {size === null ? "them" : plural(size, "text")} in it</Button>
      <JobProgress job={job} />
    </div>
  );
}

// Out of the open library, the texts of the selection; they keep their status.
function TakeOutForm({ selection, size, library, onDone }: { selection: Selection; size: number | null; library: Library; onDone: (a: Answer<unknown>, words: string) => void }) {
  const { send, job, busy } = useWrite();
  const take = async () => {
    const a = await send<TakenOut>(`/api/libraries/${encodeURIComponent(library.id)}/taken-out`, { selection: sent(selection) });
    onDone(a, a.ok ? `${plural(a.data.taken_out, "text")} taken out of ${library.name}, each keeping its status${a.data.not_held ? `; ${a.data.not_held.toLocaleString("en")} it did not hold` : ""}.${leftOut(a.data.left)}` : "");
  };
  return (
    <div className="action-form">
      <p className="small">They leave {library.name} and keep their status and decisions: an admitted text stays admitted and can be put back. No declaration is needed: it changes the library, not the texts.</p>
      <Button variant="primary" disabled={size === 0 || busy} onClick={take}>Take {size === null ? "them" : plural(size, "text")} out of it</Button>
      <JobProgress job={job} />
    </div>
  );
}
