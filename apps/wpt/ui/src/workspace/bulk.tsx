// What the workspace does with the texts ticked in its playlist: a bar at
// the bottom of the screen whose actions (admit, refuse, put in a library,
// and, when one library is open, take them out of it, report a problem on
// them or fork it without them) open the review of the ticked texts
// (review.tsx); the operation then names exactly the texts left ticked
// there. Taking out leaves the texts admitted.

import { Button, Dialog, Select } from "@workers-intelligence-commons/components";
import { useState } from "react";
import { post, type Answer } from "../api";
import { DecideForm, ReportForm } from "../decide-forms";
import { EVERY } from "../filter";
import { ForkForm } from "../pages/libraries";
import { Outcome, plural, required, RequiredNote, type Library } from "../parts";
import { Review } from "../review";

type Action = "admitted" | "refused" | "put" | "take-out" | "report" | "fork";
type Done = { decision?: string; texts?: string[]; joined?: string[]; left_out?: string[]; taken_out?: string[]; not_held?: string[] };

const TITLES: Record<Action, (library: string) => string> = {
  admitted: () => "Admit the ticked texts", refused: () => "Refuse the ticked texts", put: () => "Put the ticked texts in a library",
  "take-out": (l) => `Take the ticked texts out of ${l}`, report: (l) => `Report a problem with texts of ${l}`, fork: (l) => `Fork ${l} without these texts`,
};

// ticked: the ticked ids; library: the one library open in the workspace, if any; onDone: after a change, with what it did in words.
export function Bulk({ ticked, libraries, library, into, onUntick, onDone }: {
  ticked: Set<string>; libraries: Library[]; library: Library | null; into: string | null; onUntick: () => void; onDone: (words: string) => void;
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
  if (ticked.size === 0) return null;
  return (
    <>
      <div className="selection-bar" role="region" aria-label="The ticked texts">
        <strong className="cue">{plural(ticked.size, "text")} ticked</strong>
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
            <Review filter={{ ...EVERY, ids: [...ticked] }} libraries={libraries} decision={action === "admitted" || action === "refused" ? action : undefined}
              unticks={action === "put" ? [{ label: "Untick those not admitted", part: { statuses: ["under review", "refused"] } }] : []}>
              {(chosen) => action === "admitted" || action === "refused"
                ? <DecideForm status={action} chosen={chosen} onDone={(a) => finished(a, a.ok ? `Decision ${a.data.decision}: ${action}, on ${plural(a.data.texts.length, "text")}.` : "")} />
                : action === "put" ? <PutForm chosen={chosen} libraries={libraries} into={into} onDone={finished} />
                : action === "take-out" ? <TakeOutForm chosen={chosen} library={takes!} onDone={finished} />
                : action === "report" ? <ReportForm library={own!} chosen={chosen} onDone={finished} />
                : <ForkForm library={library!} without={chosen} onDone={finished} />}
            </Review>
            {failed && <Outcome answer={failed} done={() => ""} />}
          </div>
        )}
      </Dialog>
    </>
  );
}

function PutForm({ chosen, libraries, into, onDone }: { chosen: string[]; libraries: Library[]; into: string | null; onDone: (a: Answer<unknown>, words: string) => void }) {
  const open = libraries.filter((l) => !l.locked_at && !l.upstream);
  const [library, setLibrary] = useState(open.find((l) => l.id === into)?.id ?? open[0]?.id ?? "");
  const put = async () => {
    const a = await post<Done>(`/api/libraries/${encodeURIComponent(library)}/texts`, { texts: chosen });
    onDone(a, a.ok ? `${plural(a.data.joined!.length, "text")} joined the library; ${a.data.left_out!.length.toLocaleString("en")} left out, not admitted.` : "");
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
      <Button variant="primary" disabled={!library || chosen.length === 0} onClick={put}>Put {plural(chosen.length, "text")} in it</Button>
    </div>
  );
}

// Out of the open library, exactly the chosen texts; they keep their status.
function TakeOutForm({ chosen, library, onDone }: { chosen: string[]; library: Library; onDone: (a: Answer<unknown>, words: string) => void }) {
  const take = async () => {
    const a = await post<Done>(`/api/libraries/${encodeURIComponent(library.id)}/taken-out`, { texts: chosen });
    onDone(a, a.ok ? `${plural(a.data.taken_out!.length, "text")} taken out of ${library.name}, each keeping its status${a.data.not_held!.length ? `; ${a.data.not_held!.length.toLocaleString("en")} it did not hold` : ""}.` : "");
  };
  return (
    <div className="action-form">
      <p className="small">They leave {library.name} and keep their status and decisions: an admitted text stays admitted and can be put back. No declaration is needed: it changes the library, not the texts.</p>
      <Button variant="primary" disabled={chosen.length === 0} onClick={take}>Take {plural(chosen.length, "text")} out of it</Button>
    </div>
  );
}
