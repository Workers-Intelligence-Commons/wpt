// What the Texts page does with the texts ticked in its list: a bar at
// the bottom of the screen whose actions (admit, refuse, put back under
// review, which undoes an admission or a refusal, set or remove one score
// (scores.tsx), add tags and remove tags (retag.tsx)) open the review of the
// ticked texts (review.tsx). The ticked texts are a selection (filter.tsx):
// those ticked one by one, or every text the filter matches less those
// unticked; the operation sends it, runs as a job on the server and shows
// its stage while it runs. A change of tags shows its cost before its
// confirm. A hand-picked set of texts is a tag: tick them, then "Add a tag".

import { Button, Dialog } from "@workers-intelligence-commons/components";
import { useState, type ReactNode } from "react";
import { type Answer } from "../api";
import type { Deciding } from "../blocked";
import { DecideForm, leftOut } from "../decide-forms";
import { type Selection } from "../filter";
import { matched, useCounts } from "../counts";
import { Outcome, plural } from "../parts";
import { href } from "../route";
import { Review } from "../review";
import { ScoreForm } from "../scores";
import { RetagForm } from "./retag";

type Action = Deciding | "score" | "tag" | "untag";
const decides = (a: Action | null): a is Deciding => a === "admitted" || a === "refused" || a === "under review";
const DONE: Record<Deciding, string> = { admitted: "Admitted", refused: "Refused", "under review": "Put back under review" };

const TITLES: Record<Action, string> = {
  admitted: "Admit the ticked texts", refused: "Refuse the ticked texts", "under review": "Put the ticked texts back under review",
  score: "Set or remove a score on the ticked texts", tag: "Add tags to the ticked texts", untag: "Remove tags from the ticked texts",
};

// selection: the ticked texts, null when none is; count: how many, when
// known; onDone: after a change, with what it did in words.
// admittedHere: the address of the open filter's admitted texts.
export function Bulk({ selection, count, admittedHere, onUntick, onDone }: {
  selection: Selection | null; count: number | null; admittedHere: string; onUntick: () => void; onDone: (words: ReactNode) => void;
}) {
  const [action, setAction] = useState<Action | null>(null);
  const [failed, setFailed] = useState<Answer<unknown> | null>(null);
  const finished = (a: Answer<unknown>, words: ReactNode) => {
    setFailed(a.ok ? null : a);
    if (a.ok) { setAction(null); onDone(words); }
  };
  const start = (a: Action) => () => { setFailed(null); setAction(a); };
  // the main button follows what is ticked: once every ticked text is admitted, admitting them does nothing
  const counting = useCounts(selection);
  const allAdmitted = Boolean((counting?.done || counting?.settled) && !counting.problems && matched(counting) > 0 && counting.statuses.admitted === matched(counting));
  // and once every ticked text is under review, putting them back under review does nothing
  const allUnderReview = Boolean((counting?.done || counting?.settled) && !counting.problems && matched(counting) > 0 && counting.statuses["under review"] === matched(counting));
  if (selection === null) return null;
  return (
    <>
      <div className="selection-bar" role="region" aria-label="The ticked texts">
        <strong className="cue">{count === null ? "Every text ticked" : `${plural(count, "text")} ticked`}{allAdmitted && <span className="aside">, every one admitted</span>}</strong>
        {!allAdmitted && <Button variant="primary" onClick={start("admitted")}>Admit</Button>}
        <Button variant="quiet" onClick={start("refused")}>Refuse</Button>
        {!allUnderReview && <Button variant="quiet" onClick={start("under review")} title="Undo their admission or refusal: a public decision">Back to review</Button>}
        <Button variant={allAdmitted ? "primary" : "quiet"} onClick={start("tag")} title="Add one or more tags: a hand-picked set of texts is a tag">Add a tag</Button>
        <Button variant="quiet" onClick={start("untag")}>Remove a tag</Button>
        <Button variant="quiet" onClick={start("score")}>Score</Button>
        <Button variant="quiet" onClick={onUntick}>Untick all</Button>
      </div>
      <Dialog open={action !== null} onClose={() => setAction(null)} title={action ? TITLES[action] : ""}>
        {action && (
          <div className="wide">
            <Review selection={selection} decision={decides(action) ? action : undefined}>
              {(chosen, size, already) => decides(action)
                ? <DecideForm status={action} selection={chosen} size={size} already={already} leaveBlocked leaveSame
                    onDone={(a) => finished(a, a.ok ? <>{DONE[action]} {plural(a.data.texts, "text")}.{leftOut(a.data.left)}{" "}
                      {action === "admitted" && <><a href={admittedHere}>Show the admitted texts of this filter</a>, from where they can be ticked and tagged. </>}
                      <a href={href(["decisions", a.data.decision])}>See the decision</a></> : "")} />
                : action === "score" ? <ScoreForm selection={chosen} size={size} onDone={finished} />
                : <RetagForm selection={chosen} size={size} removing={action === "untag"} onDone={finished} />}
            </Review>
            {failed && <Outcome answer={failed} done={() => ""} />}
          </div>
        )}
      </Dialog>
    </>
  );
}
