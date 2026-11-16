// The forms that send a decision on texts: admitting or refusing the texts
// of a selection, or putting them back under review, which undoes their
// admission or refusal (POST /api/decisions), changing their licenses, which the
// depot records as a decision under review after which their decisions
// start over (POST /api/texts/licenses), and reporting a problem on texts
// of a library or on the library as a whole (POST /api/libraries/ID/problems).
// Each sends a selection (filter.tsx) and follows the write's job to its
// end, showing its stage while it runs (WriteProgress). size: how many texts
// the selection takes, when counted.
// variant: the confirm button's, strong (primary) unless the form sits beside
// the main action of its view.

import { TextArea, TextInput, type ButtonVariant } from "@workers-intelligence-commons/components";
import { useState } from "react";
import { write, type Answer, type Job } from "./api";
import type { Deciding } from "./blocked";
import { sent, type Selection } from "./filter";
import { LicensePicker } from "./license-picker";
import { JobProgress, JobState, useNow } from "./pages/jobs";
import { Confirm, optional, plural, required, RequiredNote, type Declaration, type Library } from "./parts";

// left: how many texts a write left out, by reason ("changed since": changed after the page was read)
export type Left = Record<string, number>;
export type Decided = { decision: string; texts: number; left: Left };
// one decision for each distinct list of licenses the texts had; skipped: how
// many named these licenses already, left out
export type Relicensed = { decisions: { decision: string; texts: number }[]; skipped: number; left: Left };

// What a write left out, in words, or nothing.
export const leftOut = (left: Left) => {
  const parts = Object.entries(left).filter(([, n]) => n > 0).map(([reason, n]) => `${n.toLocaleString("en")} ${reason}`);
  return parts.length ? ` Left out: ${parts.join(", ")}.` : "";
};

// What a license change did, in words.
export const relicensed = (d: Relicensed, licenses: string[]) =>
  (d.decisions.length === 0 ? "No license changed."
    : `Licenses changed to ${licenses.join(" or ")} on ${plural(d.decisions.reduce((n, one) => n + one.texts, 0), "text")}, `
      + `recorded by ${d.decisions.length === 1 ? "decision" : "decisions"} ${d.decisions.map((one) => one.decision).join(", ")}; under review again.`)
  + (d.skipped ? ` ${plural(d.skipped, "text")} left out, naming these licenses already.` : "") + leftOut(d.left);

// Where a running write stands: its state with how long it waited or ran, and its progress.
export function WriteProgress({ job }: { job: Job<unknown> | null }) {
  const now = useNow(job !== null);
  if (!job) return null;
  return <p className="small aside" role="status">The write: <JobState job={job} now={now} />{job.state === "running" && <>: <JobProgress job={job} /></>}.</p>;
}

// A write over the selection, followed to its end: send, the job while it runs, and whether one is running.
export function useWrite() {
  const [job, setJob] = useState<Job<unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const send = async <T,>(path: string, body: unknown): Promise<Answer<T>> => {
    setBusy(true);
    try {
      return await write<T>(path, body, (j) => setJob(j as Job<unknown>));
    } finally {
      setBusy(false);
      setJob(null);
    }
  };
  return { send, job, busy };
}

const CONFIRMS: Record<Deciding, string> = { admitted: "Confirm and admit", refused: "Confirm and refuse", "under review": "Confirm and put back under review" };

const ALREADY: Record<Deciding, string> = { admitted: "admitted", refused: "refused", "under review": "under review" };

// A decision on the texts of the selection; leaveBlocked: the texts it
// cannot cover (blocked.tsx) are left out of it rather than refusing it;
// leaveSame: an admission or a refusal leaves out the texts that have its
// status already, so it records no decision that changes nothing (a
// putting back under review always does); already: how many of the size
// have the decision's status, when counted.
export function DecideForm({ status, selection, size, already = null, leaveBlocked = false, leaveSame = false, onDone, variant }: {
  status: Deciding; selection: Selection; size: number | null; already?: number | null; leaveBlocked?: boolean; leaveSame?: boolean;
  onDone: (a: Answer<Decided>) => void; variant?: ButtonVariant;
}) {
  const [reason, setReason] = useState("");
  const { send, job, busy } = useWrite();
  const decide = async (declaration: Declaration) => onDone(await send<Decided>("/api/decisions", {
    status, selection: sent(selection), leave_blocked: leaveBlocked, ...(leaveSame ? { leave_same: true } : {}),
    ...(reason.trim() ? { reason: reason.trim() } : {}), declaration }));
  const skips = leaveSame || status === "under review"; // the texts of its status already are left out
  const known = skips && size !== null && already !== null && already > 0; // and how many is counted
  const decided = known ? size! - already! : size;
  const texts = size === null ? "the ticked texts" : known ? `${decided!.toLocaleString("en")} of the ${plural(size, "ticked text")}` : `the ${plural(size, "ticked text")}`;
  const apart = known ? <> {already!.toLocaleString("en")} {already === 1 ? "is" : "are"} {ALREADY[status]} already and {already === 1 ? "is" : "are"} left out.</> : null;
  const verb = status === "admitted" ? "Admits" : status === "refused" ? "Refuses" : "Puts back under review";
  return (
    <div className="action-form">
      <TextInput label={optional("Reason")} hint="Public, with the decision." value={reason} onChange={(e) => setReason(e.target.value)} />
      <Confirm label={CONFIRMS[status]} disabled={size === 0 || decided === 0 || busy} onConfirm={decide} variant={variant}
        note={decided === 0 && size !== 0
          ? <>Every ticked text is {ALREADY[status]} already: there is nothing to decide.</>
          : status === "under review"
          ? <>{verb} {texts}, undoing their admission or refusal: the decisions before it no longer count, and they leave every open library until admitted again; a locked library keeps the revision it pinned.{apart} {known ? "A text" : "A text under review already, or one"} changed since this page was read is left out.</>
          : <>{verb} {texts}{status === "refused" && ", taking them out of every open library; a locked library keeps the revision it pinned"}.{apart} A text {leaveSame && !known ? `${ALREADY[status]} already, or one ` : ""}changed since this page was read is left out.</>} />
      <WriteProgress job={job} />
    </div>
  );
}

// New licenses for the texts of the selection, chosen with the license picker.
// subject: the texts in words ("the 3 ticked texts", "this text").
export function RelicenseForm({ selection, subject, onDone, variant }: { selection: Selection; subject: string; onDone: (a: Answer<Relicensed>, licenses: string[]) => void; variant?: ButtonVariant }) {
  const [licenses, setLicenses] = useState<string[]>([]);
  const { send, job, busy } = useWrite();
  const change = async (declaration: Declaration) =>
    onDone(await send<Relicensed>("/api/texts/licenses", { selection: sent(selection), licenses, declaration }), licenses);
  return (
    <div className="action-form">
      <RequiredNote />
      <LicensePicker chosen={licenses} onChange={setLicenses} />
      <Confirm label="Confirm and change the licenses" variant={variant} disabled={licenses.length === 0 || busy} onConfirm={change}
        note={<>Changes the licenses of {subject} to {licenses.length ? licenses.join(" or ") : "no license yet"}. The decisions made before no longer count:
          under review again, they leave every open library until admitted anew; a locked library keeps the revision it pinned.</>} />
      <WriteProgress job={job} />
    </div>
  );
}

// A problem on the texts of the selection that the library holds, or on the whole library (selection null).
export function ReportForm({ library, selection, size, onDone, variant }: {
  library: Library; selection: Selection | null; size?: number | null; onDone: (a: Answer<unknown>, words: string) => void; variant?: ButtonVariant;
}) {
  const [problem, setProblem] = useState("");
  const { send, job, busy } = useWrite();
  const report = async (declaration: Declaration) => {
    const a = await send<{ decision: string; texts: number; left: Left }>(`/api/libraries/${encodeURIComponent(library.id)}/problems`,
      { problem, ...(selection ? { selection: sent(selection) } : {}), declaration });
    onDone(a, a.ok ? `Problem reported, recorded by decision ${a.data.decision}${a.data.texts ? `, on ${plural(a.data.texts, "text")}` : ""}.${leftOut(a.data.left)}` : "");
  };
  return (
    <div className="action-form">
      <RequiredNote />
      <TextArea label={required("The problem")} required value={problem} onChange={(e) => setProblem(e.target.value)} hint="Public. It changes no text and no membership." />
      <Confirm label="Confirm and report" variant={variant} disabled={!problem.trim() || size === 0 || busy} onConfirm={report}
        note={selection ? <>Reports it on the {size == null ? "ticked texts" : plural(size, "ticked text")}.</> : <>Reports it on the library as a whole.</>} />
      <WriteProgress job={job} />
    </div>
  );
}
