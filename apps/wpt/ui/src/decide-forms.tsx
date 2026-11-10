// The forms that send a decision on texts: admitting or refusing exactly
// the chosen texts (POST /api/decisions), changing their licenses, which the
// depot records as a decision under review after which their decisions
// start over (POST /api/texts/licenses), and reporting a problem on texts
// of a library or on the library as a whole (POST /api/libraries/ID/problems).
// variant: the confirm button's, strong (primary) unless the form sits beside
// the main action of its view.

import { TextArea, TextInput, type ButtonVariant } from "@workers-intelligence-commons/components";
import { useState } from "react";
import { post, type Answer } from "./api";
import { LicensePicker } from "./license-picker";
import { Confirm, optional, plural, required, RequiredNote, type Declaration, type Library } from "./parts";

export type Decided = { decision: string; texts: string[] };
// one decision for each distinct list of licenses the texts had; skipped: the texts
// that named these licenses already, left out
export type Relicensed = { decisions: Decided[]; skipped: string[] };

// What a license change did, in words.
export const relicensed = (d: Relicensed, licenses: string[]) =>
  (d.decisions.length === 0 ? "No license changed."
    : `Licenses changed to ${licenses.join(" or ")} on ${plural(d.decisions.reduce((n, one) => n + one.texts.length, 0), "text")}, `
      + `recorded by ${d.decisions.length === 1 ? "decision" : "decisions"} ${d.decisions.map((one) => one.decision).join(", ")}; under review again.`)
  + (d.skipped.length ? ` Left out, naming these licenses already: ${d.skipped.join(", ")}.` : "");

// A decision on exactly the chosen texts, as a new revision of each; a
// locked library holding one keeps reading the revision it pinned.
export function DecideForm({ status, chosen, onDone, variant }: { status: "admitted" | "refused"; chosen: string[]; onDone: (a: Answer<Decided>) => void; variant?: ButtonVariant }) {
  const [reason, setReason] = useState("");
  const send = async (declaration: Declaration) =>
    onDone(await post<Decided>("/api/decisions", { status, texts: chosen, ...(reason.trim() ? { reason: reason.trim() } : {}), declaration }));
  return (
    <div className="action-form">
      <TextInput label={optional("Reason")} hint="Public, with the decision." value={reason} onChange={(e) => setReason(e.target.value)} />
      <Confirm label={status === "admitted" ? "Confirm and admit" : "Confirm and refuse"} disabled={chosen.length === 0} onConfirm={send} variant={variant}
        note={<>{status === "admitted" ? "Admits" : "Refuses"} the {plural(chosen.length, "ticked text")}{status === "refused" && ", taking them out of every open library; a locked library keeps the revision it pinned"}.</>} />
    </div>
  );
}

// New licenses for exactly the chosen texts, chosen with the license picker.
// subject: the texts in words ("the 3 ticked texts", "this text").
export function RelicenseForm({ chosen, subject, onDone, variant }: { chosen: string[]; subject: string; onDone: (a: Answer<Relicensed>, licenses: string[]) => void; variant?: ButtonVariant }) {
  const [licenses, setLicenses] = useState<string[]>([]);
  const send = async (declaration: Declaration) =>
    onDone(await post<Relicensed>("/api/texts/licenses", { texts: chosen, licenses, declaration }), licenses);
  return (
    <div className="action-form">
      <RequiredNote />
      <LicensePicker chosen={licenses} onChange={setLicenses} />
      <Confirm label="Confirm and change the licenses" variant={variant} disabled={chosen.length === 0 || licenses.length === 0} onConfirm={send}
        note={<>Changes the licenses of {subject} to {licenses.length ? licenses.join(" or ") : "no license yet"}. The decisions made before no longer count:
          under review again, {chosen.length === 1 ? "it leaves" : "they leave"} every open library until admitted anew; a locked library keeps the revision it pinned.</>} />
    </div>
  );
}

// A problem on exactly the chosen texts of the library, or on the whole library (chosen null).
export function ReportForm({ library, chosen, onDone, variant }: { library: Library; chosen: string[] | null; onDone: (a: Answer<unknown>, words: string) => void; variant?: ButtonVariant }) {
  const [problem, setProblem] = useState("");
  const send = async (declaration: Declaration) => {
    const a = await post<{ decision: string; texts: string[] }>(`/api/libraries/${encodeURIComponent(library.id)}/problems`,
      { problem, ...(chosen ? { texts: chosen } : {}), declaration });
    onDone(a, a.ok ? `Problem reported, recorded by decision ${a.data.decision}${a.data.texts.length ? `, on ${plural(a.data.texts.length, "text")}` : ""}.` : "");
  };
  return (
    <div className="action-form">
      <RequiredNote />
      <TextArea label={required("The problem")} required value={problem} onChange={(e) => setProblem(e.target.value)} hint="Public. It changes no text and no membership." />
      <Confirm label="Confirm and report" variant={variant} disabled={!problem.trim() || (chosen !== null && chosen.length === 0)} onConfirm={send}
        note={chosen ? <>Reports it on the {plural(chosen.length, "ticked text")}.</> : <>Reports it on the library as a whole.</>} />
    </div>
  );
}
