// A text's scores (docs/data-selection.md, "Scores"): each a name, a number
// and the content version it was made from. The "Scores" tab of the open
// text lists them, changes or removes them, and adds one under a name the
// depot knows or a new one, all saved at once through POST
// /api/texts/scores with the decider's declaration: a decision under
// review, the text keeping its status. A score set here is
// made from the text's current content version. The bulk form sets one
// score on, or removes it from, the texts of a selection (POST
// /api/texts/score), as a job when the selection is large. The names the
// depot knows (GET /api/scores) come from its counts, asked once a page,
// when a name box is first entered.

import { Button, Select, TextInput } from "@workers-intelligence-commons/components";
import { useEffect, useId, useState, type ReactNode } from "react";
import { get, post, type Answer } from "./api";
import { leftOut, useWrite, WriteProgress, type Left } from "./decide-forms";
import { sent, type Selection } from "./filter";
import { Confirm, Help, Outcome, plural, required, RequiredNote, type Declaration } from "./parts";
import { href } from "./route";

export type Score = { name: string; value: number; content_version: number };
type Named = { name: string; texts: number };

// The names of the scores texts have, with how many have each: asked once per page, its answer (a failure
// included) kept for every later box.
let namesRead: Promise<Answer<Named[]>> | null = null;

// The names, null until a box asked and while read or when the read failed: a name is then typed. load asks.
function useScoreNames(): { names: Named[] | null; problems: string[] | null; load: () => void } {
  const [answer, setAnswer] = useState<Answer<Named[]> | null>(null);
  const [asked, setAsked] = useState(false);
  useEffect(() => {
    if (!asked) return;
    let live = true;
    namesRead ??= get<Named[]>("/api/scores");
    namesRead.then((a) => { if (live) setAnswer(a); });
    return () => { live = false; };
  }, [asked]);
  return { names: answer?.ok ? answer.data : null, problems: answer && !answer.ok ? answer.problems : null, load: () => setAsked(true) };
}

// A finite number typed, or null.
const numberOf = (typed: string) => (typed.trim() !== "" && Number.isFinite(Number(typed)) ? Number(typed) : null);

// The box of a score's name, which offers the names the depot knows (most used first) and takes a new one.
// optional: the box may stay empty. onFocus: asks for the names the depot knows.
function NameBox({ value, onChange, names, onFocus, optional, hint }: {
  value: string; onChange: (name: string) => void; names: Named[] | null; onFocus: () => void; optional?: boolean; hint?: ReactNode;
}) {
  const list = useId();
  return (
    <>
      <TextInput label={optional ? "Name" : required("Name")} required={!optional} value={value} onChange={(e) => onChange(e.target.value)} onFocus={onFocus}
        list={list} autoComplete="off" hint={hint} />
      <datalist id={list}>
        {[...(names ?? [])].sort((a, b) => b.texts - a.texts || a.name.localeCompare(b.name)).map((n) => (
          <option key={n.name} value={n.name}>{plural(n.texts, "text")}</option>
        ))}
      </datalist>
    </>
  );
}

// The "Scores" tab of one text. readOnly: an upstream text, which only its upstream changes.
export function TextScores({ id, scores, contentVersion, readOnly, onDone }: {
  id: string; scores: Score[]; contentVersion: number; readOnly: boolean; onDone: (words: ReactNode) => void;
}) {
  const { names, problems, load } = useScoreNames();
  const [values, setValues] = useState<Record<string, string>>(() => Object.fromEntries(scores.map((s) => [s.name, String(s.value)])));
  const [removed, setRemoved] = useState<Set<string>>(new Set());
  const [added, setAdded] = useState<{ name: string; value: string }[]>([]);
  const [name, setName] = useState("");
  const [value, setValue] = useState("");
  const [answer, setAnswer] = useState<Answer<{ decision: string; reason: string }> | null>(null);
  const held = new Set([...scores.map((s) => s.name), ...added.map((a) => a.name)]);
  const changed = scores.filter((s) => !removed.has(s.name) && numberOf(values[s.name]) !== null && numberOf(values[s.name]) !== s.value);
  const invalid = [...scores.filter((s) => !removed.has(s.name) && numberOf(values[s.name]) === null).map((s) => s.name),
    ...added.filter((a) => numberOf(a.value) === null).map((a) => a.name)];
  const newName = name.trim();
  // a new name and a number typed in the add line count without pressing "Add"
  const typed = newName && !held.has(newName) && numberOf(value) !== null ? [{ name: newName, value }] : [];
  const set = [...changed.map((s) => ({ name: s.name, value: numberOf(values[s.name])!, content_version: contentVersion })),
    ...[...added, ...typed].filter((a) => numberOf(a.value) !== null).map((a) => ({ name: a.name, value: numberOf(a.value)!, content_version: contentVersion }))];
  const remove = [...removed];
  const nothing = set.length === 0 && remove.length === 0;
  const add = () => {
    if (!newName || held.has(newName) || numberOf(value) === null) return;
    setAdded([...added, { name: newName, value }]);
    setName("");
    setValue("");
  };
  const send = async (declaration: Declaration) => {
    const a = await post<{ decision: string; reason: string }>("/api/texts/scores", { texts: [{ id, scores: set, remove }], declaration });
    setAnswer(a);
    if (a.ok) onDone(<>Decision <a href={href(["decisions", a.data.decision])}>{a.data.decision}</a>, under review: {a.data.reason}.</>);
  };
  return (
    <div className="text-scores small">
      {scores.length === 0 && added.length === 0 && <p className="aside">No score yet.</p>}
      {(scores.length > 0 || added.length > 0) && (
        <table className="ledger-table scores-table">
          <thead><tr><th scope="col">name</th><th scope="col">value</th><th scope="col">made from</th>{!readOnly && <th scope="col"><span className="visually-hidden">remove</span></th>}</tr></thead>
          <tbody>
            {scores.map((s) => (
              <tr key={s.name} className={removed.has(s.name) ? "removed" : undefined}>
                <th scope="row" className="ledger">{s.name}</th>
                <td>{readOnly ? <span className="ledger">{s.value}</span>
                  : <input className="wic-input wic-focus score-value" type="number" step="any" aria-label={`The value of ${s.name}`} disabled={removed.has(s.name)}
                      value={values[s.name] ?? ""} onChange={(e) => setValues({ ...values, [s.name]: e.target.value })} />}</td>
                <td className="ledger">version {s.content_version}{s.content_version < contentVersion && <span className="aside">, an older one</span>}</td>
                {!readOnly && <td><label className="tick"><input type="checkbox" className="wic-check-box wic-focus" checked={removed.has(s.name)}
                  aria-label={`Remove ${s.name}`} onChange={(e) => { const next = new Set(removed); if (e.target.checked) next.add(s.name); else next.delete(s.name); setRemoved(next); }} /> <span>Remove</span></label></td>}
              </tr>
            ))}
            {added.map((a, at) => (
              <tr key={a.name} className="added">
                <th scope="row" className="ledger">{a.name} <span className="aside">new</span></th>
                <td><input className="wic-input wic-focus score-value" type="number" step="any" aria-label={`The value of ${a.name}`}
                  value={a.value} onChange={(e) => setAdded(added.map((one, i) => (i === at ? { ...one, value: e.target.value } : one)))} /></td>
                <td className="ledger">version {contentVersion}</td>
                <td><Button variant="quiet" onClick={() => setAdded(added.filter((_, i) => i !== at))}>Drop</Button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {!readOnly && <>
        <form className="score-add" onSubmit={(e) => { e.preventDefault(); add(); }} aria-label="Add a score">
          <NameBox value={name} onChange={setName} names={names} onFocus={load} optional
            hint={held.has(newName) && newName ? "It has this score already: change its value above." : undefined} />
          <TextInput label="Value" type="number" step="any" value={value} onChange={(e) => setValue(e.target.value)} />
          <Button type="submit" variant="quiet" disabled={!newName || held.has(newName) || numberOf(value) === null}>Add</Button>
        </form>
        {problems && <p className="aside">The known names could not be read ({problems.join(" ")}): type the name.</p>}
        <Confirm label="Confirm and save the scores" disabled={nothing || invalid.length > 0} onConfirm={send}
          note={invalid.length > 0 ? `A value is no number: ${invalid.join(", ")}.` : nothing ? "Nothing changed yet."
            : <>{set.length > 0 && <>Sets {set.map((s) => s.name).join(", ")}. </>}{remove.length > 0 && <>Removes {remove.join(", ")}. </>}
              <Help label="saving scores">Saved as a decision under review whose reason names the scores set and removed. The text keeps its status, and a snapshot made before keeps the revision it holds. A score set here is made from content version {contentVersion}, the text as it is now.</Help></>} />
        <Outcome answer={answer} done={() => ""} />
      </>}
    </div>
  );
}

type Scored = { decision: string | null; changed: number; left_out: number; left: Left };

// One score set on, or removed from, the texts of a selection, each made from its current content version.
export function ScoreForm({ selection, size, onDone }: { selection: Selection; size: number | null; onDone: (a: Answer<unknown>, words: ReactNode) => void }) {
  const { names, load } = useScoreNames();
  const [name, setName] = useState("");
  const [action, setAction] = useState<"set" | "remove">("set");
  const [value, setValue] = useState("");
  const { send, job, busy } = useWrite();
  const given = name.trim();
  const ready = given !== "" && (action === "remove" || numberOf(value) !== null) && size !== 0 && !busy;
  const them = size === null ? "the ticked texts" : plural(size, "ticked text");
  const go = async (declaration: Declaration) => {
    const a = await send<Scored>("/api/texts/score", { selection: sent(selection), name: given, ...(action === "remove" ? { remove: true } : { value: numberOf(value) }), declaration });
    onDone(a, a.ok ? a.data.decision === null
      ? <>Nothing changed: {plural(a.data.left_out, "text")} {action === "remove" ? "had no score" : "had this score"} {given}{action === "set" ? ` at ${value}` : ""} already.{leftOut(a.data.left)}</>
      : <>{action === "remove" ? `Removed ${given} from` : `Set ${given} to ${numberOf(value)} on`} {plural(a.data.changed, "text")}{a.data.left_out ? `; ${a.data.left_out.toLocaleString("en")} unchanged` : ""}.{leftOut(a.data.left)}{" "}
        <a href={href(["decisions", a.data.decision])}>See the decision</a></> : "");
  };
  return (
    <div className="action-form">
      <RequiredNote />
      <div className="fields">
        <NameBox value={name} onChange={setName} names={names} onFocus={load} />
        <Select label={required("Change")} required value={action} onChange={(e) => setAction(e.target.value as "set" | "remove")}>
          <option value="set">Set it</option>
          <option value="remove">Remove it</option>
        </Select>
        {action === "set" && <TextInput label={required("Value")} required type="number" step="any" value={value} onChange={(e) => setValue(e.target.value)} />}
      </div>
      <Confirm label={action === "remove" ? "Confirm and remove the score" : "Confirm and set the score"} disabled={!ready} onConfirm={go}
        note={<>{action === "remove" ? `Removes ${given || "the score"} from ${them}.` : `Sets ${given || "the score"} to ${numberOf(value) ?? "a value"} on ${them}, made from each one's current content version.`}
          <Help label="setting a score">One decision under review on every text it changes, whose reason names the score. The texts keep their status, and a snapshot made before keeps the revision it holds. A text whose scores stay as they are is left out.</Help></>} />
      <WriteProgress job={job} />
    </div>
  );
}
