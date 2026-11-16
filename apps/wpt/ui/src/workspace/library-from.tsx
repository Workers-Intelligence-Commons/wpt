// A new library holding the admitted texts of the open filter, in one
// dialog: its name, then the library is made (POST /api/libraries) and the
// texts the filter matches go into it as a write over a selection
// (POST /api/libraries/ID/texts), at the change number the playlist's page
// was read at. Only admitted texts join, so the button names how many of
// the open texts are admitted; it stands there while they are counted, and
// when none is, unable to act, so the workspace under it never moves.

import { Button, Dialog, TextInput } from "@workers-intelligence-commons/components";
import { useState, type ReactNode } from "react";
import { post } from "../api";
import { WriteProgress, leftOut, useWrite, type Left } from "../decide-forms";
import { sent, type TextFilter } from "../filter";
import { plural, Problems, required, RequiredNote } from "../parts";
import { href } from "../route";

export function LibraryFrom({ filter, change, admitted, onDone }: {
  filter: TextFilter; change: number | null; admitted: number | null; onDone: (words: ReactNode) => void; // null: being counted
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [problems, setProblems] = useState<string[] | null>(null);
  const [making, setMaking] = useState(false);
  const { send, job, busy } = useWrite();
  if (admitted === null || admitted === 0)
    return <Button variant="quiet" disabled title={admitted === null ? "The open texts are being counted." : "Only admitted texts join a library, and none of the open texts is admitted."}>
      {admitted === null ? "New library of the admitted texts (counting)" : "New library: no admitted text here"}</Button>;
  const make = async () => {
    setProblems(null);
    setMaking(true);
    const made = await post<{ id: string }>("/api/libraries", { name: name.trim() });
    if (!made.ok) { setMaking(false); setProblems(made.problems); return; }
    const id = made.data.id;
    const put = await send<{ joined: number; left_out: number; left: Left }>(`/api/libraries/${encodeURIComponent(id)}/texts`,
      { selection: sent({ filter, ...(change !== null ? { change } : {}) }) });
    setMaking(false);
    const opened = <a href={href(["texts"], { library: id })}>Open {name.trim()}</a>;
    if (!put.ok) {
      setProblems([`The library ${name.trim()} is made, but its texts were not put in it:`, ...put.problems]);
      onDone(<>The library {name.trim()} is made, empty: its texts were not put in it. {opened}.</>);
      return;
    }
    setOpen(false);
    setName("");
    onDone(<>Made the library {name.trim()} with {plural(put.data.joined, "text")}{put.data.left_out ? `; ${put.data.left_out.toLocaleString("en")} not admitted left out` : ""}.{leftOut(put.data.left)}{" "}
      {opened}, or <a href={href(["datasets"], { new: "1", library: id })}>make a dataset from it</a>.</>);
  };
  return (
    <>
      <Button variant="quiet" onClick={() => { setProblems(null); setOpen(true); }}>{admitted === 1 ? "New library of this admitted text" : `New library of these ${plural(admitted, "admitted text")}`}</Button>
      <Dialog open={open} onClose={() => !making && setOpen(false)} title={`A new library of ${plural(admitted, "admitted text")}`}>
        {open && (
          <form className="action-form" onSubmit={(e) => { e.preventDefault(); if (name.trim() && !making) make(); }}>
            <RequiredNote />
            <TextInput label={required("Name")} required autoFocus value={name} placeholder="first model, October 2026" onChange={(e) => setName(e.target.value)} />
            <p className="small act">Makes the library and puts in it the {plural(admitted, "admitted text")} this filter shows; a text not admitted stays out. It stays open: texts can be put in or taken out later, and nothing is decided.</p>
            <Button type="submit" variant="primary" disabled={!name.trim() || making || busy}>Make it with {plural(admitted, "text")}</Button>
            <WriteProgress job={job} />
            {problems && <Problems problems={problems} />}
          </form>
        )}
      </Dialog>
    </>
  );
}
