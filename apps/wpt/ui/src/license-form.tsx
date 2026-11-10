// The form of a new license: its id, its name and its official text, from
// a file or pasted, saved through POST /api/licenses (wpt depot license add).

import { Button, TextArea, TextInput } from "@workers-intelligence-commons/components";
import { useState } from "react";
import { post } from "./api";
import { optional, Problems, required, RequiredNote } from "./parts";

// onSaved gets the new license's id once the depot holds it.
export function LicenseForm({ initialId = "", onSaved }: { initialId?: string; onSaved: (id: string) => void }) {
  const [id, setId] = useState(initialId);
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [problems, setProblems] = useState<string[]>([]);
  const send = async () => {
    const a = await post("/api/licenses", { id: id.trim(), name: name.trim(), text });
    setProblems(a.ok ? [] : a.problems);
    if (a.ok) onSaved(id.trim());
  };
  return (
    <form onSubmit={(e) => { e.preventDefault(); send(); }}>
      <RequiredNote />
      <div className="fields">
        <TextInput label={required("Id")} required value={id} placeholder="CC-BY-SA-4.0" onChange={(e) => setId(e.target.value)}
          hint="An SPDX name, a license with its exception, or licenses that apply together ('MIT AND CC-BY-4.0')." />
        <TextInput label={required("Name")} required value={name} placeholder="Creative Commons Attribution Share Alike 4.0 International" onChange={(e) => setName(e.target.value)} />
      </div>
      <div className="wic-field">
        <label className="wic-label" htmlFor="license-file">{optional("Official text from a file")}</label>
        <input id="license-file" className="wic-focus" type="file" accept="text/plain,.txt,.md"
          onChange={async (e) => { const file = e.target.files?.[0]; if (file) setText(await file.text()); }} />
      </div>
      <TextArea label={required("Official text")} required rows={8} value={text} onChange={(e) => setText(e.target.value)} hint="Filled from the file, or pasted. For combined licenses, every part's text." />
      <Problems problems={problems} />
      <Button type="submit" variant="primary" disabled={!id.trim() || !name.trim() || !text.trim()}>Save the license</Button>
    </form>
  );
}
