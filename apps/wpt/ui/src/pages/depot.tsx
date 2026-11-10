// This depot: its about row and change number, a pull from the depot it
// follows (wpt depot pull), and the edit of its maintainer and takedown
// contact (wpt depot about); its name stays, since it begins every id. A new
// depot and a clone are made with the commands (wpt depot new, clone).

import { Button, Dialog, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, post, type Answer } from "../api";
import { forgetAbout, type About } from "../about";
import { Facts, Outcome, PageHead, Problems, required, RequiredNote } from "../parts";
import { href } from "../route";
import type { Page } from "./index";

type DepotFacts = About["about"];

function Depot() {
  const [about, setAbout] = useState<Answer<About> | null>(null);
  const [pulled, setPulled] = useState<Answer<{ erasures: string[]; change: number }> | null>(null);
  const [editing, setEditing] = useState(false);
  const [edited, setEdited] = useState<Answer<{ about: DepotFacts }> | null>(null);
  const load = useCallback(() => { get<About>("/api/about").then(setAbout); }, []);
  useEffect(load, [load]);
  const pull = async () => { setEdited(null); setPulled(await post("/api/pull", {})); load(); };
  const follows = about?.ok && about.data.about.upstream;
  return (
    <>
      <PageHead title="Depot"
        chain={<>Every <a href={href(["texts"])}>text</a>, <a href={href(["libraries"])}>library</a>, <a href={href(["datasets"])}>dataset</a> and <a href={href(["erasures"])}>erasure request</a> lives in this depot. <span className="ledger">wpt depot new</span> and <span className="ledger">wpt depot clone</span> make another.</>}
        actions={<>
          {follows && <Button variant="primary" onClick={pull}>Pull its updates</Button>}
          {about?.ok && <Button variant={follows ? "quiet" : "primary"} onClick={() => { setPulled(null); setEdited(null); setEditing(true); }}>Edit its facts</Button>}
        </>}
        note={follows ? "Fetches the depot it follows, and carries out the erasures it admitted." : undefined} />
      <Outcome answer={pulled} done={(d) => `Pulled; now at its change ${d.change}` + (d.erasures.length ? `; carried out the erasures ${d.erasures.join(", ")}.` : ".")} />
      {!editing && <Outcome answer={edited} done={() => "The depot's facts are saved."} />}
      {about && !about.ok && <Problems problems={about.problems} />}
      {about?.ok && (
        <Facts rows={[
          ["name", about.data.about.name],
          ["address", <span className="ledger">{about.data.address}</span>],
          ["change", <span className="ledger">{about.data.change}</span>],
          ["maintainer", about.data.about.maintainer],
          ["takedown contact", about.data.about.takedown_contact],
          ["follows", about.data.about.upstream ? <span className="ledger">{about.data.about.upstream}</span> : "no other depot"],
        ]} />
      )}
      <Dialog open={editing} onClose={() => setEditing(false)} title="Edit the depot's facts">
        {editing && about?.ok && <EditFacts facts={about.data.about} onDone={(a) => { setEdited(a); if (a.ok) { setEditing(false); forgetAbout(); load(); } }} />}
        {editing && edited && !edited.ok && <Problems problems={edited.problems} />}
      </Dialog>
    </>
  );
}

function EditFacts({ facts, onDone }: { facts: DepotFacts; onDone: (a: Answer<{ about: DepotFacts }>) => void }) {
  const [maintainer, setMaintainer] = useState(facts.maintainer);
  const [contact, setContact] = useState(facts.takedown_contact);
  const changed = maintainer.trim() !== facts.maintainer || contact.trim() !== facts.takedown_contact;
  const save = async () => onDone(await post<{ about: DepotFacts }>("/api/about", {
    ...(maintainer.trim() !== facts.maintainer ? { maintainer: maintainer.trim() } : {}),
    ...(contact.trim() !== facts.takedown_contact ? { takedown_contact: contact.trim() } : {}),
  }));
  return (
    <form className="action-form" onSubmit={(e) => { e.preventDefault(); save(); }}>
      <RequiredNote />
      <p className="form-note small">Its name, {facts.name}, stays: it begins every id of the depot.</p>
      <div className="fields">
        <TextInput label={required("Maintainer")} required value={maintainer} onChange={(e) => setMaintainer(e.target.value)} />
        <TextInput label={required("Takedown contact")} required value={contact} hint="Where to ask for an erasure." onChange={(e) => setContact(e.target.value)} />
      </div>
      <Button type="submit" variant="primary" disabled={!changed || !maintainer.trim() || !contact.trim()}>Save the facts</Button>
    </form>
  );
}

export const depot: Page = { path: "depot", title: "Depot", group: "setup", Component: Depot };
