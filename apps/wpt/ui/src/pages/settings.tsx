// Who decides from this browser: the legal name and email every decision
// form sends, kept in this browser's localStorage only.

import { Button, Notice, TextInput } from "@workers-intelligence-commons/components";
import { useState } from "react";
import { readDeclarant, saveDeclarant } from "../declarant";
import { PageHead, required, RequiredNote } from "../parts";
import { href } from "../route";
import type { Page } from "./index";

function Settings() {
  const [who, setWho] = useState(readDeclarant);
  const [kept, setKept] = useState<boolean | null>(null);
  const [saved, setSaved] = useState(readDeclarant);
  const changed = who.name.trim() !== saved.name || who.email.trim() !== saved.email;
  const save = () => {
    const trimmed = { name: who.name.trim(), email: who.email.trim() };
    const ok = saveDeclarant(trimmed);
    setKept(ok);
    if (ok) { setSaved(trimmed); setWho(trimmed); }
  };
  return (
    <>
      <PageHead title="Settings"
        chain={<>Every decision on <a href={href(["texts"])}>texts</a> and <a href={href(["erasures"])}>erasure requests</a>, and every problem reported to a <a href={href(["libraries"])}>library</a>, records this name and email. Both are public and nobody verifies them.</>} />
      <form className="form-small" onSubmit={(e) => { e.preventDefault(); save(); }}>
        <RequiredNote />
        <p className="form-note small">Both stay in this browser only, and are never sent until you confirm a decision.</p>
        <TextInput label={required("Legal name")} required value={who.name} autoComplete="name" onChange={(e) => setWho({ ...who, name: e.target.value })} />
        <TextInput label={required("Email")} required type="email" value={who.email} autoComplete="email" onChange={(e) => setWho({ ...who, email: e.target.value })} />
        <Button type="submit" variant="primary" disabled={!changed}>Save them</Button>
      </form>
      {kept === true && <p className="done" role="status">Saved in this browser.</p>}
      {kept === false && <Notice title="not kept">This browser refuses to keep them (its storage is off or full), so the decision forms cannot use them.</Notice>}
    </>
  );
}

export const settings: Page = { path: "settings", title: "Settings", group: "setup", Component: Settings };
