// One click that refuses one text, and one click that undoes it: the
// refusal of a list row (the strips read whole, the size map's biggest
// texts) and the Undo of the Texts page's refusal line. A refusal is a public
// decision under the declaration this browser keeps (declarant.ts), sent at
// once with no question (POST /api/decisions, its job followed); Undo is the
// decision that puts the text back under review, sent the same way, so a
// stray click costs one more click and no typing. Without a name and email,
// the button says where to set them and decides nothing.

import { useState } from "react";
import { write, type Answer } from "../api";
import { declared, readDeclarant } from "../declarant";
import type { Decided } from "../decide-forms";
import { named, sent } from "../filter";
import { settingsHref } from "../parts";
import { href } from "../route";

type To = "refused" | "under review";

// One decision on one text, as the Texts page sends it.
export function decideOne(id: string, status: To): Promise<Answer<Decided>> {
  const who = readDeclarant();
  return write<Decided>("/api/decisions", { status, selection: sent(named([id])),
    declaration: { name: who.name.trim(), email: who.email.trim(), accepted: true } });
}

// The refusal of one row: "Refuse", then "Refusing…", then "Refused" with the decision and Undo, then "Undoing…",
// then "Refuse" again; a refusal or an undo the depot refused says why beside it.
export function RefuseRow({ id, title, refused = false }: { id: string; title: string; refused?: boolean }) {
  const [state, setState] = useState<{ at: "idle" | "refusing" | "refused" | "undoing"; decision?: string; problem?: string }>(
    { at: refused ? "refused" : "idle" });
  const [noName, setNoName] = useState(false);
  const go = async (to: To) => {
    if (!declared(readDeclarant())) { setNoName(true); return; }
    setNoName(false);
    setState((s) => ({ ...s, at: to === "refused" ? "refusing" : "undoing", problem: undefined }));
    const a = await decideOne(id, to);
    if (a.ok) setState({ at: to === "refused" ? "refused" : "idle", decision: a.data.decision });
    else setState((s) => ({ ...s, at: to === "refused" ? "idle" : "refused", problem: a.problems.join(" ") }));
  };
  return (
    <span className="row-refuse small" role="group" aria-label={`Refuse ${title}`}>
      {state.at === "idle" && <button type="button" className="choice refuse-choice wic-focus" onClick={() => go("refused")}
        title={`Refuse ${title} at once, a public decision; Undo puts it back`}>Refuse</button>}
      {state.at === "refusing" && <span className="act" role="status">Refusing…</span>}
      {state.at === "undoing" && <span className="act" role="status">Undoing…</span>}
      {state.at === "refused" && <>
        <span className="refused-mark" role="status">Refused{state.decision && <> (<a href={href(["decisions", state.decision])}>decision</a>)</>}</span>
        {(state.decision || !refused) && <button type="button" className="choice wic-focus" onClick={() => go("under review")}
          title="Put it back under review, a public decision too">Undo</button>}
      </>}
      {noName && <span className="act">Set a name and email in <a href={settingsHref()}>Settings</a> first.</span>}
      {state.problem && <span className="act">{state.problem}</span>}
    </span>
  );
}
