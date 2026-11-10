// The licenses the depot holds: for each, whether the allowed list accepts
// it (a combined row only if every part is) and how many texts name it.
// Ticked licenses open the texts table filtered to them. A new license is
// added with its official text (wpt depot license add). #/licenses/ID is one
// license's view: its official text, whether it is allowed, and the texts
// that name it.

import { Badge, Button, ButtonLink, Dialog, Skeleton } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, part, type Answer } from "../api";
import { LicenseForm } from "../license-form";
import { EVERY } from "../filter";
import { PageHead, plural, Problems, Section } from "../parts";
import { MatchedTexts } from "../table";
import { href } from "../route";
import type { Page } from "./index";

export type License = { id: string; name: string; allowed: boolean; texts: number };

function Licenses() {
  const [rows, setRows] = useState<Answer<License[]> | null>(null);
  const [ticked, setTicked] = useState<string[]>([]);
  const [adding, setAdding] = useState(false);
  const [added, setAdded] = useState<string | null>(null);
  const load = useCallback(() => { get<License[]>("/api/licenses").then(setRows); }, []);
  useEffect(load, [load]);
  const flip = (id: string, on: boolean) => setTicked(on ? [...ticked, id] : ticked.filter((t) => t !== id));
  return (
    <>
      <PageHead title="Licenses"
        chain={<>Every <a href={href(["texts"])}>text</a> names its licenses; a text is admitted only under an allowed one. Tick licenses to see their texts.</>}
        actions={<Button variant="primary" onClick={() => setAdding(true)}>Add a license</Button>} />
      {added && <p className="done cue" role="status">License {added} added.</p>}
      {!rows && <Skeleton lines={4} label="Reading the licenses" />}
      {rows && !rows.ok && <Problems problems={rows.problems} />}
      {rows?.ok && (rows.data.length === 0 ? <p>The depot holds no license yet.</p> : (
        <div className="table-scroll">
          <table className="ledger-table" aria-label="The licenses">
            <thead><tr><th scope="col" className="tick-cell"><span className="visually-hidden">ticked</span></th><th scope="col">license</th><th scope="col">name</th><th scope="col">allowed</th><th scope="col" className="number">texts</th></tr></thead>
            <tbody>
              {rows.data.map((l) => (
                <tr key={l.id} className={ticked.includes(l.id) ? "ticked" : undefined}>
                  <td className="tick-cell"><input type="checkbox" className="wic-check-box wic-focus" aria-label={`Tick ${l.id}`} checked={ticked.includes(l.id)} onChange={(e) => flip(l.id, e.target.checked)} /></td>
                  <td className="nowrap"><a href={href(["licenses", l.id])} title={`The license ${l.id}`}><Badge status="license">{l.id}</Badge></a></td>
                  <td className="wide-cell"><a href={href(["licenses", l.id])} className="cut" title={l.name}>{l.name}</a></td>
                  <td className="nowrap">{l.allowed ? <span><span aria-hidden="true">■ </span>allowed</span> : <span className="not-allowed"><span aria-hidden="true">✕ </span>not allowed</span>}</td>
                  <td className="number ledger"><a href={href(["texts"], { license: l.id })}>{l.texts.toLocaleString("en")}</a></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      {ticked.length > 0 && (
        <div className="selection-bar" role="region" aria-label="The ticked licenses">
          <strong className="cue">{plural(ticked.length, "license")} ticked</strong>
          <ButtonLink variant="ink" href={href(["texts"], { license: ticked })}>Show their texts</ButtonLink>
          <Button variant="quiet" onClick={() => setTicked([])}>Untick all</Button>
        </div>
      )}
      <Dialog open={adding} onClose={() => setAdding(false)} title="Add a license">
        {adding && <LicenseForm onSaved={(id) => { setAdding(false); setAdded(id); load(); }} />}
      </Dialog>
    </>
  );
}

// One license: whether it is allowed, its official text, and the texts that name it.
function OneLicense({ id }: { id: string }) {
  const [found, setFound] = useState<Answer<License & { text: string }> | null>(null);
  useEffect(() => { get<License & { text: string }>(`/api/licenses/${part(id)}`).then(setFound); }, [id]);
  if (!found) return <Skeleton lines={6} label="Reading the license" />;
  if (!found.ok) return <Problems problems={found.problems} />;
  const l = found.data;
  return (
    <>
      <PageHead title={l.name}
        meta={<>
          <Badge status="license">{l.id}</Badge>
          {l.allowed ? <span><span aria-hidden="true">■ </span>allowed</span> : <span className="not-allowed"><span aria-hidden="true">✕ </span>not allowed</span>}
          <a href={href(["texts"], { license: l.id })}>{plural(l.texts, "text")}</a>
        </>}
        chain={<>One of the <a href={href(["licenses"])}>licenses</a>. A <a href={href(["texts"])}>text</a> is admitted only under an allowed one{l.allowed ? "" : "; this one is not on the allowed list, so it alone admits no text"}.</>} />
      <Section title="Texts that name it" tools={<a className="small" href={href(["texts"], { license: l.id })}>Open them in the texts</a>}>
        <MatchedTexts filter={{ ...EVERY, licenses: [l.id] }} label="The texts that name it" empty="No text names it." />
      </Section>
      <Section title="Official text">
        <pre className="license-text">{l.text}</pre>
      </Section>
    </>
  );
}

export const licenses: Page = {
  path: "licenses", title: "Licenses", group: "setup",
  Component: ({ rest }) => (rest[0] ? <OneLicense key={rest[0]} id={rest[0]} /> : <Licenses />),
};
