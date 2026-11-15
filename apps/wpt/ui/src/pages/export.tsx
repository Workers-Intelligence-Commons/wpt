// Export: #/export/ID is a dataset's texts as files, at an address that never
// changes (GET /api/datasets/ID/export), with no tokenizer and no build: every
// text of its current revision a corpus may take, whatever its max_tokens,
// in numbered parts, each a range of a seeded order of the texts, made as it
// is sent and the same bytes every time for the same revision, in JSONL
// compressed with zstd or in Parquet, with each part's size once it was sent
// whole. Nothing on this page starts a job.

import { Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, part, type Answer } from "../api";
import { PageHead, plural, Problems, Section, type Dataset } from "../parts";
import { href } from "../route";
import { bytes } from "./download";
import type { Page } from "./index";

type Form = { name: string; bytes: number | null; sha256: string | null };
type Shown = { dataset: Dataset; export: string; texts: number; texts_per_part: number; parts: { part: number; forms: Record<string, Form> }[] };

const FORMS: [string, string][] = [["jsonl.zst", "JSONL, zstd"], ["parquet", "Parquet"]];

function Export({ id }: { id: string }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  useEffect(() => { get<Shown>(`/api/datasets/${part(id)}/export`).then(setShown); }, [id]);
  if (!shown) return <Skeleton lines={6} label="Reading the dataset's export" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const { dataset, parts, texts } = shown.data;
  const base = `/api/datasets/${part(id)}/export`;
  return (
    <>
      <PageHead title={`Export ${dataset.name}`}
        meta={<><span>revision {dataset.revision}</span><span>{plural(parts.length, "part")}</span><span className="ledger" title={shown.data.export}>export {shown.data.export.slice(0, 12)}</span></>}
        chain={<>The texts of the <a href={href(["datasets", dataset.id])}>dataset {dataset.name}</a> as it is now, with no tokenizer: every plain text a share takes and every conversation and task, the whole dataset whatever its max_tokens. This address stays the same; its parts change only with a new revision or an erasure. The token corpora are on <a href={href(["download", dataset.id])}>its download page</a>.</>} />
      <Section title="Texts, in parts">
        <p className="small aside">Its libraries hold {plural(texts, "text")} (a text two libraries hold counts twice, and so does one no share takes), so each part holds about {Math.round(shown.data.texts_per_part).toLocaleString("en")} of them, in a fixed shuffled order. Each part is made as it is sent, the same bytes every time. At most two parts are sent at once, one to each visitor.</p>
        <div className="table-scroll">
          <table className="ledger-table" aria-label="The parts of its texts">
            <thead><tr><th scope="col" className="number">part</th>{FORMS.map(([form, words]) => <th key={form} scope="col">{words}</th>)}</tr></thead>
            <tbody>
              {parts.map((p) => (
                <tr key={p.part}>
                  <td className="number ledger">{p.part}</td>
                  {FORMS.map(([form]) => (
                    <td key={form} className="ledger small">
                      <a href={`${base}/${part(p.forms[form].name)}`} download={p.forms[form].name}>{p.forms[form].name}</a>{" "}
                      <span className="aside">{p.forms[form].bytes === null ? "size known once sent" : bytes(p.forms[form].bytes!)}</span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}

export const exported: Page = {
  path: "export", title: "Export", group: null,
  Component: ({ rest }) => (rest[0] ? <Export key={rest[0]} id={rest[0]} /> : <p>A dataset's export page is reached from <a href={href(["datasets"])}>its dataset</a>.</p>),
};
