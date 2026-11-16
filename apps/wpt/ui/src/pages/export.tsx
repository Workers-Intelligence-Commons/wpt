// Export: #/export/ID is a dataset's texts as files, at an address that never
// changes (GET /api/datasets/ID/export), with no tokenizer and no build: every
// text of its current revision a corpus may take, whatever its max_tokens,
// in numbered parts, each a range of a seeded order of the texts, made as it
// is sent and the same bytes every time for the same revision, in JSONL
// compressed with zstd or in Parquet, with each part's size once it was sent
// whole (read again every few seconds after a download starts until its
// size is known, "downloading" shown meanwhile, then "downloaded" with it). Before its first download,
// a part shows a ceiling of its size: the characters its dataset's libraries
// hold, shared out over the parts, as text before compression, which both
// forms compress. Parts are counted from 1
// for people; the files keep their names, numbered from 0, which the
// browser saves after the dataset's name. The download buttons come first,
// one per part and form; below them the page says what each form is, what a line or a
// row holds, what it leaves out (titles, authors, licenses, notices) and
// where those are. Nothing on this page starts a job.

import { ButtonLink, Skeleton } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, part, type Answer } from "../api";
import { CharactersNote, ownLibraries, PageHead, plural, Problems, Section, useCharacters, type Dataset } from "../parts";
import { href } from "../route";
import { bytes } from "./download";
import type { Page } from "./index";

type Form = { name: string; download: string; bytes: number | null; sha256: string | null };
type Shown = { dataset: Dataset; export: string; texts: number; texts_per_part: number; parts: { part: number; forms: Record<string, Form> }[] };

// each form, its name, what it is in plain words, to choose one, and its button's words
const FORMS: [string, string, string, string][] = [
  ["jsonl.zst", "JSONL, compressed", "one text per line as JSON, compressed with zstd: the smallest file, read line by line", "Download JSONL"],
  ["parquet", "Parquet", "a table that data tools open directly", "Download Parquet"],
];
const EVERY_SECONDS = 2; // seconds between two readings of the sizes while a download's size is not known
const MOST_READINGS = 90; // readings after which the page stops asking: about three minutes

function Export({ id }: { id: string }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  const load = useCallback(() => { get<Shown>(`/api/datasets/${part(id)}/export`).then(setShown); }, [id]);
  useEffect(load, [load]);
  const [measuring, setMeasuring] = useState<Set<string>>(new Set()); // the files clicked whose size is not read yet
  const [readings, setReadings] = useState(0);
  const unknown = shown?.ok ? shown.data.parts.some((p) => FORMS.some(([form]) => measuring.has(p.forms[form].name) && p.forms[form].bytes === null)) : false;
  useEffect(() => {
    if (!unknown || readings >= MOST_READINGS) return;
    const timer = window.setTimeout(() => { setReadings((n) => n + 1); load(); }, EVERY_SECONDS * 1000);
    return () => window.clearTimeout(timer);
  }, [unknown, readings, shown, load]);
  const mine = shown?.ok ? ownLibraries(shown.data.dataset) : [];
  const size = useCharacters(mine);
  if (!shown) return <Skeleton lines={6} label="Reading the dataset's export" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const { dataset, parts, texts } = shown.data;
  const base = `/api/datasets/${part(id)}/export`;
  // a ceiling of a part's size before its first download: its share of the libraries' characters, uncompressed
  const ceiling = typeof size === "number" && parts.length ? `at most about ${bytes(Math.ceil(size / parts.length))} before compression; exact after its first download` : "size known after its first download";
  const known = (form: string) => parts.map((p) => p.forms[form].bytes).filter((b): b is number => b !== null);
  return (
    <>
      <PageHead title={`Export ${dataset.name}`}
        meta={<><span>{plural(parts.length, "part")}</span><span>{dataset.libraries.length === 1 ? "" : "about "}{plural(texts, "text")}</span>{mine.length > 0 && <CharactersNote found={size} />}</>}
        chain={<>The texts of the <a href={href(["datasets", dataset.id])}>dataset {dataset.name}</a> as it is now, with no tokenizer and no size limit: every plain text a share takes, and every conversation and task. Nothing here starts a job. The token files for training are on <a href={href(["download", dataset.id])}>its corpora page</a>.</>} />
      <Section title="Download its texts">
        <p>{parts.length === 1 ? <>One part holds the whole dataset, in two forms holding the same texts: take one.</>
          : <>The texts are split into {plural(parts.length, "part")} of about {plural(Math.round(shown.data.texts_per_part), "text")} each: download every part, in one of the two forms, to get the whole dataset.</>}</p>
        <div className="table-scroll">
          <table className="ledger-table" aria-label="The parts of its texts">
            <thead><tr><th scope="col" className="number">part</th>{FORMS.map(([form, words]) => <th key={form} scope="col">{words}</th>)}</tr></thead>
            <tbody>
              {parts.map((p) => (
                <tr key={p.part}>
                  <td className="number ledger">{parts.length === 1 ? "all" : `${p.part + 1} of ${parts.length}`}</td>
                  {FORMS.map(([form, , , button]) => (
                    <td key={form}>
                      <ButtonLink variant={form === FORMS[0][0] ? "primary" : "quiet"} href={`${base}/${part(p.forms[form].name)}`} download={p.forms[form].download}
                        onClick={() => { setMeasuring(new Set([...measuring, p.forms[form].name])); setReadings(0); }}>{button}{parts.length > 1 ? `, part ${p.part + 1}` : ""}</ButtonLink>
                      <br /><span className="aside small part-size">{p.forms[form].bytes !== null ? `${measuring.has(p.forms[form].name) ? "downloaded, " : ""}${bytes(p.forms[form].bytes!)}` : measuring.has(p.forms[form].name) ? (readings >= MOST_READINGS ? "reload the page for its size" : "downloading; its size shows once it is sent whole") : ceiling}</span>
                      <br /><span className="aside small ledger">saved as {p.forms[form].download}</span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {FORMS.some(([form]) => known(form).length === parts.length) && (
          <p className="small">In all: {FORMS.filter(([form]) => known(form).length === parts.length).map(([form, words]) => `${bytes(known(form).reduce((t, b) => t + b, 0))} as ${words}`).join(", ")}.</p>
        )}
      </Section>
      <Section title="What the files hold">
        <ul className="small">
          {FORMS.map(([form, name, what]) => <li key={form}><strong>{name}</strong> (<span className="ledger">.{form}</span>): {what}.</li>)}
        </ul>
        <p className="small">One text per JSON line, or per Parquet row, with five fields, the same in both forms:</p>
        <dl className="facts small">
          <dt className="aside ledger">depot</dt><dd>the depot that holds the text, by the name a corpus records for it</dd>
          <dt className="aside ledger">id</dt><dd>the text's id in that depot, the one its page in the workspace shows</dd>
          <dt className="aside ledger">revision</dt><dd>the revision of the text the dataset reads</dd>
          <dt className="aside ledger">kind</dt><dd>empty for a plain text, else the item's kind, such as <span className="ledger">conversation</span> or <span className="ledger">task</span></dd>
          <dt className="aside ledger">text</dt><dd>the plain text itself, or an item's content as JSON text</dd>
        </dl>
        <p className="small">Titles, authors, tags, licenses and notices are left out. Each text's page in the <a href={href(["texts"])}>workspace</a> shows them, and a corpus built from this dataset on <a href={href(["download", dataset.id])}>its corpora page</a> holds them as tables: its texts with their credits and licenses, the licenses' texts, and the notices.</p>
        <details className="small aside">
          <summary>More about the parts</summary>
          The texts are in a fixed shuffled order, and a part is made as it is sent, the same bytes every time, so its size is known after its first download. A text two libraries hold is counted twice above, so a part may hold a few fewer. At most two parts are sent at once, one to each visitor.
        </details>
      </Section>
    </>
  );
}

export const exported: Page = {
  path: "export", title: "Export", group: null, under: "datasets",
  Component: ({ rest }) => (rest[0] ? <Export key={rest[0]} id={rest[0]} /> : <p>A dataset's export page is reached from <a href={href(["datasets"])}>its dataset</a>.</p>),
};
