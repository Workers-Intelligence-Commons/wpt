// The head of the Texts page when one library is open: its name, whether it
// is open or locked, how many texts it holds and how many characters, those of
// plain text and those of conversations and tasks apart (read again after each change), the library it was forked
// from and the datasets that read it, its reported problems, and its
// actions: make a dataset from it while none reads it, put texts in it (the admitted texts it does not hold, opened as
// the filter to tick from), read it whole (pages/strip.tsx; there, back to its texts), report a problem with it as a whole, fork it,
// lock it for good, rename it, and, locked, the revisions it pinned, the
// actions after the first two folded under "More". Taking
// texts out of it, and a report or a fork without some of its texts, act on
// the texts ticked in the playlist (texts.tsx). A library that belongs
// to the upstream this depot follows says so, and keeps only the fork,
// which makes a library of this depot's own.

import { Button, ButtonLink, Dialog } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, post, type Answer } from "../api";
import { ReportForm } from "../decide-forms";
import { EVERY, toQuery, type TextFilter } from "../filter";
import { ForkForm, Pinned, RenameForm } from "../pages/libraries";
import { characters, figures, Help, LibraryLink, MoreMenu, Outcome, plural, Problems, readers, useDatasets, useMeasure, type Library } from "../parts";
import { ITEMS } from "../mix";
import { href } from "../route";

type Act = "report" | "fork" | "lock" | "rename" | "pins";

// The admitted texts a library does not hold, as the filter "Put texts in it" opens.
export const outsideOf = (library: string): TextFilter => ({ ...EVERY, statuses: ["admitted"], outside: [library] });

// whole: the head shows on the page that reads the library whole, which it links back from to its texts.
export function LibraryHead({ library, libraries, onChanged, version, whole = false }: { library: Library; libraries: Library[]; onChanged: () => void; version: number; whole?: boolean }) {
  const { datasets } = useDatasets();
  const [acting, setActing] = useState<Act | null>(null);
  const [done, setDone] = useState<Answer<unknown> | null>(null);
  const [said, setSaid] = useState("");
  const parent = libraries.find((l) => l.id === library.forked_from);
  const reading = readers(datasets?.ok ? datasets.data : [], library.id);
  const open = !library.locked_at;
  const theirs = Boolean(library.upstream);
  const finished = (a: Answer<unknown>, words: string) => {
    setDone(a);
    setSaid(words);
    if (a.ok) { setActing(null); onChanged(); }
  };
  const act = (a: Act) => () => { setDone(null); setActing(a); };
  const lock = async () => {
    const a = await post(`/api/libraries/${encodeURIComponent(library.id)}/lock`, {});
    finished(a, `Locked ${library.name} for good: it reads the revision it pinned of each of its texts.`);
  };
  const outside = outsideOf(library.id);
  const measured = useMeasure([library.id], null, version);
  const found = figures(measured);
  // how many texts it holds (GET /api/libraries/ID/size: every membership, where the measure counts only the texts
  // whose content it finds); none yet: putting texts in it leads, not a dataset
  const [size, setSize] = useState<Answer<{ texts: number }> | null>(null);
  useEffect(() => { get<{ texts: number }>(`/api/libraries/${encodeURIComponent(library.id)}/size`).then(setSize); }, [library.id, version]);
  const empty = size?.ok === true && size.data.texts === 0;
  return (
    <section className="open-head" aria-label={`The library ${library.name}`}>
      <div className="open-head-main">
        <h1 className="title open-title" title={`Its id, which the command line names it by: ${library.id}`}>{library.name}</h1>
        <div className="page-meta small aside">
          {open || theirs ? <span>{open ? "open" : `locked ${library.locked_at!.replace("T", " ").slice(0, 16)} UTC`}</span> : (
            <details className="inline-fold">
              <summary>locked {library.locked_at!.replace("T", " ").slice(0, 16)} UTC <span className="aside">(what that means)</span></summary>
              <p className="fold-body">It reads the revision it pinned of each of its texts, and no text joins or leaves it; to correct it, fork it without the texts concerned (tick them first). Its texts still take decisions and edits, as new revisions it does not read; only an erasure changes what it reads.</p>
            </details>
          )}
          {!size ? <span className="aside">counting its texts</span> : size.ok ? <span>{plural(size.data.texts, "text")}</span> : <span className="not-allowed">{size.problems.join(" ")}</span>}
          {found ? <span title={`${plural(found.plain.characters, "character")} of plain text, ${plural(found.items.characters, "character")} of ${ITEMS}`}>{characters(found.characters)}</span>
            : typeof measured === "string" ? <span title={measured}>size not read</span> : null}
          {found && found.items.characters > 0 && found.plain.characters > 0 && <span>{characters(found.items.characters).replace(/ characters?$/, "")} in {ITEMS}</span>}
          {theirs && <span className="aside">the upstream's</span>}
          {parent && <span>forked from <LibraryLink library={parent} /></span>}
          {reading.length > 0 && <span>read by {reading.map((d, i) => <span key={d.id}>{i > 0 && ", "}<a href={href(["datasets", d.id])}>{d.name}</a></span>)}</span>}
        </div>
        {theirs && <p className="chain small">Only the upstream changes it.<Help label="an upstream's library">It belongs to the upstream this depot follows: only the upstream names it, locks it, puts texts in it, takes them out or takes a problem on it, and a pull brings what it did here. A fork of it is a library of this depot's own.</Help></p>}
      </div>
      <div className="head-buttons">
        {open && !theirs && <ButtonLink variant={empty ? "ink" : "quiet"} href={href(["texts"], { ...toQuery(outside), into: library.id })}>Put texts in it</ButtonLink>}
        {reading.length === 0 && <ButtonLink variant={empty ? "quiet" : "ink"} href={href(["datasets"], { new: "1", library: library.id })}>Make a dataset</ButtonLink>}
        {whole ? <ButtonLink variant="quiet" href={href(["texts"], { library: library.id })}>Its texts</ButtonLink>
          : !empty && <ButtonLink variant="quiet" href={href(["strip", library.id])}>Read it whole</ButtonLink>}
        <MoreMenu label="More" side="right">
          {!theirs && <Button variant="quiet" onClick={act("report")}>Report a problem</Button>}
          <Button variant="quiet" onClick={act("fork")}>Fork it</Button>
          {open && !theirs && <Button variant="quiet" onClick={act("lock")}>Lock it</Button>}
          {!theirs && <Button variant="quiet" onClick={act("rename")}>Rename it</Button>}
          {!open && <Button variant="quiet" onClick={act("pins")}>Revisions it pinned</Button>}
        </MoreMenu>
      </div>
      {!acting && <Outcome answer={done} done={() => said} />}
      {(library.problems ?? []).length > 0 && (
        <details className="problems">
          <summary className="cue small">{plural(library.problems!.length, "reported problem")}</summary>
          <ul className="plain small">
            {library.problems!.map((p) => (
              <li key={p.decision}>
                <a href={href(["libraries", library.id, "problems", p.decision])} className="cut" title={p.problem}>{p.problem}</a> <a className="ledger aside" href={href(["decisions", p.decision])}>{p.decision}</a>
                {p.texts.length > 0 ? <> <a href={href(["texts"], { id: p.texts, what: `the ${plural(p.texts.length, "text")} of this problem` })}>{plural(p.texts.length, "text")}</a></> : <span className="aside"> the whole library</span>}
              </li>
            ))}
          </ul>
        </details>
      )}
      <Dialog open={acting === "report"} onClose={() => setActing(null)} title={`Report a problem with ${library.name}`}>
        {acting === "report" && <ReportForm library={library} selection={null} onDone={finished} />}
        {acting === "report" && done && !done.ok && <Problems problems={done.problems} />}
      </Dialog>
      <Dialog open={acting === "fork"} onClose={() => setActing(null)} title={`Fork ${library.name}`}>
        {acting === "fork" && <ForkForm library={library} without={null} onDone={finished} />}
        {acting === "fork" && done && !done.ok && <Problems problems={done.problems} />}
      </Dialog>
      <Dialog open={acting === "rename"} onClose={() => setActing(null)} title={`Rename ${library.name}`}>
        {acting === "rename" && <RenameForm library={library} onDone={finished} />}
        {acting === "rename" && done && !done.ok && <Problems problems={done.problems} />}
      </Dialog>
      <Dialog open={acting === "lock"} onClose={() => setActing(null)} title={`Lock ${library.name} for good`}>
        <p>Locking {library.name} cannot be undone: it pins the current revision of each of its texts, reads those for good, and no text joins or leaves it again. A later decision or edit of one of its texts makes a new revision it does not read.</p>
        <p className="small aside">Not needed to build a corpus.<Help label="locking">Lock a library to keep a fixed set of texts that a model's training can be traced back to. A dataset reads an open library too, as it stood when the dataset was made or last edited.</Help></p>
        <Button variant="primary" onClick={lock}>Lock it for good</Button>
        {acting === "lock" && done && !done.ok && <Problems problems={done.problems} />}
      </Dialog>
      <Dialog open={acting === "pins"} onClose={() => setActing(null)} title={`The revisions ${library.name} pinned`}>
        {acting === "pins" && <div className="wide"><Pinned id={library.id} /></div>}
      </Dialog>
    </section>
  );
}
