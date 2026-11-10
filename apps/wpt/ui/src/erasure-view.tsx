// What one erasure request asks to erase (GET /api/erasures/{id}), shown as
// a code review shows a removal: the needles grouped by text, each text
// linked, and for each needle its place and a few lines of the text around
// it, the lines that lose characters marked "-" in the gutter, the passage
// struck through on a red ground. A needle reads the content version of its
// text it names, and the passage also goes from every other version of the
// text where it appears. Once the request is carried out, the depot
// keeps only the passage's length and the place it left, so a marker stands
// there (wpt_depot.erasure.shown).

import { Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, type Answer } from "./api";
import { plural, Problems, StatusBadge, type Status } from "./parts";
import { codePoints } from "./reader";
import { href } from "./route";

type Place = {
  text: string; field: string | null; start: number | null; end: number | null; version: number | null; title: string | null; kind: string | null;
  state: "passage" | "erased" | "gone" | "misplaced"; before: string; passage: string | null; omitted: number; passage_end: string;
  after: string; from: number; length: number | null; erased: number | null;
};
type Shown = { id: string; reason: string; status: Status; carried_out: boolean; needles: Place[] };

const LINES = 3; // lines of context on each side of a passage
const HEAD = 12; // lines of a whole text's head

type Piece = { text: string; kind: "context" | "removed" | "marker" };

// The pieces cut into lines; a line holds the pieces, or parts of them, between two line breaks.
function lines(pieces: Piece[]): Piece[][] {
  const found: Piece[][] = [[]];
  for (const piece of pieces) {
    if (piece.kind === "marker") { found[found.length - 1].push(piece); continue; }
    piece.text.split("\n").forEach((part, i) => {
      if (i > 0) found.push([]);
      if (part || piece.kind === "removed") found[found.length - 1].push({ ...piece, text: part });
    });
  }
  return found;
}

// The last n lines of a text, and whether lines were left out before them.
function lastLines(text: string, n: number): [string, boolean] {
  const all = text.split("\n");
  return all.length > n + 1 ? [all.slice(-(n + 1)).join("\n"), true] : [text, false];
}

// The first n lines of a text, and whether lines were left out after them.
function firstLines(text: string, n: number): [string, boolean] {
  const all = text.split("\n");
  return all.length > n + 1 ? [all.slice(0, n + 1).join("\n"), true] : [text, false];
}

function Diff({ pieces, cutBefore, cutAfter, label }: { pieces: Piece[]; cutBefore: boolean; cutAfter: boolean; label: string }) {
  return (
    <div className="diff ledger" role="group" aria-label={label}>
      {cutBefore && <div className="diff-line diff-gap"><span className="diff-gutter" aria-hidden="true"> </span><span className="diff-text">…</span></div>}
      {lines(pieces).map((line, i) => {
        const removed = line.some((p) => p.kind !== "context");
        return (
          <div key={i} className={`diff-line${removed ? " diff-removed" : ""}`}>
            <span className="diff-gutter" aria-hidden="true">{removed ? "-" : " "}</span>
            <span className="diff-text">
              {line.length === 0 ? " " : line.map((p, j) => p.kind === "context" ? <span key={j}>{p.text}</span>
                : p.kind === "removed" ? <del key={j}>{p.text}</del> : <span key={j} className="erased-mark">{p.text}</span>)}
            </span>
          </div>
        );
      })}
      {cutAfter && <div className="diff-line diff-gap"><span className="diff-gutter" aria-hidden="true"> </span><span className="diff-text">…</span></div>}
    </div>
  );
}

// One needle: its place, then the lines around it.
function Hunk({ place }: { place: Place }) {
  const where = place.start === null ? (place.kind ? "the whole item" : "the whole text")
    : <>characters {place.start} to {place.end}{place.field !== null && <> of the field <span className="ledger">{place.field}</span></>}, the end excluded{place.version !== null && <>, in content version {place.version}</>}</>;
  let body;
  if (place.state === "gone") body = <p className="small aside">The depot no longer holds this text: an erasure removed it whole.</p>;
  else if (place.state === "misplaced") body = <p className="small aside">This place no longer fits the text: another erasure cut it since the request was made.</p>;
  else if (place.start === null && place.state === "erased")
    body = <Diff label="What was erased" cutBefore={false} cutAfter={false} pieces={[{ kind: "marker", text: place.erased === null ? "[the whole text erased]" : `[the whole text erased: ${plural(place.erased, "character")}]` }]} />;
  else if (place.start === null) {
    const [head, more] = firstLines(place.passage ?? "", HEAD);
    const sent = codePoints(place.passage ?? "");
    body = (
      <>
        <p className="small">The whole {place.kind ? "item" : "text"} goes: {plural(place.length ?? 0, "character")}. Its head:</p>
        <Diff label="What goes" cutBefore={false} cutAfter={more || (place.length ?? 0) > sent} pieces={[{ kind: "removed", text: head }]} />
      </>
    );
  } else {
    const [before, cutBefore] = lastLines(place.before, LINES);
    const [after, cutAfter] = firstLines(place.after, LINES);
    const middle: Piece[] = place.state === "erased"
      ? [{ kind: "marker", text: `[${plural(place.erased ?? 0, "character")} erased]` }]
      : [{ kind: "removed", text: place.passage ?? "" },
         ...(place.omitted > 0 ? [{ kind: "marker" as const, text: `[${plural(place.omitted, "more character")} of the passage]` }, { kind: "removed" as const, text: place.passage_end }] : [])];
    const reachesEnd = place.from + codePoints(place.before) + (place.state === "erased" ? 0 : place.end! - place.start) + codePoints(place.after) >= (place.length ?? 0);
    body = <Diff label={place.state === "erased" ? "Where the passage was" : "The passage and the text around it"}
      cutBefore={cutBefore || place.from > 0} cutAfter={cutAfter || !reachesEnd}
      pieces={[{ kind: "context", text: before }, ...middle, { kind: "context", text: after }]} />;
  }
  return (
    <div className="hunk">
      <p className="hunk-head small">{where}</p>
      {body}
    </div>
  );
}

export function ErasureView({ id }: { id: string }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  useEffect(() => { setShown(null); get<Shown>(`/api/erasures/${encodeURIComponent(id)}`).then(setShown); }, [id]);
  if (!shown) return <Skeleton lines={4} label="Reading the request" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const request = shown.data;
  const texts: string[] = [...new Set(request.needles.map((n) => n.text))];
  return (
    <div className="erasure-view">
      <p className="row tight">
        <StatusBadge status={request.status} />
        <span>{request.carried_out ? "Carried out: the depot keeps only each passage's length and hashes." : "Not carried out yet: the passages are still in their texts. Once it is, each passage also goes from every content version of its text where it appears."}</span>
      </p>
      <p><span className="aside">Reason:</span> {request.reason}</p>
      {texts.map((text) => {
        const needles = request.needles.filter((n) => n.text === text);
        const first = needles[0];
        const name = first.title ?? (first.kind ? `an item of kind ${first.kind}` : first.state === "gone" || (first.start === null && first.state === "erased") ? "a text no longer in the depot" : "an untitled text");
        return (
          <section key={text} className="diff-file" aria-label={`${text}: ${plural(needles.length, "passage")}`}>
            <h3 className="diff-file-head cue">
              {first.title !== null || first.kind !== null || first.length !== null ? <a href={href(["texts", text])}>{name}</a> : <span>{name}</span>}
              {" "}<span className="ledger aside">{text}</span>
              {" "}<span className="aside small">{plural(needles.length, "passage")}</span>
            </h3>
            {needles.map((n, i) => <Hunk key={i} place={n} />)}
          </section>
        );
      })}
    </div>
  );
}
