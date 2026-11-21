// Datasets: what to train on (wpt depot dataset new), and written out as a
// remote (export); a remote is read back in (wpt depot dataset import) by
// the commands. A new dataset names a tokenizer recipe of the depot, chosen
// in a list (one of the Tokenizers page's, or the name a recipe made next
// will take, filled in from the dataset's name, or another new name), its libraries of this depot ticked (the one the
// address names, ?new=1&library=ID, ticked already), its shares (each tag
// offered with how many texts hold it, and one no ticked library's text
// holds marked; a share left empty takes an even part of what the numbers leave) and its optional max_tokens; while it is filled in, it says
// what the ticked libraries hold, read by the server from the depot's counts
// (POST /api/libraries/measure): texts and characters, characters per share, the
// characters the mix keeps and the share that limits it, the texts no share
// takes, and the conversations and tasks every dataset takes; a share's tag
// is checked once typing pauses. Once made, the dataset's page opens;
// ?edit=1 on a dataset's page opens its edit form. A dataset that
// reads a library of another depot is made and changed by the commands
// only. #/datasets/ID is one dataset's view: its libraries with the moment
// of their depot each is read at (the time of that change, from POST
// /api/changes/times, or its number when the time is not known), its shares, its limit, what
// its revision reads, as the revision stores it, with how many more or fewer texts its libraries
// hold now (folded, opened from the line of the share that
// limits the mix; "unknown" when the revision does not know it), its stages as cards from texts to corpus (the corpus card
// the corpus at the dataset's own size, the others on its corpora page), the next stage's button the main one (make
// its tokenizer recipe, train its vocabulary, publish it, build a corpus),
// the list of datasets showing each one's stage and next step,
// a "Read again" button where reading again changes what it shows (the server says),
// the links to its download page (download.tsx),
// where a build starts, and to its export page (export.tsx), its remote
// (its description for another machine, folded away),
// the vocabulary it publishes (the corpus of that vocabulary at a recipe's
// total_tokens is the one training takes), set or cleared among the trained
// vocabularies in a fold, and the edit of its name, libraries, shares and limit, each
// change a new revision (wpt depot dataset edit). A dataset that belongs to
// the upstream this depot follows says so and has no edit: only the
// upstream changes it.

import { Button, ButtonLink, Dialog, Select, Skeleton, TextArea, TextInput } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { get, part, post, type Answer } from "../api";
import { characters, Cut, heldOf, Help, itemKinds, kindHeld, local, MoreMenu, optional, ownLibraries, Outcome, PageHead, PLAIN, plural, Problems, required, RequiredNote, requiredGroup, Section, figures, SHARES_NOTE, Since, SUMMED_NOTE, TAKEN, takenNames, takenShare, useLibraries, useMeasure, type Dataset, type MeasureState } from "../parts";
import { href } from "../route";
import { TickList } from "../ticks";
import { asked, mixLine, whole } from "../mix";
import { publishVocabulary, StageMark, StepButton, standing, trainedWords, useCorpora, usePipelineChange, useRecipesAndVocabularies, type Trained } from "../pipeline";
import { Stages, type Shown } from "../stages";
import type { Page } from "./index";

type Other = { depot: string; library: string };
type Share = { tag: string; share: string };

// The shares' numbers as the depot stores them (spread_shares): a share left
// empty takes an even part of what the shares given as numbers leave of 1;
// null when the numbers reach 1 already and an empty share has nothing left.
function spread(shares: Share[]): number[] | null {
  const blank = shares.filter((s) => s.share.trim() === "").length;
  const given = shares.reduce((t, s) => t + (s.share.trim() === "" ? 0 : Number(s.share) || 0), 0);
  if (blank && given >= 1) return null;
  const part = blank ? Math.round(((1 - given) / blank) * 1e12) / 1e12 : 0;
  return shares.map((s) => (s.share.trim() === "" ? part : Number(s.share)));
}

function Datasets({ query }: { query: URLSearchParams }) {
  const { libraries } = useLibraries();
  const [rows, setRows] = useState<Answer<Dataset[]> | null>(null);
  const [answer, setAnswer] = useState<Answer<any> | null>(null);
  const [said, setSaid] = useState<(d: any) => string>(() => () => "");
  const [making, setMaking] = useState(query.get("new") === "1");
  const [remote, setRemote] = useState<{ name: string; text: string } | null>(null); // the dialog of an exported remote
  const load = useCallback(() => { get<Dataset[]>("/api/datasets").then(setRows); }, []);
  useEffect(load, [load]);
  const { recipes, trained } = useRecipesAndVocabularies();
  const corpora = useCorpora(rows?.ok ? rows.data.map((d) => d.id) : []);
  const show = (a: Answer<any>, words: (d: any) => string) => { setSaid(() => words); setAnswer(a); load(); };
  const at = (d: Dataset) => standing(d, recipes?.ok ? recipes.data : null, trained?.ok ? trained.data : null, corpora.get(d.id));
  // the one main button of the page: the next step of the first dataset that has one, else New dataset
  const firstNext = rows?.ok ? rows.data.find((d) => at(d).next)?.id : undefined;

  const exported = async (dataset: string) => {
    const a = await get<unknown>(`/api/datasets/${part(dataset)}/remote`);
    if (a.ok) { setAnswer(null); setRemote({ name: dataset, text: JSON.stringify(a.data, null, 2) }); } else show(a, () => "");
  };

  return (
    <>
      <PageHead title="Datasets"
        chain={<>A dataset reads <a href={href(["libraries"])}>libraries</a>, each as it stood when the dataset was made or last edited, and builds corpora of their texts for training, with the vocabulary of its <a href={href(["tokenizers"])}>tokenizer recipe</a>.</>}
        actions={<Button variant="quiet" onClick={() => setMaking(true)}>New dataset</Button>} />
      <Outcome answer={answer} done={said} />
      {rows && !rows.ok && <Problems problems={rows.problems} />}
      {rows?.ok && (rows.data.length === 0 ? <p>No dataset yet.</p> : (
        <div className="table-scroll cards">
          <table className="ledger-table" aria-label="The datasets">
            <thead><tr><th scope="col">dataset</th><th scope="col">stage</th><th scope="col">next step</th><th scope="col">tokenizer recipe</th><th scope="col">libraries</th><th scope="col">mix</th><th scope="col"><span className="visually-hidden">pages</span></th></tr></thead>
            <tbody>
              {rows.data.map((d) => {
                const stands = at(d);
                return (
                  <tr key={d.id}>
                    <td className="nowrap" data-label=""><a href={href(["datasets", d.id])}><strong>{d.name}</strong></a>{d.upstream ? <span className="aside"> the upstream's</span> : null}</td>
                    <td className="nowrap" data-label="stage"><StageMark at={stands} /></td>
                    <td className="nowrap" data-label=""><StepButton at={stands} primary={d.id === firstNext} /></td>
                    <td data-label="tokenizer recipe"><a href={href(["tokenizers", d.tokenizer])}><Cut text={d.tokenizer} /></a></td>
                    <td data-label="libraries">
                      <span className="cut">
                        {d.libraries.map((r, i) => (
                          <span key={`${r.depot}-${r.library}`}>
                            {i > 0 && ", "}
                            {!local(d, r) ? <span className="ledger">{r.depot} {r.library}</span> : <a href={href(["libraries", r.library])}>{libraries.find((l) => l.id === r.library)?.name ?? r.library}</a>}
                          </span>
                        ))}
                      </span>
                    </td>
                    <td data-label="mix" className="small wrap-cell"><MixCell dataset={d} /></td>
                    <td className="nowrap" data-label=""><MoreMenu label="More" side="right">
                      <ButtonLink variant="quiet" href={href(["download", d.id])}>Corpora</ButtonLink>
                      <ButtonLink variant="quiet" href={href(["export", d.id])}>Export texts</ButtonLink>
                      <Button variant="quiet" onClick={() => exported(d.name)} title="The description of the dataset that the command wpt depot dataset import reads on another machine">Describe for import</Button>
                    </MoreMenu></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}

      <Dialog open={making} onClose={() => setMaking(false)} title="New dataset">
        {making && <DatasetForm library={query.get("library")} onDone={(a) => {
          if (!a.ok) return;
          setMaking(false);
          window.location.hash = href(["datasets", (a.data as { id: string }).id], { made: "1" }); // its page, with its next step
        }} />}
      </Dialog>
      <Dialog open={remote !== null} onClose={() => setRemote(null)} title={`${remote?.name ?? ""}, described for another machine`}>
        {remote && <Remote text={remote.text} />}
      </Dialog>
    </>
  );
}

// A dataset's mix in a list's cell (mix.tsx): the conversations' expected
// fraction, read from the counts of its own libraries, then its shares of the rest.
function MixCell({ dataset }: { dataset: Dataset }) {
  const measured = useMeasure(ownLibraries(dataset), dataset.shares?.length ? dataset.shares : null);
  return <>{mixLine(asked(dataset.shares, measured))}</>;
}

// The form of a dataset: a new one (POST /api/datasets), or, given the
// dataset, its edit as a new revision (POST /api/datasets/ID), its tokenizer fixed.
function DatasetForm({ dataset, library, onDone }: { dataset?: Dataset; library?: string | null; onDone: (a: Answer<any>, name: string) => void }) {
  const { libraries } = useLibraries();
  const [recipes, setRecipes] = useState<Answer<{ name: string }[]> | null>(null);
  const [tags, setTags] = useState<{ tag: string; texts: number }[]>([]);
  const [name, setName] = useState(dataset?.name ?? "");
  const [tokenizer, setTokenizer] = useState(dataset?.tokenizer ?? "");
  const typedTokenizer = useRef(false); // until a recipe is chosen, it follows the dataset's name
  const [custom, setCustom] = useState(false); // a new recipe named by hand
  const [mine, setMine] = useState<string[]>(dataset ? dataset.libraries.filter((r) => local(dataset, r)).map((r) => r.library) : library ? [library] : []);
  const [others] = useState<Other[]>(dataset ? dataset.libraries.filter((r) => !local(dataset, r)).map((r) => ({ depot: r.depot ?? "", library: r.library })) : []);
  const [shares, setShares] = useState<Share[]>(dataset?.shares?.map((s) => ({ tag: s.tag, share: String(s.share) })) ?? []);
  const [limit, setLimit] = useState(dataset?.max_tokens ? String(dataset.max_tokens) : "");
  const [problems, setProblems] = useState<string[]>([]);
  const id = useId(); // names the groups of ticks and of shares, whose "?" stays out of their names
  useEffect(() => { if (!dataset) get<{ name: string }[]>("/api/tokenizers").then(setRecipes); }, [dataset]);
  useEffect(() => { get<{ tag: string; texts: number }[]>("/api/tags").then((a) => a.ok && setTags(a.data)); }, []);
  const named = (n: string) => { setName(n); if (!dataset && !typedTokenizer.current) setTokenizer(recipeName(n)); };
  // a new dataset of the library the address names takes that library's name until a name is typed
  const typedName = useRef(false);
  const from = !dataset && library ? libraries.find((l) => l.id === library)?.name : undefined;
  useEffect(() => { if (from && !typedName.current) named(from); }, [from]);
  const known = recipes?.ok ? recipes.data.some((r) => r.name === tokenizer.trim()) : true;
  const send = async () => {
    const body = {
      name,
      libraries: mine.map((library) => ({ library })),
      shares: shares.length ? shares.map((s) => ({ tag: s.tag, share: s.share.trim() === "" ? null : Number(s.share) })) : null,
      max_tokens: limit.trim() ? Number(limit) : null,
      vocabulary: dataset?.vocabulary ?? null,
    };
    const a = dataset ? await post(`/api/datasets/${part(dataset.id)}`, { ...body, revision: dataset.revision }) : await post("/api/datasets", { ...body, tokenizer });
    setProblems(a.ok ? [] : a.problems);
    onDone(a, name);
  };
  const move = (i: number, by: number) => {
    const next = [...shares];
    [next[i], next[i + by]] = [next[i + by], next[i]];
    setShares(next);
  };
  const count = mine.length + others.length;
  // the share tags Holds checks: as typed, once typing paused, so a tag half typed is never called missing
  const parts = spread(shares);
  const typedTags = parts ? shares.map((s, i) => ({ tag: s.tag.trim(), share: parts[i] })).filter((s) => s.tag && s.share >= 0) : [];
  const [settled, setSettled] = useState(typedTags);
  const typedKey = JSON.stringify(typedTags);
  useEffect(() => { const timer = window.setTimeout(() => setSettled(JSON.parse(typedKey)), 800); return () => window.clearTimeout(timer); }, [typedKey]);
  const measured = useMeasure(mine, settled.length ? settled : null); // read once, for the summary and the line by the button
  // the tags a share can take: those of plain texts in the ticked libraries (every tag's plain texts, from the
  // counts); a tag only conversations or tasks hold is no share, since they take their own part
  const plainTags = useMeasure(mine, tags.length ? tags.map((t) => ({ tag: t.tag, share: 1 })) : null);
  const offered = figures(plainTags)?.shares.filter((t) => t.texts > 0) ?? null;
  // a share whose tag no plain text of the ticked libraries has, once measured: it takes nothing, and is saved
  // with a warning beside the button
  const empty = shares.map((s) => s.tag.trim()).filter((tag) => tag && figures(measured)?.shares.some((h) => h.tag === tag && h.texts === 0));
  // what the submit waits for, named beside it while it is off
  const missingWhat = !name.trim() ? "needs a name" : !dataset && !tokenizer.trim() ? "needs a recipe" : count === 0 ? "needs a library"
    : shares.some((s) => !s.tag.trim()) ? "a share needs a tag"
    : shares.some((s) => s.share.trim() !== "" && !(Number(s.share) > 0 && Number(s.share) <= 1)) ? "a share is above 0, at most 1"
    : !parts ? "the numbers reach 1: nothing is left for an empty share"
    : typedTags.length && mine.length && (JSON.stringify(settled) !== typedKey || measured === null) ? "checking the shares' tags"
    : limit.trim() !== "" && !(Number(limit) >= 1) ? "most tokens: a whole number" : null;
  return (
    <form onSubmit={(e) => { e.preventDefault(); send(); }}>
      <RequiredNote />
      {dataset && <p className="form-note small">Id and tokenizer recipe stay.<Help label="an edit">Its id <span className="ledger">{dataset.id}</span> and its tokenizer recipe <span className="ledger">{dataset.tokenizer}</span> stay. A library it reads already keeps the change it is read at; a library added is read at its depot's change now. A corpus built before keeps its own copy of the dataset as it was.</Help></p>}
      <div className="fields">
        <TextInput label={required("Name")} required value={name} onChange={(e) => { typedName.current = true; named(e.target.value); }} />
        {!dataset && (
          <>
            <Select label={required("Tokenizer recipe")} required
              value={custom ? OTHER : tokenizer} onChange={(e) => { typedTokenizer.current = true; if (e.target.value === OTHER) { setCustom(true); setTokenizer(""); } else { setCustom(false); setTokenizer(e.target.value); } }}>
              {!custom && tokenizer.trim() && !known && <option value={tokenizer}>{tokenizer} (new, made next)</option>}
              {!tokenizer.trim() && !custom && <option value="" disabled>choose a recipe</option>}
              {recipes?.ok && recipes.data.map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}
              <option value={OTHER}>new, another name</option>
            </Select>
            {custom && <TextInput label={required("New recipe's name")} required value={tokenizer} onChange={(e) => setTokenizer(e.target.value)} />}
          </>
        )}
        <TextInput label={optional("Most tokens")} type="number" min="1" step="1" value={limit} placeholder="no limit"
          onChange={(e) => setLimit(e.target.value)}
          hint={sizeSplit(limit, settled, measured)} />
      </div>
      {recipes && !recipes.ok && <Problems problems={recipes.problems} />}
      <fieldset className="form-part" aria-labelledby={`${id}-libraries`}>
        <div className="title-row"><span id={`${id}-libraries`} className="wic-label">{requiredGroup("Libraries")}</span><Help label="libraries">At least one. Each is read as it stands now, and a later edit keeps that reading. "Read again" on the dataset's page reads every one as it is then. A library of another depot is named with <span className="ledger">wpt depot dataset new</span> or <span className="ledger">wpt depot dataset edit</span>.</Help></div>
      <TickList label={others.length ? "Of this depot" : <span className="visually-hidden">Libraries</span>} chosen={mine} onChange={setMine}
        options={libraries.map((l) => ({ value: l.id, words: `${l.id} ${l.name}`, label: <>{l.name}{l.locked_at ? <span className="aside"> locked</span> : null}</> }))} />
      {others.length > 0 && <p className="form-note small">It reads libraries of another depot ({others.map((o) => `${o.depot} ${o.library}`).join(", ")}), so only the command line changes it.</p>}
      </fieldset>
      <fieldset className="form-part" aria-labelledby={`${id}-shares`}>
        <div className="title-row"><span id={`${id}-shares`} className="wic-label">{optional("Shares of plain text, in order")}</span><Help label="shares">Conversations and tasks are no share: every one is in, and they take their own part of the corpus, the part of the bytes they hold in the corpus of every token. The shares split the rest, the plain text. A plain text counts for the first share whose tag it has; a plain text no share takes is left out. Inside a share, its part splits over the other tags its texts carry, each tag's part following the square root of what it holds: a tag a hundred times bigger aims at ten times the tokens. A share left empty takes an even part of what the numbers leave of 1: 0.6 and two empty shares save 0.6, 0.2 and 0.2. No share: every plain text.</Help></div>
        <datalist id="share-tags">{(offered ?? []).map((t) => <option key={t.tag} value={t.tag}>{plural(t.texts, "plain text")}</option>)}</datalist>
        {shares.map((s, i) => (
          <div key={i} className="entry-row">
            <TextInput label={required("Tag")} required value={s.tag} list="share-tags" onChange={(e) => setShares(shares.map((o, j) => (j === i ? { ...o, tag: e.target.value } : o)))} />
            <TextInput label={optional("Share")} type="number" step="any" min="0" max="1" value={s.share} placeholder="even part of the rest"
              onChange={(e) => setShares(shares.map((o, j) => (j === i ? { ...o, share: e.target.value } : o)))} />
            <ShareSaid share={s} part={parts?.[i] ?? null} parts={parts} found={measured} />
            <Button variant="quiet" disabled={i === 0} onClick={() => move(i, -1)}>Up</Button>
            <Button variant="quiet" disabled={i === shares.length - 1} onClick={() => move(i, 1)}>Down</Button>
            <Button variant="quiet" onClick={() => setShares(shares.filter((_, j) => j !== i))}>Remove</Button>
          </div>
        ))}
        <Button variant="quiet" onClick={() => setShares([...shares, { tag: "", share: "" }])}>Add a share</Button>
      </fieldset>
      <HoldsView libraries={mine} shares={settled} found={measured} />
      <Problems problems={problems} />
      <div className="sticky-submit">
      <HoldsLine libraries={mine} found={measured} />
      {missingWhat && <span className="small blocking">{missingWhat}</span>}
      {!missingWhat && empty.length > 0 && <span className="small not-allowed" role="status">{empty.join(", ")}: no plain text, takes nothing</span>}
      <Button type="submit" variant="primary" disabled={Boolean(missingWhat) || others.length > 0}>
        {dataset ? "Save the dataset" : "Make the dataset"}
      </Button>
      </div>
    </form>
  );
}

const OTHER = "\u0000other"; // the choice of a new recipe named by hand
// The name a dataset's tokenizer recipe is given until one is typed: the
// dataset's name in the letters a recipe's name takes, then "-tok".
export const recipeName = (dataset: string) => {
  const base = dataset.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  return base ? `${base}-tok` : "";
};

// What libraries hold now (useMeasure, read from the depot's counts), as a
// table: the conversations and tasks, which a dataset always takes, then the
// plain texts that have each share's tag, or the plain texts as one row
// without shares, then what no share
// takes and the items of other kinds, which no corpus takes; with shares,
// what the corpus of every token keeps of each part. The share that holds
// the least for its proportion keeps all its texts and limits the others,
// so a few short texts of one share can hold a whole mix down: it is named
// under the table. A share that takes no text is marked. Nothing shows until
// the figures are read, so no number shows before it is true. A "?" says how
// several libraries and several share tags of one text are counted.

// A number of characters, short: "about 5.4 million".
const chars = (n: number) => characters(n).replace(/ characters?$/, "");

// Beside a share of the dataset form, as it is typed: its part of the plain
// texts once the shares are spread (an empty one takes an even part of what the
// numbers leave) and summed (0.5 alone is 100 %), and, once measured, how many
// plain texts it takes, marked when none.
function ShareSaid({ share, part, parts, found: state }: { share: Share; part: number | null; parts: number[] | null; found: MeasureState }) {
  const sum = (parts ?? []).reduce((t, p) => t + (p > 0 ? p : 0), 0);
  const mine = part ?? NaN;
  const held = figures(state)?.shares.find((h) => h.tag === share.tag.trim());
  if (!(mine >= 0)) return null;
  return (
    <span className="share-said small">
      <strong>{sum ? percent(mine / sum) : "0%"}</strong>
      <span className="aside">of the rest</span>
      {held && <span className={held.texts === 0 ? "not-allowed" : "aside"}>{held.texts === 0 ? "no plain text" : plural(held.texts, "text")}</span>}
    </span>
  );
}

// The line beside the dataset form's button, which stays in view while the
// form scrolls: what the mix keeps, so the click is made knowing it.
function HoldsLine({ libraries, found: state }: { libraries: string[]; found: MeasureState }) {
  const found = figures(state);
  if (!libraries.length || !found) return null;
  const limiting = found.shares.find((s) => s.limits);
  return (
    <span className="small" role="status">
      {plural(found.texts, "text")} · {characters(found.characters)}
      {found.mix && <> · keeps {chars(found.mix.kept_characters + heldOf(found, TAKEN).characters)}{limiting && <>, limited by {limiting.tag}</>}</>}
    </span>
  );
}

function HoldsView({ libraries, shares, found: state }: { libraries: string[]; shares: { tag: string; share: number }[]; found: MeasureState }) {
  if (!libraries.length) return null;
  if (state === null) return null;
  if (typeof state === "string") return <div className="holds small"><Problems problems={[state]} /></div>;
  const found = state;
  const missing = found.shares.filter((s) => s.texts === 0).map((s) => s.tag);
  const limiting = found.shares.find((s) => s.limits);
  const sum = shares.reduce((t, s) => t + s.share, 0) || 1;
  const kept = (all: number, keeps: number | undefined) => (keeps === undefined ? "" : keeps >= all ? "all" : chars(keeps));
  return (
    <div className="holds small" aria-live="polite">
      <div className="table-scroll">
        <table className="ledger-table" aria-label="What its libraries hold">
          <thead><tr><th scope="col">part</th><th scope="col" className="number">its part</th><th scope="col" className="number">texts</th><th scope="col" className="number">characters</th>{found.mix && <th scope="col" className="number" title="What a corpus of every token keeps; a size limit keeps less">kept, no size limit</th>}</tr></thead>
          <tbody>
            {itemKinds(found, true).map(([k, h]) => (
              <tr key={k}><td>{k}s</td><td className="number ledger">~{whole(takenShare(found, k))} of all</td><td className="number ledger">{h.texts.toLocaleString("en")}</td><td className="number">{chars(h.characters)}</td>{found.mix && <td className="number">all</td>}</tr>
            ))}
            {found.shares.length ? found.shares.map((s) => (
              <tr key={s.tag}>
                <td className={s.texts === 0 ? "not-allowed" : undefined}>{s.limits ? <strong title="It keeps all its texts and holds the other shares down">{s.tag}, scarcest</strong> : s.tag}</td>
                <td className="number ledger">{whole(s.share! / sum)} of the rest</td>
                <td className="number ledger">{s.texts.toLocaleString("en")}</td>
                <td className="number">{chars(s.characters)}</td>
                {found.mix && <td className="number">{s.texts > 0 ? kept(s.characters, s.kept_characters) : ""}</td>}
              </tr>
            )) : <tr><td>plain texts</td><td className="number ledger">the rest</td><td className="number ledger">{kindHeld(found, PLAIN).texts.toLocaleString("en")}</td><td className="number">{chars(kindHeld(found, PLAIN).characters)}</td></tr>}
            {found.unshared && found.unshared.texts > 0 && <tr className="aside"><td>no share: left out</td><td></td><td className="number ledger">{found.unshared.texts.toLocaleString("en")}</td><td className="number">{chars(found.unshared.characters)}</td>{found.mix && <td></td>}</tr>}
            {itemKinds(found, false).map(([k, h]) => <tr key={k} className="aside"><td>kind {k}: left out</td><td></td><td className="number ledger">{h.texts.toLocaleString("en")}</td><td className="number">{chars(h.characters)}</td>{found.mix && <td></td>}</tr>)}
          </tbody>
        </table>
      </div>
      {(libraries.length > 1 || found.shares.length > 0) && (
        <p className="aside">How texts count.<Help label="how texts count">{libraries.length > 1 && SUMMED_NOTE}{libraries.length > 1 && found.shares.length > 0 && " "}{found.shares.length > 0 && SHARES_NOTE}</Help></p>
      )}
      {found.mix && limiting && (
        <p>No size limit: {characters(found.mix.kept_characters)} of plain text.<Help label="the limit of the mix">The corpus of every token keeps all of {limiting.tag} and only its proportion of each other share, since a few short texts of {limiting.tag} hold the others down. Add texts tagged {limiting.tag}, lower its share, or accept the smaller corpus. A size limit keeps less of every part, conversations included, in whole texts.</Help></p>
      )}
      {missing.length > 0 && <p className="not-allowed"><strong>Empty share: {missing.join(", ")}.</strong><Help label="an empty share">No plain text of these libraries has the tag. Check its spelling.</Help></p>}
    </div>
  );
}

const percent = (n: number) => `${Math.round(n * 1000) / 10}%`;

// The hint of the dataset form's size, once the size and the measure are there: how a corpus of the size typed splits
// between the conversations and tasks (their part of the bytes of the
// corpus of every token, from the measure) and each share (its proportion
// of the rest), in tokens, each a part the build fills up to, never past.
function sizeSplit(limit: string, shares: { tag: string; share: number }[], state: MeasureState): string | undefined {
  const size = Number(limit);
  if (!limit.trim() || !(size >= 1)) return undefined;
  const found = figures(state);
  if (!found) return undefined;
  const itemShare = takenShare(found);
  const items = Math.floor(size * itemShare);
  const whole = shares.reduce((t, s) => t + s.share, 0) || 1;
  const about = (n: number) => `~${Math.round(n).toLocaleString("en")}`;
  const parts = [...(items > 0 ? [`${takenNames(found)} ${about(items)}`] : []),
    ...(shares.length ? shares.map((s) => `${s.tag} ${about((size - items) * s.share / whole)}`) : [`plain ${about(size - items)}`])];
  return parts.join(" · ");
}

// The vocabulary a dataset publishes, set or cleared among every trained one
// (publishVocabulary), folded under the stages: the stage card publishes the
// dataset's own recipe's; this picks another or none. It starts at the one
// published (none when none is), its own recipe's first, and its button acts
// only once the choice differs. The page reads the dataset again after any
// answer, so after a refusal for a change made since it was read, the next one
// sends the current revision.
function Published({ dataset, trained, onDone }: { dataset: Dataset; trained: Answer<Trained[]> | null; onDone: (a: Answer<any>, words: string) => void }) {
  const [chosen, setChosen] = useState(dataset.vocabulary ?? "");
  const mineFirst = trained?.ok ? [...trained.data].sort((a, b) => Number(b.name === dataset.tokenizer) - Number(a.name === dataset.tokenizer)) : [];
  useEffect(() => setChosen(dataset.vocabulary ?? ""), [dataset.vocabulary, trained]);
  const send = async () => {
    const a = await publishVocabulary(dataset, chosen || null);
    const picked = mineFirst.find((v) => v.sha256 === chosen);
    onDone(a, chosen ? `Published ${picked ? trainedWords(picked) : "the vocabulary chosen"}.` : "Published no vocabulary.");
  };
  if (!trained) return <Skeleton lines={2} label="Reading the vocabularies" />;
  if (!trained.ok) return <Problems problems={trained.problems} />;
  if (dataset.upstream || trained.data.length === 0) return null;
  return (
    <details className="fold small">
      <summary className="cue">Published vocabulary: {dataset.vocabulary ? (() => { const v = mineFirst.find((t) => t.sha256 === dataset.vocabulary); return v ? trainedWords(v) : "not held here"; })() : "none"}</summary>
      <form className="entry-row" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <Select id="published-vocabulary" label="Publish" value={chosen} onChange={(e) => setChosen(e.target.value)}>
          <option value="">none</option>
          {mineFirst.map((v) => <option key={v.sha256} value={v.sha256}>{trainedWords(v)}{v.name === dataset.tokenizer ? "" : ", another recipe's"}</option>)}
        </Select>
        <Button type="submit" variant="quiet" disabled={chosen === (dataset.vocabulary ?? "")}>{chosen ? "Publish it" : "Publish none"}</Button>
      </form>
    </details>
  );
}

// The button that reads the dataset's libraries of this depot again as they
// are now (POST /api/datasets/ID/read-again), shown only where it changes
// something, as the server says (read_again of GET /api/datasets/ID): a
// library's figures become known, or a library holds more or fewer texts now. It changes what the next build takes, so a click first asks on
// one line.
function ReadAgain({ dataset, since, onDone }: { dataset: Dataset; since: number | null; onDone: (a: Answer<any>, words: string) => void }) {
  const [asking, setAsking] = useState(false);
  const change = since ? ` (${since > 0 ? "+" : "−"}${plural(Math.abs(since), "text")})` : "";
  const send = async () => {
    setAsking(false);
    const a = await post(`/api/datasets/${part(dataset.id)}/read-again`, { revision: dataset.revision });
    onDone(a, `Read ${dataset.name}'s libraries again.`);
  };
  if (!asking) return <p className="small"><Button variant="quiet" onClick={() => setAsking(true)}>Read again{change}</Button></p>;
  return (
    <p className="small entry-row" style={{ alignItems: "center" }} role="group" aria-label="Read again">
      <span>The next build takes its libraries as they are now{change}.</span>
      <Button variant="ink" onClick={send}>Read again</Button>
      <Button variant="quiet" onClick={() => setAsking(false)}>Cancel</Button>
    </p>
  );
}

// The UTC time of each of this depot's changes asked for (POST
// /api/changes/times), "YYYY-MM-DD HH:MM"; null until read, and empty when
// the reading fails, so the change numbers show instead.
function useChangeTimes(changes: number[]): Map<number, string> | null {
  const [found, setFound] = useState<Map<number, string> | null>(null);
  const key = [...new Set(changes)].sort((a, b) => a - b).join(",");
  useEffect(() => {
    if (!key) return;
    post<{ change: number; made_at: string | null }[]>("/api/changes/times", { changes: key.split(",").map(Number) })
      .then((a) => setFound(new Map(a.ok ? a.data.filter((r) => r.made_at).map((r) => [r.change, r.made_at!.replace("T", " ").slice(0, 16)]) : [])));
  }, [key]);
  return found;
}

// One dataset: its stages as cards (Stages), the next one's button the
// page's main one, then, folded, what its libraries hold, and the details:
// the libraries it reads, the vocabulary it publishes and its remote; and its edit.
// edit: the address asked for its edit (?edit=1, from a corpus page whose shares came out uneven), opened at once
function OneDataset({ id, made, edit = false }: { id: string; made: boolean; edit?: boolean }) {
  const [shown, setShown] = useState<Answer<Shown> | null>(null);
  const { recipes, trained, reload } = useRecipesAndVocabularies();
  const [editing, setEditing] = useState(edit);
  useEffect(() => { if (edit) setEditing(true); }, [edit]); // the address can ask for it while the page is open
  const [answer, setAnswer] = useState<Answer<any> | null>(null);
  const [said, setSaid] = useState("");
  const read = useCallback(() => { get<Shown>(`/api/datasets/${part(id)}`).then(setShown); }, [id]);
  useEffect(read, [read]);
  usePipelineChange(read);
  const load = useCallback(() => { read(); reload(); }, [read, reload]);
  const changeTimes = useChangeTimes(shown?.ok ? shown.data.libraries.filter((r) => !r.depot).map((r) => r.change) : []);
  const mine = shown?.ok ? shown.data.libraries.filter((r) => r.local).map((r) => r.library) : [];
  useEffect(() => { // while a vocabulary trains, the stages are read again until it ends
    if (!recipes?.ok || !shown?.ok || !recipes.data.find((r) => r.name === shown.data.dataset.tokenizer)?.job) return;
    const timer = window.setTimeout(load, 3000);
    return () => window.clearTimeout(timer);
  }, [recipes, shown, load]);
  if (!shown) return <Skeleton lines={6} label="Reading the dataset" />;
  if (!shown.ok) return <Problems problems={shown.problems} />;
  const { dataset, libraries, corpora } = shown.data;
  const times = changeTimes;
  const done = (a: Answer<any>, words: string) => { setSaid(words); setAnswer(a); load(); };
  return (
    <>
      <PageHead title={dataset.name}
        meta={<><a href={href(["datasets"])}>datasets</a>{shown.data.upstream ? <span>the upstream's</span> : null}<span title="Each edit of a dataset makes a new revision; a corpus records the revision it was built from.">revision {dataset.revision}</span></>}
        note={shown.data.upstream ? "Only the upstream edits it." : undefined}
        actions={<>{!shown.data.upstream && <Button variant="quiet" onClick={() => { setAnswer(null); setEditing(true); }}>Edit</Button>}<ButtonLink variant="quiet" href={href(["export", dataset.id])}>Export texts</ButtonLink></>}
        chain={<>It reads libraries, and gives their texts as files on <a href={href(["export", dataset.id])}>its export page</a>, with no tokenizer, and token files for training on <a href={href(["download", dataset.id])}>its corpora page</a>. {shown.data.upstream && "It belongs to the upstream this depot follows: only the upstream edits it, and a pull brings its edits here. Its corpus can still be built here."}</>}
      />
      {made && !answer && <p className="done cue" role="status">Made {dataset.name}.</p>}
      {!editing && <Outcome answer={answer} done={() => said} />}
      <Stages shown={shown.data} recipes={recipes} trained={trained} here="dataset" onPublished={done} />
      {shown.data.read_again && <ReadAgain dataset={dataset} since={shown.data.figures.since} onDone={done} />}
      {corpora === null && <p className="small aside">Store unreadable here. {shown.data.store}</p>}
      {mine.length > 0 && (
        <details className="fold small">
          <summary className="cue">What its libraries hold</summary>
          {shown.data.figures.measure
            ? <HoldsView libraries={mine} shares={dataset.shares ?? []} found={shown.data.figures.measure} />
            : <p className="small aside">Not known for this revision.<Help label="unknown figures">This revision cannot say what its libraries held at the change it reads each one at: a library changed after that change, one is of another depot, the depot did not store these figures when it was saved, or an erasure reached its texts since. "Read again" reads this depot's libraries as they are now, with their figures.</Help></p>}
          {shown.data.figures.since ? <p className="small"><Since since={shown.data.figures.since} /></p> : null}
        </details>
      )}
      <details className="fold small">
      <summary className="cue">More</summary>
      <Section title="Libraries" help={`Each library is read as it stood when it was ticked, and an edit keeps that reading: a text put in or taken out after that changes nothing here. "Read again" reads this depot's libraries as they are now; a library of another depot keeps its reading.`}>
        <div className="table-scroll cards">
          <table className="ledger-table" aria-label="The libraries it reads">
            <thead><tr><th scope="col">library</th><th scope="col">now</th><th scope="col">depot</th><th scope="col">read as of (UTC)</th></tr></thead>
            <tbody>
              {libraries.map((r) => (
                <tr key={`${r.depot}-${r.library}`}>
                  <td className="wide-cell" data-label="">{r.local ? <a href={href(["libraries", r.library])}>{r.name}</a> : <span className="ledger">{r.library}</span>}</td>
                  <td data-label="now">{r.local ? (r.locked_at ? "locked" : "open") : <span className="aside">unknown</span>}</td>
                  <td className="ledger" data-label="depot">{r.depot ? <Cut text={r.depot} /> : "this depot"}</td>
                  <td className="ledger nowrap" data-label="read as of" title={`change ${r.change} of its depot`}>{!r.depot && times?.get(r.change) ? times.get(r.change) : `change ${r.change}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <Published dataset={dataset} trained={trained} onDone={done} />
      <details className="fold small">
        <summary className="cue">For another machine</summary>
        <p className="aside">Read there by <span className="ledger">wpt depot dataset import</span>.</p>
        <Button variant="quiet" onClick={() => navigator.clipboard?.writeText(JSON.stringify(shown.data.remote, null, 2))}>Copy it</Button>
        <pre className="remote ledger small">{JSON.stringify(shown.data.remote, null, 2)}</pre>
      </details>
      </details>
      <Dialog open={editing} onClose={() => setEditing(false)} title={`Edit ${dataset.name}`}>
        {editing && <DatasetForm dataset={dataset} onDone={(a, name) => { setSaid(`Saved ${name}. Older corpora keep the dataset as it was.`); setAnswer(a); if (a.ok) { setEditing(false); load(); } }} />}
      </Dialog>
    </>
  );
}

// An exported remote, to copy to another machine.
function Remote({ text }: { text: string }) {
  return (
    <>
      <TextArea label="Its description" hint={<>Copy it to another machine, then import it there with <span className="ledger">wpt depot dataset import</span>.</>} className="code" rows={14} value={text} readOnly />
      <Button variant="ink" onClick={() => navigator.clipboard?.writeText(text)}>Copy it</Button>
    </>
  );
}

export const datasets: Page = {
  path: "datasets", title: "Datasets", group: "data",
  Component: ({ rest, query }) => (rest[0] ? <OneDataset key={rest[0]} id={rest[0]} made={query.get("made") === "1"} edit={query.get("edit") === "1"} /> : <Datasets key={query.toString()} query={query} />),
};
