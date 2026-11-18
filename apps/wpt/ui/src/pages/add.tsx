// Add one plain text, or one item of a kind whose content is edited with a
// form built from the kind's JSON Schema, under review, with its licenses,
// notice, tags, title, authors and address (wpt depot add). Licenses are
// chosen with the license picker (license-picker.tsx). A plain text can be
// read from a file, chosen or dropped on the text box, the title then taken
// from the file's name. "Add another" empties the text, the title and the
// address, and keeps the licenses, notice, authors and tags, which it says,
// then puts the cursor back in the text box; choosing another kind for the
// next text takes the kept tags off, since they told the last kind apart,
// and says so with a button that puts them back. The tags box holds only the
// tag being typed, which Enter or a comma adds. The depot's ten most used tags
// stand under it in a fixed order, each a toggle that adds or removes it and
// shows whether it is chosen, so a click never moves the others, and a
// button shows the rest; typing narrows them to those holding the words typed. A chosen tag none of those
// toggles shows shows as a chip, removed by a click. An item's content can
// also be read from a JSON file, chosen with its own button. The
// texts added since the page opened are linked as one filter of the
// Texts page.

import { Button, Select, TextArea, TextInput } from "@workers-intelligence-commons/components";
import { useEffect, useRef, useState, type DragEvent, type KeyboardEvent } from "react";
import { get, post, type Answer } from "../api";
import { LicensePicker } from "../license-picker";
import { Confirm, Help, only, optional, Outcome, PageHead, plural, Problems, required, RequiredNote, Section, type Declaration } from "../parts";
import { href } from "../route";
import { KIND_WORDS, kindStart } from "../kind-words";
import { SchemaForm, initial, type Schema } from "../schema-form";
import type { Page } from "./index";

type Kind = { id: string; meaning: string; schema: Schema };

const list = (text: string) => text.split(",").map((s) => s.trim()).filter(Boolean);
const TOP_TAGS = 10; // the depot's tags shown as toggles before "more"
const an = (word: string) => `${/^[aeiou]/.test(word) ? "an" : "a"} ${word}`;

function Add() {
  const [kinds, setKinds] = useState<Kind[]>([]);
  const [kind, setKind] = useState("");
  const [text, setText] = useState("");
  const [content, setContent] = useState<unknown>(null);
  const [chosen, setChosen] = useState<string[]>([]);
  const [notice, setNotice] = useState("");
  const [title, setTitle] = useState("");
  const [authors, setAuthors] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [typing, setTyping] = useState(""); // the tag being typed, not added yet
  const [address, setAddress] = useState("");
  const [answer, setAnswer] = useState<Answer<{ id: string }> | null>(null);
  const [added, setAdded] = useState(""); // the title or start of the text last added, for the line that says so
  const [kept, setKept] = useState(false); // the fields kept from the text added before
  const [addedKind, setAddedKind] = useState<string | null>(null); // the kind of the text added before ("" a plain text)
  const [cleared, setCleared] = useState<string[]>([]); // kept tags taken off when the kind changed
  const [allTags, setAllTags] = useState(false); // every depot tag shown as a toggle, not only the most used
  const [fileProblem, setFileProblem] = useState("");
  const [addedIds, setAddedIds] = useState<string[]>([]); // every text added on this page since it opened
  const [known, setKnown] = useState<{ tag: string; texts: number }[]>([]); // the depot's tags, to suggest
  const top = useRef<HTMLDivElement>(null);
  useEffect(() => { get<Kind[]>("/api/kinds").then((a) => a.ok && setKinds(a.data)); }, []);
  useEffect(() => { get<{ tag: string; texts: number }[]>("/api/tags").then((a) => a.ok && setKnown([...a.data].sort((x, y) => y.texts - x.texts))); }, []);
  const schema = kinds.find((k) => k.id === kind)?.schema;
  const start = (id: string) => {
    const found = kinds.find((k) => k.id === id);
    return found ? kindStart(id, initial(found.schema, found.schema)) : null;
  };
  const pick = (id: string) => {
    setKind(id);
    setContent(start(id));
    // the kept tags told the last text's kind apart: another kind takes them off, a way back offered
    if (kept && addedKind !== null) {
      if (id !== addedKind && tags.length) { setCleared(tags); setTags([]); }
      else if (id === addedKind && cleared.length) { setTags([...new Set([...tags, ...cleared])]); setCleared([]); }
    }
  };
  const send = async (declaration: Declaration) => {
    const a = await post<{ id: string }>("/api/texts", {
      ...(kind ? { kind, content } : { text }), licenses: chosen, notice: notice || null, title: title || null,
      authors: authors ? list(authors) : null, tags: [...new Set([...tags, ...list(typing)])], address: address || null, declaration,
    });
    setAnswer(a);
    if (a.ok) {
      setTags([...new Set([...tags, ...list(typing)])]); // a tag left typed was sent: it shows as one now
      setTyping("");
      setAdded(title.trim() || (kind ? an(kind) : `the text starting “${text.trim().slice(0, 40)}”`));
      setAddedKind(kind);
      setCleared([]);
      setAddedIds([...addedIds, a.data.id]);
    }
  };
  const another = () => {
    setAnswer(null);
    setText("");
    setContent(kind && schema ? start(kind) : null);
    setTitle("");
    setAddress("");
    setKept(true);
    // back to the top, the cursor in the text box, ready for the next text
    window.scrollTo({ top: 0 });
    window.setTimeout(() => (top.current?.querySelector<HTMLElement>("textarea") ?? top.current?.querySelector<HTMLElement>("input, select"))?.focus(), 0);
  };
  // the texts added on this page, opened on the Texts page by their ids, named in words
  const addedHere = href(["texts", addedIds.at(-1) ?? ""], { id: addedIds, what: `the ${plural(addedIds.length, "text")} added just now` });
  const fragment = typing.trim().toLowerCase();
  const matching = known.filter((t) => !fragment || t.tag.toLowerCase().includes(fragment));
  const suggested = allTags ? matching : matching.slice(0, TOP_TAGS);
  const more = matching.length - suggested.length;
  const sentTags = [...new Set([...tags, ...list(typing)])];
  const toggleTag = (tag: string) => { setTags(tags.includes(tag) ? tags.filter((t) => t !== tag) : [...tags, tag]); if (fragment) setTyping(""); };
  // a comma or Enter adds what is typed; Backspace in an empty box takes back the last tag
  const typed = (value: string) => {
    if (!value.includes(",")) { setTyping(value); return; }
    const words = value.split(",");
    setTags([...new Set([...tags, ...list(words.slice(0, -1).join(","))])]);
    setTyping(words.at(-1) ?? "");
  };
  const tagKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && typing.trim()) { e.preventDefault(); typed(`${typing},`); }
    else if (e.key === "Backspace" && typing === "" && tags.length) setTags(tags.slice(0, -1));
  };
  const picker = useRef<HTMLInputElement>(null);
  // a file chosen or dropped: its text, and its name as the title when none is typed
  const read = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    setFileProblem("");
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      setText(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
      if (!title.trim()) setTitle(file.name);
    } catch {
      setFileProblem(`${file.name} is not UTF-8 text, so it cannot be added as a plain text.`);
    }
  };
  const jsonPicker = useRef<HTMLInputElement>(null);
  // an item's content read from a JSON file: the form then shows it, to check or change before adding
  const readJson = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    setFileProblem("");
    try {
      setContent(JSON.parse(await file.text()));
      if (!title.trim()) setTitle(file.name);
    } catch (error) {
      setFileProblem(`${file.name} is not JSON: ${error instanceof Error ? error.message : String(error)}.`);
    }
  };
  const dropped = (e: DragEvent) => { e.preventDefault(); read(e.dataTransfer.files); };
  const why = chosen.length === 0 ? "Tick a license first." : !kind && text === "" ? "Write or choose the text first." : null;
  return (
    <>
      <PageHead title="Add a text"
        chain={<>What you add is under review among the <a href={href(["texts"])}>texts</a>, and joins no <a href={href(["libraries"])}>library</a> until it is admitted. Its licenses are those of the <a href={href(["licenses"])}>licenses</a> page.</>} />
      <div className="form-narrow" ref={top}>
        <RequiredNote />
        <Section title="The text">
          <Select label={required("What it is")} required value={kind} onChange={(e) => pick(e.target.value)}>
            <option value="">A plain text: prose, code or any other text file</option>
            {kinds.map((k) => <option key={k.id} value={k.id}>An item: {k.id}</option>)}
          </Select>
          {kind && <p className="wic-field-hint">{(kinds.find((k) => k.id === kind)?.meaning ?? "").replace(/^./, (c) => c.toUpperCase())}.</p>}
          {cleared.length > 0 && (
            <p className="form-note small row tight">
              <span>Tags {cleared.join(", ")} taken off.<Help label="the tags taken off">They came from the text before, which was {addedKind ? an(addedKind) : "a plain text"}.</Help></span>
              <Button variant="quiet" onClick={() => { setTags([...new Set([...tags, ...cleared])]); setCleared([]); }}>Put them back</Button>
            </p>
          )}
          {kind && schema
            ? <>
                <div className="file-pick small">
                  <Button variant="quiet" onClick={() => jsonPicker.current?.click()}>Load it from a JSON file</Button>
                  <span className="aside">or fill in below.</span>
                  <input ref={jsonPicker} type="file" accept=".json,application/json" hidden onChange={(e) => { readJson(e.target.files); e.target.value = ""; }} />
                </div>
                {fileProblem && <Problems problems={[fileProblem]} />}
                <SchemaForm value={content} schema={schema} root={schema} name={`The ${kind}`} words={KIND_WORDS[kind]} need="required" onChange={setContent} />
              </>
            : (
              <div onDragOver={(e) => e.preventDefault()} onDrop={dropped}>
                <TextArea label={required(only("Text", "for a plain text"))} required rows={8} value={text} className="code" onChange={(e) => setText(e.target.value)}
                  hint="Type, paste or drop a file." />
                <div className="file-pick small">
                  <Button variant="quiet" onClick={() => picker.current?.click()}>Choose a file</Button>
                  <input ref={picker} type="file" hidden onChange={(e) => { read(e.target.files); e.target.value = ""; }} />
                </div>
                {fileProblem && <Problems problems={[fileProblem]} />}
              </div>
            )}
        </Section>

        {kept && <p className="form-note small">Kept from the text before.<Help label="what is kept">The licenses, notice{cleared.length ? " and authors" : ", authors and tags"} are kept from the text added before{tags.length && !cleared.length ? ` (tags ${tags.join(", ")})` : ""}; change them if this one differs.</Help></p>}
        <Section title="Licenses and notice">
          <LicensePicker chosen={chosen} onChange={setChosen} />
          <TextArea label={optional("Notice")} hint="The copyright notice exactly as the source gives it, or why the work is in the public domain." rows={3} value={notice} onChange={(e) => setNotice(e.target.value)} />
        </Section>

        <Section title="Credits">
          <div className="fields">
            <TextInput label={optional("Title")} value={title} onChange={(e) => setTitle(e.target.value)} />
            <TextInput label={optional("Authors")} value={authors} hint="Separated by commas." onChange={(e) => setAuthors(e.target.value)} />
            <div className="full-row">
              <TextInput label={optional("Tags")} hint="Enter or a comma adds the tag typed; the depot's tags below add or remove one with a click." value={typing} placeholder={tags.length ? "another tag" : "wikipedia, en"} onChange={(e) => typed(e.target.value)} onKeyDown={tagKey} />
              {tags.some((t) => !suggested.some((s) => s.tag === t)) && (
                <div className="row tight small tag-chips" aria-label="Its other tags">
                  {tags.filter((t) => !suggested.some((s) => s.tag === t)).map((t) => (
                    <span key={t} className="chip">{t}
                      <button type="button" className="forget wic-focus" aria-label={`Remove the tag ${t}`} title="Remove it"
                        onClick={() => setTags(tags.filter((x) => x !== t))}>✕</button>
                    </span>
                  ))}
                </div>
              )}
              {suggested.length > 0 && (
                <div className="row tight small tag-suggestions" aria-label="Tags the depot uses">
                  {suggested.map((t) => {
                    const on = tags.includes(t.tag);
                    return <Button key={t.tag} variant="quiet" aria-pressed={on} onClick={() => toggleTag(t.tag)}
                      title={`${t.texts.toLocaleString("en")} texts have it; ${on ? "a click removes it" : "a click adds it"}`}>{on ? "✓" : "+"} {t.tag}</Button>;
                  })}
                  {more > 0 && <Button variant="quiet" onClick={() => setAllTags(true)} aria-expanded={false}>{plural(more, "more tag")}</Button>}
                  {allTags && matching.length > TOP_TAGS && <Button variant="quiet" onClick={() => setAllTags(false)} aria-expanded>Fewer tags</Button>}
                  {tags.length > 1 && <Button variant="quiet" onClick={() => setTags([])}>Remove every tag</Button>}
                </div>
              )}
            </div>
            <TextInput className="full-row" label={optional("Address")} value={address} placeholder="https://example.org/where-it-came-from" onChange={(e) => setAddress(e.target.value)} />
          </div>
        </Section>

        <Section>
          {answer?.ok && (
            <div className="row tight">
              <Outcome answer={answer} done={(d) => <>Added {added}, under review: <a href={href(["texts", d.id])}>open it</a>{addedIds.length > 1 && <>, or <a href={addedHere}>see the {addedIds.length} texts added here</a></>}.</>} />
              <Button variant="primary" onClick={another}>Add another</Button>
            </div>
          )}
          {answer && !answer.ok && <Outcome answer={answer} done={() => null} />}
          {!answer?.ok && addedIds.length > 0 && <p className="small aside"><a href={addedHere}>{plural(addedIds.length, "text")} added here</a>.</p>}
          <Confirm label="Confirm and add" disabled={why !== null || answer?.ok === true} onConfirm={send}
            note={answer?.ok ? <>Added.</> : why ?? <>Under {chosen.join(" or ")}, {sentTags.length ? `tagged ${sentTags.join(", ")}` : "no tag"}, under review.</>} />
        </Section>
      </div>
    </>
  );
}

export const add: Page = { path: "add", title: "Add a text", group: null, Component: Add };
