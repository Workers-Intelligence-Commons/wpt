// Add one plain text, or one item of a kind whose content is edited with a
// form built from the kind's JSON Schema, under review, with its licenses,
// notice, tags, title, authors and address (wpt depot add). Licenses are
// chosen with the license picker (license-picker.tsx).

import { Button, Select, TextArea, TextInput } from "@workers-intelligence-commons/components";
import { useEffect, useState } from "react";
import { get, post, type Answer } from "../api";
import { LicensePicker } from "../license-picker";
import { Confirm, only, optional, Outcome, PageHead, required, RequiredNote, Section, type Declaration } from "../parts";
import { href } from "../route";
import { SchemaForm, initial, type Schema } from "../schema-form";
import type { Page } from "./index";

type Kind = { id: string; meaning: string; schema: Schema };

const list = (text: string) => text.split(",").map((s) => s.trim()).filter(Boolean);

function Add() {
  const [kinds, setKinds] = useState<Kind[]>([]);
  const [kind, setKind] = useState("");
  const [text, setText] = useState("");
  const [content, setContent] = useState<unknown>(null);
  const [chosen, setChosen] = useState<string[]>([]);
  const [notice, setNotice] = useState("");
  const [title, setTitle] = useState("");
  const [authors, setAuthors] = useState("");
  const [tags, setTags] = useState("");
  const [address, setAddress] = useState("");
  const [answer, setAnswer] = useState<Answer<{ id: string }> | null>(null);
  useEffect(() => { get<Kind[]>("/api/kinds").then((a) => a.ok && setKinds(a.data)); }, []);
  const schema = kinds.find((k) => k.id === kind)?.schema;
  const pick = (id: string) => {
    setKind(id);
    const found = kinds.find((k) => k.id === id);
    setContent(found ? initial(found.schema, found.schema) : null);
  };
  const send = async (declaration: Declaration) => {
    const a = await post<{ id: string }>("/api/texts", {
      ...(kind ? { kind, content } : { text }), licenses: chosen, notice: notice || null, title: title || null,
      authors: authors ? list(authors) : null, tags: list(tags), address: address || null, declaration,
    });
    setAnswer(a);
    if (a.ok) { setText(""); setContent(kind && schema ? initial(schema, schema) : null); }
  };
  return (
    <>
      <PageHead title="Add a text"
        chain={<>What you add is under review among the <a href={href(["texts"])}>texts</a>, and joins no <a href={href(["libraries"])}>library</a> until it is admitted. Its licenses are those of the <a href={href(["licenses"])}>licenses</a> page.</>} />
      <div className="form-narrow">
        <RequiredNote />
        <Section title="The text">
          <Select label={required("What it is")} required value={kind} onChange={(e) => pick(e.target.value)}>
            <option value="">A plain text</option>
            {kinds.map((k) => <option key={k.id} value={k.id}>An item of kind {k.id}: {k.meaning}</option>)}
          </Select>
          {kind && schema
            ? <SchemaForm value={content} schema={schema} root={schema} name={kind} need="required" onChange={setContent} />
            : <TextArea label={required(only("Text", "for a plain text"))} required rows={8} value={text} onChange={(e) => setText(e.target.value)} />}
        </Section>

        <Section title="Licenses and notice">
          <LicensePicker chosen={chosen} onChange={setChosen} />
          <TextArea label={optional("Notice")} rows={3} value={notice} onChange={(e) => setNotice(e.target.value)} hint="The copyright notice exactly as the source gives it, or why the work is in the public domain." />
        </Section>

        <Section title="Credits">
          <div className="fields">
            <TextInput label={optional("Title")} value={title} onChange={(e) => setTitle(e.target.value)} />
            <TextInput label={optional("Authors")} value={authors} onChange={(e) => setAuthors(e.target.value)} hint="Separated by commas." />
            <TextInput label={optional("Tags")} value={tags} placeholder="wikipedia, en" onChange={(e) => setTags(e.target.value)} hint="Separated by commas." />
            <TextInput label={optional("Address")} value={address} placeholder="https://example.org/where-it-came-from" onChange={(e) => setAddress(e.target.value)} hint="Where it came from." />
          </div>
        </Section>

        <Section>
          <Confirm label="Confirm and add" disabled={chosen.length === 0 || (!kind && text === "")} onConfirm={send}
            note={<>Adds {kind ? `this item of kind ${kind}` : "this text"} under {chosen.length ? chosen.join(" or ") : "no license yet"}, under review.</>} />
          <Outcome answer={answer} done={(d) => <>Added <a href={href(["texts", d.id])}>{d.id}</a>, under review.</>} />
          {answer?.ok && <p><Button variant="quiet" onClick={() => setAnswer(null)}>Add another</Button></p>}
        </Section>
      </div>
    </>
  );
}

export const add: Page = { path: "add", title: "Add a text", group: null, Component: Add };
