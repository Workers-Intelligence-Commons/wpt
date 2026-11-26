// An editor for any JSON document, built from its JSON schema: every field
// the schema names is shown and editable, so a new field in a shape needs no
// change here. A value that may take one of several shapes (oneOf, anyOf)
// shows the shape it has and lets a person pick another; an object whose
// keys are free (additionalProperties, or no properties named) lists its
// keys and takes new ones, a value of no schema as text, a number or a
// boolean. An
// object whose allOf holds "if/then/else" rules shows the fields of the rule
// its value meets (a block of type text shows its text field, one of type
// tool_call its id, name and arguments), and when a change makes it meet
// another rule, drops the fields no rule names any more and adds the newly
// required ones. A rule's condition is read from its properties' const and
// enum alone; any other condition never adds fields. It only shapes the
// input; the server's rules decide what is valid. Words (kind-words.ts)
// give the fields, the entries of a list and the choices plain names, and
// a list's new entry its start, where a kind has them; a field without them
// shows its own name.

import { Button, Select, TextArea, TextInput } from "@workers-intelligence-commons/components";
import { useState, type ReactNode } from "react";
import { optional, required } from "./parts";

export type Schema = {
  type?: string | string[];
  properties?: Record<string, Schema>;
  additionalProperties?: Schema | boolean;
  required?: string[];
  items?: Schema;
  enum?: string[];
  const?: unknown;
  oneOf?: Schema[];
  anyOf?: Schema[];
  $ref?: string;
  description?: string;
  allOf?: Schema[];
  if?: Schema;
  then?: Schema;
  else?: Schema;
};

type Json = unknown;
// need: whether the object holding this value requires it (its label then
// carries the red star, or says "optional"); none for an entry of a list.
// remove: the button that removes this value from what holds it, shown on
// the value's own line (beside a group's name, or beside a field).
// field: the value's own name in the object that holds it, which words look
// up (name, for an entry of a list, is the entry's name in words).
type Props = { value: Json; schema: Schema; root: Schema; name: string; field?: string; words?: Words; need?: "required" | "optional"; remove?: ReactNode; onChange: (value: Json) => void };

// Plain names for a kind's fields (names), for one entry of a list field
// (entries: "messages" gives "message"), and for the choices of a field
// (values); start: a list field's new entry, given the entries it holds,
// or undefined for the schema's own start.
export type Words = {
  names?: Record<string, string>; entries?: Record<string, string>; values?: Record<string, string>;
  start?: (field: string, entries: Json[]) => Json | undefined;
};

const ANY_SCALAR: Schema = { type: ["string", "number", "boolean"] };
const LONG = new Set(["text", "attribution", "note", "reason", "question"]);

function resolve(schema: Schema, root: Schema): Schema {
  if (!schema.$ref) return schema;
  const found = schema.$ref.replace(/^#\//, "").split("/").reduce<any>((at, key) => at?.[key], root);
  return found ? resolve(found, root) : schema;
}

// Whether an object value meets an "if" condition, read from its
// properties' const and enum (a property the value lacks meets it, as in
// JSON Schema); a condition of any other shape is not met.
function meets(condition: Schema, value: Json): boolean {
  if (!condition.properties || Object.keys(condition).some((key) => key !== "properties")) return false;
  const object = (value && typeof value === "object" ? value : {}) as Record<string, Json>;
  return Object.entries(condition.properties).every(([key, rule]) => {
    if (!(key in object)) return true;
    if (rule.const !== undefined) return object[key] === rule.const;
    if (rule.enum) return rule.enum.includes(object[key] as string);
    return false;
  });
}

// An object schema with the fields its allOf rules add for this value: each
// rule's then (when met) or else, its properties the schema does not name
// yet, and its required fields.
function effective(s: Schema, value: Json): Schema {
  if (!s.allOf) return s;
  const properties = { ...(s.properties ?? {}) };
  const needed = new Set(s.required ?? []);
  for (const rule of s.allOf) {
    if (!rule.if) continue;
    const branch = meets(rule.if, value) ? rule.then : rule.else;
    for (const [key, field] of Object.entries(branch?.properties ?? {}))
      if (!(key in properties) && typeof field === "object") properties[key] = field;
    for (const key of branch?.required ?? []) needed.add(key);
  }
  return { ...s, properties, required: [...needed] };
}

// A list's name for one of its entries: "messages" gives "message".
const one = (name: string) => (name.length > 3 && name.endsWith("s") && !name.endsWith("ss") ? name.slice(0, -1) : name);

// The JSON kind of a value, as a schema's type names it.
function kind(value: Json): string {
  if (Array.isArray(value)) return "array";
  if (value === null) return "null";
  if (typeof value === "number") return "number";
  return typeof value;
}

function fits(schema: Schema, value: Json): boolean {
  const types = Array.isArray(schema.type) ? schema.type : schema.type ? [schema.type] : [];
  const found = kind(value);
  return types.includes(found) || (found === "number" && types.includes("integer"));
}

// The alternative of a oneOf or anyOf the value has: of those of its kind,
// the first whose required fields it holds, else the first of its kind.
function chosen(choices: Schema[], value: Json, root: Schema): number {
  const resolved = choices.map((choice) => resolve(choice, root));
  const ofKind = resolved.map((s, i) => [s, i] as const).filter(([s]) => fits(s, value));
  const whole = ofKind.find(([s]) => (s.required ?? []).every((key) => value && typeof value === "object" && key in value));
  return (whole ?? ofKind[0] ?? [null, 0])[1];
}

// A value typed in a field that may be a number, a boolean or text.
function scalar(text: string): Json {
  if (text === "true" || text === "false") return text === "true";
  return /^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(text) ? Number(text) : text;
}

// A new value of that schema: its required fields, each at its own start.
export function initial(schema: Schema, root: Schema): Json {
  const s = resolve(schema, root);
  const choices = s.oneOf ?? s.anyOf;
  if (choices) return initial(choices[0], root);
  if (s.const !== undefined) return s.const;
  if (s.enum) return s.enum[0];
  if (s.type === "object") {
    const start = Object.fromEntries((s.required ?? []).map((key) => [key, initial(s.properties?.[key] ?? {}, root)]));
    const shaped = effective(s, start);
    return Object.fromEntries((shaped.required ?? []).map((key) => [key, key in start ? start[key] : initial(shaped.properties?.[key] ?? {}, root)]));
  }
  if (s.type === "array") return [];
  if (s.type === "integer" || s.type === "number") return 0;
  return "";
}

// A group's name, with the buttons that act on the group on the same line.
function Head({ label, children }: { label: ReactNode; children?: ReactNode }) {
  return <legend className="schema-head"><span className="wic-label">{label}</span>{children}</legend>;
}

// A field with the button that removes it beside it.
function Line({ remove, children }: { remove?: ReactNode; children: ReactNode }) {
  return remove ? <div className="entry-row">{children}{remove}</div> : <>{children}</>;
}

function Remove({ what, onClick }: { what: string; onClick: () => void }) {
  return <Button variant="quiet" onClick={onClick}>Remove {what}</Button>;
}

export function SchemaForm({ value, schema, root, name: given, field = given, words, need, remove, onChange }: Props) {
  const s = resolve(schema, root);
  const hint = s.description;
  const name = words?.names?.[field] ?? given;
  const label: ReactNode = need === "required" ? required(name) : need === "optional" ? optional(name) : name;
  const isRequired = need === "required";

  if (s.const !== undefined) return <Line remove={remove}><p className="field-line"><span className="aside">{name}</span> <code className="ledger">{JSON.stringify(value)}</code></p></Line>;

  const choices = s.oneOf ?? s.anyOf;
  if (choices) {
    const at = chosen(choices, value, root);
    return (
      <div>
        {choices.length > 1 && (
          <Select label={need === "required" ? required(`${name}: its shape`) : `${name}: its shape`} required={isRequired} value={at} onChange={(e) => onChange(initial(choices[Number(e.target.value)], root))}>
            {choices.map((choice, i) => {
              const r = resolve(choice, root);
              return <option key={i} value={i}>{r.description ?? String(r.type ?? `shape ${i + 1}`)}</option>;
            })}
          </Select>
        )}
        <SchemaForm value={value} schema={choices[at]} root={root} name={given} field={field} words={words} need={need} remove={remove} onChange={onChange} />
      </div>
    );
  }

  if (s.enum)
    return (
      <Line remove={remove}>
        <Select label={label} required={isRequired} hint={hint} value={String(value)} onChange={(e) => onChange(e.target.value)}>
          {s.enum.map((option) => <option key={option} value={option}>{words?.values?.[option] ?? option}</option>)}
        </Select>
      </Line>
    );

  if (s.type === "object" && !s.properties && s.additionalProperties !== false) // free keys; with no schema for their values, each takes text, a number or a boolean
    return <FreeObject value={value} schema={typeof s.additionalProperties === "object" ? s : { ...s, additionalProperties: ANY_SCALAR }} root={root} name={name} words={words} need={need} remove={remove} onChange={onChange} />;

  if (s.type === "object") {
    const object = (value && typeof value === "object" ? value : {}) as Record<string, Json>;
    const shaped = effective(s, object);
    const needed = new Set(shaped.required ?? []);
    const keys = Object.keys(shaped.properties ?? {});
    // a change that makes the value meet other rules: the fields no rule names go, the newly required ones come
    const change = (next: Record<string, Json>) => {
      if (!s.allOf) return onChange(next);
      const after = effective(s, next);
      const named = new Set(Object.keys(after.properties ?? {}));
      const kept = Object.fromEntries(Object.entries(next).filter(([key]) => named.has(key) || !(key in (shaped.properties ?? {}))));
      for (const key of after.required ?? []) if (!(key in kept)) kept[key] = initial(after.properties?.[key] ?? {}, root);
      onChange(kept);
    };
    const missing = keys.filter((key) => !(key in object));
    return (
      <fieldset className="schema-object">
        <Head label={label}>
          {missing.map((key) => <Button key={key} variant="quiet" onClick={() => change({ ...object, [key]: initial(shaped.properties![key], root) })}>Add {words?.names?.[key] ?? key}</Button>)}
          {remove}
        </Head>
        {hint && <p className="wic-field-hint">{hint}</p>}
        {keys.filter((key) => key in object).map((key) => (
          <SchemaForm key={key} value={object[key]} schema={shaped.properties![key]} root={root} name={key} words={words} need={needed.has(key) ? "required" : "optional"}
            remove={needed.has(key) ? undefined : <Remove what={words?.names?.[key] ?? key} onClick={() => { const { [key]: _, ...rest } = object; change(rest); }} />}
            onChange={(v) => change({ ...object, [key]: v })} />
        ))}
      </fieldset>
    );
  }

  if (s.type === "array") {
    const list = Array.isArray(value) ? value : [];
    const entry = words?.entries?.[field] ?? one(name);
    const added = () => words?.start?.(field, list) ?? initial(s.items ?? {}, root);
    return (
      <fieldset className="schema-object">
        <Head label={label}>
          <Button variant="quiet" onClick={() => onChange([...list, added()])}>{entry === name ? `Add to ${name}` : `Add ${/^[aeiou]/i.test(entry) ? "an" : "a"} ${entry}`}</Button>
          {remove}
        </Head>
        {hint && <p className="wic-field-hint">{hint}</p>}
        {list.map((item, i) => (
          <SchemaForm key={i} value={item} schema={s.items ?? {}} root={root} name={`${entry[0].toUpperCase()}${entry.slice(1)} ${i + 1}`} field={`${field} ${i + 1}`} words={words}
            remove={<Remove what={`${entry} ${i + 1}`} onClick={() => onChange(list.filter((_, j) => j !== i))} />}
            onChange={(v) => onChange(list.map((old, j) => (j === i ? v : old)))} />
        ))}
      </fieldset>
    );
  }

  if (s.type === "integer" || s.type === "number")
    return <Line remove={remove}><TextInput label={label} required={isRequired} hint={hint} type="number" value={Number(value)}
      onChange={(e) => onChange(s.type === "integer" ? parseInt(e.target.value || "0", 10) : Number(e.target.value))} /></Line>;

  if (Array.isArray(s.type)) // a number, a boolean or text, as typed
    return <Line remove={remove}><TextInput label={label} required={isRequired} hint={hint} value={typeof value === "string" ? value : JSON.stringify(value)} onChange={(e) => onChange(scalar(e.target.value))} /></Line>;

  const text = typeof value === "string" ? value : JSON.stringify(value);
  const long = LONG.has(name) || text.includes("\n") || text.length > 90;
  return (
    <Line remove={remove}>
      {long
        ? <TextArea label={label} required={isRequired} hint={hint} value={text} rows={Math.min(20, Math.max(3, text.split("\n").length + 1))} onChange={(e) => onChange(e.target.value)} />
        : <TextInput label={label} required={isRequired} hint={hint} value={text} onChange={(e) => onChange(e.target.value)} />}
    </Line>
  );
}

// An object whose keys are free: each key with its value, removable, and a
// name to add a new key.
function FreeObject({ value, schema, root, name, words, need, remove, onChange }: Props) {
  const [key, setKey] = useState("");
  const object = (value && typeof value === "object" && !Array.isArray(value) ? value : {}) as Record<string, Json>;
  const item = schema.additionalProperties as Schema;
  return (
    <fieldset className="schema-object">
      <Head label={need === "required" ? required(name) : need === "optional" ? optional(name) : name}>{remove}</Head>
      {schema.description && <p className="wic-field-hint">{schema.description}</p>}
      {Object.keys(object).map((k) => (
        <SchemaForm key={k} value={object[k]} schema={item} root={root} name={k} words={words}
          remove={<Remove what={k} onClick={() => { const { [k]: _, ...rest } = object; onChange(rest); }} />}
          onChange={(v) => onChange({ ...object, [k]: v })} />
      ))}
      <div className="entry-row">
        <TextInput label={`A new name in ${name}`} value={key} onChange={(e) => setKey(e.target.value)} />
        <Button variant="quiet" disabled={!key || key in object} onClick={() => { onChange({ ...object, [key]: initial(item, root) }); setKey(""); }}>Add it</Button>
      </div>
    </fieldset>
  );
}
