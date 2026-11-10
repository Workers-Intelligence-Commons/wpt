// An editor for any JSON document, built from its JSON schema: every field
// the schema names is shown and editable, so a new field in a shape needs no
// change here. A value that may take one of several shapes (oneOf, anyOf)
// shows the shape it has and lets a person pick another; an object whose
// keys are free (additionalProperties) lists its keys and takes new ones. It
// only shapes the input; the server's rules decide what is valid
// (conditions such as "if/then" are left to them).

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
};

type Json = unknown;
// need: whether the object holding this value requires it (its label then
// carries the red star, or says "optional"); none for an entry of a list.
// remove: the button that removes this value from what holds it, shown on
// the value's own line (beside a group's name, or beside a field).
type Props = { value: Json; schema: Schema; root: Schema; name: string; need?: "required" | "optional"; remove?: ReactNode; onChange: (value: Json) => void };

const LONG = new Set(["text", "attribution", "note", "reason", "question"]);

function resolve(schema: Schema, root: Schema): Schema {
  if (!schema.$ref) return schema;
  const found = schema.$ref.replace(/^#\//, "").split("/").reduce<any>((at, key) => at?.[key], root);
  return found ? resolve(found, root) : schema;
}

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
  if (s.type === "object")
    return Object.fromEntries((s.required ?? []).map((key) => [key, initial(s.properties?.[key] ?? {}, root)]));
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

export function SchemaForm({ value, schema, root, name, need, remove, onChange }: Props) {
  const s = resolve(schema, root);
  const hint = s.description;
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
        <SchemaForm value={value} schema={choices[at]} root={root} name={name} need={need} remove={remove} onChange={onChange} />
      </div>
    );
  }

  if (s.enum)
    return (
      <Line remove={remove}>
        <Select label={label} required={isRequired} hint={hint} value={String(value)} onChange={(e) => onChange(e.target.value)}>
          {s.enum.map((option) => <option key={option}>{option}</option>)}
        </Select>
      </Line>
    );

  if (s.type === "object" && !s.properties && typeof s.additionalProperties === "object")
    return <FreeObject value={value} schema={s} root={root} name={name} need={need} remove={remove} onChange={onChange} />;

  if (s.type === "object") {
    const object = (value && typeof value === "object" ? value : {}) as Record<string, Json>;
    const needed = new Set(s.required ?? []);
    const keys = Object.keys(s.properties ?? {});
    const missing = keys.filter((key) => !(key in object));
    return (
      <fieldset className="schema-object">
        <Head label={label}>
          {missing.map((key) => <Button key={key} variant="quiet" onClick={() => onChange({ ...object, [key]: initial(s.properties![key], root) })}>Add {key}</Button>)}
          {remove}
        </Head>
        {hint && <p className="wic-field-hint">{hint}</p>}
        {keys.filter((key) => key in object).map((key) => (
          <SchemaForm key={key} value={object[key]} schema={s.properties![key]} root={root} name={key} need={needed.has(key) ? "required" : "optional"}
            remove={needed.has(key) ? undefined : <Remove what={key} onClick={() => { const { [key]: _, ...rest } = object; onChange(rest); }} />}
            onChange={(v) => onChange({ ...object, [key]: v })} />
        ))}
      </fieldset>
    );
  }

  if (s.type === "array") {
    const list = Array.isArray(value) ? value : [];
    return (
      <fieldset className="schema-object">
        <Head label={label}>
          <Button variant="quiet" onClick={() => onChange([...list, initial(s.items ?? {}, root)])}>Add to {name}</Button>
          {remove}
        </Head>
        {hint && <p className="wic-field-hint">{hint}</p>}
        {list.map((item, i) => (
          <SchemaForm key={i} value={item} schema={s.items ?? {}} root={root} name={`${name} ${i + 1}`}
            remove={<Remove what={`${name} ${i + 1}`} onClick={() => onChange(list.filter((_, j) => j !== i))} />}
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
function FreeObject({ value, schema, root, name, need, remove, onChange }: Props) {
  const [key, setKey] = useState("");
  const object = (value && typeof value === "object" && !Array.isArray(value) ? value : {}) as Record<string, Json>;
  const item = schema.additionalProperties as Schema;
  return (
    <fieldset className="schema-object">
      <Head label={need === "required" ? required(name) : need === "optional" ? optional(name) : name}>{remove}</Head>
      {schema.description && <p className="wic-field-hint">{schema.description}</p>}
      {Object.keys(object).map((k) => (
        <SchemaForm key={k} value={object[k]} schema={item} root={root} name={k}
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
