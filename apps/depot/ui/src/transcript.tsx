// A conversation (an item of kind conversation, docs/chat-format.md) read as
// a transcript, in the design system's messages: the harness it was recorded
// from, when it names one, above them; each message under the name of its role
// (a user's on the right, the others on the left), its text with its line
// breaks, a text the harness added set apart and marked so, a thinking block
// set apart, a tool call as the design system's tool call, folded, with its
// arguments, or with the raw text the model wrote when they could not be parsed
// (under its name, or, when not even that could be read, as the whole call), a
// tool's result under the call it answers, with what the tool returned in
// structured form and a link to the sub-agent's conversation when it has them,
// the tools a tools block defines, a JSON value and a file by their form, an
// excerpt as the passage it quotes under the id of the quoted text, and under
// an assistant's message how its turn ended. The tools offered from the start
// and the metadata, never read as tokens, stand in a fold at the end. The words
// looked for (find), when given, are marked in every text, in any case, as the
// search found them.

import { Message, ToolCall } from "@workers-intelligence-commons/components";
import { Fragment, type ReactNode } from "react";
import { END_MEANING, ROLE_LABEL, formOf, type Block, type Conversation, type Tool } from "./conversation";
import { href } from "./route";
import { wordsPattern } from "./search-words";

// Whether a content is a conversation this transcript can show: an object with a list of messages.
export function isConversation(content: unknown): content is Conversation {
  return Boolean(content && typeof content === "object" && Array.isArray((content as Conversation).messages)
    && (content as Conversation).messages.every((m) => m && typeof m === "object" && Array.isArray(m.content)));
}

// The words looked for, each one marked wherever it stands in text, in any case.
function Marked({ text, find }: { text: string; find?: string }) {
  const pattern = wordsPattern(find, true);
  if (!pattern) return <>{text}</>;
  return <>{text.split(pattern).map((part, i) => (i % 2 ? <mark key={i}>{part}</mark> : <Fragment key={i}>{part}</Fragment>))}</>;
}

const pretty = (value: unknown) => JSON.stringify(value, null, 2);

// Tools, each by its name, with its description when it is known.
function Tools({ tools }: { tools: Tool[] }) {
  return <>{tools.map((t) => <p key={t.name} className="turn-text"><strong>{t.name}</strong>{t.description ? <>: {t.description}</> : <span className="aside"> (named alone)</span>}</p>)}</>;
}

function Blocks({ blocks, find }: { blocks: Block[]; find?: string }) {
  return <>{blocks.map((b, i) => <BlockView key={i} block={b} find={find} />)}</>;
}

function CallView({ block }: { block: Block }) {
  const form = formOf(block);
  if (form === "arguments") return <ToolCall name={block.name ?? "a tool"} state="done" progress={`call ${block.id}`}><pre className="turn-json">{pretty(block.arguments)}</pre></ToolCall>;
  return (
    <ToolCall name={form === "named" ? block.name ?? "" : "a call not read"} state="done" progress={`call ${block.id}, ${form === "named" ? "arguments not parsed" : "kept whole"}`}>
      <span className="turn-label">{form === "named" ? "Its arguments as the model wrote them, not parsed" : "The whole call as the model wrote it, not read"}</span>
      <pre className="turn-json">{block.raw}</pre>
    </ToolCall>
  );
}

function BlockView({ block, find }: { block: Block; find?: string }): ReactNode {
  switch (block.type) {
    case "text":
      return block.by
        ? <div className="turn-aside turn-harness"><span className="turn-label">Added by the harness</span><p className="turn-text"><Marked text={block.text ?? ""} find={find} /></p></div>
        : <p className="turn-text"><Marked text={block.text ?? ""} find={find} /></p>;
    case "thinking":
      return <div className="turn-aside"><span className="turn-label">Thinking</span><p className="turn-text"><Marked text={block.text ?? ""} find={find} /></p></div>;
    case "tool_call":
      return <CallView block={block} />;
    case "tool_result":
      return (
        <div className="turn-aside">
          <span className="turn-label">{block.error ? "Error" : "Result"} of the call <span className="ledger">{block.call}</span></span>
          <Blocks blocks={block.content ?? []} find={find} />
          {"value" in block && <><span className="turn-label">What the tool returned, structured</span><pre className="turn-json ledger">{pretty(block.value)}</pre></>}
          {block.conversation !== undefined && <span className="turn-label">From the sub-agent's conversation <a className="ledger" href={href(["texts", block.conversation])}>{block.conversation}</a></span>}
        </div>
      );
    case "tools":
      return <div className="turn-aside"><span className="turn-label">Tools defined from here on</span><Tools tools={block.tools ?? []} /></div>;
    case "json":
      return <pre className="turn-json ledger">{pretty(block.value)}</pre>;
    case "file":
      return <p className="turn-text aside">A file, <span className="ledger">{block.media_type}</span>, sha256 <span className="ledger">{block.sha256?.slice(0, 12)}…</span></p>;
    case "excerpt":
      return <div className="turn-aside"><span className="turn-label">Quoted from <span className="ledger">{block.source}</span></span><p className="turn-text"><Marked text={block.selector?.exact ?? ""} find={find} /></p></div>;
    default:
      return <pre className="turn-json ledger">{pretty(block)}</pre>;
  }
}

export function Transcript({ content, find }: { content: Conversation; find?: string }) {
  const tools = content.tools ?? [];
  const meta = content.metadata && Object.keys(content.metadata).length > 0 ? content.metadata : null;
  const harness = content.harness;
  return (
    <div className="transcript">
      {harness && <p className="small aside turn-harness-name">Recorded from the harness <strong>{harness.name}</strong>{harness.version ? <> {harness.version}</> : null}</p>}
      {content.messages.map((m, i) => (
        <Message key={i} from={m.role === "user" ? "user" : "assistant"} label={ROLE_LABEL[m.role] ?? m.role}
          details={m.metadata?.draft ? `drafted by the model ${m.metadata.draft.model}${m.metadata.draft.edited ? ", then edited" : ", unchanged"}` : undefined}>
          <Blocks blocks={m.content} find={find} />
          {m.end && <p className="small aside turn-end" title={END_MEANING[m.end]}>{m.end === "stop" ? "Its turn ended: the model stopped." : m.end === "tool_calls" ? "Its turn ended on its tool calls." : "Its turn was cut by a limit."}</p>}
        </Message>
      ))}
      {(tools.length > 0 || meta) && (
        <details className="turn-extras small">
          <summary>{[tools.length ? `${tools.length === 1 ? "1 tool" : `${tools.length} tools`} offered (${tools.map((t) => t.name).join(", ")})` : null,
            meta ? "its metadata, never read as tokens" : null].filter(Boolean).join("; ")}</summary>
          <Tools tools={tools} />
          {meta && <pre className="turn-json ledger">{pretty(meta)}</pre>}
        </details>
      )}
    </div>
  );
}
