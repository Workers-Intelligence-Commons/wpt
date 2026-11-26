// A conversation (an item of kind conversation, docs/chat-format.md) read as
// a transcript, in the design system's messages: each message under the name
// of its role (a user's on the right, the others on the left), its text with
// its line breaks, a thinking block set apart, a tool call as the design
// system's tool call, folded, with its arguments, a tool's result under the
// call it answers, a JSON value and a file by their form, an excerpt as the
// passage it quotes under the id of the quoted text. The tools offered and the metadata, never read as tokens, stand in a
// fold at the end. The words looked for (find), when given, are marked in
// every text, in any case, as the search found them.

import { Message, ToolCall } from "@workers-intelligence-commons/components";
import { Fragment, type ReactNode } from "react";
import { wordsPattern } from "./search-words";

type Block = { type: string; text?: string; id?: string; name?: string; arguments?: unknown; call?: string; error?: boolean;
  content?: Block[]; value?: unknown; sha256?: string; media_type?: string; source?: string;
  selector?: { exact?: string; prefix?: string; suffix?: string } };
type Message = { role: string; content: Block[]; metadata?: { draft?: { model: string; edited: boolean } } & Record<string, unknown> };
type Conversation = { messages: Message[]; tools?: { name: string; description: string }[]; metadata?: Record<string, unknown> };

const ROLES: Record<string, string> = { system: "system", user: "user", assistant: "assistant", tool: "tool" };

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

function Blocks({ blocks, find }: { blocks: Block[]; find?: string }) {
  return <>{blocks.map((b, i) => <BlockView key={i} block={b} find={find} />)}</>;
}

function BlockView({ block, find }: { block: Block; find?: string }): ReactNode {
  switch (block.type) {
    case "text":
      return <p className="turn-text"><Marked text={block.text ?? ""} find={find} /></p>;
    case "thinking":
      return <div className="turn-aside"><span className="turn-label">Thinking</span><p className="turn-text"><Marked text={block.text ?? ""} find={find} /></p></div>;
    case "tool_call":
      return <ToolCall name={block.name ?? "a tool"} state="done" progress={`call ${block.id}`}><pre className="turn-json">{pretty(block.arguments)}</pre></ToolCall>;
    case "tool_result":
      return <div className="turn-aside"><span className="turn-label">{block.error ? "Error" : "Result"} of the call <span className="ledger">{block.call}</span></span><Blocks blocks={block.content ?? []} find={find} /></div>;
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
  const extras = (content.tools?.length ?? 0) > 0 || (content.metadata && Object.keys(content.metadata).length > 0);
  return (
    <div className="transcript">
      {content.messages.map((m, i) => (
        <Message key={i} from={m.role === "user" ? "user" : "assistant"} label={ROLES[m.role] ?? m.role}
          details={m.metadata?.draft ? `drafted by the model ${m.metadata.draft.model}${m.metadata.draft.edited ? ", then edited" : ", unchanged"}` : undefined}>
          <Blocks blocks={m.content} find={find} />
        </Message>
      ))}
      {extras && (
        <details className="turn-extras small">
          <summary>{[content.tools?.length ? `${content.tools.length === 1 ? "1 tool" : `${content.tools.length} tools`} offered (${content.tools.map((t) => t.name).join(", ")})` : null,
            content.metadata && Object.keys(content.metadata).length ? "its metadata, never read as tokens" : null].filter(Boolean).join("; ")}</summary>
          {content.tools?.map((t) => <p key={t.name}><strong>{t.name}</strong>: {t.description}</p>)}
          {content.metadata && <pre className="turn-json ledger">{pretty(content.metadata)}</pre>}
        </details>
      )}
    </div>
  );
}
