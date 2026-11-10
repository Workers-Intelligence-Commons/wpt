# Chat format

A conversation is always stored as one JSON object. A template turns that
object into tokens for a model, with special tokens marking its structure.
The object is the record; the tokens are only one way of reading it.

## Everything is recorded

A conversation holds every element of what happened: the text, the
thinking, each tool call with its arguments, each tool result, and each
structured output. Every element is recorded, even when no training uses
it yet. The template decides which elements become tokens, so a later
template can train on an element that an earlier one skipped, without
recording anything again.

Recording everything does not widen what may be recorded: the rules of
[training.md](training.md) still apply to every element. An
`assistant` message in a demonstration is written by a person, never by
another company's model; it may start as a draft of a WPT model that the
person then edits or confirms, and its metadata records that
([Model drafts](#model-drafts)).

## The object

```json
{
  "tools": [
    {"name": "read_file", "description": "Read a file of the project.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
  ],
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "What does setup.py install?"}]},
    {"role": "assistant", "content": [
      {"type": "thinking", "text": "The answer is in the script itself."},
      {"type": "tool_call", "id": "c1", "name": "read_file", "arguments": {"path": "scripts/setup.py"}}
    ]},
    {"role": "tool", "content": [
      {"type": "tool_result", "call": "c1", "content": [{"type": "text", "text": "..."}]}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "The PyTorch build for this machine's GPU."}],
     "metadata": {"draft": {"model": "wpt-small", "weights": {"model.safetensors": "<sha256>"},
                            "sha256": "<sha256 of the draft's content>", "edited": true}}}
  ],
  "metadata": {"step": "finish-tasks", "task": "<task id>", "round": 1}
}
```

- `tools`: the tools offered in the conversation, each with its `name`, its
  `description` and the JSON Schema of its `parameters`. Empty or absent
  when none is offered.
- `messages`: in order. Each message has a `role` (`system`, `user`,
  `assistant` or `tool`) and a `content` list of blocks, in the order they
  happened: thinking, then a call, then more thinking, then text, stays in
  that order. An `assistant` message may have no block: when a model closes
  its turn without writing anything, the record says exactly that. A
  message of any other role holds at least one block. A message may also
  hold `metadata`, a JSON object never written as tokens; its one named
  field is `draft` ([Model drafts](#model-drafts)).
- `metadata`: optional. What identifies the conversation where it was
  recorded, as any JSON object: a training attempt's step, task, round
  and score ([training.md](training.md), "Runs"). It is never
  written as tokens.

A conversation without any `metadata`, or whose messages hold none, stays
valid.

## Model drafts

An `assistant` message may start as a reply a WPT model drafted, which a
person then edited or confirmed as it is. The message's `metadata.draft`
records where it started:

- `model`: the name of the model folder that wrote the draft.
- `weights`: the sha256 of each weights file of that folder, by file name,
  so the exact model is known even after the folder is renamed.
- `sha256`: the sha256 of the draft's `content` as the model wrote it: the
  JSON text of the block list with sorted keys, no spaces, characters
  outside ASCII kept as they are, and a whole number written as an
  integer (`2.0` as `2`).
- `edited`: whether the person changed it, which is whether the sha256 of
  the message's `content` now differs from `sha256`.

`metadata.draft` is a claim the person who adds the conversation makes.
The depot checks only its shape, through the kind's schema, and refuses a
`draft` on any message but an `assistant` one: nothing checks that a WPT
model wrote the draft, its `model`, `weights` or `sha256`, or that
`edited` matches the content. An unedited confirmed draft is allowed:
confirming it is the person's choice that the reply is right. A message
with no `draft` was written by the person from the start.

## Blocks

| `type` | Fields | What it holds |
| --- | --- | --- |
| `text` | `text` | text shown to the other side |
| `thinking` | `text` | reasoning written before or between other blocks |
| `tool_call` | `id`, `name`, `arguments` | a call to one of `tools`; `arguments` is a JSON object that matches the tool's `parameters` |
| `tool_result` | `call`, `content`, `error` (optional, `true` when the call failed) | the answer to the `tool_call` whose `id` is `call`; `content` is a list of blocks |
| `json` | `value` | a structured value: any JSON value; a harness builds it, the model never writes one ([The answer](#the-answer)) |
| `file` | `sha256`, `media_type` | a media file such as a screenshot, stored next to the record and named by its sha256 |

A new kind of element gets a new block type, and existing blocks keep
their meaning.

## The answer

The model never writes JSON, except its tool calls, which other harnesses
parse as JSON: it writes text and special tokens, and the harness builds
every `json` block and value in the records, in a shape that can later go
back into a depot, as conversations for the `answer-as-assistant` step
for example.

The model's final reply, its last assistant message, one with no tool call,
is its answer, as text. The agent loop (`wpt_agent.loop`) then adds one
user turn that asks for its confidence, a number from 0 to 100 and nothing
else, the same question for every task (`CONFIDENCE` in
`wpt_agent/loop.py`, the one place it is written). The model's reply to it
is read, divided by 100, and the loop builds the attempt's
`metadata.answer`, which the task's check reads
([training.md](training.md), "Scores"):

```json
{"answer": "calc.py adds instead of subtracting; fixed line 12.", "confidence": 0.8}
```

- `answer`: the final reply's text, its text blocks one per line, its
  thinking left out. In a file-changing task it is the reply's text too.
- `confidence`: the reply to the question, from 0 to 100, divided by 100.
  In a file-changing task it is how sure the model is that the task's check
  will pass.

The reply counts when its text, spaces around it aside, is a number from 0
to 100 written with digits and at most one decimal point (`80`, `72.5`),
and nothing else: thinking before it is not part of its text, and a tool
call, a `%`, a word or a number above 100 make it count as none. A final
reply closed by the end of a document instead of `<|im_end|>` is a final
reply too. A reply that is not such a number, or that a limit cuts, scores
the attempt 0; so does an attempt that ends with no
final reply, stopped by a limit or by the end of a document written in a
turn with a tool call: its `metadata.answer` is `null`, the shared check
library that every check calls scores its `correct` part 0, and the step
that learns makes the whole attempt worth 0, whatever its other parts
(training.md, "Rewards"). The confidence turn counts against
`max_tokens` and the context, not against `max_turns`. No task's
instructions ask for a block or say how to answer. There is no way to
abstain: "I don't know" is never a way out that costs nothing.

## In a depot

A conversation is a depot item of kind `conversation`, its content the
object above ([data-selection.md](data-selection.md), "Items"). The depot
checks it against the kind's JSON Schema when it is added
(`packages/wpt-depot/src/wpt_depot/kinds/conversation.schema.json`): only
an `assistant` message holds `thinking` or `tool_call` blocks or a
`draft`, and only a `tool` message holds `tool_result` blocks.

A conversation is never a plain text that the `read-text` step reads. A
corpus keeps every conversation of its dataset as the tokens of the
template below, in token files of their own, and the `answer-as-assistant`
step reads them ([training.md](training.md), "The answer-as-assistant
step").

## The template

`wpt_tokenizer.chat` turns a conversation into tokens with the layout of
ChatML and the tags of Qwen and Hermes:

```text
END
<|im_start|>system\n{text}\n\n<tools>\n{each tool as JSON, one per line}\n</tools><|im_end|>\n
<|im_start|>user\n{text}<|im_end|>\n
<|im_start|>assistant\n<think>\n{thinking}\n</think>\n\n{text}\n<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call><|im_end|>\n
<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n
<|im_start|>user\n{text}\n<json>\n{value}\n</json><|im_end|>\n
```

- A message's blocks keep their order. Blocks after the first are
  separated by a line break, and a thinking block ends with a blank line.
- Tools offered in a conversation without a system message get a system
  message of their own.
- A `tool` message is written as a `user` turn of `<tool_response>` blocks,
  as Qwen does.
- A `json` block of a system or user message, or of a tool result, is
  written as its JSON text between `<json>` and `</json>`, so the model
  sees a structured value and tells it apart from text.
- The model never writes JSON but its tool calls ([The answer](#the-answer)):
  a `json` block in an assistant message, which only a harness builds, is
  never written as tokens, so no conversation teaches the model to write
  one, and when the model writes `<json>` tags they stay text in its
  message (`chat.reply` reads text, thinking and tool calls only).
- Recorded but not written as tokens yet: `file` blocks (no model reads
  media yet) and a `tool_result`'s `error` flag.
- Never written as tokens: `metadata`.

A model learns to predict only the assistant's turns: their content and
their closing `<|im_end|>`. The other turns are context it reads.

Each tag is one special token. Encoding text never produces a special
token, so text typed inside a message, such as `<|im_end|>`, stays text and
cannot imitate the structure.

## Special token ids

Special tokens have fixed ids, the same in every vocabulary, so the
template never depends on a vocabulary
([tokenizer.md](tokenizer.md)):

- Ids 257 to 512 are reserved for special tokens. 256 ids, 12 in use.
- Merges start at 513.

| Id | Token | Id | Token |
| --- | --- | --- | --- |
| 257 | `<\|im_start\|>` | 262 | `</tools>` |
| 258 | `<\|im_end\|>` | 263 | `<tool_call>` |
| 259 | `<think>` | 264 | `</tool_call>` |
| 260 | `</think>` | 265 | `<tool_response>` |
| 261 | `<tools>` | 266 | `</tool_response>` |
|  |  | 267 | `<json>` |
|  |  | 268 | `</json>` |

An id once given never changes. A new special token takes the next free id
up to 512, in `SPECIAL` (`packages/wpt-tokenizer/src/wpt_tokenizer/__init__.py`);
unused ids stay reserved. Giving a new id needs no new vocabulary, since
every vocabulary already reserves it.
