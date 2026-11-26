// The editor of a plain text: CodeMirror 6 over the text, with undo and
// redo, its standard keys, and its find and replace panel (Ctrl+F or
// Cmd+F, or openFind), whose search takes plain words or a regular
// expression and works inside this one text only. Its colours, type and
// focus ring are the design system's tokens, so it follows the light and
// dark themes. It owns its document: it starts from initial, again whenever
// initial changes, and reports every edit through onChange with the
// document and whether it differs from initial. Lines break at "\n" only,
// so a "\r" stays a character of its line and the text comes back from
// toString() exactly as it went in. The words a search looked for (find)
// are marked wherever they stand, in any case, and the first one is scrolled
// into view when the text opens; the cursor stays where it was, so the
// playlist's keys keep working.

import { history, defaultKeymap, historyKeymap } from "@codemirror/commands";
import { openSearchPanel, search, searchKeymap } from "@codemirror/search";
import { Compartment, EditorState, type Text } from "@codemirror/state";
import { Decoration, drawSelection, EditorView, keymap, lineNumbers, MatchDecorator, ViewPlugin, type DecorationSet, type ViewUpdate } from "@codemirror/view";
import { useEffect, useRef } from "react";
import { wordsPattern } from "./search-words";

// What a view around the editor can ask of it.
export type EditorHandle = { openFind: () => void; focus: () => void };

const theme = EditorView.theme({
  "&": { backgroundColor: "var(--paper-raised)", color: "var(--ink)", fontSize: "15px", height: "100%" },
  "&.cm-focused": { outline: "3px solid var(--ink)", outlineOffset: "-3px" },
  ".cm-scroller": { fontFamily: "var(--text-font, var(--font-serif))", lineHeight: "23px" },
  ".cm-content": { caretColor: "var(--ink)", padding: "var(--space-4) 0" },
  ".cm-line": { padding: "0 var(--space-4)" },
  ".cm-cursor, .cm-dropCursor": { borderLeftColor: "var(--ink)", borderLeftWidth: "2px" },
  "&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground, ::selection":
    { backgroundColor: "color-mix(in srgb, var(--ochre) 40%, transparent)" },
  ".cm-gutters": { backgroundColor: "var(--paper)", color: "var(--ink-muted)", borderRight: "1px solid var(--line)", fontFamily: "var(--font-typewriter)", fontSize: "13px" },
  ".cm-found": { backgroundColor: "var(--red-soft)", boxShadow: "0 2px 0 var(--red)" },
  ".cm-searchMatch": { backgroundColor: "var(--red-soft)", boxShadow: "0 2px 0 var(--red)" },
  ".cm-searchMatch.cm-searchMatch-selected": { backgroundColor: "var(--ochre)", color: "var(--on-ochre)" },
  ".cm-panels": { backgroundColor: "var(--paper)", color: "var(--ink)" },
  ".cm-panels.cm-panels-top": { borderBottom: "2px solid var(--ink)" },
  ".cm-panel.cm-search": { padding: "var(--space-2) var(--space-4)", fontFamily: "var(--font-serif)", fontSize: "15px", display: "flex", flexWrap: "wrap", alignItems: "center", gap: "var(--space-2)" },
  ".cm-panel.cm-search br": { flexBasis: "100%", height: 0 },
  ".cm-panel.cm-search input.cm-textfield": { backgroundColor: "var(--paper-raised)", color: "var(--ink)", border: "1px solid var(--line-strong)", borderRadius: "var(--radius-0)", minHeight: "32px", padding: "0 var(--space-2)", fontFamily: "var(--font-typewriter)", fontSize: "14px", margin: 0 },
  ".cm-panel.cm-search input.cm-textfield:focus-visible, .cm-panel.cm-search .cm-button:focus-visible, .cm-panel.cm-search input[type=checkbox]:focus-visible":
    { outline: "3px solid var(--ink)", outlineOffset: "2px" },
  ".cm-panel.cm-search .cm-button": { backgroundImage: "none", backgroundColor: "var(--paper)", color: "var(--ink)", border: "1px solid var(--ink)", borderRadius: "var(--radius-0)", minHeight: "32px", padding: "0 var(--space-2)", fontFamily: "var(--font-serif)", fontSize: "14px", margin: 0, cursor: "pointer" },
  ".cm-panel.cm-search .cm-button:hover": { backgroundColor: "var(--paper-raised)" },
  ".cm-panel.cm-search label": { fontSize: "14px", display: "inline-flex", alignItems: "center", gap: "var(--space-1)" },
  ".cm-panel.cm-search input[type=checkbox]": { accentColor: "var(--ink)", width: "16px", height: "16px", margin: 0 },
  ".cm-panel.cm-search button[name=close]": { color: "var(--ink)", fontSize: "20px", right: "var(--space-2)", top: "var(--space-1)" },
});

// The marks of every place a pattern matches, kept up to date as the text is edited.
function marks(found: RegExp) {
  const decorator = new MatchDecorator({ regexp: found, decoration: Decoration.mark({ class: "cm-found" }) });
  return ViewPlugin.fromClass(class {
    decorations: DecorationSet;
    constructor(view: EditorView) { this.decorations = decorator.createDeco(view); }
    update(u: ViewUpdate) { this.decorations = decorator.updateDeco(u, this.decorations); }
  }, { decorations: (v) => v.decorations });
}

export function CodeEditor({ initial, onChange, label, handle, find }: {
  initial: string; onChange: (doc: Text, changed: boolean) => void; label: string; handle?: (h: EditorHandle | null) => void; find?: string;
}) {
  const box = useRef<HTMLDivElement>(null);
  const changed = useRef(onChange);
  changed.current = onChange;
  const view = useRef<EditorView | null>(null);
  const finding = useRef(new Compartment());
  useEffect(() => {
    const state = EditorState.create({
      doc: initial,
      extensions: [
        EditorState.lineSeparator.of("\n"), lineNumbers(), history(), drawSelection(), EditorView.lineWrapping, search({ top: true }),
        keymap.of([...defaultKeymap, ...historyKeymap, ...searchKeymap]), theme,
        EditorView.contentAttributes.of({ "aria-label": label }),
        EditorView.updateListener.of((u) => { if (u.docChanged) changed.current(u.state.doc, !u.state.doc.eq(state.doc)); }),
        finding.current.of([]),
      ],
    });
    const opened = new EditorView({ parent: box.current!, state });
    view.current = opened;
    handle?.({ openFind: () => { openSearchPanel(opened); }, focus: () => opened.focus() });
    return () => { handle?.(null); view.current = null; opened.destroy(); };
    // a new document starts a new editor, its history included; the label and handle are read once per document
  }, [initial]);
  // the words looked for, marked again whenever they change, the first place scrolled to; the edits stay
  useEffect(() => {
    const open = view.current;
    if (!open) return;
    const found = wordsPattern(find);
    const first = found ? new RegExp(found.source, "i").exec(open.state.doc.toString()) : null;
    open.dispatch({ effects: [finding.current.reconfigure(found ? marks(found) : []), ...(first ? [EditorView.scrollIntoView(first.index, { y: "center" })] : [])] });
  }, [initial, find]);
  return <div ref={box} className="code-editor" />;
}
