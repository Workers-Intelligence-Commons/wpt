// Help: how the depot works, then every page of the interface: what it is
// for, what it shows, how to use it and what is where on the screen. A list
// of links to each part leads the page; a link scrolls to its part and puts
// it in the address (#/help?on=PART), and an address that names a part
// scrolls to it when the page opens.

import { useEffect, type MouseEvent, type ReactNode } from "react";
import { PageHead, Section } from "../parts";
import { href, usePlace } from "../route";
import type { Page } from "./index";

// Each part of the page: its key in the address and its title in the list of links.
const DEPOT: [string, string][] = [
  ["texts-and-items", "Texts and their kinds"],
  ["decisions", "Statuses and decisions"],
  ["tags-explained", "Tags: hand-picked sets"],
  ["snapshots-explained", "Snapshots"],
  ["orders", "Orders: vocabularies and corpora"],
  ["costs", "What things cost"],
  ["jobs-explained", "Jobs"],
  ["erasures-explained", "Erasures"],
  ["older-models", "Models trained before snapshots"],
  ["who-decides", "Who decides, and Settings"],
];
const PAGES: [string, string][] = [
  ["overview", "Overview"],
  ["texts", "Texts"],
  ["snapshots", "Snapshots"],
  ["jobs", "Jobs"],
  ["erasures", "Erasures"],
  ["licenses", "Licenses"],
  ["depot", "Depot"],
  ["settings", "Settings"],
  ["add", "Add a text"],
  ["views", "Views of one decision or tag"],
];

const partId = (key: string) => `help-${key}`;
const scrollTo = (key: string) => document.getElementById(partId(key))?.scrollIntoView({ block: "start" });

// A link to a part of this page: it scrolls there and puts the part in the address without leaving the page.
function PartLink({ on, children }: { on: string; children: ReactNode }) {
  const to = href(["help"], { on });
  const go = (e: MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    window.history.replaceState(null, "", to);
    scrollTo(on);
  };
  return <a href={to} onClick={go}>{children}</a>;
}

// One part of the page, with the id its links scroll to; page: the page it explains, linked from its title.
function Part({ on, title, page, children }: { on: string; title: string; page?: string[]; children: ReactNode }) {
  return (
    <div id={partId(on)}>
      <Section title={page ? <a href={href(page)}>{title}</a> : title}>{children}</Section>
    </div>
  );
}

const Cmd = ({ children }: { children: ReactNode }) => <span className="ledger">{children}</span>;

export function HelpPage() {
  const { query } = usePlace();
  const on = query.get("on");
  useEffect(() => { if (on) scrollTo(on); }, [on]);
  return (
    <div className="form-narrow">
      <PageHead title="Help" chain="How the depot works, then what each page of this interface does and where things are on it." />

      <nav aria-label="Parts of the help" className="section">
        <div className="two-lists">
          <div>
            <h2 className="cue section-title">How the depot works</h2>
            <ul>{DEPOT.map(([key, title]) => <li key={key}><PartLink on={key}>{title}</PartLink></li>)}</ul>
          </div>
          <div>
            <h2 className="cue section-title">The pages</h2>
            <ul>{PAGES.map(([key, title]) => <li key={key}><PartLink on={key}>{title}</PartLink></li>)}</ul>
          </div>
        </div>
      </nav>

      <Section>
        <p>
          The depot is one file that holds texts, their licenses, the decisions people make on them, and the snapshots that training
          reads. This web interface reads and changes it directly. There is no login: anyone who reaches it can read and change
          everything through it, and every change that decides something records a declared name and email.
        </p>
        <p>
          The path to a model is: texts are <strong>admitted</strong>; a <strong>snapshot</strong> freezes the set of admitted texts at one
          moment; an <strong>order</strong>, your file, names a snapshot and says what to take from it; the freight, outside this
          interface, learns its vocabulary and prepares its corpus; training reads them and records the depot, the snapshot, the order
          and the code. Daily work on texts never touches a snapshot, so nothing done today can change a model already published.
        </p>
        <p className="small">
          A few operations exist only as commands run where the depot is: making a new depot or a clone (<Cmd>depot new</Cmd>,{" "}
          <Cmd>clone</Cmd>), pulling from the depot it follows (<Cmd>pull</Cmd>), upgrading its format (<Cmd>upgrade</Cmd>), backups,
          changing the maintainer and takedown contact (<Cmd>about</Cmd>), making the depot's counts again (<Cmd>recount</Cmd>), and
          cancelling a job.
        </p>
      </Section>

      <Part on="texts-and-items" title="Texts and their kinds">
        <p>A text is one piece of training material, kept exactly as its source gives it. Every text has a <strong>kind</strong>, chosen when it is added and fixed for good: <Cmd>plain</Cmd> (an article, a book, a code file, read as plain text), or a kind whose content is JSON, such as <Cmd>conversation</Cmd> or <Cmd>task</Cmd>, which the depot checks against the kind's schema. A training step names the kinds it reads.</p>
        <p>Each text has:</p>
        <ul>
          <li><strong>licenses</strong>: one or more, as a choice, so one allowed license is enough to admit it;</li>
          <li>a <strong>notice</strong>: the copyright notice exactly as the source gives it, or why the work is in the public domain;</li>
          <li><strong>credits</strong>: a title, authors and the address it came from, each optional;</li>
          <li><strong>tags</strong>: free words that say what it is (a language, a form, a subject, a hand-picked set); an order's shares pick texts by tag;</li>
          <li><strong>scores</strong>, and its <strong>decisions</strong> in order.</li>
        </ul>
        <p>A text never changes by itself. Every change makes a new revision under the same id, except a change of tags: tags are kept beside the text, each with the change it was given and the change it was taken away. A change of its content or of its licenses starts its decisions over: it goes back under review. A change of its credits, tags or scores keeps its status. A snapshot made before any of these keeps the revision it holds and the tags the text had then.</p>
      </Part>

      <Part on="decisions" title="Statuses and decisions">
        <p>Every text, and every erasure request, has one of three statuses: <strong>under review</strong>, <strong>admitted</strong> or <strong>refused</strong>. Only a person changes it, by a decision. A text is admitted only if one of its licenses is on the allowed list.</p>
        <p>A decision records who made it: a legal name, an email and the confirmation that they read the requirements, accept them, and that what they submit meets them. The name and email are public and nobody verifies them. Adding a text, changing its content, licenses, credits, tags or scores, and asking for an erasure are decisions too, under review.</p>
        <p>A decision is sent only by a click on a button, or a key, that says beside it the confirmation it makes, links to the requirements and shows the name it records; most open a confirming line first, while scores, tags, Admit &amp; next and Refuse on the Texts page decide at the click. Decisions are appended, never removed: "back to review" is a later decision that undoes an admission or a refusal.</p>
        <p>Problems reported on libraries before snapshots took their place stay in the depot as decisions on no text; their pages show them read only.</p>
      </Part>

      <Part on="tags-explained" title="Tags: hand-picked sets">
        <p>A set of texts chosen by hand is a tag. Tick the texts on the Texts page, click "Add a tag", and click an existing tag or type a new one. "Remove a tag" does the reverse. Before the confirm, the bar shows how many texts change, the rows written, what the depot's file grows by and how long it takes. Tags cost nothing to keep and never change a snapshot.</p>
      </Part>

      <Part on="snapshots-explained" title="Snapshots">
        <p>A snapshot is one change number of the depot, named by the UTC time it was made, such as <Cmd>2026-11-26T14-03-27Z</Cmd>, with the set of texts admitted at that change. A text in a snapshot is always read at its revision as of that change: later edits, new decisions and new texts never reach it. That is what makes a model reproducible: the same snapshot gives the same texts next year.</p>
        <ul>
          <li>Making one is a background job of a few minutes on a large depot, and stores a few megabytes: the set of text numbers, compressed, and a small summary by kind and by tag. Its cost shows before the click.</li>
          <li>A snapshot is never changed or deleted. Its <strong>digest</strong>, a fingerprint of its set of texts, lets anyone check that a copy holds the same texts.</li>
          <li>Only an admitted erasure reaches into a snapshot: it cuts the passage, or removes the text, everywhere. The snapshot's page says how many erasures were carried out since it was made.</li>
          <li>When the depot was upgraded to snapshots, each locked library became a snapshot named after it, and each dataset a snapshot named after the dataset, so what was published stays readable. A snapshot with another name, such as one the upgrade made, keeps it.</li>
        </ul>
      </Part>

      <Part on="orders" title="Orders: vocabularies and corpora">
        <p>An order is a JSON file that says how a model is trained; it is yours, kept wherever you keep it. Its data section names a snapshot of this depot (its name, change number and digest), the kinds to take, optional <strong>shares</strong> by tag, a size and a vocabulary. The freight reads it outside this interface (<Cmd>wpt build-resources</Cmd>): it learns the vocabulary, chooses the texts from the snapshot and prepares the corpus, and leaves each on the dock. This depot only gives it the snapshot.</p>
        <ul>
          <li><strong>Shares</strong> split the plain text by tag, each with a proportion: a plain text counts for the first share whose tag it has, and a plain text no share takes is left out. Without shares, every plain text of the kinds taken is in.</li>
          <li>Conversations and tasks are never a share: every one is in, and they take their own part, the part of the bytes they hold.</li>
        </ul>
        <p>A depot may hold tokenizer recipes and the vocabularies they trained: they stay as they are, read only. A trained model records the depot's address, the snapshot with its change number and digest, the order and the code commits. Anyone who has these and the code can check what it was trained on.</p>
      </Part>

      <Part on="costs" title="What things cost">
        <p>Every heavy action shows what it costs before the click that starts it, and the server refuses it if the cost grew meanwhile, showing the new one to confirm again:</p>
        <ul>
          <li>making a snapshot: the admitted texts, the time, the disk, the change it reads;</li>
          <li>adding or removing a tag over ticked texts: the texts that change, the rows written, the bytes added and the time.</li>
        </ul>
        <p>Reading, reviewing and deciding on texts cost nothing extra: snapshots are the only heavy work.</p>
      </Part>

      <Part on="jobs-explained" title="Jobs">
        <p>Long work runs as jobs in three lanes, each one job at a time, in a queue that survives a restart: the <strong>write</strong> lane (a decision, a tag change or a license change over many texts), the <strong>heavy</strong> lane (making a snapshot) and the <strong>read</strong> lane (ordering a snapshot by size for its strip), so a snapshot never holds up a write. Anyone can start one; only the command <Cmd>depot jobs cancel ID</Cmd>, run where the depot is, cancels a job.</p>
      </Part>

      <Part on="erasures-explained" title="Erasures">
        <p>An erasure request names passages of texts by their place (from one character to another, in a field for a JSON kind) or a whole text, with a public reason. It never repeats the passage itself.</p>
        <p>Admitted, it is carried out at once and cannot be undone: every place where the passage appears in that text is cut, in every version of it, snapshots included; a whole text leaves the depot. The depot marks nothing on the dock: a corpus holding the passage goes stale at the freight's next check, so run <Cmd>freight check</Cmd> right after the decision. Refused, the texts stay as they are. Every request is kept.</p>
      </Part>

      <Part on="older-models" title="Models trained before snapshots">
        <p>A model trained before snapshots records a dataset of this depot, not a snapshot. The depot keeps what it held of its libraries and datasets as the upgrade found it, and nothing writes or serves it any more.</p>
        <p>Its locked libraries and datasets became snapshots at the upgrade (see Snapshots), so a new order can name the same texts.</p>
      </Part>

      <Part on="who-decides" title="Who decides, and Settings">
        <p>The name and email every decision records are kept in this browser only, until a decision sends them. Another browser, a private window or cleared site data has none. A form that needs them and finds none says so and links to Settings, which comes back to the form once they are saved.</p>
        <p>When this depot follows another one (its upstream), what belongs to the upstream is marked "the upstream's" and changes only there; a pull brings its changes and its snapshots here.</p>
      </Part>

      <Part on="overview" title="Overview" page={[]}>
        <p>What waits for the maintainer. It opens from the depot's name at the left of the top bar.</p>
        <ul>
          <li>At the top, the texts admitted, the jobs open (how many in each lane) and the erasures open, each linked.</li>
          <li><strong>Jobs open</strong>: the ones queued or running, while there are some.</li>
          <li><strong>Snapshots</strong>: the latest ones, with "Make a snapshot".</li>
          <li><strong>Under review</strong>: how many texts wait, counted by tag and by license. Each tag and license opens its view; each number opens those texts.</li>
          <li><strong>Open erasure requests</strong> and <strong>Latest decisions</strong>, each opening its view.</li>
        </ul>
      </Part>

      <Part on="texts" title="Texts" page={["texts"]}>
        <p>Where texts are read, scored, tagged, edited and decided, one after the other. The content of the open text fills the middle, its review panel stands at the right, and the list of the open texts hides at the left edge.</p>
        <p><strong>Choosing the texts</strong>, at the top:</p>
        <ul>
          <li>The quick filters: <strong>Under review</strong>, <strong>Not scored</strong> (no score yet; an SQL condition, slower on a large depot), <strong>Newest first</strong> and <strong>All</strong>. A second click takes one away.</li>
          <li>The filter bar: words of the title, status, tags, licenses and kinds, each a menu of ticks (Kinds lists the kinds the depot has texts of, with their counts, once there are two). A tick applies at once, and the words of the title once typing stops: the first text of the new filter opens, with "Reading the first texts of this filter" until it does; when the texts read so far hold no match, it says so and reads on by itself, until a text matches or the count finds none. The Tags and Licenses menus count within the open texts, and say so; until that count is done they show the whole depot's. "More filters" adds an SQL condition, a search in the texts themselves and whole words.</li>
          <li><strong>Open ▾</strong>: every text, each status, each kind, each tag and license with its count, and the filters saved in this browser, with "Save the open filter". Escape or a click outside closes it.</li>
          <li><strong>Biggest first</strong>: the open texts as one strip, the longest first, to find what takes the most room. A second click goes back to one text at a time.</li>
          <li>Text, Blob, Size map and Mono at the right: Blob reads the open texts one after the other as one page, each text's length in characters in the margin with a Refuse button (it refuses at once, then offers Undo); "3 lines" shows three lines of each text (the default with Biggest first), "long" the start of each; a click on a text opens it. Size map shows which tags, kinds and licenses take the most characters, and the biggest texts. Mono shows the texts in a fixed-width font.</li>
        </ul>
        <p>The quick filters and the views make one row, the filter bar the next; a long choice in a menu is cut short, so the bar never changes height when a filter changes.</p>
        <p><strong>The list</strong>: rest the mouse against the left edge of the window and the list slides out; it goes back when the mouse leaves it. A click on a text opens it. Each text has a tick, and the box at the top ticks every text the filter matches. The ticked texts get a bar at the bottom: Admit, Refuse, Back to review, <strong>Add a tag</strong>, <strong>Remove a tag</strong> and Score. Each opens the ticked texts for a last look, where unticking one leaves it out. Adding a tag shows the depot's tags as chips, the most used first: click one or more, or type a new one; removing shows the tags the ticked texts have. The cost shows before the confirm, and the confirm records the name and email set in Settings. On a phone, the "☰ List" button opens the list.</p>
        <p><strong>The content</strong> is always editable, with no Save button: a change is saved when you move to another text (Next, ‹, a text of the list, J or K), before a decision on it, with Ctrl+S, and when you leave the text any other way; "Revert" drops it. While a change is not saved, "● edited, not saved" stands beside the title, above the editor, and the bar under the editor says when it will be saved. Saving makes a new content version and puts the text back under review, an admitted one too; Admit &amp; next saves first, then admits the new version. When a save is refused (no name and email, a problem in a conversation), the text stays open and the line at the bottom says why. Escape leaves the editor, so the keys below work again.</p>
        <p><strong>A text too long to edit</strong> (more than 2,000,000 characters) opens read only, a passage of 200,000 characters at a time: scrolling near the end reads the next passage, and near the start the one before, the line in view staying still. "Go to character", Start and End jump anywhere in it. You can still score it, tag it and decide on it, and Translate translates the passage at the top of the view.</p>
        <p><strong>Where you were</strong>: the Texts page keeps in this browser what you opened (the filter or search, the order, the view) and the text you are on. The "Texts" link of the top bar leads back there from any page, and a reload keeps it too, so the texts you passed stay behind you. A new filter or search replaces it.</p>
        <p><strong>Translate</strong>, beside Find, shows a plain text translated in place of the editor, read only, so you can judge a text in a language you do not read. The text's language is what Firefox's language detector finds in it when it is sure, else what a tag names, else your choice (kept for that text until the tab closes); "From" says which. The language you read in is kept in this browser, and the translation stays on as you move from text to text. Mozilla's models translate it in this browser: the first use of a language fetches its model through the server, this browser keeps it for the next times, and no text leaves your machine. Nothing of the translation is saved; "Back to the text" shows the editor again, with any change not saved still there.</p>
        <p><strong>The review panel</strong>, from top to bottom:</p>
        <ul>
          <li>One row per score the depot knows, with buttons that follow the values the depot's texts hold. A score of whole grades has one button per grade, 0 to 10, and "other", which opens a number pad for a value between them (7.5); the pad keeps to 0 to 10 and says so when a value typed is outside. Any other score has − and + to step and its value, which opens the pad. The pad opens over the panel, never over the text.</li>
          <li>Nine tags as toggles: the last four you put on a text in this browser first, since texts read one after another often take the same tags, then the depot's most used; the text's other tags with ✕. "+ tag" opens every tag of the depot, read again as you type, each added with a click or a new one with Enter; "+ score" does the same with the score names.</li>
          <li>The buttons: ‹ (previous), Next ›, Refuse, Admit &amp; next, and "Back to review" on a text already decided.</li>
          <li>The presets: a click sets their scores and adds and removes their tags. "+ preset from this text" keeps this text's scores and tags under a name, in this browser only.</li>
          <li>The text's facts: license, address, decisions and id; then "More on this text": its record, a decision with a reason, its licenses, its credits and asking for an erasure.</li>
        </ul>
        <p>Clicks on scores and tags are saved together about two seconds after the last one, or at once when another text opens; they keep the text's status. Admit &amp; next and Refuse decide at once. After a refusal, a line at the bottom names the text refused, its id and its decision, with <strong>Undo</strong> first (or the key Z): it puts the text back under review at once, a public decision too, and opens it again. The same line offers reasons as chips: "bad quality", "bad license", "license not compatible" and your own; a click (or the key 1 to 9) makes it that refusal's public reason, another click replaces it, and no reason is fine; the line goes once another decision is made or the filter changes. On a phone it sits above the buttons of the next text. Type another reason and press Enter, or "+ chip" to attach it and keep it as a chip in this browser. Back to review and admitting a refused text first ask for a confirmation, with an optional public reason.</p>
        <p><strong>Keys</strong>, outside the editor: J next, K previous, A admit (and next), R refuse (and next), 1 to 9 a reason for the refusal just made, Z undo that refusal, U back to review, X tick the open text, E into the editor, Escape out of it.</p>
      </Part>

      <Part on="snapshots" title="Snapshots" page={["snapshots"]}>
        <p>Every snapshot, the latest first, with its change, its texts and its size.</p>
        <ol>
          <li>Click "Make a snapshot". It shows what it costs: the admitted texts, the time, the disk and the change it reads. A note is optional; the depot names the snapshot by the UTC time it is made.</li>
          <li>Click the button. The job is followed on the page, and the snapshot's page opens when it ends. One snapshot is made at a time: while one is being made, the dialog follows it instead.</li>
        </ol>
        <p><strong>One snapshot's page</strong>: its change, its digest (with Copy), its texts by kind and by tag, the erasures carried out since, and its sets to download for anyone who checks the digest.</p>
        <p><strong>Read it whole</strong>, on that page, shows every text of the snapshot as one strip, the longest first (the switch "Longest first / Shortest first" shows the order and turns it around), three lines of each ("long" shows more), each as the snapshot holds it, its length in characters and a Refuse button in the margin, and "refused now" or "under review now" when it changed since. A click on a text opens it as it is now in a pane at the right, to edit, admit or refuse it; Close or Escape puts the pane away and the strip stays where it was. The first time, the snapshot is ordered by size once for good: the page says so at once and shows how many texts are read, and every other page works meanwhile. <strong>Size map</strong> shows which of its tags, kinds and licenses take the most characters, and its biggest texts, each with a bar and a Refuse button.</p>
      </Part>

      <Part on="jobs" title="Jobs" page={["jobs"]}>
        <p>The server's queue in its three lanes, read again every few seconds: the jobs waiting or running in the order they run, then the latest that ended. Each row says what the job does, linked to its snapshot or decision, its state and how long it waited or ran, and what it gave. While a job waits or runs, a mark in the top bar says how many and links here.</p>
      </Part>

      <Part on="erasures" title="Erasures" page={["erasures"]}>
        <p>Every erasure request, and where they are asked for and decided.</p>
        <ul>
          <li>The table: each request, its status, its reason, its passages by text and place, and whether it was carried out. A request opens what it asks to erase, with the text around each passage.</li>
        </ul>
        <p>To ask for one: click "Ask for an erasure", pick the text, give the place (or tick the whole text) and a reason, then "Confirm and ask".</p>
        <p>To decide: tick open requests, click "Admit" or "Refuse" in the bar that appears, give an optional reason, then confirm. Admitting cannot be undone. A request of the upstream has no tick.</p>
      </Part>

      <Part on="licenses" title="Licenses" page={["licenses"]}>
        <p>Every license the depot holds: its id, its name, whether the allowed list accepts it (a combined license only if every part is allowed), and how many texts name it. Tick licenses, then "Show their texts" to open them on the Texts page. "Add a license" takes an id, a name and the official text. A license's own page shows its official text and the texts that name it.</p>
      </Part>

      <Part on="depot" title="Depot" page={["depot"]}>
        <p>This depot's facts: its name, its address, its current change number, its maintainer and takedown contact, and the depot it follows, if any. They change only through the commands.</p>
      </Part>

      <Part on="settings" title="Settings" page={["settings"]}>
        <p>The legal name and email this browser sends with every decision. Type both and click "Save them". Opened from a form, it goes back there once saved.</p>
      </Part>

      <Part on="add" title="Add a text" page={["add"]}>
        <p>The button at the right of the top bar. What it adds is under review and is in no snapshot until it is admitted and a snapshot is made.</p>
        <ol>
          <li>Choose its kind from the cards, each with a short line; the "?" beside them says what each kind is used for. The kind picked last in this browser is picked again. The kind cannot change later: on a text's page it shows as a plain word beside the title.</li>
          <li>Type, paste or drop a file for a plain text. A conversation is written as a chat: each message shows its role; "Add a … message" proposes the role that usually comes next, with the other roles beside it, and each message takes its blocks (text, thinking, a tool call, a tool result, an excerpt) from buttons, each moved with ↑ ↓ or removed with ✕. An excerpt's passage is selected in its text: "Pick the passage" opens the text quoted last, or a list of plain texts (quoted here, newest, or by tag); select the passage in it and the excerpt is filled in, with enough text before and after it to find it at one place. What the depot would refuse is listed under the editor and holds the confirm. Any other kind is filled in through its form, or loaded from a JSON file.</li>
          <li>Tick its licenses and give its notice.</li>
          <li>Give its title, authors, tags and address, all optional. The depot's most used tags stand under the tags box; a click adds or removes one.</li>
          <li>Click "Confirm and add".</li>
          <li>"Add another" keeps the licenses, notice, authors and tags for the next one.</li>
        </ol>
      </Part>

      <Part on="views" title="Views of one decision or tag">
        <p>Not in the top bar; the lists that name them link to them, and the top bar keeps Texts marked while one shows.</p>
        <ul>
          <li><strong>A decision</strong>: what it did, its status, who decided, when, its reason, and its texts or the erasure requests it decides. A decision that put texts back under review shows as a correction. A decision on no text (an old problem report, or texts erased since) says so.</li>
          <li><strong>A tag</strong>: how many texts have it, by status, and those texts. <a href={href(["tags"])}>Tags</a> lists every tag, the most used first.</li>
        </ul>
      </Part>
    </div>
  );
}

export const help: Page = { path: "help", title: "Help", group: "help", Component: () => <HelpPage /> };
