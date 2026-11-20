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
  ["texts-and-items", "Texts and items"],
  ["decisions", "Statuses and decisions"],
  ["libraries-explained", "Libraries"],
  ["datasets-explained", "Datasets"],
  ["tokenizers-explained", "Tokenizers and vocabularies"],
  ["corpora", "Corpora"],
  ["jobs-explained", "Jobs"],
  ["erasures-explained", "Erasures"],
  ["who-decides", "Who decides, and Settings"],
];
const PAGES: [string, string][] = [
  ["overview", "Overview"],
  ["texts", "Texts"],
  ["libraries", "Libraries"],
  ["datasets", "Datasets, with Corpora and Export"],
  ["tokenizers", "Tokenizers"],
  ["jobs", "Jobs"],
  ["erasures", "Erasures"],
  ["licenses", "Licenses"],
  ["depot", "Depot"],
  ["settings", "Settings"],
  ["add", "Add a text"],
  ["views", "Views of one decision, tag or problem"],
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
          The depot is one file that holds texts, their licenses, the decisions people make on them, and what is built from them for
          training. This web interface reads and changes it directly. There is no login: anyone who reaches it can read and change
          everything through it, and every change that decides something records a declared name and email.
        </p>
        <p className="small">
          A few operations exist only as commands run where the depot is: making a new depot or a clone (<Cmd>wpt depot new</Cmd>,{" "}
          <Cmd>clone</Cmd>), pulling from the depot it follows (<Cmd>pull</Cmd>), upgrading its format, backups, changing the maintainer
          and takedown contact (<Cmd>about</Cmd>), importing a dataset or a tokenizer recipe from a file, a dataset that reads another
          depot's library, making the depot's counts again (<Cmd>recount</Cmd>), and cancelling a job.
        </p>
      </Section>

      <Part on="texts-and-items" title="Texts and items">
        <p>A text is one piece of training text, such as an article, a book or a code file, kept exactly as its source gives it. An item is a text whose content is JSON instead of plain text, of a kind such as <Cmd>conversation</Cmd> or <Cmd>task</Cmd>; the depot checks its content against its kind's schema.</p>
        <p>Each text or item has:</p>
        <ul>
          <li><strong>licenses</strong>: one or more, as a choice, so one allowed license is enough to admit it;</li>
          <li>a <strong>notice</strong>: the copyright notice exactly as the source gives it, or why the work is in the public domain;</li>
          <li><strong>credits</strong>: a title, authors and the address it came from, each optional;</li>
          <li><strong>tags</strong>: free words that say what it is (a language code, a form, a subject); shares of a dataset pick texts by tag;</li>
          <li><strong>scores</strong>, its <strong>decisions</strong> in order, and the libraries that hold it.</li>
        </ul>
        <p>A text never changes by itself. Every change makes a new revision under the same id. A change of its content or of its licenses starts its decisions over: it goes back under review and leaves every open library, to be admitted and put back again. A change of its credits or tags keeps its status and its libraries.</p>
      </Part>

      <Part on="decisions" title="Statuses and decisions">
        <p>Every text, and every erasure request, has one of three statuses: <strong>under review</strong>, <strong>admitted</strong> or <strong>refused</strong>. Only a person changes it, by a decision. A text is admitted only if one of its licenses is on the allowed list.</p>
        <p>A decision records who made it: a legal name, an email and the confirmation that they read the requirements, accept them, and that what they submit meets them. The name and email are public and nobody verifies them. Adding a text, changing its content, licenses, credits or scores, asking for an erasure and reporting a problem on a library are decisions too, under review.</p>
        <p>A decision is sent only by a click on a button, or a key, that says beside it the confirmation it makes, links to the requirements and shows the name it records; most open a confirming line first, while scores, tags and Admit &amp; next on the Texts page decide at the click. Decisions are appended, never removed: "back to review" is a later decision that undoes an admission or a refusal.</p>
      </Part>

      <Part on="libraries-explained" title="Libraries">
        <p>A library is a named collection of admitted texts, which datasets read. A text under review is in no library, and a refusal, a return to review or a change of licenses or content takes it out of every open one.</p>
        <ul>
          <li>An <strong>open</strong> library takes texts in and out, and reads each one at its current revision.</li>
          <li>A <strong>locked</strong> library is fixed for good: it pins the revision of each text it held when it was locked. No text joins or leaves it, only an erasure can still cut or remove one, and newer revisions do not reach it.</li>
        </ul>
        <p>A library can be renamed, open or locked, with no decision. A fork is a new open library with the same texts, less any chosen ones: it is how a locked library is corrected. A problem reported on a library changes nothing in it; it records the problem and who reported it.</p>
      </Part>

      <Part on="datasets-explained" title="Datasets">
        <p>A dataset says what to train on: its libraries, its tokenizer recipe, optional shares and an optional size.</p>
        <ul>
          <li>Each library is read as it stood when it was ticked. A text put in or taken out later changes nothing until "Read again".</li>
          <li><strong>Shares</strong> split the plain text by tag, in order, each with a proportion: a plain text counts for the first share whose tag it has, and a plain text no share takes is left out. Inside a share, its part splits over the other tags its texts carry, each tag's part following the square root of what it holds: a tag a hundred times bigger aims at ten times the tokens. Without shares, every plain text is in.</li>
          <li>Conversations and tasks are never a share: every one is in, and they take their own part, the part of the bytes they hold.</li>
          <li><strong>Most tokens</strong> is only the size the build form starts with.</li>
        </ul>
        <p>Every edit makes a new revision of the dataset; a corpus records the revision it was built from. A dataset also <strong>publishes</strong> one vocabulary, the one a build picks first and training needs.</p>
      </Part>

      <Part on="tokenizers-explained" title="Tokenizers and vocabularies">
        <p>A tokenizer recipe says how a vocabulary is trained: on a sample of a named dataset (of this depot or of another one), at most a number of bytes of plain text, into a number of tokens (at least 513).</p>
        <p>The depot keeps at most one vocabulary for each version of a recipe, for good. Changing the dataset, the sample size or the number of tokens makes a new version, which trains its own vocabulary; a change of the note alone keeps the version. Adding texts never retrains a vocabulary.</p>
      </Part>

      <Part on="corpora" title="Corpora">
        <p>A corpus is the token files of a dataset for training, built for one trained vocabulary and one size: a number of tokens, or all. A corpus never goes past its size.</p>
        <p>Each dataset, vocabulary and size has one link that never changes. It gives the latest fresh corpus built from the dataset's current revision, or says why there is none: being built, stale, not built, or checking an erasure. A corpus is <strong>stale</strong> once an admitted erasure reaches one of its texts; nothing builds it again by itself.</p>
        <p>The store keeps only the latest corpus of each dataset, vocabulary and size. When the store has a size cap, a build deletes the corpus downloaded least recently to make room.</p>
      </Part>

      <Part on="jobs-explained" title="Jobs">
        <p>Long work runs as jobs, one at a time, in a queue that survives a restart of the server: training a vocabulary, building a corpus, and a write over many texts. Anyone can start one. A vocabulary training and a corpus build each start only after their estimate is shown and confirmed. Only the command <Cmd>wpt depot jobs cancel ID</Cmd>, run where the depot is, cancels a job.</p>
      </Part>

      <Part on="erasures-explained" title="Erasures">
        <p>An erasure request names passages of texts by their place (from one character to another, in a field for an item) or a whole text, with a public reason. It never repeats the passage itself.</p>
        <p>Admitted, it is carried out at once and cannot be undone: every place where the passage appears in that text is cut, in every version of it, locked libraries included; a whole text leaves the depot. Every corpus holding it goes stale. Refused, the texts stay as they are. Every request is kept.</p>
      </Part>

      <Part on="who-decides" title="Who decides, and Settings">
        <p>The name and email every decision records are kept in this browser only, until a decision sends them. Another browser, a private window or cleared site data has none. A form that needs them and finds none says so and links to Settings, which comes back to the form once they are saved.</p>
        <p>When this depot follows another one (its upstream), what belongs to the upstream is marked "the upstream's" and changes only there; a pull brings its changes here.</p>
      </Part>

      <Part on="overview" title="Overview" page={[]}>
        <p>What waits for a decision, and where each dataset stands. It opens from the depot's name at the left of the top bar.</p>
        <ul>
          <li>At the top, the texts admitted out of all texts and the jobs open, each linked. A notice shows when no name and email are set.</li>
          <li><strong>Datasets</strong>: one line per dataset with its stages and the button of its next step; the first dataset with a next step has the main button. "New dataset" starts one.</li>
          <li><strong>Under review</strong>: how many texts wait, counted by tag and by license. The count is made in steps and grows as it goes. Each tag and license opens its view; each number opens those texts.</li>
          <li><strong>Open erasure requests</strong> and <strong>Problems on libraries</strong>, the first eight of each. When there is none of either, two zero counts stand at the top instead.</li>
          <li><strong>Latest decisions</strong>: the ten latest, each opening its view.</li>
        </ul>
      </Part>

      <Part on="texts" title="Texts" page={["texts"]}>
        <p>Where texts are read, scored, tagged, edited and decided, one after the other. The content of the open text fills the middle, its review panel stands at the right, and the list of the open texts hides at the left edge.</p>
        <p><strong>Choosing the texts</strong>, at the top:</p>
        <ul>
          <li>The quick filters: <strong>Under review</strong>, <strong>Not scored</strong> (no score yet; an SQL condition, slower on a large depot), <strong>Newest first</strong> (the texts added last first) and <strong>All</strong>. Under review replaces the choice of statuses, and Not scored replaces any SQL condition; a second click on either takes it away. Newest first changes only the order.</li>
          <li>The filter bar: words of the title, status, tags, licenses and libraries. "More filters" adds an SQL condition, a search in the texts themselves and whole words.</li>
          <li><strong>Open ▾</strong>: every text, each status, library, tag and license with its count, "New library", and the filters saved in this browser, with "Save the open filter". Escape or a click outside closes it.</li>
          <li>Text, Blob and Mono at the right: Blob reads the open texts one after the other as one page; Mono shows them in a fixed-width font.</li>
        </ul>
        <p><strong>The list</strong>: rest the mouse against the left edge of the window, or on the thin strip left of the text, and the list slides out; it goes back when the mouse leaves it. A click on a text opens it. Each text has a tick: the ticked texts get a bar at the bottom to admit, refuse, put back under review, put in a library, score, take out of the open library, report a problem on them or fork the library without them. On a phone, the "Texts" button opens the list, and choosing a text or a tap outside closes it.</p>
        <p><strong>The content</strong> is always editable. The line above it gives the title, the status, the content version and the characters, then the lines and Find. A change is saved only by "Save" or Ctrl+S, which shows what it records first; "Revert" drops it. Saving makes a new content version: the text goes back under review and leaves every open library. On an admitted text, Save asks once more ("Save anyway" or "Keep editing"). Escape leaves the editor, so the keys below work again.</p>
        <p><strong>The review panel</strong>, from top to bottom:</p>
        <ul>
          <li>One row per score the depot knows. Its buttons are the values the depot uses most for that score, or round values spread over its range; the value set last from this browser is underlined. − and + step by a twentieth of the range. The box beside them opens a number pad: click the digits, then ✓; "none" removes the score, ✕ or Escape closes it. The keyboard's digits and Enter work too.</li>
          <li>The depot's nine most used tags as toggles; the text's other tags on the "also" line, each with ✕. "+ tag" opens every tag of the depot, most used first, each added with a click; the box above them narrows them, and Enter in it adds what it holds, a new tag included; Done or Escape closes it. "+ score" takes the name of a new score.</li>
          <li>The buttons: ‹ (previous), Next ›, Refuse, Admit &amp; next; "Back to review" on a text already decided, and "Take out" when an open library is open.</li>
          <li>The presets: a click sets their scores and adds and removes their tags. "+ preset from this text" keeps this text's scores and tags under a name, in this browser only; ✕ forgets one.</li>
          <li>The text's facts: license, address, decisions, libraries and id; then "More on this text": its record, a decision with a reason, its licenses, its credits, taking it out of a library, reporting a problem and asking for an erasure.</li>
        </ul>
        <p>Clicks on scores and tags are gathered and saved together about two seconds after the last one, or at once when another text opens; the line at the bottom of the page says what was saved, as which decision. They keep the text's status and libraries. Admit &amp; next decides at once, under the name the panel shows, and on a text admitted already it only opens the next one; Refuse, Back to review and admitting a refused text first ask for a confirmation, with an optional public reason. The next two texts are read ahead, so Next shows them at once.</p>
        <p><strong>Keys</strong>, outside the editor: J next, K previous, A admit (and next), R refuse, U back to review, T take out of the open library, X tick the open text, E into the editor, Escape out of it.</p>
        <p><strong>A library opened</strong> (from Libraries, or Open ▾) shows its head above the text: "Put texts in it", "Make a dataset", "Read it whole", and under "More": report a problem, fork it, lock it, rename it, and the revisions it pinned. "Put texts in it" opens the admitted texts with a banner; the ticked ones then go into that library.</p>
        <p><strong>Read it whole</strong> opens the library as one long text, its texts one after another in the order of their ids, each after a thin rule with its name (a click opens it). The scrollbar at the right stands for the whole library: drag it or click on it to jump, or type a number of characters or a percentage in "go to"; Start and End, or the keys Home and End, go to either end. The first time a library is read whole, the depot makes its map, which jumps need, in the background: a few minutes for millions of texts; the start and the end read meanwhile.</p>
      </Part>

      <Part on="libraries" title="Libraries" page={["libraries"]}>
        <p>Every library, with its size and the datasets that read it.</p>
        <ul>
          <li>The table: the library's name, its texts, the characters of its plain texts and of its conversations and tasks, its state (open, or locked with the day), and the datasets that read it, or a link to make a dataset from it.</li>
          <li>A library's name opens it on the Texts page, where texts are put in or taken out and where it is renamed, forked or locked.</li>
        </ul>
        <p>To make one:</p>
        <ol>
          <li>Click "New library".</li>
          <li>Give it a name and choose what it starts with: every admitted text, the admitted texts of a filter saved in this browser, or nothing.</li>
          <li>Click the button; the library opens on the Texts page.</li>
        </ol>
      </Part>

      <Part on="datasets" title="Datasets" page={["datasets"]}>
        <p>Every dataset, where it stands, and the forms that make and change one.</p>
        <ul>
          <li>The table: each dataset's stage and next step, its tokenizer recipe, its libraries, its mix, and under "More" its corpora, its export, and its description for <Cmd>wpt depot dataset import</Cmd> on another machine.</li>
        </ul>
        <p>To make one:</p>
        <ol>
          <li>Click "New dataset", and give it a name.</li>
          <li>Choose its tokenizer recipe, or keep the new name filled in from the dataset's name: the recipe is made next.</li>
          <li>Tick its libraries. Add shares if the plain text should be split by tag: a tag and a proportion each, ordered with Up and Down.</li>
          <li>Read the table of what the libraries hold, and the line beside the button: the texts, the characters, what the mix keeps and which share limits it. A share whose tag no plain text has is marked.</li>
          <li>Click "Make the dataset"; its page opens.</li>
        </ol>
        <p><strong>One dataset's page</strong> shows its revision, "Edit" and "Export texts" at the top, then five cards: texts, libraries, dataset, vocabulary, corpus. The first card not done is the next step and holds its button: make the recipe, train it, publish it, build a corpus. "Read again" shows when reading its libraries of this depot again would change something. Folded below: what its libraries hold, then under "More" the libraries with the moment each is read at, the published vocabulary (choose another or none), and its description for another machine.</p>
        <p><strong>Corpora</strong> (from a dataset's page): the corpora built, one row per vocabulary and size, with their state, texts, tokens and mix, and "Download". Below, the form of a new build:</p>
        <ol>
          <li>Choose a trained vocabulary (the published one is filled in) and a size.</li>
          <li>Click "Estimate": tokens, bytes in the store, time, whether it fits the store's cap, and which share would come out short, with the smallest size that keeps every share even.</li>
          <li>Click "Confirm and build". The job is followed at the top of the page.</li>
        </ol>
        <p>Corpora of older revisions are folded at the bottom. A corpus's own page gives its files, each with its size (its sha256 on hover), "Download everything" as one archive, and "Copy its link".</p>
        <p><strong>Export</strong> (from a dataset's page): the dataset's texts as files, with no tokenizer, no size limit and no job, in JSONL compressed with zstd or in Parquet, split into numbered parts when large. Download every part in one form to get the whole dataset. Folded below: what a line or a row holds. Titles, authors, licenses and notices are not in these files; a corpus holds them as tables.</p>
      </Part>

      <Part on="tokenizers" title="Tokenizers" page={["tokenizers"]}>
        <p>The tokenizer recipes, and the vocabularies they trained.</p>
        <ul>
          <li>The table: each recipe's version, the dataset it trains on, its sample, its vocabulary size, and whether it is trained, being trained, or "train it".</li>
          <li>A recipe a dataset names but that does not exist yet shows above the table, with "Make the recipe".</li>
          <li>"New recipe" opens the form: a name (kept for good), a dataset of this depot or of another one, a sample size and a number of tokens. The sizes filled in are small, for a quick test.</li>
        </ul>
        <p><strong>One recipe's page</strong>:</p>
        <ol>
          <li>Click "Train its vocabulary": an estimate of the sample and the time opens.</li>
          <li>Click "Confirm and start the training"; the job is followed on the page.</li>
          <li>Under "Vocabularies", click "Publish it on" its dataset, then "Build a corpus".</li>
        </ol>
        <p>Each vocabulary also has its file to download and, folded, the texts it trained on. "Edit the recipe" saves a new revision; the revisions are folded at the bottom.</p>
      </Part>

      <Part on="jobs" title="Jobs" page={["jobs"]}>
        <p>The server's queue, read again every few seconds: the jobs waiting or running in the order they run, then the latest that ended. Each row says what the job does, its state and how long it waited or ran, its progress or what it gave with a link to the result, and when it was queued. Below, "Can start next" lists each dataset's next step. While a job waits or runs, a mark in the top bar says how many and links here.</p>
      </Part>

      <Part on="erasures" title="Erasures" page={["erasures"]}>
        <p>Every erasure request, and where they are asked for and decided.</p>
        <ul>
          <li>The table: each request, its status, its reason, its passages by text and place, and whether it was carried out. A request opens what it asks to erase, with the text around each passage.</li>
        </ul>
        <p>To ask for one:</p>
        <ol>
          <li>Click "Ask for an erasure".</li>
          <li>Type words of the text's title, or its exact id, and pick it.</li>
          <li>Give the place (or tick the whole text) and a reason, then "Confirm and ask".</li>
        </ol>
        <p>To decide: tick open requests, click "Admit" or "Refuse" in the bar that appears, give an optional reason, then confirm. Admitting cannot be undone. A request of the upstream has no tick.</p>
      </Part>

      <Part on="licenses" title="Licenses" page={["licenses"]}>
        <p>Every license the depot holds: its id, its name, whether the allowed list accepts it (a combined license only if every part is allowed), and how many texts name it. Tick licenses, then "Show their texts" to open them on the Texts page. "Add a license" takes an id (an SPDX name, or licenses that apply together), a name and the official text. A license's own page shows its official text and the texts that name it.</p>
      </Part>

      <Part on="depot" title="Depot" page={["depot"]}>
        <p>This depot's facts: its name, its address, its current change number, its maintainer and takedown contact, and the depot it follows, if any. They change only through the commands.</p>
      </Part>

      <Part on="settings" title="Settings" page={["settings"]}>
        <p>The legal name and email this browser sends with every decision. Type both and click "Save them". Opened from a form, it goes back there once saved.</p>
      </Part>

      <Part on="add" title="Add a text" page={["add"]}>
        <p>The button at the right of the top bar. What it adds is under review and joins no library until admitted.</p>
        <ol>
          <li>Choose what it is: a plain text, or an item of a kind. Type, paste or drop a file for a plain text; fill in the form, or load a JSON file, for an item.</li>
          <li>Tick its licenses and give its notice.</li>
          <li>Give its title, authors, tags and address, all optional. The depot's most used tags stand under the tags box; a click adds or removes one.</li>
          <li>Click "Confirm and add".</li>
          <li>"Add another" keeps the licenses, notice, authors and tags for the next one. The texts added since the page opened are linked together.</li>
        </ol>
      </Part>

      <Part on="views" title="Views of one decision, tag or problem">
        <p>Not in the top bar; the lists that name them link to them, and the top bar keeps Texts marked while a decision or a tag shows, Libraries while a problem shows.</p>
        <ul>
          <li><strong>A decision</strong>: what it did, its status, who decided, when, its reason, and what it concerns: its texts, the erasure requests it decides, or the problem it records. A decision that put texts back under review shows as a correction.</li>
          <li><strong>A tag</strong>: how many texts have it, by status, and those texts. <a href={href(["tags"])}>Tags</a> lists every tag, the most used first, with a search.</li>
          <li><strong>A problem on a library</strong>: the problem, who reported it and when, and its texts.</li>
        </ul>
      </Part>
    </div>
  );
}

export const help: Page = { path: "help", title: "Help", group: "help", Component: () => <HelpPage /> };
