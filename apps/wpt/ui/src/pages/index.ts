import type { ComponentType } from "react";
import { add } from "./add";
import { datasets } from "./datasets";
import { decisions } from "./decisions";
import { depot } from "./depot";
import { download } from "./download";
import { erasures } from "./erasures";
import { help } from "./help";
import { exported } from "./export";
import { jobs } from "./jobs";
import { libraries } from "./libraries";
import { licenses } from "./licenses";
import { overview } from "./overview";
import { settings } from "./settings";
import { strip } from "./strip";
import { tags } from "./tags";
import { tokenizers } from "./tokenizers";
import { texts } from "./texts";

// One entry per page; each calls the routes of apps/wpt/src/wpt/depot/web/routes.py.
// group: where the navigation lists it, the data apart from the setup, and
// the help last (null:
// not listed: "Add a text" is a button, the views of one decision or one
// tag are reached from the lists that name them, a dataset's download and
// export pages from the dataset; the address of one library leads into the
// Texts page). under: for a page not listed, the path of the listed page the
// navigation marks while it shows (a dataset's corpora and export pages mark
// Datasets). rest: the parts of the address after the page's own; query: the
// address's query. full: the page fills the screen, scrolling inside itself (the Texts page, a library read whole).
export type Page = {
  path: string; title: string; group: "data" | "setup" | "help" | null; under?: string; full?: boolean;
  Component: ComponentType<{ rest: string[]; query: URLSearchParams }>;
};

export const pages: Page[] = [overview, texts, libraries, datasets, tokenizers, jobs, erasures, licenses, depot, settings, add, decisions, tags,
  download, exported, strip, help];
