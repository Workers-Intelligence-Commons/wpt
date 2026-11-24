import type { ComponentType } from "react";
import { add } from "./add";
import { build } from "./build";
import { decisions } from "./decisions";
import { depot } from "./depot";
import { erasures } from "./erasures";
import { help } from "./help";
import { jobs } from "./jobs";
import { licenses } from "./licenses";
import { overview } from "./overview";
import { settings } from "./settings";
import { sizes } from "./sizes";
import { snapshots } from "./snapshots";
import { tags } from "./tags";
import { tokenizers } from "./tokenizers";
import { texts } from "./texts";

// One entry per page; each calls the routes of apps/wpt/src/wpt/depot/web/routes.py.
// group: where the navigation lists it, the data apart from the setup, and
// the help last (null: not listed: "Add a text" is a button, the views of one
// decision or one tag are reached from the lists that name them). under: for
// a page not listed, the path of the listed page the navigation marks while
// it shows, or a function of the address's query that names it. rest: the parts of the address after the page's own; query: the
// address's query. full: the page fills the screen, scrolling inside itself
// (the Texts page).
export type Page = {
  path: string; title: string; group: "data" | "setup" | "help" | null; under?: string | ((query: URLSearchParams) => string); full?: boolean;
  Component: ComponentType<{ rest: string[]; query: URLSearchParams }>;
};

export const pages: Page[] = [overview, texts, snapshots, build, tokenizers, jobs, erasures, licenses, depot, settings, add, decisions, tags, sizes, help];
