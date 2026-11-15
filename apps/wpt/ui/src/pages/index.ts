import type { ComponentType } from "react";
import { add } from "./add";
import { datasets } from "./datasets";
import { decisions } from "./decisions";
import { depot } from "./depot";
import { download } from "./download";
import { erasures } from "./erasures";
import { exported } from "./export";
import { jobs } from "./jobs";
import { libraries } from "./libraries";
import { licenses } from "./licenses";
import { overview } from "./overview";
import { settings } from "./settings";
import { tags } from "./tags";
import { tokenizers } from "./tokenizers";
import { workspace } from "./workspace";

// One entry per page; each calls the routes of apps/wpt/src/wpt/depot/web/routes.py.
// group: where the navigation lists it, the data apart from the setup (null:
// not listed: "Add a text" is a button, the views of one decision or one
// tag are reached from the lists that name them, a dataset's download and
// export pages from the dataset, and the addresses of the libraries lead into the
// workspace). rest: the parts of the address
// after the page's own; query: the address's query.
export type Page = {
  path: string; title: string; group: "data" | "setup" | null;
  Component: ComponentType<{ rest: string[]; query: URLSearchParams }>;
};

export const pages: Page[] = [overview, workspace, datasets, tokenizers, jobs, erasures, licenses, depot, settings, add, decisions, tags,
  libraries, download, exported];
