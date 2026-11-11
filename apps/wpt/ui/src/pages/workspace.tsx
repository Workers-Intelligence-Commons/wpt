// The workspace's page: #/texts opens it, #/texts/ID with the text ID in
// its editor, and the address's query names what is open (filter.tsx) and
// its mode (workspace/workspace.tsx).

import { Workspace } from "../workspace/workspace";
import type { Page } from "./index";

export const workspace: Page = {
  path: "texts", title: "Workspace", group: "data",
  Component: ({ rest, query }) => <Workspace current={rest[0] ?? null} query={query} />,
};
