// The Texts page: #/texts opens it, #/texts/ID with the text ID open, and
// the address's query names what is open (filter.tsx) and its mode
// (texts/texts.tsx).

import { Texts } from "../texts/texts";
import type { Page } from "./index";

export const texts: Page = {
  path: "texts", title: "Texts", group: "data", full: true,
  Component: ({ rest, query }) => <Texts current={rest[0] ?? null} query={query} />,
};
