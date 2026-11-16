import { ButtonLink, Mark } from "@workers-intelligence-commons/components";
import { useEffect, useRef } from "react";
import { useAbout } from "./about";
import { pages, type Page } from "./pages";
import { RunningJobs } from "./pages/jobs";
import { href, usePlace } from "./route";

const GROUPS: { group: Page["group"]; label: string }[] = [{ group: "data", label: "Data" }, { group: "setup", label: "Setup" }];

export function App() {
  const place = usePlace();
  const { about } = useAbout();
  const page = pages.find((p) => p.path === (place.parts[0] ?? "")) ?? pages[0];
  // on a phone the navigation is one line that scrolls sideways: the current page is brought into it
  const nav = useRef<HTMLElement>(null);
  useEffect(() => {
    const at = nav.current?.querySelector<HTMLElement>("[aria-current]");
    if (at && nav.current && nav.current.scrollWidth > nav.current.clientWidth) nav.current.scrollLeft = at.offsetLeft - nav.current.offsetLeft - 16;
  }, [page]);
  return (
    <div className={`wrap${page.path === "texts" ? " full" : ""}`}>
      <header className="top">
        <a className="home wic-focus" href={href([])} aria-label="Overview of the depot" aria-current={page.path === "" ? "page" : undefined}>
          <Mark name="wpt" height={24} />
          <span className="cue">{about ? about.about.name : "depot"}</span>
        </a>
        <nav ref={nav} aria-label="Pages of the depot" className="pages">
          {GROUPS.map(({ group, label }) => (
            <ul key={label} className="page-group cue" aria-label={label}>
              {pages.filter((p) => p.group === group).map((p) => (
                <li key={p.path}>
                  <a href={href([p.path])} aria-current={p === page ? "page" : p.path === page.under ? "true" : undefined}>{p.title}</a>
                </li>
              ))}
            </ul>
          ))}
        </nav>
        <RunningJobs />
        <ButtonLink variant="quiet" href={href(["add"])}>Add a text</ButtonLink>
      </header>
      <main>
        <page.Component rest={place.parts.slice(1)} query={place.query} />
      </main>
    </div>
  );
}
