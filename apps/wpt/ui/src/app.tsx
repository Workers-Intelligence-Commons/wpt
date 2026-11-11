import { ButtonLink, Mark } from "@workers-intelligence-commons/components";
import { useAbout } from "./about";
import { pages, type Page } from "./pages";
import { href, usePlace } from "./route";

const GROUPS: { group: Page["group"]; label: string }[] = [{ group: "data", label: "Data" }, { group: "setup", label: "Setup" }];

export function App() {
  const place = usePlace();
  const { about } = useAbout();
  const page = pages.find((p) => p.path === (place.parts[0] ?? "")) ?? pages[0];
  return (
    <div className={`wrap${page.path === "texts" ? " full" : ""}`}>
      <header className="top">
        <a className="home wic-focus" href={href([])} aria-label="Overview of the depot" aria-current={page.path === "" ? "page" : undefined}>
          <Mark name="wpt" height={24} />
          <span className="cue">{about ? about.about.name : "depot"}</span>
        </a>
        <nav aria-label="Pages of the depot" className="pages">
          {GROUPS.map(({ group, label }) => (
            <ul key={label} className="page-group cue" aria-label={label}>
              {pages.filter((p) => p.group === group).map((p) => (
                <li key={p.path}>
                  <a href={href([p.path])} aria-current={p === page ? "page" : undefined}>{p.title}</a>
                </li>
              ))}
            </ul>
          ))}
        </nav>
        <ButtonLink variant="quiet" href={href(["add"])}>Add a text</ButtonLink>
      </header>
      <main>
        <page.Component rest={place.parts.slice(1)} query={place.query} />
      </main>
    </div>
  );
}
