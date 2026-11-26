// Tags: #/tags/TAG is one tag's view: how many texts have it, by status, and
// those texts, with the link to the Texts page filtered to it. #/tags lists
// every tag. The page is not in the navigation, where it marks the texts; lists link to it.

import { Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { get, type Answer } from "../api";
import { MatchedCount, useCounts, STATUSES_ONLY } from "../counts";
import { EVERY } from "../filter";
import { PageHead, Problems, Section, Stat, STATUSES } from "../parts";
import { href } from "../route";
import { MatchedTexts } from "../table";
import type { Page } from "./index";

function OneTag({ tag }: { tag: string }) {
  const filter = useMemo(() => ({ ...EVERY, tags: [tag] }), [tag]);
  const found = useCounts(filter, 0, STATUSES_ONLY); // the statuses alone
  return (
    <>
      <PageHead title={tag}
        meta={<a href={href(["texts"], { tag })}><MatchedCount counting={found} /></a>}
        chain={<>One of the <a href={href(["tags"])}>tags</a> the <a href={href(["texts"])}>texts</a> have. A hand-picked set of texts is a tag; an order's shares choose texts by tag.</>} />
      {found?.problems && <Problems problems={found.problems} />}
      {found && found.counted > 0 && (
        <div className="stats">
          {STATUSES.map((s) => <Stat key={s} href={href(["texts"], { tag, status: s })} value={found.statuses[s].toLocaleString("en")} label={s} />)}
        </div>
      )}
      <Section title="Texts that have it" tools={<a className="small" href={href(["texts"], { tag })}>Open them in the texts</a>}>
        <MatchedTexts filter={{ ...EVERY, tags: [tag] }} label="The texts that have it" empty="No text has it." />
      </Section>
    </>
  );
}

// Every tag, the most used first, each with a bar of its count against the largest, on a log scale so a small count still shows; a search narrows the list.
function Tags() {
  const [tags, setTags] = useState<Answer<{ tag: string; texts: number }[]> | null>(null);
  const [find, setFind] = useState("");
  useEffect(() => { get<{ tag: string; texts: number }[]>("/api/tags").then(setTags); }, []);
  const sorted = tags?.ok ? [...tags.data].sort((a, b) => b.texts - a.texts || a.tag.localeCompare(b.tag)) : [];
  const most = sorted[0]?.texts || 1;
  const shown = sorted.filter((t) => t.tag.toLowerCase().includes(find.trim().toLowerCase()));
  return (
    <>
      <PageHead title="Tags" meta={tags?.ok ? <span>{tags.data.length.toLocaleString("en")} tags</span> : undefined}
        chain={<>What the <a href={href(["texts"])}>texts</a> are, each tag with how many texts have it.</>} />
      {!tags && <Skeleton lines={4} />}
      {tags && !tags.ok && <Problems problems={tags.problems} />}
      {tags?.ok && (tags.data.length === 0 ? <p>No text has a tag.</p> : (
        <>
          <input className="wic-input wic-focus tag-find" type="search" aria-label="Find a tag" placeholder="find a tag" value={find} onChange={(e) => setFind(e.target.value)} />
          <ul className="plain counted small tag-list">
            {shown.map((t) => (
              <li key={t.tag}>
                <a href={href(["tags", t.tag])} className="cut" title={t.tag}>{t.tag}</a>
                <span className="number">{t.texts.toLocaleString("en")}</span>
                <span className="tag-bar" aria-hidden="true" style={{ "--share": `${Math.max(4, (Math.log1p(t.texts) / Math.log1p(most)) * 100)}%` } as CSSProperties} />
              </li>
            ))}
          </ul>
        </>
      ))}
    </>
  );
}

export const tags: Page = {
  path: "tags", title: "Tags", group: null, under: "texts",
  Component: ({ rest }) => (rest[0] ? <OneTag key={rest[0]} tag={rest[0]} /> : <Tags />),
};
