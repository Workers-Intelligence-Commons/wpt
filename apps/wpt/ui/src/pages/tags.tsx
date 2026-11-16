// Tags: #/tags/TAG is one tag's view: how many texts have it, by status, and
// those texts, with the link to the workspace filtered to it. #/tags lists
// every tag. The page is not in the navigation: lists link to it.

import { Skeleton } from "@workers-intelligence-commons/components";
import { useEffect, useMemo, useState } from "react";
import { get, type Answer } from "../api";
import { MatchedCount, useCounts } from "../counts";
import { EVERY } from "../filter";
import { PageHead, Problems, Section, STATUSES, StatusBadge } from "../parts";
import { href } from "../route";
import { MatchedTexts } from "../table";
import type { Page } from "./index";

function OneTag({ tag }: { tag: string }) {
  const filter = useMemo(() => ({ ...EVERY, tags: [tag] }), [tag]);
  const found = useCounts(filter);
  return (
    <>
      <PageHead title={tag}
        meta={<a href={href(["texts"], { tag })}><MatchedCount counting={found} /></a>}
        chain={<>One of the <a href={href(["tags"])}>tags</a> the <a href={href(["texts"])}>texts</a> have; a dataset's shares choose texts by tag.</>} />
      {found?.problems && <Problems problems={found.problems} />}
      {found && found.counted > 0 && (
        <ul className="plain counted small status-counts" aria-label="Its texts by status">
          {STATUSES.map((s) => (
            <li key={s}><a href={href(["texts"], { tag, status: s })}><StatusBadge status={s} /></a><span className="number">{found.statuses[s].toLocaleString("en")}</span></li>
          ))}
          {found.locked > 0 && <li><span className="aside">in a locked library</span><span className="number">{found.locked.toLocaleString("en")}</span></li>}
        </ul>
      )}
      <Section title="Texts that have it" tools={<a className="small" href={href(["texts"], { tag })}>Open them in the texts</a>}>
        <MatchedTexts filter={{ ...EVERY, tags: [tag] }} label="The texts that have it" empty="No text has it." />
      </Section>
    </>
  );
}

function Tags() {
  const [tags, setTags] = useState<Answer<{ tag: string; texts: number }[]> | null>(null);
  useEffect(() => { get<{ tag: string; texts: number }[]>("/api/tags").then(setTags); }, []);
  return (
    <>
      <PageHead title="Tags" chain={<>What the <a href={href(["texts"])}>texts</a> are, each tag with how many texts have it.</>} />
      {!tags && <Skeleton lines={4} />}
      {tags && !tags.ok && <Problems problems={tags.problems} />}
      {tags?.ok && (tags.data.length === 0 ? <p>No text has a tag.</p> : (
        <ul className="plain counted small tag-list">
          {tags.data.map((t) => <li key={t.tag}><a href={href(["tags", t.tag])} className="cut" title={t.tag}>{t.tag}</a><span className="number">{t.texts.toLocaleString("en")}</span></li>)}
        </ul>
      ))}
    </>
  );
}

export const tags: Page = {
  path: "tags", title: "Tags", group: null, under: "texts",
  Component: ({ rest }) => (rest[0] ? <OneTag key={rest[0]} tag={rest[0]} /> : <Tags />),
};
