// Build: a corpus built from a model recipe the person hands over. The depot
// reads no repository, so a recipe is the person's file: chosen with the
// button or dropped anywhere on the page (a model recipe or a series; only its
// data section counts), then checked, measured, estimated and built
// (recipe-build.tsx). A data section composed on a snapshot's page comes here
// by its "Build it" button. #/build?key=KEY shows the corpus of a build key
// (GET /api/builds/KEY) with its downloads and the job building it, where a
// corpus job links to, first and in place of the recipe kept from before.

import { Skeleton } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, part, type Answer } from "../api";
import { PageHead, Problems, Section } from "../parts";
import { Corpus, keepHanded, keptHanded, RecipeBuild, RecipeDrop, type BuildShown, type Handed } from "../recipe-build";
import { href } from "../route";
import type { Page } from "./index";
import { JobLine } from "./jobs";

// The corpus of a build key on its own, with the job building it.
function BuildKey({ buildKey }: { buildKey: string }) {
  const [shown, setShown] = useState<Answer<BuildShown> | null>(null);
  const [version, setVersion] = useState(0);
  useEffect(() => { get<BuildShown>(`/api/builds/${part(buildKey)}`).then(setShown); }, [buildKey, version]);
  return (
    <Section title="Corpus of this build key">
      {!shown && <Skeleton lines={2} />}
      {shown && !shown.ok && <Problems problems={shown.problems} />}
      {shown?.ok && shown.data.job && <JobLine key={shown.data.job.id} job={shown.data.job} onEnded={() => setVersion((v) => v + 1)} />}
      <Corpus buildKey={buildKey} version={version} />
    </Section>
  );
}

function Build({ query }: { query: URLSearchParams }) {
  const key = query.get("key");
  // the recipe kept from before, unless the address names a build key: that corpus is what the page shows then
  const [handed, setHanded] = useState<Handed | null>(() => (key ? null : keptHanded()));
  const take = useCallback((h: Handed) => { keepHanded(h); setHanded(h); }, []);
  const clear = () => { keepHanded(null); setHanded(null); };
  return (
    <>
      <PageHead title="Build"
        chain={<>A model recipe is your file: its data section names a <a href={href(["snapshots"])}>snapshot</a> of this depot with its change and digest, the kinds and the shares by tag it takes, a size and a <a href={href(["tokenizers"])}>vocabulary</a>. Hand it over here, a model recipe or a series (whose one corpus every size reads), and its build gives a corpus, a cache that the same recipe always gives again. A new data section is made of clicks on a snapshot's page.</>} />
      {key && !handed && <BuildKey buildKey={key} />}
      <RecipeDrop onHanded={take} whole primary={!handed && !key} label={handed ? "Choose another recipe file" : "Choose a recipe file"} />
      {handed && <RecipeBuild key={JSON.stringify(handed)} handed={handed} onClear={clear} />}
    </>
  );
}

export const build: Page = { path: "build", title: "Build", group: "data", Component: ({ query }) => <Build key={query.get("key") ?? ""} query={query} /> };
