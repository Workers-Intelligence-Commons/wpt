// This depot: its about row and change number. A new depot, a clone, a pull
// from the depot it follows and a change of its maintainer or takedown
// contact are made with the commands (wpt depot new, clone, pull, about).

import { useCallback, useEffect, useState } from "react";
import { get, type Answer } from "../api";
import { type About } from "../about";
import { Facts, PageHead, Problems } from "../parts";
import { href } from "../route";
import type { Page } from "./index";

function Depot() {
  const [about, setAbout] = useState<Answer<About> | null>(null);
  const load = useCallback(() => { get<About>("/api/about").then(setAbout); }, []);
  useEffect(load, [load]);
  return (
    <>
      <PageHead title="Depot"
        chain={<>Every <a href={href(["texts"])}>text</a>, <a href={href(["libraries"])}>library</a>, <a href={href(["datasets"])}>dataset</a> and <a href={href(["erasures"])}>erasure request</a> lives in this depot. <span className="ledger">wpt depot new</span> and <span className="ledger">wpt depot clone</span> make another; <span className="ledger">wpt depot pull</span> fetches the depot it follows, and <span className="ledger">wpt depot about</span> changes its maintainer and takedown contact.</>} />
      {about && !about.ok && <Problems problems={about.problems} />}
      {about?.ok && (
        <Facts rows={[
          ["name", about.data.about.name],
          ["address", <span className="ledger">{about.data.address}</span>],
          ["change", <span className="ledger">{about.data.change}</span>],
          ["maintainer", about.data.about.maintainer],
          ["takedown contact", about.data.about.takedown_contact],
          ["follows", about.data.about.upstream ? <span className="ledger">{about.data.about.upstream}</span> : "no other depot"],
        ]} />
      )}
    </>
  );
}

export const depot: Page = { path: "depot", title: "Depot", group: "setup", Component: Depot };
