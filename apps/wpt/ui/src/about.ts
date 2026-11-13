// The depot's about route, read once per page load (and again after a
// failure): its name and address, and the words of the confirmation and the
// address of the requirements that every decision form shows. A page that needs the depot's change
// number now reads /api/about itself.

import { useEffect, useState } from "react";
import { get, type Answer } from "./api";

export type About = {
  address: string;
  change: number;
  about: { name: string; maintainer: string; takedown_contact: string; upstream: string | null; revision: number };
  confirmation: string;
  requirements: string;
};

// The answer once one came; a failed question is asked again by the next
// page that needs it.
let known: About | null = null;
let asking: Promise<Answer<About>> | null = null;

export function useAbout(): { about: About | null; problems: string[] } {
  const [about, setAbout] = useState<About | null>(known);
  const [problems, setProblems] = useState<string[]>([]);
  useEffect(() => {
    if (known) return;
    let live = true;
    asking ??= get<About>("/api/about").then((a) => {
      asking = null;
      if (a.ok) known = a.data;
      return a;
    });
    asking.then((a) => {
      if (!live) return;
      if (a.ok) setAbout(a.data);
      else setProblems(a.problems);
    });
    return () => { live = false; };
  }, []);
  return { about, problems };
}
