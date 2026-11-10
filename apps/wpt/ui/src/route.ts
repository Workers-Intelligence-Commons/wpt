// The address after "#/" picks the page and what it shows: #/texts/fox-12
// is the page "texts" with the rest ["fox-12"], and #/texts?status=admitted
// gives it the query status=admitted. Links are plain <a href="#/...">.

import { useEffect, useState } from "react";

export type Place = { parts: string[]; query: URLSearchParams };

function place(): Place {
  const [path, query = ""] = window.location.hash.replace(/^#\/?/, "").split("?", 2);
  return { parts: path.split("/").filter(Boolean).map(decodeURIComponent), query: new URLSearchParams(query) };
}

export function usePlace(): Place {
  const [found, setFound] = useState(place);
  useEffect(() => {
    const changed = () => setFound(place());
    window.addEventListener("hashchange", changed);
    return () => window.removeEventListener("hashchange", changed);
  }, []);
  return found;
}

// The address of a page: its parts, each one part of the path, and its query.
export function href(parts: string[], query?: Record<string, string | string[] | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query ?? {}))
    for (const one of Array.isArray(value) ? value : value == null ? [] : [value]) search.append(key, one);
  const text = search.toString();
  return `#/${parts.map(encodeURIComponent).join("/")}${text ? `?${text}` : ""}`;
}
