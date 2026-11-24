// One line under the bar of pages that every page shows, for what happened after the page that caused it closed
// (an edit that could not be saved when its text was left by the browser's back button): it stays until dismissed
// or replaced.

import { Button } from "@workers-intelligence-commons/components";
import { useEffect, useState, type ReactNode } from "react";

let shown: ReactNode = null;
const watching = new Set<(words: ReactNode) => void>();

export function tell(words: ReactNode) {
  shown = words;
  watching.forEach((one) => one(words));
}

export function Notice() {
  const [words, setWords] = useState<ReactNode>(shown);
  useEffect(() => {
    watching.add(setWords);
    return () => { watching.delete(setWords); };
  }, []);
  if (!words) return null;
  return (
    <div className="notice-line small" role="alert">
      <span>{words}</span>
      <Button variant="quiet" onClick={() => tell(null)}>Dismiss</Button>
    </div>
  );
}
