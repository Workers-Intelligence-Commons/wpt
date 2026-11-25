// What every page reads of the depot's snapshots, the same way: a snapshot
// (GET /api/snapshots) is a name on a change number, the UTC time it was
// made, with the set of admitted texts then, which an order to the freight
// names. This file keeps what the pages read them through: one request per
// address at a time (shared), read again after a page's own action
// (pipelineChanged) or a move to another address, and the words and address
// of a snapshot.

import { useCallback, useEffect, useState } from "react";
import { get, type Answer, type Job } from "./api";
import { href } from "./route";

// [texts, characters, bytes]
export type Sized = [number, number, number];
// A snapshot as GET /api/snapshots lists it; GET /api/snapshots/ID adds its
// summary and what erasures carried out since did to it.
export type Snapshot = {
  id: string; name: string; owner: string; at_change: number; change: number; made_at: string; origin: { kind: string } & Record<string, unknown>;
  note: string | null; texts: number; characters: number; bytes: number; digest: string;
};
export type Summary = {
  version: number; kinds: Record<string, Sized>; tags: Record<string, Sized>; plain_tagsets: Record<string, Sized>;
  licenses: Record<string, Sized>; owners: Record<string, number>;
};
export type SnapshotShown = Snapshot & {
  summary: Summary; erased_since: { texts: number; erasures: string[] };
};
// What GET /api/snapshots/new gives: the snapshot job queued or running, if any, and the cost of making one now.
export type SnapshotEstimate = { texts: number; seconds: number; bytes: number; set_bytes: number; summary_bytes: number; at: string; basis: string; change: number };
export type SnapshotOffer = { job: Job<unknown> | null; estimate: SnapshotEstimate };

// What the pages read changes after a page's own action (a job that ends):
// pipelineChanged() tells every part of the page that reads them to read
// again, as does a move to another address, since a page kept for a new
// address would otherwise show what it read first.
//
// The parts of a page that read the same address at the same time share one
// request (shared). A change forgets the requests under way, so what is read
// after it never comes from a request sent before it.
const CHANGED = "wpt-pipeline-changed";
const reading = new Map<string, Promise<Answer<unknown>>>();
export function shared<T>(address: string): Promise<Answer<T>> {
  let answer = reading.get(address);
  if (!answer) {
    const sent: Promise<Answer<unknown>> = get<unknown>(address).finally(() => { if (reading.get(address) === sent) reading.delete(address); });
    reading.set(address, sent);
    answer = sent;
  }
  return answer as Promise<Answer<T>>;
}
export const pipelineChanged = () => { reading.clear(); window.dispatchEvent(new Event(CHANGED)); };
window.addEventListener("hashchange", () => reading.clear(), { capture: true });
export function usePipelineChange(read: () => void) {
  useEffect(() => {
    window.addEventListener(CHANGED, read);
    window.addEventListener("hashchange", read);
    return () => { window.removeEventListener(CHANGED, read); window.removeEventListener("hashchange", read); };
  }, [read]);
}

// One address read once, again on a change (pipelineChanged) and by reload.
function useShared<T>(address: string | null): { answer: Answer<T> | null; reload: () => void } {
  const [answer, setAnswer] = useState<Answer<T> | null>(null);
  const reload = useCallback(() => { if (address) shared<T>(address).then(setAnswer); }, [address]);
  useEffect(reload, [reload]);
  usePipelineChange(reload);
  return { answer, reload };
}
export const useSnapshots = () => useShared<Snapshot[]>("/api/snapshots");

// A snapshot's page, and the words that name it with its change.
export const snapshotHref = (s: { id: string }, query?: Record<string, string | null>) => href(["snapshots", s.id], query);
export const snapshotWords = (s: { name: string; at_change: number }) => `${s.name} (change ${s.at_change.toLocaleString("en")})`;
