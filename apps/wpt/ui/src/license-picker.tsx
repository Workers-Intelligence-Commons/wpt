// The choice of a text's licenses among those the depot holds (GET
// /api/licenses), found by typing; one the depot lacks is added on the
// spot through the license dialog (license-form.tsx), then chosen.

import { Badge, Button, Dialog, Skeleton } from "@workers-intelligence-commons/components";
import { useCallback, useEffect, useState } from "react";
import { get, type Answer } from "./api";
import { LicenseForm } from "./license-form";
import { Problems, requiredGroup } from "./parts";
import type { License } from "./pages/licenses";
import { TickList } from "./ticks";

export function LicensePicker({ chosen, onChange }: { chosen: string[]; onChange: (ids: string[]) => void }) {
  const [licenses, setLicenses] = useState<Answer<License[]> | null>(null); // null while the first answer is awaited
  const [adding, setAdding] = useState<string | null>(null); // the id typed for a license to add, while its form is open
  const load = useCallback(() => { get<License[]>("/api/licenses").then(setLicenses); }, []);
  useEffect(load, [load]);
  return (
    <>
      {!licenses && <Skeleton lines={3} label="Reading the licenses" />}
      {licenses && !licenses.ok && <Problems problems={licenses.problems} />}
      {licenses?.ok && <TickList label={requiredGroup("Licenses")} legend="Licenses" search chosen={chosen} onChange={onChange}
        hint={<>One allowed license is enough. <Button variant="quiet" onClick={() => setAdding("")}>Add a license the depot lacks</Button></>}
        options={licenses.data.map((l) => ({ value: l.id, words: `${l.id} ${l.name}`, label: <><Badge status="license">{l.id}</Badge> <span className="aside">{l.name}</span></> }))}
        empty={(typed) => (
          <p className="small">The depot holds no license {typed ? <>matching <q>{typed}</q></> : "yet"}. <Button variant="quiet" onClick={() => setAdding(typed)}>Add {typed ? `the license ${typed}` : "a license"}</Button></p>
        )} />}
      <Dialog open={adding !== null} onClose={() => setAdding(null)} title="Add a license">
        {adding !== null && <LicenseForm initialId={adding} onSaved={(id) => { setAdding(null); load(); onChange(chosen.includes(id) ? chosen : [...chosen, id]); }} />}
      </Dialog>
    </>
  );
}
