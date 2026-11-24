// The language a text is written in, by Firefox's language detector: CLD2 compiled to asm.js (vendor/cld2/), in one
// Web Worker of this page, started on the first detection. The worker says "ready" once loaded, then answers each
// {text, isHTML} in the order asked with {language, confident, languages}, as Firefox's LanguageDetector reads it.
// Nothing leaves the browser.

// What CLD2 found: its language code (CLD2's own, languages.ts's fromDetector turns it into the models') and whether
// it is sure of it.
export type Detected = { language: string; confident: boolean };

const SAMPLE = 20000; // characters read from the start of the text: plenty for CLD2, quick on a text of any size

let worker: Promise<Worker> | null = null;
const waiting: { accept: (d: Detected) => void; reject: (e: Error) => void }[] = [];

function start(): Promise<Worker> {
  return new Promise((ready, failed) => {
    const one = new Worker(new URL("cld2/cld-worker.js", document.baseURI));
    one.onmessage = ({ data }) => {
      if (data === "ready") return ready(one);
      waiting.shift()?.accept({ language: data.language, confident: data.confident });
    };
    one.onerror = (event) => {
      const error = new Error(event.message || "the language detector stopped");
      failed(error);
      waiting.splice(0).forEach((w) => w.reject(error));
      worker = null;
    };
  });
}

export async function detect(text: string): Promise<Detected> {
  if (!worker) worker = start();
  const one = await worker;
  return new Promise((accept, reject) => {
    waiting.push({ accept, reject });
    one.postMessage({ text: text.slice(0, SAMPLE), isHTML: false });
  });
}
