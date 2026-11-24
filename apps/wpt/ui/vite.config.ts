import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { defineConfig } from "vite";

// The translation engine and the language detector each run in a classic Web Worker that loads its other files by
// name, beside its own address, so their files are copied as they are: the engine's to dist/bergamot/
// (src/translation/engine.ts), the detector's to dist/cld2/ (src/translation/detect.ts, vendor/cld2/).
const ENGINE = join(dirname(createRequire(import.meta.url).resolve("@browsermt/bergamot-translator/package.json")), "worker");
const COPIED = [
  ...["translator-worker.js", "bergamot-translator-worker.js", "bergamot-translator-worker.wasm"]
    .map((name) => [join(ENGINE, name), `bergamot/${name}`]),
  ...["cld-worker.js", "cld-worker.js.mem"].map((name) => [join(import.meta.dirname, "vendor", "cld2", name), `cld2/${name}`]),
];

// Vite turns TSX into JavaScript itself, following tsconfig.json's "jsx".
export default defineConfig({
  build: { outDir: "dist", emptyOutDir: true },
  plugins: [{
    name: "translation-workers",
    generateBundle() {
      for (const [source, fileName] of COPIED) this.emitFile({ type: "asset", fileName, source: readFileSync(source) });
    },
  }],
});
