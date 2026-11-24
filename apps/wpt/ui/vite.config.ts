import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { defineConfig } from "vite";

// The translation engine runs in a classic Web Worker that loads its Emscripten glue and its WebAssembly by name,
// beside its own address, so its three files are copied as they are to dist/bergamot/ (src/translation/engine.ts).
const ENGINE = join(dirname(createRequire(import.meta.url).resolve("@browsermt/bergamot-translator/package.json")), "worker");
const ENGINE_FILES = ["translator-worker.js", "bergamot-translator-worker.js", "bergamot-translator-worker.wasm"];

// Vite turns TSX into JavaScript itself, following tsconfig.json's "jsx".
export default defineConfig({
  build: { outDir: "dist", emptyOutDir: true },
  plugins: [{
    name: "translation-engine",
    generateBundle() {
      for (const name of ENGINE_FILES) this.emitFile({ type: "asset", fileName: `bergamot/${name}`, source: readFileSync(join(ENGINE, name)) });
    },
  }],
});
