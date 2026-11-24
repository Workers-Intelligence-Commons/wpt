# CLD2, as Firefox builds it

These files are Firefox's language detector, copied unchanged from
`toolkit/components/translations/cld2/` of
https://github.com/mozilla-firefox/firefox at commit
755782163bca76b2d4925bc184d3e8bbcbaad2cd.

`cld-worker.js` is CLD2 (Compact Language Detector 2, Apache-2.0,
https://github.com/CLD2Owners/cld2) compiled by Mozilla to asm.js as a Web
Worker; it reads `cld-worker.js.mem` beside itself. `LICENSE` and `moz.yaml`
are Mozilla's: the license text and where the code comes from.

The interface's build copies the two worker files to `dist/cld2/`
(`vite.config.ts`), and `src/translation/detect.ts` speaks to the worker.
