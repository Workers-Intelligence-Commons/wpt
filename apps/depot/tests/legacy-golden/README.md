# Golden answers of the legacy dataset routes

This folder holds the answers that the web app of commit 6b470328c7 gives to the dataset routes that code at
earlier commits calls. They were recorded on the format-12 fixture depot and its dock
(`packages/depot/tests/fixtures/`). The legacy API's tests compare their answers with these.

| File | What it is |
| --- | --- |
| `answers.json` | every request in the order it was asked: its file, route, request headers and status |
| `answers/NNN.json` | one answer: `method`, `route`, `asked_headers`, `status`, `headers` (Content-Type, Content-Length, Content-Disposition, Content-Range, Accept-Ranges, ETag, when sent) and `body` |
| `answers/NNN.bin` | the bytes of an answer that is not JSON or is a file to save, named by `body_file` with its length in `body_bytes` |
| `record_golden.py` | the script that recorded them |

The order matters. Reading an export part records its size, so the second listing of an export names the sizes
of its parts, and the first one does not.

## Routes

`{id}` is a dataset's id, `{vocabulary}` a vocabulary's sha256 and `{size}` a number of tokens or `all`. Old
clients read the fields named in the last column. Every route is a GET.

| Route | Who calls it | Fields old clients read |
| --- | --- | --- |
| `/api/datasets` | `wpt train` through `order_depot.WebDepot.dataset` | `id`, `name` of each row |
| `/api/datasets/{id}/download` | `wpt train` through `WebDepot.built` | `state`, `problem`, `erasures`, `corpora[].vocabulary.sha256`, `corpora[].max_tokens`, `corpora[].latest_fresh.id` |
| `/api/tokenizers/{name}` | `wpt train` through `WebDepot.vocab_size` | `recipe.version`, `recipe.vocab_size`, `trained` |
| `/api/tokenizers/{name}/vocabularies/{version}/file` | `wpt train` through `WebDepot.vocab_size` | the whole body, a vocabulary file |
| `/api/datasets/{id or name}/sample?bytes=N` | another depot's `depot tokenizer train` (`corpus.remote_sample`) | lines `{"depot", "text", "bytes"}` each followed by that many bytes, `{"waiting": true}`, and a last `{"end": true}` |
| `/api/datasets/{id}` | a person, from the dataset's page | none |
| `/api/datasets/{name}/remote` | a person: the dataset as `depot dataset import` reads it | the remote (`depot.remote`) |
| `/api/datasets/{id}/download/{vocabulary}/{size}` | a person, from the download page | `state`, `files[].name`, `archive` |
| `/api/datasets/{id}/download/{vocabulary}/{size}/files/{name}` | a person downloading a corpus file, with or without a Range | the bytes |
| `/api/datasets/{id}/download/{vocabulary}/{size}/everything.tar` | a person downloading every file of a corpus, with or without a Range | the bytes |
| `/api/datasets/{id}/export` | a person, from the export page | `parts[].forms` |
| `/api/datasets/{id}/export/part-NNNNN.jsonl.zst`, `.parquet` | a person downloading the texts of a dataset | the bytes |
| `/api/tokenizers`, `/api/vocabularies`, `/api/tokenizers/{name}/vocabularies/{version}`, `.../texts` | a person, from the tokenizer pages, which the web app no longer has | none: the test expects 404 for them |

The answers also cover the refusals: a dataset, a recipe, a corpus file, an export part and a vocabulary version
that do not exist (404), a size never built (`state` "not built"), a sample with no size (400) and a Range past the
end of a file (416).

## Recording them again

Run the script with the code of 6b470328c7, extracted as the fixtures' README says, on a free port between 8500 and
8599:

`UV_PROJECT_ENVIRONMENT=VENV-12 uv run --frozen --no-sync python -I PATH/record_golden.py FIXTURES THIS-FOLDER PORT`

It serves copies of the depot and dock on 127.0.0.1 and stops the server when it ends. Two runs on the same
fixtures give the same files.
