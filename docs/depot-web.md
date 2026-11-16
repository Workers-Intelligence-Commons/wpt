# The public depot web app

`deploy/depot-web/` serves `wpt depot web` for one depot at
https://depot.workers-intelligence-commons.org, on one host with Docker,
behind Cloudflare's proxy. Anyone who reaches it reads and changes the
depot through every route, with no login ([architecture.md](architecture.md),
"Two interfaces"); the limits below only slow each address down.

## The containers

`deploy/depot-web/compose.yaml` runs three containers:

- **app**: `wpt depot web --port 8780`, on 127.0.0.1. Its image
  (`deploy/depot-web/Dockerfile`) holds the repository's code, its Python
  libraries and the browser interface, built when the image is built, so
  the container never builds anything. `WPT_DEPOT` names
  `wpt-depot.sqlite` in the host folder `WPT_DEPOT_FOLDER`, mounted whole,
  since SQLite writes its `-wal` and `-shm` files, the depot its
  `.waiting` folder, the queue its `wpt-depot.sqlite.jobs` file and the
  jobs their `wpt-depot.sqlite.work` folder beside the file, and the store
  its local copies in `cache/` there. `WPT_DEPOT_URL` is
  `https://depot.workers-intelligence-commons.org`, the address the
  corpora it builds name the depot by. The app runs as `WPT_DEPOT_USER`,
  which must be able to write in that folder. It reads no `.env` file. The
  store settings Compose passes it name the store that holds the corpora
  it builds and serves, with a key that reads and writes it; without
  `WPT_STORE`, it builds and serves no corpus, and an erasure carried out
  through it marks no corpus stale.
- **nginx**: the reverse proxy, with the limits below. It takes the
  visitor's address from the `CF-Connecting-IP` header of a request that
  comes from one of Cloudflare's address ranges. It answers only for
  depot.workers-intelligence-commons.org: a request for any other name, on
  port 80 or 443, gets its connection closed with no answer. Port 80
  serves Let's Encrypt's challenges and redirects everything else to
  HTTPS. The HTTPS server starts once the certificate exists, and nginx
  reloads every 12 hours, which picks up a renewed certificate.
- **certbot**: gets the Let's Encrypt certificate through the HTTP-01
  challenge, in a folder it shares with nginx, then renews it when it is
  due. It tries every 12 hours, and 30 minutes after a failed attempt.

Every container uses the host's network. Ports that Docker publishes pass
around ufw, and ufw lets only Cloudflare's address ranges reach ports 80
and 443. Nothing else on the host may listen on ports 80, 443 or 8780.

## The limits

nginx counts each request against one limit, by the visitor's address.
A request over its limit gets 429.

| Requests | Rate | Burst |
| --- | --- | --- |
| `POST` to `/api/sql`, `/api/texts/search`, `/api/texts/passages`, `/api/texts/counts` and `/api/texts/blocked`, `GET /api/datasets/{id}/estimate` | 1 a second | 5, delayed |
| every other `POST`, except `/api/libraries/measure` | 1 a second | 10, delayed |
| every other request, `POST` to `/api/libraries/measure` included (it reads only the depot's counts) | 10 a second | 40, at once |

A filter of every text, or of one status, tag, license or library, gets
how many texts it matches, their split by status and how many a locked
library holds at once from the depot's counts, when those hold them. The
pages count the rest of it, and every other filter, one step at a time, at most
one step every two seconds, and send a step refused with 429 or for lack
of a free reader again after a wait, so counting leaves at least half of the first limit to the other
requests.

An address holds at most 20 connections at a time. A request body holds
at most 100 MB, and the app has 100 seconds to answer. The files of a
corpus, the parts of a dataset's export and a dataset's sample stream for
as long as they last:
nginx passes each piece on as it comes, without a temporary file, and waits
up to 600 seconds between two pieces. The limits above count them as any
other request. A build's estimate counts as a heavy read.

## Jobs and downloads

The web app runs its long work as jobs, one at a time, in a process of
their own: the training of a tokenizer recipe's vocabulary, the build of a
dataset's corpus, and a write over many texts. The queue lives in
`wpt-depot.sqlite.jobs`, which nothing backs up, so a job survives a
restart of the app: a queued job stays queued, and a job the restart cut
short runs again, a corpus build going on from its last token file. Anyone
may start a job. Each dataset has at most one corpus job open, and each
recipe one vocabulary job. The page "Jobs" shows the queue and each job's
progress. Only `wpt depot jobs cancel ID`, run where the depot is, cancels
one; `wpt depot jobs` lists them.

A dataset's page "Export" (`GET /api/datasets/{id}/export`) gives the
texts of its current revision with no tokenizer, no build and no job: every
plain text a share takes and every conversation and task, whatever its
`max_tokens`, in numbered parts, each in JSONL compressed with zstd and in
Parquet, made as it is sent and the same bytes every time for the same
revision. Each text's export key, a seeded hash of its id that its depot
stores with it and with each library's row of it, places it, and each part
is one range of the keys, so no pass orders the dataset. Reading a part
reads, for each of the dataset's libraries, that range of the library's
rows by key, so it never reads the texts of the other parts nor the rows
of other libraries. A part
names each depot by the name a corpus records for it, and the app reads a
text of another local depot from that depot's file. At most 2 parts or
samples are sent at once in all, and 1 to each visitor; past that the app
answers 429.

A corpus is built for a dataset, one vocabulary the depot holds (by its
sha256) and a size: a number of tokens, or all. A build at a size counts
everything, and the corpus never passes that size: conversations and
tasks take the share of the bytes they hold in the corpus of every token,
and the shares split the rest ([data-selection.md](data-selection.md), "Datasets"). The dataset's `max_tokens` is only the size the build form
fills in. The page "Download" lists the
dataset's corpora and holds the form of a new build: a trained vocabulary,
the one the dataset publishes filled in, else the latest of its own recipe, and a size, then an estimate (`GET /api/datasets/{id}/estimate`: the
tokens, the bytes in the store, the time of tokenizing on this machine
and whether it fits under the cap, from a small sample tokenized, writing
nothing), and
only after it a start that restates the estimate and asks for a
confirmation.
At a size, the estimate's tokens are a forecast that walks the dataset's
order as the build does, where a share stops at its first text that does
not fit. Its headline, above the start, says how many tokens will come
out and which share will come out short. It also names the smallest size
at which every share comes within 10 percentage points of its proportion; a built
corpus whose shares came out uneven offers the same size. The forecast
counts the texts as the build does: the plain texts, and how many of the
dataset's conversations and tasks the size takes. The estimate and the
build read the bytes of each share and of the conversations and tasks from
the figures the dataset's revision stores. When that revision's figures are
not known, both count the same figures from every text instead, and the
estimate warns that it did. When reading the dataset again would change
what it reads, the form also links to the dataset's page, whose "Read
again" stores figures for the depot's own libraries. A build never trains a vocabulary: the training is the
recipe page's own action, with its own estimate and confirmation.

Each dataset, vocabulary and size has one fixed download link
(`GET /api/datasets/{id}/download/{vocabulary}/{size}`). It gives the
latest fresh corpus of that vocabulary and size built from the dataset's
current revision, or from a revision that differs from it only in its
`max_tokens` or the vocabulary it publishes, which a build never reads.
Otherwise it says why there is
none: being built (with the job's progress), stale (an
admitted erasure reached one of its texts), not built, or checking an
erasure (an erasure decided here is being carried into the corpora's
records; no corpus is sent meanwhile). Opening it starts nothing, and
nothing builds a stale corpus again by itself. The corpus's files stream
from the store, each with its size and sha256, and take a `Range` of bytes.

The listing of a dataset's corpora (`GET /api/datasets/{id}/download`)
names, for each vocabulary and size, the latest fresh corpus built from
any revision of the dataset, with that revision. Training through the web
app takes that corpus, by the same rule as from the depot's file
([training.md](training.md), "Training").

The store keeps only the latest fresh corpus of each dataset, vocabulary
and size: a build deletes the one before of the same three once its own is
whole, and the corpora of other vocabularies and sizes stay.
`WPT_CORPORA_CAP` bounds the bytes all corpora take together, the files a
stopped build left without a record included. To make room, a build
deletes those files first, then the corpus downloaded least recently, a
file at a time, never one being built or downloaded. A corpus whose
estimated size passes the cap fails before its first upload, with that
reason. A deleted corpus keeps its record, and a build of the same dataset
state, vocabulary and size writes the same corpus into its folder again.

## Cloudflare's address ranges

`deploy/depot-web/wpt-depot-cloudflare.py` runs on the host, as root,
from the systemd timer `wpt-depot-cloudflare.timer` every week. It reads
Cloudflare's IPv4 and IPv6 ranges, then:

- writes `/etc/wpt-depot-web/cloudflare-real-ip.conf`, the list of
  addresses nginx takes the visitor's address from; the nginx container
  mounts the folder read-only and does not start without the file;
- adds a ufw rule that allows ports 80 and 443 from each range, then
  deletes every other ufw rule that allows either port; it leaves rules on
  other ports, SSH's included, as they are;
- keeps a rule whose ports it cannot read, an application profile or a
  service name that no longer exists, and names it in its error, and the
  service fails;
- reloads nginx when its list changed and its container runs.

An answer that cannot be read, is empty, or holds anything but networks of
its address family changes nothing: the previous list and rules stay, and
the service fails.

## Setting up the host

The host needs Docker with Compose, ufw with its SSH rule, and Python 3.
Cloudflare proxies the domain's DNS record, with the SSL/TLS mode "Full
(strict)" and "Always Use HTTPS" off: nginx redirects to HTTPS itself, and
Let's Encrypt's challenges must reach port 80 over plain HTTP.

1. Clone the repository with its submodule (`git clone --recurse-submodules`).
2. Put the depot's file, `wpt-depot.sqlite`, in its own folder, writable by
   `WPT_DEPOT_USER`.
3. Write the settings below into `deploy/depot-web/.env` (git-ignored). The
   repository's `.env` stays out of it: the app needs a key to the corpora
   only.
4. Install the script and its timer, then run it once:

   ```sh
   sudo install -m 755 deploy/depot-web/wpt-depot-cloudflare.py /usr/local/sbin/wpt-depot-cloudflare
   sudo install -m 644 deploy/depot-web/wpt-depot-cloudflare.service deploy/depot-web/wpt-depot-cloudflare.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl start wpt-depot-cloudflare.service
   sudo systemctl enable --now wpt-depot-cloudflare.timer
   ```

5. From `deploy/depot-web/`, start the containers:

   ```sh
   docker compose up -d
   ```

   certbot gets the first certificate over port 80, and nginx starts its
   HTTPS server within a minute of it.

6. Bring the tokenizer recipes into the depot. For each recipe file
   (JSON with `sample_bytes`, `vocab_size` and an optional `note`) put in
   the depot's folder, from `deploy/depot-web/`:

   ```sh
   docker compose run --rm app wpt depot tokenizer import NAME /depot/NAME.json --dataset DATASET
   ```

   with `--lock /depot/LOCK.json` to keep the vocabulary a lock file names
   ([tokenizer.md](tokenizer.md), "Importing one"). Without a vocabulary,
   the first build of a dataset that names the recipe trains it.

7. Name what backs the depot up. Every write leaves a change file in
   `wpt-depot.sqlite.waiting/` until `wpt depot backup` sends it to the
   store, and the app sends none by itself, so without one of these the
   folder grows until the disk fills
   ([architecture.md](architecture.md), "Backups"):
   - a timer on the host, as `WPT_DEPOT_USER`, runs `wpt depot backup` from
     a checkout whose `.env` names the store, with `WPT_DEPOT` set to the
     depot's file;
   - or, from `deploy/depot-web/`, `docker compose run --rm app wpt depot
     backup --off` marks the depot that nothing backs up.

To update, `git pull --recurse-submodules`, then `docker compose up -d
--build` from `deploy/depot-web/`. A change to the script or its systemd
files needs step 4 again. A new depot format needs the upgrade first, the
app stopped, since the app refuses a depot of another format:

```sh
docker compose stop app
docker compose run --rm --build app wpt depot upgrade
docker compose up -d --build
```

| Setting | Meaning |
| --- | --- |
| `WPT_DEPOT_FOLDER` | the host folder that holds `wpt-depot.sqlite`; required |
| `WPT_DEPOT_USER` | `uid:gid` the app runs as; default `1000:1000` |
| `WPT_STORE` | the store that holds the corpora (`s3://BUCKET/PREFIX`); unset, no corpus is built or served |
| `WPT_STORE_ACCESS_KEY_ID`, `WPT_STORE_SECRET_ACCESS_KEY` | a key that reads and writes that store |
| `WPT_STORE_ENDPOINT_URL`, `WPT_STORE_REGION`, `WPT_STORE_SESSION_TOKEN` | optional, as for any store ([architecture.md](architecture.md), "The store") |
| `WPT_CORPORA_CAP` | the bytes all corpora may take together, such as `3TB`; unset, no cap |
| `CERTBOT_EMAIL` | the address Let's Encrypt writes to about the certificate; required |
