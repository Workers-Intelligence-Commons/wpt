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
  through it marks no corpus stale. Compose gives it no `WPT_REPOSITORY`,
  so its page "Recipes" lists no recipe and says no repository is set.
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
| `POST` to `/api/sql`, `/api/texts/search`, `/api/texts/passages`, `/api/texts/counts`, `/api/texts/blocked` and `/api/recipes/compose`, `GET /api/recipes/{name}` (both work out a build's estimate) | 1 a second | 5, delayed |
| every other `POST` | 1 a second | 10, delayed |
| every other request | 10 a second | 40, at once |

A filter of every text, or of statuses with one tag or license or with
kinds, gets how many texts it matches and their split by status at once
from the depot's counts, when those hold them. The
pages count the rest of it, and every other filter, one step at a time, at most
one step every two seconds, and send a step refused with 429 or for lack
of a free reader again after a wait, so counting leaves at least half of the first limit to the other
requests.

An address holds at most 20 connections at a time. A request body holds
at most 100 MB, and the app has 100 seconds to answer. The files of a
corpus and its archive, and on the frozen routes of datasets a corpus's
files, an export's parts and a sample, stream for as long as they last:
nginx passes each piece on as it comes, without a temporary file, and waits
up to 600 seconds between two pieces. The limits above count them as any
other request. A build's estimate counts as a heavy read.

## Jobs and downloads

The web app runs its long work as jobs, each in a process of its own, in
two lanes that each run one job at a time. The write lane runs the writes
over many texts (a decision, a license change, a score, a change of tags);
the heavy lane runs the making of a snapshot, the training of a tokenizer
recipe's vocabulary and the build of a recipe's corpus. A heavy job holds
the depot's write lock for one short transaction at most, so a decision
over many texts never waits behind a build of hours, and a small write runs
in the request while no write job is open. The queue lives in
`wpt-depot.sqlite.jobs`, which nothing backs up, so a job survives a
restart of the app: a queued job stays queued, and a job the restart cut
short runs again, a corpus build going on from its last token file (its
checkpoint is kept by its build key), a snapshot read again from the start,
so it reads the change current when it finishes. Anyone may start a job.
Each build key has at most one corpus job open, each snapshot name one
snapshot job, and each tokenizer recipe one vocabulary job. The page "Jobs"
shows the queue, each job's lane and progress. Only `wpt depot jobs cancel
ID`, run where the depot is, cancels one; `wpt depot jobs` lists them. A
queue file written by an older app is brought up when the app opens it:
a job of an operation the app no longer has fails, saying so, and so does
every write still queued or running, since its selection could name a
library, which no longer exists, and run now it would reach more texts
than were chosen. Nothing of it is written; the texts are chosen again.

Every heavy action shows its cost before the click that starts it, and its
route refuses it without `confirmed: true` and the estimate it showed; when
the estimate now differs (a cost more than 10% above the one shown, another
build key, a corpus the store's cap would now delete), it answers 409 with
the new estimate to confirm instead:

- a snapshot (`GET /api/snapshots/new`: the name offered, the year and
  month, then `2026-11b`, `2026-11c`, ... once taken, with the admitted
  texts now, the seconds and the bytes; `POST /api/snapshots`);
- a build (`GET /api/recipes/{name}`: tokens, bytes, seconds, from a
  sample, what the cap would delete, whether its corpus is built already;
  `POST /api/recipes/{name}/build`);
- a change of tags over a selection (`POST /api/texts/retag/estimate`:
  the texts changed, the rows and bytes written, the seconds;
  `POST /api/texts/retag`, a job of the write lane past 100 texts).

`GET /api/snapshots` lists the snapshots; `GET /api/snapshots/{id}` gives
one with its summary (by kind, tag, plain tag set, license and owner), how
many of its texts an erasure carried out after it cut or removed (worked
out once for each last erasure and kept in the queue's file) and the
repository's recipes that name it; `GET /api/snapshots/{id}/sets` gives its
sets as portable roaring bitmaps in base64 with its digest, so anyone can
check it without a copy of the depot; `POST /api/snapshots/{id}/texts`
gives its texts a page at a time, each at its revision then.

The page of recipes lists the model recipes of the repository checkout
`WPT_REPOSITORY` names on the machine that serves the app, read through
`wpt_recipes`, with the commit the checkout is at; without the setting the
page says so. Each series follows as `SERIES:`, read at its largest size:
the one corpus every size of the series reads. Each recipe shows its state (ready, a recipe of the legacy
dataset shape, which is never built here, its snapshot not here or not the
one it names, its vocabulary missing), its measure from its snapshot's summary, its estimate
and the corpus of its build key. `POST /api/recipes/compose` composes a
data section from a snapshot, kinds, shares, a size and a vocabulary, with
its measure and estimate, and writes nothing: the recipe file is committed
by a person. A corpus downloads by its build key: `GET /api/builds/{key}`
(its state and files), `.../files/{name}` and `.../everything.tar`, each
taking a `Range` of bytes.

A corpus is a cache: the same recipe on the same snapshot builds it again,
so a model never needs a kept corpus to be checked
([training.md](training.md), "What is exactly repeatable").

### Models trained before snapshots

The dataset routes are frozen: they answer what they answered for a depot
of format 12, read-only, from the depot's legacy tables
(`wpt_depot/legacy.py`, served by `wpt/depot/web/legacy.py`), so a model
trained before snapshots can still be checked with the depot's address and
the repository at the commit it records ([training.md](training.md),
"Models trained before snapshots"). They have no page. They are `GET
/api/datasets` and, for one dataset, its page, its `remote` fields, the
listing of its corpora (`download`), each corpus's fixed link
(`download/{vocabulary}/{size}`) with its files and `everything.tar`, its
export and its parts, and its sample (`sample?bytes=N`), the plain texts a
vocabulary of another depot trained on. Their answers are the depot's
legacy rows and the corpora the store kept; nothing builds a corpus of a
dataset. The export gives a dataset revision's texts in numbered parts,
each in JSONL compressed with zstd and in Parquet, made as it is sent and
the same bytes every time, a conversation whose quotes it cannot hold left
out ([data-selection.md](data-selection.md), "Conversations that quote
texts"). At most 2 parts and samples are sent at once in all, and 1 to
each visitor; past that the app answers 429.

A build deletes no other corpus but those the cap makes it delete.
`WPT_CORPORA_CAP` bounds the bytes all corpora take together, the files a
stopped build left without a record included. To make room, a build
deletes those files first, then the corpus downloaded least recently, a
file at a time, never one being built or downloaded. A corpus whose
estimated size passes the cap fails before its first upload, with that
reason. A deleted corpus keeps its record.

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
   docker compose run --rm app wpt depot tokenizer import NAME /depot/NAME.json --snapshot SNAPSHOT
   ```

   with `--lock /depot/LOCK.json` to keep the vocabulary a lock file names
   ([tokenizer.md](tokenizer.md), "Importing one"). Without a vocabulary,
   the recipe's page trains it, its cost shown first.

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
app stopped, since the app refuses a depot of another format. One run
brings the depot up through every format between, so a depot of format 11
goes to 12 and then to 13. The check, `--check`, only reads: it says what
the upgrade does (each locked library and each dataset revision models were
trained on made a snapshot, the open libraries left as they are), what it
costs (time, WAL, disk, the full copy the next backup sends) and what to
check before, and writes nothing in the depot; it saves what it measured
beside the depot's file, in `wpt-depot.sqlite.upgrade-check.json`. It runs
while the app keeps serving, from the new code's image: it holds one read
transaction to its end, so the WAL keeps every page the app writes until
then (`wpt-depot.sqlite-wal` grows by what the app writes, and goes back
to its usual size at the next checkpoint after the check), and its reads
take disk time from the app, so run it when the site is quiet. Then stop
the app and back the depot up right before the upgrade (`wpt depot
backup`, which writes nothing in the depot's tables); only the wpt of the
old format opens that backup, so restoring it means checking out that wpt
first. `--yes` runs the check again, taking what the saved check measured
when nothing was written in the depot since it read it (it says so, and
reads every library again otherwise), and upgrades only when it passes; a
failed check writes nothing. Both sort the depot's tables in SQLite's
temporary folder, which `SQLITE_TMPDIR` names; inside the container it
would default to `/var/tmp`, in the container's own layer, so the commands
point it at the depot's folder, which needs the free disk the check says on
top of the disk the upgrade adds. Upgrade this depot before the depots that
clone it, since a clone of the new format refuses an upstream of the old
one:

```sh
docker compose run --rm --build -e SQLITE_TMPDIR=/depot app wpt depot upgrade --check
docker compose stop app
docker compose run --rm app wpt depot backup
docker compose run --rm -e SQLITE_TMPDIR=/depot app wpt depot upgrade --yes
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
