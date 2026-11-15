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
| `POST` to `/api/sql`, `/api/texts/search`, `/api/texts/passages`, `/api/texts/counts` and `/api/texts/blocked` | 1 a second | 5, delayed |
| every other `POST` | 1 a second | 10, delayed |
| every other request | 10 a second | 40, at once |

The pages count the texts a filter matches one step at a time, at most
one step every two seconds, and send a step refused with 429 or for lack
of a free reader again after a wait, so counting leaves at least half of the first limit to the other
requests.

An address holds at most 20 connections at a time. A request body holds
at most 100 MB, and the app has 100 seconds to answer. The files and text
parts of a corpus and a dataset's sample stream for as long as they last:
nginx passes each piece on as it comes, without a temporary file, and waits
up to 600 seconds between two pieces. The limits above count them as any
other request.

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

"Build" on a dataset queues its corpus build, after the training of its
recipe's vocabulary when the recipe's version has none. Each dataset has
one fixed download link, its page "Download"
(`GET /api/datasets/{id}/download`). It gives the latest fresh corpus built
from the dataset's current revision, or says why there is none: being
built (with the job's progress), stale (an admitted erasure reached one of
its texts), not built, or checking an erasure (an erasure decided here is
being carried into the corpora's records; no corpus is sent meanwhile). A stale corpus gives nothing until someone presses
"Build it again"; nothing builds it by itself. The corpus's files stream
from the store, each with its size and sha256, and take a `Range` of bytes.
Its texts come as numbered parts, each in JSONL compressed with zstd and in
Parquet, made as they are sent and the same bytes every time. A part names
each depot by the name its corpus records, and the app reads a text of
another local depot from that depot's file, found from the dataset's
libraries at the revision the corpus took. At most 2
text parts or samples are sent at once in all, and 1 to each visitor; past
that the app answers 429.

The store keeps only the latest fresh corpus of each dataset: a build
deletes the one before once its own is whole. `WPT_CORPORA_CAP` bounds the
bytes all corpora take together, the files a stopped build left without a
record included. To make room, a build deletes those files first, then the
corpus downloaded least recently, a file at a time, never one being built
or downloaded. A corpus whose estimated size passes the cap fails before
its first upload, with that reason. A
deleted corpus keeps its record, and "Build it again" builds it from the
same dataset revision and vocabulary, saying so when an erasure changed it
since.

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
