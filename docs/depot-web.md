# The public depot web app

`deploy/depot-web/` serves `depot web` for one depot at
https://depot.workers-intelligence-commons.org, on one host with Docker,
behind Cloudflare's proxy. Anyone who reaches it reads and changes the
depot through every route, with no login ([architecture.md](architecture.md),
"Two interfaces"); the limits below only slow each address down.

## The containers

`deploy/depot-web/compose.yaml` runs three containers:

- **app**: `depot web --port 8780`, on 127.0.0.1. Its image
  (`deploy/depot-web/Dockerfile`) holds the repository's code, its Python
  libraries and the browser interface, built when the image is built, so
  the container never builds anything. `DEPOT` names
  `depot.sqlite` in the host folder `DEPOT_FOLDER`, mounted whole,
  since SQLite writes its `-wal` and `-shm` files, the depot its
  `.waiting` folder, the queue its `depot.sqlite.jobs` file and the
  jobs their `depot.sqlite.work` folder beside the file, and the dock
  its local copies in `cache/` there. `DEPOT_URL` is
  `https://depot.workers-intelligence-commons.org`, the address an order
  names the depot by. The app runs as `DEPOT_USER`, which must be able
  to write in that folder. It reads no `.env` file. The dock settings
  Compose passes it name the dock that holds the corpora the frozen
  dataset routes serve, with a key that reads and writes it; without
  `DOCK`, it serves no corpus, and an erasure carried out through it
  marks no corpus stale. It reads no code repository, and builds no corpus
  and trains no vocabulary: the freight does ([api/freight.md](api/freight.md)).
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
| `POST` to `/api/texts/search` and `/api/texts/passages` (a page of a list or of a strip of texts, each bounded whatever the depot's size) | 5 a second | 20, at once |
| `POST` to `/api/sql`, `/api/texts/counts` and `/api/texts/blocked` | 1 a second | 5, delayed |
| every other `POST`, except `/api/texts/held` and `/api/snapshots/ID/strip` | 1 a second | 10, delayed |
| every other request, those two included (the first reads only the depot's counts; the second reads a page of a snapshot's order by length, and the page follows the job that makes that order every half second) | 10 a second | 40, at once |

A page of texts has a limit of its own, so the first page of a filter never
waits behind a step of a count. A filter of every text, or of statuses with
one tag or license or with kinds, gets how many texts it matches and their
split by status at once from the depot's counts (`/api/texts/held`), when
those hold them, and a filter of statuses alone its tags and licenses too.
The pages count what those leave out and a page shows, and every other
filter, one step at a time, at most one step every two seconds, and send a
step refused with 429 or for lack of a free reader again after a wait, so
counting leaves at least half of the second limit to the other requests.

An address holds at most 20 connections at a time. A request body holds
at most 100 MB, and the app has 100 seconds to answer. On the frozen
routes of datasets, a corpus's files, an export's parts and a sample
stream for as long as they last: nginx passes each piece on as it comes,
without a temporary file, and waits up to 600 seconds between two pieces.
The limits above count them as any other request.

## Jobs and downloads

The web app runs its long work as jobs, each in a process of its own, in
three lanes that each run one job at a time. The write lane runs the writes
over many texts (a decision, a license change, a score, a change of tags);
the heavy lane makes the snapshots, one at a time; the read lane
orders a snapshot by the length of its texts, once for good, the first
time its strip or its size map opens (`POST /api/snapshots/{id}/strip`
queues it and answers its state), and keeps that order in
`depot.sqlite.work/sizes/`, about 14 bytes a text of the snapshot. A
snapshot holds the depot's write lock for one short transaction at most,
so a decision over many texts never waits behind it, and a small write runs
in the request while no write job is open. The queue lives in
`depot.sqlite.jobs`, which nothing backs up, so a job survives a
restart of the app: a queued job stays queued, and a job the restart cut
short runs again, a snapshot read again from the start, so it reads the
change current when it finishes. Anyone may start a job. At most one
snapshot job is open, and each snapshot has at most one order by length.
The page "Jobs"
shows the queue, each job's lane and progress. Only `depot jobs cancel
ID`, run where the depot is, cancels one; `depot jobs` lists them. A
queue file written by an older app is brought up when the app opens it:
a job of an operation or a kind the app no longer has (a vocabulary's
training, a corpus's build) fails, saying so, and so does every write
still queued or running, since its selection could name a
library, which no longer exists, and run now it would reach more texts
than were chosen. Nothing of it is written; the texts are chosen again.

Every heavy action shows its cost before the click that starts it, and its
route refuses it without `confirmed: true` and the estimate it showed; when
the estimate now differs (a cost more than 10% above the one shown), it
answers 409 with the new estimate to confirm instead:

- a snapshot (`GET /api/snapshots/new`: the admitted texts now, the
  seconds and the bytes, and the snapshot job open, if any; `POST
  /api/snapshots`, which takes no name: the depot names the snapshot by
  the UTC time it is made, `YYYY-MM-DDTHH-MM-SSZ`, and refuses with 409
  while another is being made);
- a change of tags over a selection (`POST /api/texts/retag/estimate`:
  the texts changed, the rows and bytes written, the seconds;
  `POST /api/texts/retag`, a job of the write lane past 100 texts).

`GET /api/snapshots` lists the snapshots; `GET /api/snapshots/{id}` gives
one with its summary (by kind, tag, plain tag set, license and owner), how
many of its texts an erasure carried out after it cut or removed (worked
out once for each last erasure and kept in the queue's file);
`GET /api/snapshots/{id}/sets` gives its
sets as portable roaring bitmaps in base64 with its digest, so anyone can
check it without a copy of the depot; `POST /api/snapshots/{id}/texts`
gives its texts a page at a time, each at its revision then.

The depot lives apart from any code repository, and may serve models of
other code. An order names a snapshot by its name, change and digest, which
its page shows; the freight reads the snapshot through the depot's request
module and makes the vocabulary and the corpus
([api/freight.md](api/freight.md)).

### Models trained before snapshots

The dataset routes and two reads of a tokenizer recipe are frozen: they
answer what they answered for a depot
of format 12, read-only, from the depot's legacy tables
(`depot/legacy.py`, served by `depot_app/web/legacy.py`), so a model
trained before snapshots can still be checked with the depot's address and
the repository at the commit it records ([training.md](training.md),
"Models trained before snapshots"). They have no page. They are `GET
/api/datasets` and, for one dataset, its page, its `remote` fields, the
listing of its corpora (`download`), each corpus's fixed link
(`download/{vocabulary}/{size}`) with its files and `everything.tar`, its
export and its parts, and its sample (`sample?bytes=N`), the plain texts a
vocabulary of another depot trained on; and two reads of a tokenizer
recipe, `GET /api/tokenizers/{name}` (its row, revisions and vocabularies,
whether its version is trained) and `GET
/api/tokenizers/{name}/vocabularies/{version}/file`, which `wpt train`
read for a dataset's vocabulary. Their answers are the depot's legacy rows
and the corpora the dock kept; nothing builds a corpus of a dataset. The export gives a dataset revision's texts in numbered parts,
each in JSONL compressed with zstd and in Parquet, made as it is sent and
the same bytes every time, a conversation whose quotes it cannot hold left
out ([data-selection.md](data-selection.md), "Conversations that quote
texts"). At most 2 parts and samples are sent at once in all, and 1 to
each visitor; past that the app answers 429.

## Cloudflare's address ranges

`deploy/depot-web/depot-cloudflare.py` runs on the host, as root,
from the systemd timer `depot-cloudflare.timer` every week. It reads
Cloudflare's IPv4 and IPv6 ranges, then:

- writes `/etc/depot-web/cloudflare-real-ip.conf`, the list of
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
2. Put the depot's file, `depot.sqlite`, in its own folder, writable by
   `DEPOT_USER`.
3. Write the settings below into `deploy/depot-web/.env` (git-ignored). The
   repository's `.env` stays out of it: the app needs a key to the corpora
   only.
4. Install the script and its timer, then run it once:

   ```sh
   sudo install -m 755 deploy/depot-web/depot-cloudflare.py /usr/local/sbin/depot-cloudflare
   sudo install -m 644 deploy/depot-web/depot-cloudflare.service deploy/depot-web/depot-cloudflare.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl start depot-cloudflare.service
   sudo systemctl enable --now depot-cloudflare.timer
   ```

5. From `deploy/depot-web/`, start the containers:

   ```sh
   docker compose up -d
   ```

   certbot gets the first certificate over port 80, and nginx starts its
   HTTPS server within a minute of it.

6. Name what backs the depot up. Every write leaves a change file in
   `depot.sqlite.waiting/` until `depot backup` sends it to the
   dock, and the app sends none by itself, so without one of these the
   folder grows until the disk fills
   ([architecture.md](architecture.md), "Backups"):
   - a timer on the host, as `DEPOT_USER`, runs `depot backup` from
     a checkout whose `.env` names the dock, with `DEPOT` set to the
     depot's file;
   - or, from `deploy/depot-web/`, `docker compose run --rm app depot
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
beside the depot's file, in `depot.sqlite.upgrade-check.json`. It runs
while the app keeps serving, from the new code's image: it holds one read
transaction to its end, so the WAL keeps every page the app writes until
then (`depot.sqlite-wal` grows by what the app writes, and goes back
to its usual size at the next checkpoint after the check), and its reads
take disk time from the app, so run it when the site is quiet. Then stop
the app and back the depot up right before the upgrade (`depot
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
docker compose run --rm --build -e SQLITE_TMPDIR=/depot app depot upgrade --check
docker compose stop app
docker compose run --rm app depot backup
docker compose run --rm -e SQLITE_TMPDIR=/depot app depot upgrade --yes
docker compose up -d --build
```

| Setting | Meaning |
| --- | --- |
| `DEPOT_FOLDER` | the host folder that holds `depot.sqlite`; required |
| `DEPOT_USER` | `uid:gid` the app runs as; default `1000:1000` |
| `DOCK` | the dock that holds the corpora the frozen dataset routes serve (`s3://BUCKET/PREFIX`); unset, no corpus is served |
| `DOCK_ACCESS_KEY_ID`, `DOCK_SECRET_ACCESS_KEY` | a key that reads and writes that dock |
| `DOCK_ENDPOINT_URL`, `DOCK_REGION`, `DOCK_SESSION_TOKEN` | optional, as for any dock ([architecture.md](architecture.md), "The dock") |
| `CERTBOT_EMAIL` | the address Let's Encrypt writes to about the certificate; required |
