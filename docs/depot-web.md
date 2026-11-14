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
  since SQLite writes its `-wal` and `-shm` files and the depot its
  `.waiting` folder beside the file. The app runs as `WPT_DEPOT_USER`,
  which must be able to write in that folder. It reads no `.env` file and
  holds no store setting: a dataset's page cannot list its corpora, and an erasure
  carried out through it marks no corpus of the store stale.
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
| `POST` to `/api/sql`, `/api/texts/search`, `/api/texts/passages` and `/api/texts/blocked` | 1 a second | 5, delayed |
| every other `POST` | 1 a second | 10, delayed |
| every other request | 10 a second | 40, at once |

An address holds at most 20 connections at a time. A request body holds
at most 100 MB, and the app has 100 seconds to answer.

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
   repository's `.env` stays out of it: no container needs its
   credentials.
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

6. Name what backs the depot up. Every write leaves a change file in
   `wpt-depot.sqlite.waiting/` until `wpt depot backup` sends it to the
   store, and the app holds no store settings, so without one of these the
   folder grows until the disk fills
   ([architecture.md](architecture.md), "Backups"):
   - a timer on the host, as `WPT_DEPOT_USER`, runs `wpt depot backup` from
     a checkout whose `.env` names the store, with `WPT_DEPOT` set to the
     depot's file;
   - or, from `deploy/depot-web/`, `docker compose run --rm app wpt depot
     backup --off` marks the depot that nothing backs up.

To update, `git pull --recurse-submodules`, then `docker compose up -d
--build` from `deploy/depot-web/`. A change to the script or its systemd
files needs step 4 again.

| Setting | Meaning |
| --- | --- |
| `WPT_DEPOT_FOLDER` | the host folder that holds `wpt-depot.sqlite`; required |
| `WPT_DEPOT_USER` | `uid:gid` the app runs as; default `1000:1000` |
| `CERTBOT_EMAIL` | the address Let's Encrypt writes to about the certificate; required |
