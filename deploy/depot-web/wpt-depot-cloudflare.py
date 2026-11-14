#!/usr/bin/env python3
"""Keep the public depot web app's host in step with Cloudflare's address
ranges (docs/depot-web.md). Runs as root on the host, from the systemd timer
wpt-depot-cloudflare.timer, and once by hand before the containers first start.

It reads Cloudflare's IPv4 and IPv6 ranges, then:

- writes nginx's list of the addresses it takes the real visitor's address
  from, /etc/wpt-depot-web/cloudflare-real-ip.conf, which the nginx
  container mounts read-only;
- makes ufw allow ports 80 and 443 only from those ranges: it adds a rule for
  each range, then deletes every other rule that allows either port; a rule
  whose ports it cannot read (an application profile or a service name that
  no longer exists) stays, named in the error, and the exit code is 1;
- reloads nginx in its container, when the container runs.

An answer that cannot be read, is empty or holds anything but networks of its
address family changes nothing: the previous list stays, and the exit code is 1.
"""

import ipaddress
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

SOURCES = {4: "https://www.cloudflare.com/ips-v4", 6: "https://www.cloudflare.com/ips-v6"}
FOLDER = Path("/etc/wpt-depot-web")
INCLUDE = FOLDER / "cloudflare-real-ip.conf"
PORTS = {80, 443}
COMMENT = "wpt-depot-web cloudflare"
NGINX = ["--filter", "label=com.docker.compose.project=wpt-depot-web",
         "--filter", "label=com.docker.compose.service=nginx"]


def ranges(url: str, version: int) -> list:
    """The networks the answer at url lists, one per line. ValueError when it
    lists none or a line is not a network of that version."""
    # Cloudflare refuses Python's own User-Agent
    request = urllib.request.Request(url, headers={"User-Agent": "wpt-depot-cloudflare"})
    with urllib.request.urlopen(request, timeout=60) as answer:
        lines = answer.read(1_000_000).decode("ascii").split()
    if not lines:
        raise ValueError(f"{url} lists no network")
    found = [ipaddress.ip_network(line) for line in lines]
    wrong = [str(net) for net in found if net.version != version]
    if wrong:
        raise ValueError(f"{url} lists networks that are not IPv{version}: {', '.join(wrong)}")
    return found


def write_include(networks: list) -> bool:
    """Write nginx's list; whether it differs from the one there before."""
    text = ("# Cloudflare's address ranges, written by wpt-depot-cloudflare (deploy/depot-web)\n"
            + "".join(f"set_real_ip_from {net};\n" for net in networks))
    if INCLUDE.is_file() and INCLUDE.read_text(encoding="utf-8") == text:
        return False
    FOLDER.mkdir(mode=0o755, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=FOLDER, delete=False) as new:
        new.write(text)
    os.chmod(new.name, 0o644)
    os.replace(new.name, INCLUDE)  # the folder is mounted, not the file: nginx sees the new file
    return True


def ufw(*args: str) -> str:
    return subprocess.run(["ufw", *args], check=True, capture_output=True, text=True).stdout


def port_numbers(spec: str) -> set[int]:
    """The ports of a ufw port list such as 80,443/tcp or 8000:8100; a service
    name counts as its port."""
    found = set()
    for part in spec.split("/")[0].split(","):
        if ":" in part:
            low, high = part.split(":")
            found |= set(range(int(low), int(high) + 1))
        elif part.isdigit():
            found.add(int(part))
        else:
            found.add(socket.getservbyname(part))
    return found


def app_ports(name: str) -> set[int]:
    """The ports of a ufw application profile."""
    lines = ufw("app", "info", name).splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "Ports:")
    return set().union(*(port_numbers(line.strip()) for line in lines[start + 1:] if line.strip()))


def rule_ports(words: list[str]) -> set[int]:
    """The ports a rule of `ufw show added` (without its leading ufw and its
    comment) names; none for a rule on addresses that names no port."""
    if "port" in words:
        return port_numbers(words[words.index("port") + 1])
    if "app" in words:
        return app_ports(words[words.index("app") + 1])
    if "from" in words or "to" in words:
        return set()
    spec = words[-1]
    return port_numbers(spec) if spec[0].isdigit() or "/" in spec else app_ports(spec)


def stale_rules(networks: list) -> tuple[list[list[str]], list[list[str]]]:
    """The rules that let ports 80 or 443 in and are not one of this script's
    for a range of networks, and the allowing rules whose ports cannot be read."""
    stale, unread = [], []
    for line in ufw("show", "added").splitlines():
        if not line.startswith("ufw "):
            continue
        words = shlex.split(line)[1:]
        comment = words[words.index("comment") + 1] if "comment" in words else None
        bare = words[:words.index("comment")] if comment is not None else words
        if bare[0] not in ("allow", "limit") or "out" in bare[:2]:
            continue
        try:
            ports = rule_ports(bare)
        except (subprocess.CalledProcessError, OSError):
            # its ports unknown, it may be SSH's: kept, never deleted blindly
            unread.append(words)
            continue
        if not PORTS & ports:
            continue
        source = bare[bare.index("from") + 1] if "from" in bare else None
        ours = (comment == COMMENT and source not in (None, "any") and bare[-3:] == ["80,443", "proto", "tcp"]
                and ipaddress.ip_network(source, strict=False) in networks)
        if not ours:
            stale.append(words)
    return stale, unread


def reload_nginx() -> None:
    try:
        running = subprocess.run(["docker", "ps", "-q", *NGINX], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"nginx not reloaded: docker cannot list its container ({error})")
        return
    for container in running.stdout.split():
        subprocess.run(["docker", "kill", "--signal", "HUP", container], check=True, capture_output=True)
        print("nginx reloaded")
    if not running.stdout.split():
        print("nginx not reloaded: its container is not running")


def main() -> int:
    try:
        networks = [net for version, url in SOURCES.items() for net in ranges(url, version)]
    except (OSError, ValueError, UnicodeDecodeError) as error:
        print(f"Cloudflare's ranges not read, nothing changed: {error}", file=sys.stderr)
        return 1
    changed = write_include(networks)
    print(f"{len(networks)} ranges; nginx's list {'written' if changed else 'unchanged'}")
    try:
        for net in networks:
            ufw("allow", "proto", "tcp", "from", str(net), "to", "any", "port", "80,443", "comment", COMMENT)
        stale, unread = stale_rules(networks)
        for words in stale:
            ufw("delete", *words)
            print(f"ufw rule deleted: {shlex.join(words)}")
    except subprocess.CalledProcessError as error:
        print(f"ufw failed: {shlex.join(error.cmd)}: {error.stderr.strip()}", file=sys.stderr)
        return 1
    if changed:
        reload_nginx()
    for words in unread:
        print(f"ufw rule kept, its ports cannot be read; delete it by hand if it allows 80 or 443: "
              f"ufw delete {shlex.join(words)}", file=sys.stderr)
    return 1 if unread else 0


if __name__ == "__main__":
    sys.exit(main())
