"""dock: the dock's command line, a controller over dock.api
(docs/api/dock.md), on the dock DOCK names (environment or .env).

    dock find PREFIX               every file whose key starts with PREFIX, one JSON line each: {key, bytes}
    dock fetch KEY [--sha256 S] [--out FILE]
                                   the file's bytes, on standard output or into FILE, which must not
                                   exist; checked against S when it is given
    dock put KEY FILE              FILE's bytes stored under KEY, replacing any earlier file; prints
                                   {key, sha256, bytes}

Exit codes: 1 refused or failed, 2 a wrong command line.
"""

import argparse
import json
import sys
from pathlib import Path

from dock import api
from wic_utils.lines import ERROR, say


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dock", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    finding = commands.add_parser("find", help="every file whose key starts with a prefix")
    finding.add_argument("prefix")
    fetching = commands.add_parser("fetch", help="a file's bytes")
    fetching.add_argument("key")
    fetching.add_argument("--sha256")
    fetching.add_argument("--out", help="a new file to write them into (default: standard output)")
    putting = commands.add_parser("put", help="a file stored under a key")
    putting.add_argument("key")
    putting.add_argument("file")
    args = parser.parse_args(argv)
    try:
        dock = api.url()
        if args.command == "find":
            for found in api.find(dock, args.prefix):
                print(json.dumps(found, ensure_ascii=False))
        elif args.command == "fetch":
            if args.out is not None and Path(args.out).exists():
                say(f"dock fetch: {args.out} exists: give a new file", ERROR)
                return 1
            data = api.fetch(dock, args.key, args.sha256)
            if args.out is None:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            else:
                Path(args.out).write_bytes(data)
        else:
            print(json.dumps(api.put_from(dock, args.key, args.file)))
    except (api.DockError, OSError, ValueError) as error:
        say(f"dock {args.command}: {type(error).__name__}: {error}", ERROR)
        return 1
    return 0


def entry():
    sys.exit(main())
