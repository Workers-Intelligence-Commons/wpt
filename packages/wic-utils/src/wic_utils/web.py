"""Every web request WPT makes, under its own program name.

Sites behind Cloudflare refuse Python's default name (Python-urllib) with
403, so a request names the program `wpt` instead.

    urlopen(url, body=None, headers=None, method=None, timeout)   urllib.request.urlopen's answer
"""

import urllib.request

AGENT = "wpt"


def urlopen(url: str, body: bytes | None = None, headers: dict | None = None,
           method: str | None = None, timeout: float | None = None):
    """The answer to a request for url, as urllib.request.urlopen gives it,
    sent with the User-Agent `wpt`; HTTPError and OSError as it raises them."""
    request = urllib.request.Request(url, body, {"User-Agent": AGENT, **(headers or {})}, method=method)
    return urllib.request.urlopen(request, timeout=timeout)
