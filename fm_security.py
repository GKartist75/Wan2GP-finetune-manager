"""URL policy for every network request the plugin makes.

The finetune registry is community content, so every URL in a JSON file is
untrusted. This module gives the plugin one place to answer three questions:

* is this URL safe to contact at all? (scheme, userinfo, host, resolved IP)
* is it still safe after every redirect?
* is the response small enough to be a JSON metadata reply?

Downloads additionally use :func:`copy_limited`, which enforces a byte cap
while the body is streamed to disk.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

# Weight files can be large, but a response larger than this is not a model.
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_REDIRECTS = 5

_BLOCKED_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".home.arpa",
)
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class UnsafeUrl(ValueError):
    """Raised when a URL fails the policy before any bytes are sent."""


def _ip_problem(value: str) -> str | None:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return None
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return f"blocked address {ip}"
    return None


def _host_problem(
    host: str, *, allow_private: bool, resolve: bool = False
) -> str | None:
    host = str(host or "").strip().rstrip(".").lower()
    if not host:
        return "missing host"
    if "%" in host:
        return "host contains a zone identifier"
    if host == "localhost" or host.endswith(_BLOCKED_HOST_SUFFIXES):
        return "blocked local host"
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return "invalid international host name"
    if len(ascii_host) > 253:
        return "host name is too long"
    labels = ascii_host.split(".")
    if any(not _HOST_LABEL.match(label) for label in labels):
        return "invalid host name"
    if allow_private:
        return None
    try:
        ip = ipaddress.ip_address(ascii_host)
    except ValueError:
        ip = None
    if ip is not None:
        return _ip_problem(str(ip))
    if not resolve:
        return None
    # Resolve before connecting so a name cannot point at the loopback or a
    # private range. The connection resolves again, so this is not a complete
    # defence against DNS rebinding; it does stop the ordinary SSRF payloads.
    try:
        infos = socket.getaddrinfo(ascii_host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return "host does not resolve"
    for info in infos:
        address = info[4][0]
        problem = _ip_problem(address)
        if problem:
            return problem
    return None


def _host_allowed(host: str, allowed_hosts: tuple[str, ...] | list[str] | None) -> bool:
    if not allowed_hosts:
        return True
    host = host.lower().rstrip(".")
    for base in allowed_hosts:
        base = str(base or "").lower().strip().rstrip(".")
        if base and (host == base or host.endswith("." + base)):
            return True
    return False


def host_allowed(host: str, allowed_hosts: tuple[str, ...] | list[str] | None) -> bool:
    """True when *host* equals an allowlisted host or is its subdomain."""
    return _host_allowed(host, allowed_hosts)


def url_problem(
    url: str,
    *,
    allowed_hosts: tuple[str, ...] | list[str] | None = None,
    allow_http: bool = False,
    allow_private: bool = False,
    resolve: bool = False,
) -> str | None:
    """Return a human-readable reason to reject *url*, or ``None``.

    ``resolve=False`` performs only syntax checks, which is what the editor's
    keystroke validation needs. Network callers pass ``resolve=True``.
    """
    text = str(url or "").strip()
    if not text:
        return "empty URL"
    if any(ord(ch) < 0x20 or ch == "\\" for ch in text):
        return "URL contains control characters or backslashes"
    try:
        parts = urllib.parse.urlsplit(text)
    except ValueError as exc:
        return f"invalid URL: {exc}"
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return f"unsupported scheme '{scheme or 'none'}'"
    if scheme == "http" and not allow_http:
        return "plain HTTP is not allowed; use HTTPS"
    if parts.username is not None or parts.password is not None:
        return "URL must not contain credentials"
    try:
        port = parts.port
    except ValueError:
        return "invalid port"
    if port is not None and not (1 <= port <= 65535):
        return "invalid port"
    host = parts.hostname
    if not host:
        return "missing host"
    allowed = _host_allowed(host, allowed_hosts)
    if not allowed:
        return f"host '{host}' is not allowlisted"
    problem = _host_problem(host, allow_private=allow_private, resolve=resolve)
    if problem:
        return problem
    return None


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def __init__(
        self,
        *,
        allowed_hosts: tuple[str, ...] | list[str] | None,
        allow_http: bool,
        allow_private: bool,
    ):
        self.allowed_hosts = allowed_hosts
        self.allow_http = allow_http
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urljoin(req.full_url, newurl)
        problem = url_problem(
            target,
            allowed_hosts=self.allowed_hosts,
            allow_http=self.allow_http,
            allow_private=self.allow_private,
            resolve=True,
        )
        if problem:
            raise urllib.error.HTTPError(
                target,
                code,
                f"blocked redirect: {problem}",
                headers,
                fp,
            )
        redirected = super().redirect_request(req, fp, code, msg, headers, target)
        if redirected is not None:
            old_host = (urllib.parse.urlsplit(req.full_url).hostname or "").lower()
            new_host = (urllib.parse.urlsplit(target).hostname or "").lower()
            if new_host != old_host:
                redirected.remove_header("Authorization")
                redirected.remove_header("Cookie")
        return redirected


def open_url(
    url: str,
    *,
    method: str = "GET",
    headers: dict | None = None,
    data: bytes | None = None,
    timeout: float = 30,
    allowed_hosts: tuple[str, ...] | list[str] | None = None,
    allow_http: bool = False,
    allow_private: bool = False,
):
    """Open *url* after checking it and every redirect target.

    The returned response is the stdlib ``HTTPResponse``; callers own it and
    should use it as a context manager.
    """
    problem = url_problem(
        url,
        allowed_hosts=allowed_hosts,
        allow_http=allow_http,
        allow_private=allow_private,
        resolve=True,
    )
    if problem:
        raise UnsafeUrl(problem)
    request = urllib.request.Request(
        url, data=data, method=method.upper(), headers=headers or {}
    )
    opener = urllib.request.build_opener(
        _GuardedRedirect(
            allowed_hosts=allowed_hosts,
            allow_http=allow_http,
            allow_private=allow_private,
        )
    )
    return opener.open(request, timeout=timeout)


def probe_url(
    url: str,
    *,
    headers: dict | None = None,
    timeout: float = 15,
) -> tuple[int | None, str | None]:
    """Check reachability with a guarded HEAD (GET Range fallback).

    Returns ``(status_code, error)``. The URL policy is applied first, so a
    rejected URL returns ``(None, reason)`` without any connection.
    """
    try:
        try:
            with open_url(url, method="HEAD", headers=headers, timeout=timeout) as resp:
                return int(resp.status), None
        except urllib.error.HTTPError as exc:
            if "blocked redirect" in str(exc.reason):
                return None, str(exc.reason)
            if exc.code not in (403, 405, 501):
                return int(exc.code), None
        hdrs = dict(headers or {})
        hdrs["Range"] = "bytes=0-0"
        try:
            with open_url(
                url, method="GET", headers=hdrs, timeout=timeout
            ) as resp:
                return int(resp.status), None
        except urllib.error.HTTPError as exc:
            if "blocked redirect" in str(exc.reason):
                return None, str(exc.reason)
            return int(exc.code), None
    except UnsafeUrl as exc:
        return None, str(exc)
    except urllib.error.URLError as exc:
        return None, f"Connection error: {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - surfaced to the validation table
        return None, f"Error: {exc}"


def content_problem(content_type: str) -> str | None:
    """Reject HTML/JSON bodies when a weight file was expected."""
    value = str(content_type or "").split(";", 1)[0].strip().lower()
    if value in ("text/html", "application/xhtml+xml", "application/json"):
        return f"server returned {value}, not a file"
    return None


def copy_limited(
    source,
    destination,
    *,
    max_bytes: int,
    start: int = 0,
    progress=None,
    label: str = "",
) -> int:
    """Stream *source* into *destination* without exceeding *max_bytes*."""
    limit = int(max_bytes or MAX_DOWNLOAD_BYTES)
    written = int(start or 0)
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        written += len(chunk)
        if written > limit:
            raise ValueError(f"download exceeds the {limit} byte limit")
        destination.write(chunk)
        if progress:
            progress(written, limit, label)
    return written
