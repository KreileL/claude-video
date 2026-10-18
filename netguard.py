"""SSRF protection for an MCP server that downloads user-supplied video URLs.

Two layers:
  * validate_url()  - cheap up-front check of the URL the user typed.
  * EgressProxy     - local forward proxy that yt-dlp is pointed at, so every
                      connection it makes (redirects, CDN hosts) is re-checked.
                      It connects to the vetted IP, never re-resolving, which
                      defeats DNS rebinding.

Pure stdlib.
"""
import ipaddress
import os
import selectors
import socket
import socketserver
import sys
import threading
from urllib.parse import urlsplit

TEST_ALLOW_ENV = "WATCH_MCP_TEST_ALLOW_HOSTS"
MAX_URL_LEN = 2048
MAX_HEAD = 64 * 1024
CONNECT_TIMEOUT = 10.0
_DROP_HEADERS = {"proxy-connection", "proxy-authorization", "connection", "keep-alive"}


class BlockedURL(ValueError):
    """The URL/host is not allowed. The message is safe to show to the end user."""


def test_allow_hosts() -> frozenset[str]:
    """Hostnames (from $WATCH_MCP_TEST_ALLOW_HOSTS, comma-separated) that skip the
    private-address check. Exact hostname match only. Test-mode escape hatch: this
    variable MUST stay unset in production."""
    raw = os.environ.get(TEST_ALLOW_ENV, "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


test_allow_hosts.__test__ = False  # not a pytest test when imported into a test module


def _ip_ok(ip) -> bool:
    return ip.is_global and not (
        ip.is_multicast or ip.is_unspecified or ip.is_loopback
        or ip.is_link_local or ip.is_private or ip.is_reserved
    )


def is_public_ip(ip: str) -> bool:
    """True only for globally routable unicast addresses."""
    try:
        addr = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        return False
    if not _ip_ok(addr):
        return False
    if addr.version == 6:
        embedded = []
        if addr.ipv4_mapped:
            embedded.append(addr.ipv4_mapped)
        if addr.sixtofour:
            embedded.append(addr.sixtofour)
        if addr.teredo:
            embedded.extend(addr.teredo)
        return all(_ip_ok(v4) for v4 in embedded)
    return True


def resolve_checked(host: str, port: int, *, allow_hosts=None, resolver=socket.getaddrinfo) -> list[tuple]:
    """Resolve host for TCP and return getaddrinfo-style results, refusing the
    whole name if any address is not public."""
    if allow_hosts is None:
        allow_hosts = test_allow_hosts()
    name = host.lower().rstrip(".")
    allowed = name in allow_hosts
    if not allowed and (name == "localhost" or name.endswith(".localhost")):
        raise BlockedURL("blocked: localhost/private address (localhost)")
    try:
        results = resolver(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError):
        raise BlockedURL(f"cannot resolve host ({host[:100]})") from None
    if not results:
        raise BlockedURL(f"cannot resolve host ({host[:100]})")
    if not allowed:
        for *_, sockaddr in results:
            if not is_public_ip(sockaddr[0]):
                raise BlockedURL(f"blocked: localhost/private address ({sockaddr[0]})")
    return results


def validate_url(url: str, *, allow_hosts=None, resolver=socket.getaddrinfo) -> str:
    """Return url unchanged if it is a plain http(s) URL to a public host."""
    if not isinstance(url, str):
        raise BlockedURL("blocked: URL must be text")
    if len(url) > MAX_URL_LEN:
        raise BlockedURL("blocked: URL too long")
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        raise BlockedURL("blocked: URL has spaces or control characters")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise BlockedURL("blocked: malformed URL or port") from None
    if parts.scheme.lower() not in ("http", "https"):
        raise BlockedURL("blocked: only http and https URLs are allowed")
    if not parts.hostname:
        raise BlockedURL("blocked: URL has no host")
    if "@" in parts.netloc:
        raise BlockedURL("blocked: credentials in URL are not allowed")
    default = 443 if parts.scheme.lower() == "https" else 80
    resolve_checked(parts.hostname, port or default, allow_hosts=allow_hosts, resolver=resolver)
    return url


# ---------------------------------------------------------------- proxy

class _BadRequest(Exception):
    pass


class _Upstream(Exception):
    pass


def _reply(sock, code: int, reason: str, body: str = "") -> None:
    data = body.encode() + b"\n" if body else b""
    head = (f"HTTP/1.1 {code} {reason}\r\nContent-Type: text/plain\r\n"
            f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n").encode()
    try:
        sock.sendall(head + data)
    except OSError:
        pass


def _pipe(a: socket.socket, b: socket.socket, idle: float) -> None:
    """Copy bytes both ways until either side closes or idle seconds pass."""
    sel = selectors.DefaultSelector()
    sel.register(a, selectors.EVENT_READ, b)
    sel.register(b, selectors.EVENT_READ, a)
    try:
        while True:
            events = sel.select(idle)
            if not events:
                return
            for key, _ in events:
                try:
                    data = key.fileobj.recv(65536)
                    if not data:
                        return
                    key.data.sendall(data)
                except OSError:
                    return
    finally:
        sel.close()


def _split_hostport(target: str) -> tuple[str, int]:
    host, sep, port = target.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise _BadRequest("bad CONNECT target")
    return host.strip("[]"), int(port)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        client = self.request
        client.settimeout(self.server.egress.idle_timeout)
        method, hostport = "?", "?"
        try:
            head, rest = self._read_head(client)
            method, hostport, plan = self._parse(head)
            host, port, first_bytes = plan
            results = resolve_checked(host, port, allow_hosts=self.server.egress.allow_hosts,
                                      resolver=self.server.egress.resolver)
            upstream = self._connect(results)
        except _BadRequest as e:
            _reply(client, 400, "Bad Request", str(e))
            return
        except BlockedURL as e:
            print(f"[netguard] blocked {method} {hostport[:100]} ({e})", file=sys.stderr, flush=True)
            _reply(client, 403, "Forbidden", str(e))
            return
        except _Upstream as e:
            _reply(client, 502, "Bad Gateway", str(e))
            return
        except OSError:
            return
        with upstream:
            try:
                if first_bytes is None:  # CONNECT
                    client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                else:
                    upstream.sendall(first_bytes)
                if rest:
                    upstream.sendall(rest)
                _pipe(client, upstream, self.server.egress.idle_timeout)
            except OSError:
                pass

    @staticmethod
    def _read_head(sock) -> tuple[bytes, bytes]:
        buf = b""
        while b"\r\n\r\n" not in buf:
            if len(buf) > MAX_HEAD:
                raise _BadRequest("request head too large")
            chunk = sock.recv(8192)
            if not chunk:
                raise _BadRequest("incomplete request")
            buf += chunk
        if len(buf.split(b"\r\n\r\n", 1)[0]) > MAX_HEAD:
            raise _BadRequest("request head too large")
        head, _, rest = buf.partition(b"\r\n\r\n")
        return head, rest

    @staticmethod
    def _parse(head: bytes):
        """Return (method, 'host:port', (host, port, request_bytes_or_None))."""
        lines = head.split(b"\r\n")
        try:
            method, target, version = lines[0].decode("ascii").split(" ")
        except (UnicodeDecodeError, ValueError):
            raise _BadRequest("malformed request line") from None
        if not version.startswith("HTTP/"):
            raise _BadRequest("malformed request line")
        if method.upper() == "CONNECT":
            host, port = _split_hostport(target)
            return "CONNECT", f"{host}:{port}", (host, port, None)
        try:
            parts = urlsplit(target)
            port = parts.port or 80
        except ValueError:
            raise _BadRequest("malformed target") from None
        if parts.scheme.lower() != "http" or not parts.hostname:
            raise _BadRequest("proxy accepts absolute http:// URLs or CONNECT only")
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        out = [f"{method} {path} {version}".encode("ascii", "replace")]
        for line in lines[1:]:
            if line.split(b":", 1)[0].strip().lower().decode("latin-1") not in _DROP_HEADERS:
                out.append(line)
        out.append(b"Connection: close")
        return method, f"{parts.hostname}:{port}", (parts.hostname, port, b"\r\n".join(out) + b"\r\n\r\n")

    @staticmethod
    def _connect(results) -> socket.socket:
        # Connect to the vetted address itself; never resolve again.
        fam, typ, proto, _, sockaddr = results[0]
        s = socket.socket(fam, typ, proto)
        try:
            s.settimeout(CONNECT_TIMEOUT)
            s.connect(sockaddr)
            s.settimeout(None)
        except OSError:
            s.close()
            raise _Upstream("upstream connection failed") from None
        return s


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class EgressProxy:
    """Loopback-only HTTP forward proxy that refuses non-public destinations."""

    def __init__(self, *, allow_hosts=None, resolver=socket.getaddrinfo, idle_timeout: float = 60.0):
        self.allow_hosts = allow_hosts
        self.resolver = resolver
        self.idle_timeout = idle_timeout
        self._server = None
        self._thread = None

    def start(self) -> int:
        self._server = _Server(("127.0.0.1", 0), _Handler)
        self._server.egress = self
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(5)
            self._server = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
