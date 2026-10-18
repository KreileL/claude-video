"""Tests for netguard: SSRF protection for the MCP server (no external network)."""
import http.server
import socket
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import netguard  # noqa: E402
from netguard import BlockedURL, EgressProxy, resolve_checked, validate_url  # noqa: E402


def fake_resolver(mapping):
    """getaddrinfo stand-in: host -> list of IP strings; unknown host -> gaierror."""
    def resolver(host, port, *args, **kwargs):
        if host not in mapping:
            raise socket.gaierror(-2, "Name or service not known")
        out = []
        for ip in mapping[host]:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sockaddr = (ip, port, 0, 0) if fam == socket.AF_INET6 else (ip, port)
            out.append((fam, socket.SOCK_STREAM, 6, "", sockaddr))
        return out
    return resolver


PUBLIC = "157.240.0.1"
NONE = frozenset()


# ---------- validate_url ----------

def test_validate_url_accepts_public_https():
    url = "https://www.instagram.com/reel/abc/"
    r = fake_resolver({"www.instagram.com": [PUBLIC]})
    assert validate_url(url, allow_hosts=NONE, resolver=r) == url


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://x", "javascript:alert(1)", "data:text/plain,hi",
    "gopher://example.com/", "C:\\video.mp4", "/etc/passwd", "relative.mp4",
    "./a.mp4", "http://", "http://user:pw@example.com/", "http://a@example.com/",
    "http://example.com /x", "http://example.com/\nx", "http://example.com:99999/",
    "http://example.com:abc/", "", "x" * 3000,
])
def test_validate_url_rejects_malformed_or_bad_scheme(url):
    r = fake_resolver({"example.com": [PUBLIC]})
    with pytest.raises(BlockedURL):
        validate_url(url, allow_hosts=NONE, resolver=r)


def test_validate_url_rejects_non_str():
    with pytest.raises(BlockedURL):
        validate_url(None, allow_hosts=NONE)  # type: ignore[arg-type]


@pytest.mark.parametrize("url", [
    "https://localhost/", "http://LOCALHOST./", "http://foo.localhost/",
    "http://127.0.0.1/", "http://[::1]/", "http://10.0.0.1/", "http://172.17.0.2/",
    "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data",
    "http://0.0.0.0/", "http://[::ffff:127.0.0.1]/", "http://100.64.0.1/",
    "http://[fe80::1%25eth0]/", "http://[2002:7f00:1::]/",
])
def test_validate_url_rejects_private_targets(url):
    with pytest.raises(BlockedURL):
        validate_url(url, allow_hosts=NONE)  # real resolver: only literals, no network


@pytest.mark.parametrize("host", ["2130706433", "0x7f000001", "127.1", "0177.0.0.1"])
def test_numeric_oddities_blocked_with_fake_resolver(host):
    r = fake_resolver({host: ["127.0.0.1"]})
    with pytest.raises(BlockedURL):
        validate_url(f"http://{host}/", allow_hosts=NONE, resolver=r)


@pytest.mark.parametrize("host", ["127.1", "0177.0.0.1"])  # others trigger slow real DNS on Windows
def test_numeric_oddities_blocked_with_real_resolver(host):
    # Whether or not the platform resolves these, they must never pass.
    with pytest.raises(BlockedURL):
        validate_url(f"http://{host}/", allow_hosts=NONE)


def test_all_or_nothing_resolution():
    r = fake_resolver({"mixed.example": [PUBLIC, "10.0.0.5"]})
    with pytest.raises(BlockedURL, match="10.0.0.5"):
        validate_url("http://mixed.example/", allow_hosts=NONE, resolver=r)


def test_unresolvable_host():
    with pytest.raises(BlockedURL, match="resolve"):
        validate_url("http://nope.example/", allow_hosts=NONE, resolver=fake_resolver({}))


def test_is_public_ip():
    assert netguard.is_public_ip("8.8.8.8")
    assert netguard.is_public_ip("2606:4700:4700::1111")
    for bad in ["127.0.0.1", "10.1.1.1", "224.0.0.1", "0.0.0.0", "::", "::1",
                "fe80::1%eth0", "::ffff:10.0.0.1", "garbage"]:
        assert not netguard.is_public_ip(bad), bad


# ---------- test allow hosts ----------

def test_allow_hosts_empty_by_default(monkeypatch):
    monkeypatch.delenv(netguard.TEST_ALLOW_ENV, raising=False)
    assert netguard.test_allow_hosts() == frozenset()


def test_allow_hosts_parsed(monkeypatch):
    monkeypatch.setenv(netguard.TEST_ALLOW_ENV, " Watch-MCP-TestSrv ,,")
    assert netguard.test_allow_hosts() == frozenset({"watch-mcp-testsrv"})


def test_allowed_host_skips_private_check_only_for_exact_host(monkeypatch):
    monkeypatch.setenv(netguard.TEST_ALLOW_ENV, "srv.test")
    r = fake_resolver({"srv.test": ["10.0.0.9"], "other.test": ["10.0.0.9"],
                       "sub.srv.test": ["10.0.0.9"]})
    assert validate_url("http://srv.test/x", resolver=r) == "http://srv.test/x"
    with pytest.raises(BlockedURL):
        validate_url("http://other.test/x", resolver=r)
    with pytest.raises(BlockedURL):
        validate_url("http://sub.srv.test/x", resolver=r)


# ---------- EgressProxy ----------

class _Origin:
    """Local HTTP server; records hits, optionally redirects."""

    def __init__(self, redirect_to=None):
        self.hits = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.hits.append(self.path)
                if outer.redirect_to:
                    self.send_response(302)
                    self.send_header("Location", outer.redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = b"hello"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.redirect_to = redirect_to
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def origin():
    o = _Origin()
    yield o
    o.close()


def _opener(proxy):
    return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy.url}))


def _get(proxy, url):
    with _opener(proxy).open(url, timeout=5) as resp:
        return resp.read().decode()


RES = fake_resolver({"origin.test": ["127.0.0.1"], "a.test": ["127.0.0.1"],
                     "internal.test": ["10.0.0.7"]})


def test_proxy_binds_loopback_only():
    with EgressProxy(allow_hosts=NONE, resolver=RES) as p:
        assert p.url.startswith("http://127.0.0.1:")
        assert p._server.server_address[0] == "127.0.0.1"
        assert p.url.endswith(str(p._server.server_address[1]))


def test_proxy_blocks_private_by_default(origin):
    with EgressProxy(allow_hosts=NONE, resolver=RES) as p:
        with pytest.raises(urllib.error.HTTPError) as ei:
            _get(p, f"http://origin.test:{origin.port}/")
        assert ei.value.code == 403
    assert origin.hits == []


def test_proxy_allows_allowlisted_host(origin):
    with EgressProxy(allow_hosts={"origin.test"}, resolver=RES) as p:
        assert _get(p, f"http://origin.test:{origin.port}/x?q=1") == "hello"
    assert origin.hits == ["/x?q=1"]


def _raw(proxy, payload, read_timeout=5):
    s = socket.create_connection(("127.0.0.1", int(proxy.url.rsplit(":", 1)[1])), timeout=read_timeout)
    s.sendall(payload)
    return s


def _read_all(s):
    data = b""
    while True:
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            break
        if not chunk:
            break
        data += chunk
    return data


def test_connect_blocked_by_default(origin):
    with EgressProxy(allow_hosts=NONE, resolver=RES) as p:
        s = _raw(p, f"CONNECT 127.0.0.1:{origin.port} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        resp = _read_all(s)
        s.close()
    assert resp.startswith(b"HTTP/1.1 403")
    assert origin.hits == []


def test_connect_tunnel_allowed(origin):
    with EgressProxy(allow_hosts={"origin.test"}, resolver=RES) as p:
        s = _raw(p, f"CONNECT origin.test:{origin.port} HTTP/1.1\r\n\r\n".encode())
        head = b""
        while b"\r\n\r\n" not in head:
            head += s.recv(1024)
        assert head.startswith(b"HTTP/1.1 200 Connection Established")
        s.sendall(b"GET /t HTTP/1.1\r\nHost: origin.test\r\nConnection: close\r\n\r\n")
        body = _read_all(s)
        s.close()
    assert b"200 OK" in body and body.endswith(b"hello")
    assert origin.hits == ["/t"]


def test_redirect_to_loopback_is_blocked():
    b = _Origin()
    a = _Origin(redirect_to=f"http://127.0.0.1:{b.port}/secret")
    try:
        with EgressProxy(allow_hosts={"a.test"}, resolver=RES) as p:
            with pytest.raises(urllib.error.HTTPError) as ei:
                _get(p, f"http://a.test:{a.port}/")
            assert ei.value.code == 403
        assert a.hits == ["/"]
        assert b.hits == []
    finally:
        a.close()
        b.close()


def test_redirect_to_private_hostname_is_blocked():
    b = _Origin()
    a = _Origin(redirect_to=f"http://internal.test:{b.port}/")
    try:
        with EgressProxy(allow_hosts={"a.test"}, resolver=RES) as p:
            with pytest.raises(urllib.error.HTTPError) as ei:
                _get(p, f"http://a.test:{a.port}/")
            assert ei.value.code == 403
        assert b.hits == []
    finally:
        a.close()
        b.close()


def test_malformed_and_https_absolute_form_get_400():
    with EgressProxy(allow_hosts=NONE, resolver=RES) as p:
        s = _raw(p, b"GET https://example.com/ HTTP/1.1\r\n\r\n")
        assert _read_all(s).startswith(b"HTTP/1.1 400")
        s.close()
        s = _raw(p, b"garbage\r\n\r\n")
        assert _read_all(s).startswith(b"HTTP/1.1 400")
        s.close()


def test_blocked_request_logs_one_line_without_secrets(origin, capsys):
    with EgressProxy(allow_hosts=NONE, resolver=RES) as p:
        s = _raw(p, (f"GET http://127.0.0.1:{origin.port}/ HTTP/1.1\r\n"
                     "Cookie: secret=XYZ\r\n\r\n").encode())
        _read_all(s)
        s.close()
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert len(lines) == 1 and "blocked" in lines[0]
    assert "XYZ" not in err


def test_allowed_request_not_logged(origin, capsys):
    with EgressProxy(allow_hosts={"origin.test"}, resolver=RES) as p:
        _get(p, f"http://origin.test:{origin.port}/")
    assert capsys.readouterr().err == ""


def test_connect_order_prefers_one_ipv6_then_one_ipv4():
    results = fake_resolver({"h": ["1.1.1.1", "2.2.2.2", "2001:db8::1", "2001:db8::2"]})("h", 443)
    order = [r[4][0] for r in netguard._connect_order(results)]
    assert order == ["2001:db8::1", "1.1.1.1"]


def test_proxy_falls_back_to_next_vetted_address(origin):
    # ::1 refuses (origin listens on 127.0.0.1 only), so the IPv4 address must be tried next.
    res = fake_resolver({"multi.test": ["::1", "127.0.0.1"]})
    with EgressProxy(allow_hosts={"multi.test"}, resolver=res) as p:
        assert _get(p, f"http://multi.test:{origin.port}/") == "hello"


class _NoLength(http.server.BaseHTTPRequestHandler):
    """Streams 8 KiB with no Content-Length (what defeats yt-dlp's --max-filesize)."""
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        try:
            self.wfile.write(b"Z" * 8192)
        except OSError:
            pass

    def log_message(self, *a):
        pass


def test_proxy_byte_budget_is_hard_and_aggregate():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NoLength)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    res = fake_resolver({"big.test": ["127.0.0.1"]})
    try:
        with EgressProxy(allow_hosts={"big.test"}, resolver=res, max_bytes=1024) as p:
            got = b""
            try:
                with _opener(p).open(f"http://big.test:{srv.server_address[1]}/", timeout=5) as resp:
                    got = resp.read()
            except Exception:
                pass
            assert len(got) < 8192
            assert p.exceeded
            # Budget is per proxy, so a second connection is refused outright.
            with pytest.raises(urllib.error.HTTPError) as ei:
                _get(p, f"http://big.test:{srv.server_address[1]}/again")
            assert ei.value.code == 403
    finally:
        srv.shutdown()
        srv.server_close()


def test_proxy_without_budget_passes_everything():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NoLength)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    res = fake_resolver({"big.test": ["127.0.0.1"]})
    try:
        with EgressProxy(allow_hosts={"big.test"}, resolver=res) as p:
            with _opener(p).open(f"http://big.test:{srv.server_address[1]}/", timeout=5) as resp:
                assert len(resp.read()) == 8192
            assert not p.exceeded
    finally:
        srv.shutdown()
        srv.server_close()
