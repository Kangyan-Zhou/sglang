"""Connect-time address policy and client-safe errors for remote media URLs."""

import asyncio
import http.server
import os
import socket
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import requests

from sglang.srt.multimodal import media_io
from sglang.srt.multimodal.media_io import (
    MediaLoadError,
    MediaURLNotAllowedError,
    configure_media_url_security,
    describe_media_item,
    download_remote_media,
    get_mm_http_session,
    is_public_media_address,
    redact_media_url,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.llava import LlavaImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_REAL_GETADDRINFO = socket.getaddrinfo

_DENIED = (
    # ipaddress.is_global reports these as global on some Python versions.
    "224.0.0.1",
    "ff0e::1",
    "fec0::1",
    "64:ff9b::a00:1",
    "192.88.99.1",
    # IPv4
    "0.0.0.0",
    "0.1.2.3",
    "10.1.2.3",
    "100.64.0.1",
    "100.127.255.254",
    "127.0.0.1",
    "127.255.255.254",
    "169.254.169.254",
    "172.16.0.1",
    "172.31.255.255",
    "192.0.0.8",
    "192.0.2.1",
    "192.168.1.1",
    "198.18.0.1",
    "198.51.100.1",
    "203.0.113.1",
    "239.255.255.250",
    "240.0.0.1",
    "255.255.255.255",
    # IPv6
    "::",
    "::1",
    "fc00::1",
    "fd12:3456::1",
    "fe80::1",
    "fe80::1%lo0",
    "ff02::1",
    "100::1",
    "2001:db8::1",
    "3fff::1",
    "2001:0:4136:e378:8000:63bf:3fff:fdd2",  # Teredo
    "64:ff9b:1::a00:1",  # local-use NAT64
    "5f00::1",
    # IPv6 forms embedding a denied IPv4 address
    "::ffff:127.0.0.1",
    "::ffff:10.0.0.1",
    "::ffff:169.254.169.254",
    "::127.0.0.1",  # IPv4-compatible
    "::a00:1",
    "64:ff9b::7f00:1",  # NAT64 well-known prefix -> 127.0.0.1
    "64:ff9b::a9fe:a9fe",  # -> 169.254.169.254
    "2002:7f00:1::1",  # 6to4 -> 127.0.0.1
    "2002:a00:1::",  # 6to4 -> 10.0.0.1
    "2002:c0a8:101::1",  # 6to4 -> 192.168.1.1
    # not an address
    "localhost",
    "",
)

_ALLOWED = (
    "1.1.1.1",
    "8.8.8.8",
    "93.184.216.34",
    "100.128.0.1",  # just above 100.64.0.0/10
    "172.32.0.1",  # just above 172.16.0.0/12
    "2606:4700:4700::1111",
    "2a00:1450:4001::1",
    "::ffff:8.8.8.8",
    "64:ff9b::808:808",  # NAT64 -> 8.8.8.8
    "2002:808:808::1",  # 6to4 -> 8.8.8.8
)


class TestAddressPolicy(unittest.TestCase):
    def test_denied_addresses(self):
        for address in _DENIED:
            with self.subTest(address=address):
                self.assertFalse(is_public_media_address(address))

    def test_public_addresses(self):
        for address in _ALLOWED:
            with self.subTest(address=address):
                self.assertTrue(is_public_media_address(address))


class _MediaHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.requests.append((self.path, self.headers.get("Host")))
        if self.path.startswith("/media"):
            payload = b"remote-media"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/redirect-internal-name":
            self.send_response(302)
            self.send_header(
                "Location", f"http://internal.test:{self.server.server_port}/admin"
            )
            self.end_headers()
        elif self.path == "/redirect-metadata":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data")
            self.end_headers()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *_):
        pass


class _Listener:
    """A bare TCP listener that records whether anything connected."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]

    def connection_count(self) -> int:
        count = 0
        while True:
            try:
                conn, _ = self.sock.accept()
            except BlockingIOError:
                return count
            conn.close()
            count += 1

    def close(self):
        self.sock.close()


def _fake_resolver(mapping, calls=None):
    """getaddrinfo that answers the test hostnames in ``mapping`` and defers
    everything else (including IP literals) to the real resolver."""

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if calls is not None:
            calls.append(host)
        if host in mapping:
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
                for address in mapping[host]
            ]
        return _REAL_GETADDRINFO(host, port, family, type, proto, flags)

    return getaddrinfo


class _ServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MediaHandler)
        cls.server.requests = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.server.requests.clear()
        configure_media_url_security([], max_file_size_mb=64)
        self.listener = _Listener()

    def tearDown(self):
        self.listener.close()
        configure_media_url_security([], max_file_size_mb=64)

    def _treat_loopback_as_public(self):
        """Stand-in for a public host: the policy accepts only the fixture
        server's address, so a 'public' fetch can be served locally."""
        return patch.object(
            media_io, "is_public_media_address", lambda address: address == "127.0.0.1"
        )


class TestConnectTimeCheck(_ServerTestCase):
    def test_hostname_resolving_to_loopback_is_never_connected(self):
        resolver = _fake_resolver({"rebind.test": ["127.0.0.1"]})
        with patch("socket.getaddrinfo", resolver):
            for scheme in ("http", "https"):
                with self.subTest(scheme=scheme):
                    with self.assertRaises(MediaLoadError) as ctx:
                        download_remote_media(
                            f"{scheme}://rebind.test:{self.listener.port}/start_profile",
                            timeout=5,
                        )
                    self.assertEqual(str(ctx.exception), "Failed to fetch media URL")
        self.assertEqual(self.listener.connection_count(), 0)

    def test_session_itself_refuses_without_the_url_precheck(self):
        # The URL pre-check never resolves names; the session must refuse on
        # its own, so a name that rebinds between checks cannot get through.
        resolver = _fake_resolver({"rebind.test": ["127.0.0.1"]})
        with patch("socket.getaddrinfo", resolver):
            for scheme in ("http", "https"):
                with self.subTest(scheme=scheme):
                    with self.assertRaises(MediaLoadError):
                        get_mm_http_session().get(
                            f"{scheme}://rebind.test:{self.listener.port}/flush_cache",
                            timeout=5,
                        )
        self.assertEqual(self.listener.connection_count(), 0)

    def test_mixed_resolution_connects_only_to_the_allowed_address(self):
        calls = []
        resolver = _fake_resolver(
            {"media.test": ["10.255.255.1", "127.0.0.1"]}, calls=calls
        )
        with patch("socket.getaddrinfo", resolver), self._treat_loopback_as_public():
            body = download_remote_media(
                f"http://media.test:{self.port}/media", timeout=5
            )
        self.assertEqual(body, b"remote-media")
        # Resolved once; the socket connects to the validated sockaddr instead
        # of resolving the name again.
        self.assertEqual(calls, ["media.test"])

    def test_public_target_keeps_the_original_host_header(self):
        resolver = _fake_resolver({"media.test": ["127.0.0.1"]})
        with patch("socket.getaddrinfo", resolver), self._treat_loopback_as_public():
            body = download_remote_media(
                f"http://media.test:{self.port}/media?x=1", timeout=5
            )
        self.assertEqual(body, b"remote-media")
        self.assertEqual(
            self.server.requests, [("/media?x=1", f"media.test:{self.port}")]
        )

    def test_redirect_to_internal_name_is_refused(self):
        resolver = _fake_resolver(
            {"media.test": ["127.0.0.1"], "internal.test": ["10.255.255.1"]}
        )
        with patch("socket.getaddrinfo", resolver), self._treat_loopback_as_public():
            with self.assertRaises(MediaLoadError) as ctx:
                download_remote_media(
                    f"http://media.test:{self.port}/redirect-internal-name", timeout=5
                )
        self.assertEqual(str(ctx.exception), "Failed to fetch media URL")
        self.assertEqual(
            [path for path, _ in self.server.requests], ["/redirect-internal-name"]
        )

    def test_redirect_to_literal_private_ip_is_refused_before_connecting(self):
        resolver = _fake_resolver({"media.test": ["127.0.0.1"]})
        with patch("socket.getaddrinfo", resolver), self._treat_loopback_as_public():
            with self.assertRaises(MediaURLNotAllowedError):
                download_remote_media(
                    f"http://media.test:{self.port}/redirect-metadata", timeout=5
                )
        self.assertEqual(
            [path for path, _ in self.server.requests], ["/redirect-metadata"]
        )

    def test_literal_private_ip_is_refused_before_connecting(self):
        for url in (
            f"http://127.0.0.1:{self.listener.port}/start_profile",
            f"http://[::ffff:127.0.0.1]:{self.listener.port}/start_profile",
            "http://169.254.169.254/latest/meta-data",
        ):
            with self.subTest(url=url):
                with self.assertRaises(MediaURLNotAllowedError):
                    download_remote_media(url, timeout=5)
        self.assertEqual(self.listener.connection_count(), 0)

    def test_environment_proxy_is_not_used(self):
        proxy = f"http://127.0.0.1:{self.listener.port}"
        env = {"HTTP_PROXY": proxy, "http_proxy": proxy, "ALL_PROXY": proxy}
        resolver = _fake_resolver({"media.test": ["127.0.0.1"]})
        with (
            patch.dict(os.environ, env),
            patch("socket.getaddrinfo", resolver),
            self._treat_loopback_as_public(),
        ):
            configure_media_url_security([], max_file_size_mb=64)  # fresh session
            body = download_remote_media(
                f"http://media.test:{self.port}/media", timeout=5
            )
        self.assertEqual(body, b"remote-media")
        self.assertEqual(self.listener.connection_count(), 0)

    def test_explicit_proxy_is_refused(self):
        with self.assertRaises(MediaLoadError):
            get_mm_http_session().get(
                "http://example.com/",
                proxies={"http": f"http://127.0.0.1:{self.listener.port}"},
                timeout=5,
            )
        self.assertEqual(self.listener.connection_count(), 0)

    def test_policy_change_rebuilds_the_session(self):
        before = get_mm_http_session()
        configure_media_url_security([], max_file_size_mb=64)
        self.assertIsNot(get_mm_http_session(), before)


class TestAllowPrivateNetworks(_ServerTestCase):
    def test_loopback_is_refused_by_default(self):
        with self.assertRaises(MediaLoadError):
            download_remote_media(f"http://localhost:{self.port}/media", timeout=5)
        self.assertEqual(self.server.requests, [])

    def test_opt_out_allows_internal_hosts(self):
        configure_media_url_security(
            [], max_file_size_mb=64, allow_private_networks=True
        )
        for host in ("localhost", "127.0.0.1"):
            with self.subTest(host=host):
                self.assertEqual(
                    download_remote_media(
                        f"http://{host}:{self.port}/media", timeout=5
                    ),
                    b"remote-media",
                )

    def test_opt_out_keeps_the_domain_allowlist(self):
        configure_media_url_security(
            ["media.example.org"], max_file_size_mb=64, allow_private_networks=True
        )
        with self.assertRaisesRegex(MediaURLNotAllowedError, "not allowed"):
            download_remote_media(f"http://localhost:{self.port}/media", timeout=5)
        self.assertEqual(self.server.requests, [])


_SECRET_URL_PARTS = ("s3cr3t-pass", "user-name", "SIG-TOKEN")


class _StubProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class TestClientSafeErrors(_ServerTestCase):
    def _assert_safe(self, text):
        for leaked in (
            *_SECRET_URL_PARTS,
            "refused",
            "Errno",
            "timed out",
            "404",
            "Max retries",
            "HTTPConnectionPool",
        ):
            self.assertNotIn(leaked, text)

    def _url(self, path, host="media.test", port=None):
        return (
            f"http://user-name:s3cr3t-pass@{host}:{port or self.port}{path}"
            "?X-Amz-Signature=SIG-TOKEN"
        )

    def _failure_cases(self):
        # Connection refused: an allowed address with nothing listening.
        closed = _Listener()
        closed_port = closed.port
        closed.close()
        return {
            "refused": self._url("/media", port=closed_port),
            "http-404": self._url("/missing"),
            "blocked": self._url("/media", host="rebind.test"),
        }

    def test_fetch_errors_carry_no_connection_detail_or_url_secrets(self):
        resolver = _fake_resolver(
            {"media.test": ["127.0.0.1"], "rebind.test": ["10.255.255.1"]}
        )
        with patch("socket.getaddrinfo", resolver), self._treat_loopback_as_public():
            for case, url in self._failure_cases().items():
                with self.subTest(case=case):
                    with self.assertLogs(media_io.logger, "INFO") as logs:
                        with self.assertRaises(MediaLoadError) as ctx:
                            download_remote_media(url, timeout=5)
                    self.assertEqual(str(ctx.exception), "Failed to fetch media URL")
                    self._assert_safe(str(ctx.exception))
                    log_text = "\n".join(logs.output)
                    for leaked in _SECRET_URL_PARTS:
                        self.assertNotIn(leaked, log_text)
                    self.assertIn("?<redacted>", log_text)

    def test_timeout_error_is_generic(self):
        resolver = _fake_resolver({"media.test": ["127.0.0.1"]})
        with patch("socket.getaddrinfo", resolver), self._treat_loopback_as_public():
            with patch.object(
                requests.Session,
                "get",
                side_effect=requests.exceptions.ReadTimeout(
                    "HTTPConnectionPool(host='media.test', port=1): Read timed out."
                ),
            ):
                with self.assertRaises(MediaLoadError) as ctx:
                    download_remote_media(self._url("/media"), timeout=5)
        self.assertEqual(str(ctx.exception), "Failed to fetch media URL")
        self.assertTrue(ctx.exception.transient)

    def test_processor_error_is_client_safe(self):
        resolver = _fake_resolver(
            {"media.test": ["127.0.0.1"], "rebind.test": ["10.255.255.1"]}
        )
        with patch("socket.getaddrinfo", resolver), self._treat_loopback_as_public():
            for case, url in self._failure_cases().items():
                for modality in (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO):
                    with self.subTest(case=case, modality=modality.name):
                        with self.assertRaises(MediaLoadError) as ctx:
                            _StubProcessor._load_single_item(url, modality)
                        message = str(ctx.exception)
                        self._assert_safe(message)
                        self.assertIn("Failed to fetch media URL", message)
                        self.assertIn("?<redacted>", message)

    def test_raw_request_exception_text_is_replaced(self):
        leaky = requests.exceptions.ConnectionError(
            "HTTPConnectionPool(host='10.0.0.5', port=6379): Max retries exceeded "
            "with url: /?X-Amz-Signature=SIG-TOKEN ([Errno 111] Connection refused)"
        )
        with patch.object(media_io, "get_image_bytes", side_effect=leaky):
            with self.assertRaises(MediaLoadError) as ctx:
                _StubProcessor._load_single_item(self._url("/x"), Modality.IMAGE)
        self._assert_safe(str(ctx.exception))

    def test_redact_media_url(self):
        self.assertEqual(
            redact_media_url("https://user:pw@host.example:8443/a/b.png?sig=1#frag"),
            "https://host.example:8443/a/b.png?<redacted>",
        )
        self.assertEqual(redact_media_url("http://[::1]/x"), "http://[::1]/x")
        self.assertEqual(
            redact_media_url("data:image/png;base64," + "A" * 1000),
            "data:image/png;base64,...",
        )
        self.assertLessEqual(len(redact_media_url("x" * 1000)), 103)
        self.assertEqual(describe_media_item(b"abc"), "<3 bytes>")


class TestLlavaRetriesOnlyTransientFailures(unittest.TestCase):
    def _fetch(self, error):
        processor = object.__new__(LlavaImageProcessor)
        attempts = []

        def fail(url):
            attempts.append(url)
            raise error

        with ThreadPoolExecutor(max_workers=1) as processor.io_executor:
            with (
                patch("sglang.srt.multimodal.processors.llava.get_image_bytes", fail),
                patch.dict(os.environ, {"SGLANG_MM_LOAD_MAX_RETRIES": "2"}),
                patch("asyncio.sleep", return_value=None),
            ):
                with self.assertRaises(MediaLoadError):
                    asyncio.run(processor._fetch_remote_image_bytes("https://h/x"))
        return len(attempts)

    def test_transient_failure_is_retried(self):
        self.assertEqual(self._fetch(MediaLoadError("x", transient=True)), 3)

    def test_refusal_is_not_retried(self):
        self.assertEqual(self._fetch(MediaURLNotAllowedError("x")), 1)
        self.assertEqual(self._fetch(MediaLoadError("x")), 1)


if __name__ == "__main__":
    unittest.main()
