# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Loading, decoding, and encoding image, video, and audio inputs."""

from __future__ import annotations

import ipaddress
import logging
import math
import os
import socket
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, Optional, Sequence, Union
from urllib.parse import unquote, urljoin, urlparse, urlsplit

import numpy as np
import pybase64
import requests
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from typing_extensions import Literal
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError
from urllib3.exceptions import HTTPError as Urllib3HTTPError
from urllib3.exceptions import NewConnectionError
from urllib3.util.connection import allowed_gai_family

from sglang.srt.utils.device import is_cuda
from sglang.srt.utils.video_decoder import _BACKEND, VideoDecoderWrapper

logger = logging.getLogger(__name__)


_mm_http_session = threading.local()


_DEFAULT_MEDIA_URL_MAX_FILE_SIZE_MB = 64


_MAX_MEDIA_URL_REDIRECTS = 5


_MEDIA_URL_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


_allowed_media_domains: frozenset[str] = frozenset()


_media_url_max_file_size_bytes = _DEFAULT_MEDIA_URL_MAX_FILE_SIZE_MB * 1024 * 1024


_media_url_allow_private_networks = False


# Bumped by configure_media_url_security so per-thread sessions (and their
# pooled connections, each validated under the policy in force when it opened)
# are rebuilt after a policy change.
_media_url_policy_generation = 0


class MediaLoadError(ValueError):
    """A client-supplied media item could not be loaded.

    ``str()`` is returned to the client, so it never carries the underlying
    exception text, a URL query or userinfo: whether a connection was refused,
    timed out or returned an HTTP error would let a client probe ports and
    services through the server. ``log_detail`` holds the server-side reason.
    """

    def __init__(
        self,
        message: str = "Failed to load media",
        *,
        log_detail: Optional[str] = None,
        transient: bool = False,
    ):
        super().__init__(message)
        self.log_detail = log_detail or message
        # A connect or read failure that a retry may cure; never a refusal.
        self.transient = transient


class MediaURLNotAllowedError(MediaLoadError):
    """The media URL is refused by policy, judged from the URL text alone
    (scheme, domain allowlist, literal IP). A refusal that depends on what a
    hostname resolves to is reported as a generic fetch failure instead, so the
    error does not reveal which internal names exist."""


class _MediaAddressBlockedError(MediaLoadError):
    """Raised at connect time when no resolved address is public. Deriving
    from ValueError (not OSError) keeps urllib3 from retrying or rewrapping it."""


_MEDIA_FETCH_FAILED = "Failed to fetch media URL"


# Explicit deny policy for the address a media download connects to. Do not
# replace it with ipaddress.is_global/is_private: their classification has
# changed across Python versions and reports several unroutable or internal
# ranges (multicast, fec0::/10, 64:ff9b::/96, 192.88.99.0/24) as global.
_DENIED_IPV4_NETWORKS = tuple(
    ipaddress.IPv4Network(network)
    for network in (
        "0.0.0.0/8",  # "this network", includes the unspecified address
        "10.0.0.0/8",  # private
        "100.64.0.0/10",  # carrier-grade NAT
        "127.0.0.0/8",  # loopback
        "169.254.0.0/16",  # link-local, includes cloud metadata services
        "172.16.0.0/12",  # private
        "192.0.0.0/24",  # IETF protocol assignments
        "192.0.2.0/24",  # documentation
        "192.88.99.0/24",  # deprecated 6to4 relay anycast
        "192.168.0.0/16",  # private
        "198.18.0.0/15",  # benchmarking
        "198.51.100.0/24",  # documentation
        "203.0.113.0/24",  # documentation
        "224.0.0.0/4",  # multicast
        "240.0.0.0/4",  # reserved, includes the 255.255.255.255 broadcast
    )
)


# IPv6 is allowlisted to global unicast; everything outside 2000::/3 (loopback,
# unspecified, ULA fc00::/7, link-local fe80::/10, site-local fec0::/10,
# multicast ff00::/8, discard 100::/64, local-use NAT64 64:ff9b:1::/48, ...)
# is denied without being listed.
_IPV6_GLOBAL_UNICAST = ipaddress.IPv6Network("2000::/3")
_DENIED_IPV6_NETWORKS = tuple(
    ipaddress.IPv6Network(network)
    for network in (
        "2001::/23",  # IETF protocol assignments, includes Teredo 2001::/32
        "2001:db8::/32",  # documentation
        "3fff::/20",  # documentation
    )
)
_IPV4_COMPATIBLE = ipaddress.IPv6Network("::/96")
_NAT64_WELL_KNOWN = ipaddress.IPv6Network("64:ff9b::/96")


def _embedded_ipv4(address: ipaddress.IPv6Address) -> Optional[ipaddress.IPv4Address]:
    """The IPv4 address an IPv6 address reaches, for the forms that embed one.

    Covers IPv4-mapped (::ffff:0:0/96), deprecated IPv4-compatible (::/96),
    well-known-prefix NAT64 (64:ff9b::/96) and 6to4 (2002::/16). A NAT64
    gateway on a network-specific prefix (RFC 6052) cannot be recognized
    generically: such a deployment must block internal targets at the network
    layer.
    """
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address in _IPV4_COMPATIBLE or address in _NAT64_WELL_KNOWN:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return address.sixtofour


def is_public_media_address(address: str) -> bool:
    """Whether a media download may connect to ``address`` (an IP literal)."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(ip)
        if embedded is None:
            return ip in _IPV6_GLOBAL_UNICAST and not any(
                ip in network for network in _DENIED_IPV6_NETWORKS
            )
        ip = embedded
    return not any(ip in network for network in _DENIED_IPV4_NETWORKS)


def redact_media_url(url: Any, max_len: int = 100) -> str:
    """Render a client media reference for logs and error text.

    Drops URL userinfo, query and fragment (signed URLs carry credentials
    there) and truncates the rest.
    """
    if not isinstance(url, str):
        return f"<{type(url).__name__}>"
    if url.startswith("data:"):
        return url.split(",", 1)[0][:max_len] + ",..."
    if url.startswith(("http://", "https://")):
        try:
            parsed = urlsplit(url)
            host = parsed.hostname or ""
            port = parsed.port
        except ValueError:
            return "<invalid media URL>"
        if ":" in host:
            host = f"[{host}]"
        rendered = f"{parsed.scheme}://{host}"
        if port is not None:
            rendered += f":{port}"
        rendered += parsed.path
        if parsed.query or parsed.fragment:
            rendered += "?<redacted>"
    else:
        rendered = url  # a local path or inline payload
    if len(rendered) > max_len:
        rendered = rendered[:max_len] + "..."
    return rendered


def describe_media_item(data: Any) -> str:
    """A short, credential-free description of one media input."""
    url = getattr(data, "url", None)
    if isinstance(url, str):
        data = url
    if isinstance(data, (bytes, bytearray, memoryview)):
        return f"<{len(data)} bytes>"
    return redact_media_url(data)


def _normalize_media_domain(domain: str) -> str:
    if not isinstance(domain, str):
        raise ValueError("allowed media domains must be strings")

    domain = domain.strip().rstrip(".")
    if not domain:
        raise ValueError("allowed media domains cannot be empty")
    if "://" in domain or any(char in domain for char in "/?#@"):
        raise ValueError(
            f"Invalid allowed media domain {domain!r}: provide a hostname only"
        )

    # Brackets are URL syntax, not part of an IPv6 hostname.
    if domain.startswith("[") and domain.endswith("]"):
        domain = domain[1:-1]
    try:
        return str(ipaddress.ip_address(domain))
    except ValueError:
        if ":" in domain:
            raise ValueError(
                f"Invalid allowed media domain {domain!r}: ports are not supported"
            )

    try:
        normalized = domain.encode("idna").decode("ascii").lower()
    except UnicodeError as e:
        raise ValueError(f"Invalid allowed media domain {domain!r}") from e
    if not normalized:
        raise ValueError("allowed media domains cannot be empty")
    return normalized


def configure_media_url_security(
    allowed_media_domains: Optional[Sequence[str]] = None,
    max_file_size_mb: int = _DEFAULT_MEDIA_URL_MAX_FILE_SIZE_MB,
    allow_private_networks: bool = False,
) -> list[str]:
    """Configure process-wide safeguards for client-supplied media URLs.

    A serving worker hosts one engine configuration, while media loading fans
    out to worker threads. Keeping the immutable policy here makes the same
    checks apply to image, video, audio, cache, and model-specific loaders.
    """

    if max_file_size_mb < 0:
        raise ValueError("media_url_max_file_size_mb must be non-negative")

    normalized_domains = sorted(
        {_normalize_media_domain(domain) for domain in allowed_media_domains or []}
    )
    global _allowed_media_domains, _media_url_max_file_size_bytes
    global _media_url_allow_private_networks, _media_url_policy_generation
    _allowed_media_domains = frozenset(normalized_domains)
    _media_url_max_file_size_bytes = max_file_size_mb * 1024 * 1024
    _media_url_allow_private_networks = bool(allow_private_networks)
    _media_url_policy_generation += 1
    return normalized_domains


def _assert_media_url_allowed(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise MediaURLNotAllowedError(
            "Invalid media URL", log_detail=f"invalid URL {redact_media_url(url)}"
        )

    try:
        hostname = _normalize_media_domain(parsed.hostname)
    except ValueError as e:
        raise MediaURLNotAllowedError(
            "Invalid media URL",
            log_detail=f"invalid hostname in {redact_media_url(url)}",
        ) from e
    if _allowed_media_domains and hostname not in _allowed_media_domains:
        raise MediaURLNotAllowedError(
            "Media URL domain is not allowed. "
            f"Allowed domains: {sorted(_allowed_media_domains)}; "
            f"input domain: {hostname}"
        )
    # A literal IP is refused here, before any connection, with a specific
    # message: the client typed the address, so naming it leaks nothing.
    # Hostnames are checked at connect time (_GuardedConnectionMixin).
    if not _media_url_allow_private_networks:
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            return
        if not is_public_media_address(hostname):
            raise MediaURLNotAllowedError(
                "Media URL is not allowed: the address is not public"
            )


def _connect_public_address(conn: HTTPConnection) -> socket.socket:
    """Resolve ``conn``'s host once, keep only addresses the media policy
    allows, and connect to one of those exact addresses.

    Resolving here and connecting to the validated sockaddr (instead of
    checking a name and letting the socket layer resolve it again) is what
    defeats DNS rebinding. TLS SNI, certificate verification and the Host
    header still use ``conn.host``.
    """
    host = getattr(conn, "_dns_host", conn.host).strip("[]")
    try:
        infos = socket.getaddrinfo(
            host, conn.port, allowed_gai_family(), socket.SOCK_STREAM
        )
    except socket.gaierror as e:
        raise NewConnectionError(conn, f"Failed to resolve media host: {e}") from e

    allow_all = _media_url_allow_private_networks
    candidates = []
    blocked = []
    for family, socktype, proto, _, sockaddr in infos:
        if allow_all or is_public_media_address(sockaddr[0]):
            candidates.append((family, socktype, proto, sockaddr))
        else:
            blocked.append(sockaddr[0])
    if not candidates:
        raise _MediaAddressBlockedError(
            _MEDIA_FETCH_FAILED,
            log_detail=(
                f"host {host!r} resolved only to non-public addresses "
                f"{sorted(set(blocked))}; set --media-url-allow-private-networks "
                "to fetch media from internal hosts"
            ),
        )

    last_error: Optional[OSError] = None
    for family, socktype, proto, sockaddr in candidates:
        sock = socket.socket(family, socktype, proto)
        try:
            for option in conn.socket_options or ():
                sock.setsockopt(*option)
            if isinstance(conn.timeout, (int, float)):
                sock.settimeout(conn.timeout)
            if conn.source_address:
                sock.bind(conn.source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as e:
            sock.close()
            last_error = e
    if isinstance(last_error, socket.timeout):
        raise ConnectTimeoutError(
            conn, f"Connection to {conn.host} timed out"
        ) from last_error
    raise NewConnectionError(
        conn, f"Failed to establish a new connection: {last_error}"
    ) from last_error


class _GuardedConnectionMixin:
    def _new_conn(self) -> socket.socket:
        return _connect_public_address(self)


class _GuardedHTTPConnection(_GuardedConnectionMixin, HTTPConnection):
    pass


class _GuardedHTTPSConnection(_GuardedConnectionMixin, HTTPSConnection):
    pass


class _GuardedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _GuardedHTTPConnection


class _GuardedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _GuardedHTTPSConnection


class _MediaHTTPAdapter(HTTPAdapter):
    """Opens every connection, pooled or new, HTTP or HTTPS, through the
    connect-time address check."""

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _GuardedHTTPConnectionPool,
            "https": _GuardedHTTPSConnectionPool,
        }

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        # Through a proxy the check would vet the proxy, not the target.
        raise _MediaAddressBlockedError(
            _MEDIA_FETCH_FAILED, log_detail="proxies are not used for media URLs"
        )


def _describe_fetch_failure(error: BaseException) -> str:
    """A server-side reason for a failed download. Built from the exception
    type and status, never ``str(error)``: requests and urllib3 messages embed
    the request path and query."""
    if isinstance(error, MediaLoadError):
        return error.log_detail
    if isinstance(error, requests.exceptions.HTTPError):
        status = getattr(error.response, "status_code", None)
        return f"HTTP status {status}"
    if isinstance(error, requests.exceptions.SSLError):
        return "TLS error"
    if isinstance(error, requests.exceptions.Timeout):
        return "timed out"
    if isinstance(error, requests.exceptions.TooManyRedirects):
        return "too many redirects"
    cause: Optional[BaseException] = error
    seen = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, OSError) and cause.strerror:
            return f"{type(error).__name__} ({cause.strerror})"
        cause = cause.__cause__ or cause.__context__
    return type(error).__name__


def download_remote_media(url: str, timeout: float) -> bytes:
    """Download one HTTP(S) media object under the configured URL policy.

    Redirects are followed manually so every destination is validated before
    a connection is made, and every connection goes through the connect-time
    address check of ``get_mm_http_session``. The response is streamed to
    enforce both the total request deadline and the configured byte limit
    without first buffering an attacker-controlled body in memory.

    Every failure is raised as ``MediaLoadError`` with a client-safe message;
    the reason is logged here with the URL redacted.
    """

    if timeout <= 0:
        raise ValueError("media URL timeout must be positive")

    try:
        return _download_remote_media(url, timeout)
    except MediaLoadError as e:
        _log_media_fetch_failure(url, e)
        raise
    except (requests.exceptions.RequestException, Urllib3HTTPError, OSError) as e:
        error = MediaLoadError(
            _MEDIA_FETCH_FAILED,
            log_detail=_describe_fetch_failure(e),
            transient=isinstance(
                e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
            )
            and not isinstance(e, requests.exceptions.SSLError),
        )
        _log_media_fetch_failure(url, error)
        raise error from e


def _log_media_fetch_failure(url: str, error: MediaLoadError) -> None:
    log = (
        logger.warning if isinstance(error, _MediaAddressBlockedError) else logger.info
    )
    log(
        "Media URL fetch refused or failed: url=%s reason=%s",
        redact_media_url(url),
        error.log_detail,
    )


def _download_remote_media(url: str, timeout: float) -> bytes:
    session = get_mm_http_session()
    deadline = time.monotonic() + timeout
    current_url = url

    for redirect_count in range(_MAX_MEDIA_URL_REDIRECTS + 1):
        # Validate the same normalized URL representation that requests sends
        # to urllib3. This avoids parser disagreements around backslashes and
        # userinfo separators.
        try:
            prepared_url = requests.Request("GET", current_url).prepare().url
        except (requests.exceptions.RequestException, ValueError) as e:
            raise MediaURLNotAllowedError(
                "Invalid media URL",
                log_detail=f"invalid URL {redact_media_url(current_url)}",
            ) from e
        if prepared_url is None:
            raise MediaURLNotAllowedError("Invalid media URL")
        _assert_media_url_allowed(prepared_url)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise requests.exceptions.Timeout("media URL deadline exceeded")

        with session.get(
            prepared_url,
            allow_redirects=False,
            stream=True,
            timeout=remaining,
        ) as response:
            location = response.headers.get("Location")
            if response.status_code in _MEDIA_URL_REDIRECT_STATUS_CODES and location:
                if redirect_count == _MAX_MEDIA_URL_REDIRECTS:
                    raise requests.exceptions.TooManyRedirects(
                        f"more than {_MAX_MEDIA_URL_REDIRECTS} redirects"
                    )
                current_url = urljoin(response.url, location)
                continue

            response.raise_for_status()
            max_bytes = _media_url_max_file_size_bytes
            content_length = response.headers.get("Content-Length")
            if max_bytes and content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > max_bytes:
                    raise MediaLoadError(
                        f"Remote media exceeds the {max_bytes} byte download limit"
                    )

            content = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                if time.monotonic() > deadline:
                    raise requests.exceptions.Timeout("media URL deadline exceeded")
                if max_bytes and len(content) + len(chunk) > max_bytes:
                    raise MediaLoadError(
                        f"Remote media exceeds the {max_bytes} byte download limit"
                    )
                content.extend(chunk)
            return bytes(content)

    raise AssertionError("unreachable")


def _ca_bundle_from_env() -> Union[str, bool]:
    # trust_env=False also stops requests from reading these. They only choose
    # which CAs to trust (never where to connect), so keep honoring them.
    return (
        os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE") or True
    )


def get_mm_http_session() -> requests.Session:
    """Per-thread HTTP session for multimodal downloads, to pool/reuse TCP
    connections. Pid-checked so a forked worker rebuilds its own, not the parent's.

    Every connection the session opens goes through the connect-time address
    check. ``trust_env`` is off so HTTP(S)_PROXY cannot route a download
    around that check (and ``.netrc`` credentials are never sent to a
    client-chosen host); TLS verification stays on.
    """
    pid = os.getpid()
    session = getattr(_mm_http_session, "session", None)
    if (
        session is None
        or getattr(_mm_http_session, "pid", None) != pid
        or getattr(_mm_http_session, "generation", None) != _media_url_policy_generation
    ):
        if session is not None and getattr(_mm_http_session, "pid", None) == pid:
            session.close()  # drop connections opened under the old policy
        session = requests.Session()
        session.trust_env = False
        session.verify = _ca_bundle_from_env()
        adapter = _MediaHTTPAdapter()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _mm_http_session.session = session
        _mm_http_session.pid = pid
        _mm_http_session.generation = _media_url_policy_generation
    return session


def media_error_client_text(error: BaseException) -> str:
    """The text of a ``CLIENT_MEDIA_EXCEPTIONS`` error that may be returned to
    the client. A raw requests error (which names the URL and the connection
    outcome) is replaced; ``download_remote_media`` already raises only
    ``MediaLoadError``, so this guards callers that fetch some other way."""
    if isinstance(error, requests.exceptions.RequestException):
        return _MEDIA_FETCH_FAILED
    return str(error)


# Raised by the loaders below when client-supplied media cannot be fetched or
# decoded. ValueError is in the set because invalid base64 raises binascii.Error.
# Pillow raises DecompressionBombError from Image.open, before any decode, for an
# image above twice Image.MAX_IMAGE_PIXELS; it subclasses Exception, not OSError.
CLIENT_MEDIA_EXCEPTIONS = (
    ValueError,
    UnidentifiedImageError,
    Image.DecompressionBombError,
    requests.exceptions.RequestException,
)


def load_audio(
    audio_file: Union[str, bytes], sr: Optional[int] = None, mono: bool = True
) -> np.ndarray:
    if sr is None:
        sr = 16000

    # Normalize input: resolve URL / base64 / file:// to bytes or path
    if isinstance(audio_file, bytes):
        source = audio_file
    elif isinstance(audio_file, str) and audio_file.startswith("data:"):
        source = pybase64.b64decode(audio_file.split(",")[1], validate=True)
    elif isinstance(audio_file, str) and (
        audio_file.startswith("http://") or audio_file.startswith("https://")
    ):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "5"))
        source = download_remote_media(audio_file, timeout=timeout)
    elif isinstance(audio_file, str) and audio_file.startswith("file://"):
        source = unquote(urlparse(audio_file).path)
    elif isinstance(audio_file, str):
        source = audio_file
    else:
        raise ValueError(f"Invalid audio format: {audio_file}")

    from sglang.srt.multimodal.audio_from_video import (
        decode_audio_container,
        is_audio_container,
    )

    if isinstance(source, bytes):
        header = source[:16]
    else:
        with open(source, "rb") as audio_stream:
            header = audio_stream.read(16)

    if is_audio_container(header):
        return decode_audio_container(
            source,
            target_sr=sr,
            mono=mono,
        )

    if _BACKEND == "torchcodec":
        from torchcodec.decoders import AudioDecoder

        try:
            decoder = AudioDecoder(
                source,
                sample_rate=sr,
                num_channels=1 if mono else None,
            )
            samples = decoder.get_all_samples()
            if mono:
                return samples.data.squeeze(0).numpy()
            return samples.data.T.numpy()
        except Exception as e:
            # torchcodec's bytes-buffer IO can fail on WAV files that carry
            # large trailing metadata chunks. Fall back to soundfile, which reads the PCM payload directly.
            logger.warning(
                f"torchcodec AudioDecoder failed ({e}); falling back to soundfile + torchaudio."
            )

    # Fallback: soundfile + torchaudio (ARM / no FFmpeg / torchcodec failure)
    import soundfile as sf
    import torch
    import torchaudio

    try:
        if isinstance(source, bytes):
            audio, original_sr = sf.read(BytesIO(source))
        else:
            audio, original_sr = sf.read(source)
    except sf.LibsndfileError as e:
        raise ValueError(f"Could not decode audio: {e}") from e

    if mono and len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if original_sr != sr:
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        else:
            audio_tensor = audio_tensor.T
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=original_sr, new_freq=sr
        )
        if audio_tensor.shape[0] == 1:
            audio = audio_tensor.squeeze(0).numpy()
        else:
            audio = audio_tensor.T.numpy()

    return audio


@dataclass
class ImageData:
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"
    max_dynamic_patch: Optional[int] = None
    preprocess_kwargs: Optional[Dict] = None
    content_hash: Optional[str] = None
    max_long_side_pixel: Optional[int] = None


GLM_MEDIA_CONFIG_KEYS = (
    "fps",
    "max_frames",
    "max_tokens_per_frame",
    "max_image_tokens",
)


@dataclass
class VideoData:
    url: str
    preprocess_kwargs: Optional[Dict] = None
    max_long_side_pixel: Optional[int] = None


image_extension_names = (".png", ".jpg", ".jpeg", ".webp", ".gif")


GPUImageDecodeMode = Union[bool, Literal["nvjpeg_fancy"]]


def smart_to_rgb(
    image: Union[torch.Tensor, Image.Image],
) -> Union[torch.Tensor, Image.Image]:
    if not isinstance(image, Image.Image):
        return image

    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
        width, height = image.size
        edge_pixels = []

        for x in range(0, width, max(1, width // 20)):
            for y in (0, height - 1):
                pixel = image.getpixel((x, y))
                if pixel[3] > 128:
                    edge_pixels.append(pixel[:3])

        for y in range(0, height, max(1, height // 20)):
            for x in (0, width - 1):
                pixel = image.getpixel((x, y))
                if pixel[3] > 128:
                    edge_pixels.append(pixel[:3])

        if edge_pixels:
            avg_brightness = sum(sum(pixel) for pixel in edge_pixels) / (
                len(edge_pixels) * 3
            )
            background_color = (32, 32, 32) if avg_brightness > 128 else (240, 240, 240)
        else:
            background_color = (255, 255, 255)

        background = Image.new("RGB", image.size, background_color)
        background.paste(image, mask=image.getchannel("A"))
        return background

    return image.convert("RGB")


def is_jpeg_with_cuda(
    image_bytes: bytes = b"", gpu_image_decode: GPUImageDecodeMode = True
) -> bool:
    """
    Check three conditions:
    1. whether CUDA is available.
    2. whether input is recognized as JPEG.
    3. whether GPU image decode is enabled (some models such as CPM forcibly disable this).
    """
    if not is_cuda() or not gpu_image_decode:
        return False
    if image_bytes != b"":
        return image_bytes.startswith(b"\xff\xd8") and image_bytes.endswith(b"\xff\xd9")
    return False


@lru_cache(maxsize=16)
def _warn_fancy_jpeg_fallback(error: str) -> None:
    logger.warning(
        "High-fidelity GPU JPEG decode is unavailable; falling back to PIL. "
        "Install the Kimi-K3 serving image or NVIDIA nvImageCodec. Error: %s",
        error,
    )


def _load_image(
    image_bytes: bytes = b"",
    image_file: str = "",
    gpu_image_decode: GPUImageDecodeMode = True,
) -> Union[torch.Tensor, Image.Image]:
    """
    Try to decode JPEG with nvJPEG on GPU and return a torch device tensor,
    otherwise fallback to decode with PIL on CPU and return a PIL Image.
    Keep the fallback path since nvJPEG may fail on some JPEG images that are not strictly compliant with the standard, while PIL is more tolerant.
    """
    if image_file != "":
        image_bytes = get_image_bytes(image_file)
    if is_jpeg_with_cuda(image_bytes, gpu_image_decode):
        try:
            if gpu_image_decode == "nvjpeg_fancy":
                from sglang.srt.utils.nvjpeg_decoder import (
                    decode_jpeg_with_fancy_upsampling,
                )

                return decode_jpeg_with_fancy_upsampling(image_bytes)
            from torchvision.io import decode_jpeg  # lazy: ~1 s of torch._dynamo

            encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)
            image_tensor = decode_jpeg(encoded_image, device="cuda")
            return image_tensor
        except Exception as e:
            if gpu_image_decode == "nvjpeg_fancy":
                _warn_fancy_jpeg_fallback(f"{type(e).__name__}: {e}")
            else:
                logger.warning(
                    "Failed to decode JPEG on GPU, falling back to CPU. Error: %s",
                    e,
                )
    try:
        image = Image.open(BytesIO(image_bytes))
    except (OSError, SyntaxError) as e:
        raise ValueError(f"Could not decode image: {e}") from e
    return _fully_load_pil_image(image)


def _fully_load_pil_image(image: Image.Image) -> Image.Image:
    """Force PIL's lazy decode while malformed input is still request-local."""
    try:
        image.load()
    except (OSError, SyntaxError) as e:
        raise ValueError(f"Could not decode image: {e}") from e
    return image


def load_image(
    image_file: Union[Image.Image, str, ImageData, bytes],
    gpu_image_decode: GPUImageDecodeMode = True,
) -> tuple[Union[torch.Tensor, Image.Image], Optional[tuple[int, int]]]:
    """
    Load image from multiple input formats, including:
    ImageData, PIL Image, bytes, URL, file path, or base64 string.
    """
    if isinstance(image_file, ImageData):
        image_file = image_file.url

    image = None
    image_size: Optional[tuple[int, int]] = None
    if isinstance(image_file, Image.Image):
        image = _fully_load_pil_image(image_file)
        image_size = (image.width, image.height)
    elif isinstance(image_file, bytes):
        image = _load_image(image_bytes=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(image_file, str) and image_file.startswith(("http://", "https://")):
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(image_file, str) and image_file.startswith("file://"):
        image = _load_image(
            image_file=unquote(urlparse(image_file).path),
            gpu_image_decode=gpu_image_decode,
        )
    elif isinstance(image_file, str) and image_file.lower().endswith(
        image_extension_names
    ):
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(image_file, str) and image_file.startswith("data:"):
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    elif isinstance(
        image_file, str
    ):  # Other formats, try to decode as base64 by default
        image = _load_image(image_file=image_file, gpu_image_decode=gpu_image_decode)
    else:
        raise ValueError(f"Invalid image: {image_file}")
    if image_size is not None and isinstance(image, Image.Image):
        image_size = (image.width, image.height)
    return image, image_size


def get_image_bytes(image_file: Union[str, bytes]) -> bytes:
    """Normalize various image inputs into raw bytes."""
    if isinstance(image_file, bytes):
        return image_file
    if image_file.startswith(("http://", "https://")):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        return download_remote_media(image_file, timeout=timeout)
    if image_file.startswith(("file://", "/")):
        with open(image_file, "rb") as f:
            return f.read()
    if isinstance(image_file, str) and image_file.startswith("data:"):
        _, encoded = image_file.split(",", 1)
        return pybase64.b64decode(encoded, validate=True)
    if isinstance(image_file, str):
        return pybase64.b64decode(image_file, validate=True)
    raise NotImplementedError(f"Invalid image: {image_file}")


def _normalize_video_input(
    video_file: Union[str, bytes],
) -> Union[str, bytes, None]:
    """Normalize video input (URL, base64, file://, etc.) to a file path or bytes.

    Returns a file path or bytes suitable for a decoder, or None on failure.
    URLs and base64 are returned as bytes (no temp files needed since both
    torchcodec and VideoDecoderWrapper accept bytes natively).
    """
    if isinstance(video_file, bytes):
        return video_file
    elif isinstance(video_file, str):
        if video_file.startswith(("http://", "https://")):
            timeout = int(os.getenv("REQUEST_TIMEOUT", "10"))
            return download_remote_media(video_file, timeout=timeout)
        elif video_file.startswith("data:"):
            _, encoded = video_file.split(",", 1)
            return pybase64.b64decode(encoded, validate=True)
        elif video_file.startswith("file://"):
            return unquote(urlparse(video_file).path)
        elif os.path.isfile(unquote(urlparse(video_file).path)):
            return video_file
        else:
            return pybase64.b64decode(video_file, validate=True)
    else:
        return None


def get_video_bytes(video_file: Union[str, bytes, VideoData]) -> bytes:
    """Normalize a video input and return its encoded bytes."""
    if isinstance(video_file, VideoData):
        video_file = video_file.url

    source = _normalize_video_input(video_file)
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        with open(source, "rb") as f:
            return f.read()
    raise ValueError(f"Unsupported video input type: {type(video_file)}")


def load_video(video_file: Union[str, bytes, VideoData], use_gpu: bool = True):
    if isinstance(video_file, VideoData):
        # preprocess_kwargs is consumed by the multimodal processor, not here.
        video_file = video_file.url

    if isinstance(video_file, (list, tuple, torch.Tensor, np.ndarray)):
        return video_file

    source = _normalize_video_input(video_file)
    if source is None:
        raise ValueError(f"Unsupported video input type: {type(video_file)}")

    device = "cuda" if use_gpu else "cpu"
    try:
        return VideoDecoderWrapper(source, device=device)
    except (ImportError, MemoryError):
        raise  # missing backend / OOM is not a bad payload
    except Exception as e:
        # Broad on purpose: torchcodec raises RuntimeError, decord its own type.
        raise ValueError(f"Could not decode video: {e}") from e


def sample_video_frames(video, *, desired_fps: int, max_frames: int) -> list[int]:
    total_frames = len(video)
    assert total_frames > 0, "Video must have at least one frame"

    avg_fps = video.avg_fps
    duration = total_frames / avg_fps if avg_fps > 0 else 0
    fps = min(desired_fps, avg_fps)

    num_frames = math.floor(duration * fps)
    num_frames = min(max_frames, num_frames, total_frames)
    num_frames = max(1, num_frames)  # At least one frame
    if num_frames == total_frames:
        return list(range(total_frames))
    else:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()


def encode_video(video_path, frame_count_limit=None):
    if not os.path.exists(video_path):
        logger.error(f"Video {video_path} does not exist")
        return []

    if frame_count_limit == 0:
        return []

    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    decoder = VideoDecoderWrapper(video_path)
    avg_fps = decoder.avg_fps
    total_frames = len(decoder)

    sample_fps = round(avg_fps / 1)
    if sample_fps == 0:
        sample_fps = 1
    frame_indices = [i for i in range(0, total_frames, sample_fps)]
    if frame_count_limit is not None and len(frame_indices) > frame_count_limit:
        frame_indices = uniform_sample(frame_indices, frame_count_limit)

    if not frame_indices:
        return []

    frames_data = decoder.get_frames_at(frame_indices)
    frames = [Image.fromarray(v.astype("uint8")) for v in frames_data]

    return frames
