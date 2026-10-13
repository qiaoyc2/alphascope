"""Small bounded HTTP reader for fixed news and SEC endpoints."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

import requests


@dataclass(frozen=True)
class HttpFailure(Exception):
    code: str
    message: str
    retryable: bool = False
    request_count: int = 0

    def __str__(self) -> str:
        return self.message


def fetch_bytes(
    session: requests.Session,
    url: str,
    *,
    allowed_hosts: frozenset[str],
    headers: dict[str, str],
    max_bytes: int,
    timeout_seconds: float = 10.0,
    retries: int = 1,
    before_request: Callable[[], None] | None = None,
) -> tuple[bytes, int]:
    """Never follow redirects or read beyond the configured response budget."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts or parsed.port is not None or parsed.username:
        raise HttpFailure("UNSAFE_URL", "Provider URL is not an approved HTTPS endpoint")
    if max_bytes <= 0 or timeout_seconds <= 0 or retries not in (0, 1):
        raise ValueError("Invalid HTTP bounds")
    count = 0
    for attempt in range(retries + 1):
        if before_request is not None:
            before_request()
        count += 1
        try:
            with session.get(
                url,
                headers=headers,
                timeout=timeout_seconds,
                stream=True,
                allow_redirects=False,
            ) as response:
                status = response.status_code
                if status == 429:
                    failure = HttpFailure("RATE_LIMITED", "Provider rate limited the request", True, count)
                elif status == 403:
                    raise HttpFailure("ACCESS_DENIED", "Provider denied access", False, count)
                elif 300 <= status < 400:
                    raise HttpFailure("UNEXPECTED_REDIRECT", "Provider returned a redirect", False, count)
                elif status >= 500:
                    failure = HttpFailure("PROVIDER_UNAVAILABLE", f"Provider HTTP {status}", True, count)
                elif status >= 400:
                    raise HttpFailure("HTTP_ERROR", f"Provider HTTP {status}", False, count)
                else:
                    length = response.headers.get("Content-Length")
                    if length and length.isdecimal() and int(length) > max_bytes:
                        raise HttpFailure("RESPONSE_TOO_LARGE", "Provider response exceeds size limit", False, count)
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_content(chunk_size=65536):
                        size += len(chunk)
                        if size > max_bytes:
                            raise HttpFailure("RESPONSE_TOO_LARGE", "Provider response exceeds size limit", False, count)
                        chunks.append(chunk)
                    return b"".join(chunks), count
        except requests.RequestException:
            failure = HttpFailure("PROVIDER_UNAVAILABLE", "Provider request timed out or failed", True, count)
        if attempt == retries:
            raise failure
        time.sleep(min(2.0, 0.5 * 2**attempt))
    raise AssertionError("unreachable")
