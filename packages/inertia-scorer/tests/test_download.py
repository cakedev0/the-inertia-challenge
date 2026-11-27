from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import threading

import numpy as np
import pytest

from inertia_scorer import challenge, download
from inertia_scorer.challenge import Dataset


@pytest.fixture
def server(monkeypatch):
    """A server of `server.content`, with ranges. It cuts the first
    `server.drops` responses in the middle: `server.chunked` says whether they
    have a chunked transfer encoding, as the ones of archive.ics.uci.edu, or a
    Content-Length."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            content, status = server.content, 200
            if "Range" in self.headers:
                first, last = self.headers["Range"].removeprefix("bytes=").split("-")
                content = content[int(first) : int(last) + 1 if last else None]
                status = 206
            with server.lock:
                server.requests += 1
                # Not the header of DEEP1B: its download is not retried.
                drop = server.drops > 0 and len(content) > 8
                server.drops -= drop
            self.send_response(status)
            self.send_header("Connection", "close")
            if drop and server.chunked:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                half = content[: len(content) // 2]
                self.wfile.write(b"%x\r\n%b\r\n" % (len(half), half))
            else:
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content[: len(content) // 2] if drop else content)

        def log_message(self, *arguments):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.lock = threading.Lock()
    server.content, server.drops, server.chunked, server.requests = b"", 0, False, 0
    server.url = f"http://127.0.0.1:{server.server_port}/file"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(download, "BLOCK", 1000)
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    yield server
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("chunked", [True, False])
def test_fetch_resumes_a_dropped_download(server, tmp_path, chunked):
    server.content = np.random.default_rng(0).bytes(50_000)
    server.drops, server.chunked = 2, chunked

    path = download.fetch(server.url, tmp_path / "file")

    assert path.read_bytes() == server.content
    assert server.requests == 3
    assert not (tmp_path / "file.part").exists()


def test_fetch_gives_up(server, tmp_path):
    server.content = b"0123456789" * 1000
    server.drops, server.chunked = 100, True

    with pytest.raises(download.CONNECTION_ERRORS):
        download.fetch(server.url, tmp_path / "file", retries=2)

    assert server.requests == 3
    assert not (tmp_path / "file").exists()


@pytest.mark.parametrize("pwrite", [True, False])
def test_download_deep(server, tmp_path, monkeypatch, pwrite):
    if not pwrite:
        # As on Windows.
        monkeypatch.delattr(os, "pwrite", raising=False)
    elif not hasattr(os, "pwrite"):
        pytest.skip("no os.pwrite on this system")
    X = np.random.default_rng(0).normal(size=(3000, 5)).astype("<f4")
    server.content = np.array(X.shape, dtype="<i4").tobytes() + X.tobytes()
    monkeypatch.setattr(download, "DEEP_URL", server.url)
    # 15 segments for the first dataset, with 4 connections.
    monkeypatch.setattr(download, "DEEP_SEGMENT", 2000)
    small = Dataset("deep_small", 1500, 5, (4,), "", stem="deep")
    large = Dataset("deep_large", 3000, 5, (4,), "", stem="deep")
    for dataset in (small, large):
        monkeypatch.setitem(challenge.DATASETS, dataset.name, dataset)
    server.drops = 3

    download.prepare(small, tmp_path, jobs=4)

    assert challenge.is_present(small, tmp_path)
    assert not challenge.is_present(large, tmp_path)
    np.testing.assert_array_equal(
        challenge.load_dataset("deep_small", tmp_path), X[:1500]
    )

    # The second dataset only downloads the rows that are missing.
    requests = server.requests
    download.prepare(large, tmp_path, jobs=4)

    assert server.requests == requests + 1 + 15
    np.testing.assert_array_equal(challenge.load_dataset("deep_large", tmp_path), X)
    recorded = challenge.read_meta(large, tmp_path)["sha256"]
    assert recorded["deep_large"] == download.checksum(large, tmp_path)[3000]
