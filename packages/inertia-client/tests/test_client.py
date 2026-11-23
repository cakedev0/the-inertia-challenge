from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import stat
import threading
import urllib.parse

import numpy as np
import pytest

from inertia_client import Client, ServerError, check_submission, client, save_labels
from inertia_scorer import challenge, download, score
from inertia_scorer.challenge import Dataset, Reader

N_SAMPLES = 1000
PART_SIZE = 3000


class FakeServer(ThreadingHTTPServer):
    """The API of the server for one user, and the device flow of GitHub."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server_port}"
        self.submissions = {}
        self.tokens = {"token-of-ada"}
        self.failures = []  # statuses to answer to the next uploads of parts
        self.polls = 0  # of the device flow
        self.requests = []
        self.scores = []  # lines of the published scores of "toy"


class Handler(BaseHTTPRequestHandler):
    def answer(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        if status == 429:
            self.send_header("Retry-After", "3600")
        self.end_headers()
        self.wfile.write(data)

    def answer_text(self, text):
        data = text.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def view(self, submission):
        result = {key: submission[key] for key in ("id", "dataset", "status", "size")}
        result |= {"scores": submission["scores"], "error": None}
        if submission["status"] == "uploading":
            result |= {
                "part_size": PART_SIZE,
                "n_parts": -(-submission["size"] // PART_SIZE),
                "uploaded_parts": sorted(submission["parts"]),
            }
        return result

    def handle_request(self):
        server = self.server
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length)
        server.requests.append((self.command, self.path))

        # GitHub
        if self.path == "/login/device/code":
            assert urllib.parse.parse_qs(data.decode()) == {"client_id": ["client-id"]}
            return self.answer(
                200,
                {
                    "device_code": "device",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 0,
                    "expires_in": 900,
                },
            )
        if self.path == "/login/oauth/access_token":
            server.polls += 1
            if server.polls < 3:
                return self.answer(200, {"error": "authorization_pending"})
            return self.answer(200, {"access_token": "token-of-ada"})

        # The public scores, and who a token is
        if self.path == "/scores/toy.jsonl":
            return self.answer_text(
                "".join(json.dumps(s) + "\n" for s in server.scores)
            )
        if self.path == "/scores/mnist.jsonl":
            return self.answer(404, {})

        # The server
        if self.path == "/":
            return self.answer(200, {"github_client_id": "client-id"})
        token = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
        if token not in server.tokens:
            return self.answer(401, {"error": "This is not a valid GitHub token."})
        if self.path == "/user":
            return self.answer(200, {"login": "ada"})
        if self.path == "/submissions":
            body = json.loads(data)
            if any(
                s["dataset"] == body["dataset"] for s in server.submissions.values()
            ):
                return self.answer(429, {"error": "One submission per 24 hours."})
            name = f"id-{len(server.submissions)}"
            server.submissions[name] = body | {
                "id": name,
                "status": "uploading",
                "parts": {},
                "scores": [],
            }
            return self.answer(201, {"id": name})
        match = re.fullmatch(
            r"/submissions/([\w-]+)(?:/(parts/(\d+)|complete))?", self.path
        )
        submission = server.submissions.get(match.group(1)) if match else None
        if not submission:
            return self.answer(404, {"error": "You have no submission with this id."})
        if match.group(3):
            if server.failures:
                return self.answer(
                    server.failures.pop(0), {"error": "The server failed."}
                )
            submission["parts"][int(match.group(3))] = data
            return self.answer(200, {"part_number": int(match.group(3))})
        if match.group(2) == "complete":
            parts = submission["parts"]
            submission["content"] = b"".join(parts[part] for part in sorted(parts))
            assert len(submission["content"]) == submission["size"]
            submission["status"] = "queued"
        self.answer(200, self.view(submission))

    do_GET = do_POST = do_PUT = handle_request

    def log_message(self, *arguments):
        pass


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setitem(
        challenge.DATASETS, "toy", Dataset("toy", N_SAMPLES, 5, (4, 16), "")
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv(client.TOKEN_VARIABLE, raising=False)
    monkeypatch.delenv(client.SERVER_VARIABLE, raising=False)
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    server = FakeServer()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def labels(k, dtype=np.uint32):
    return np.random.default_rng(k).integers(0, k, N_SAMPLES).astype(dtype)


def test_submit(server, tmp_path):
    progress = []
    submission = Client(server.url, "token-of-ada").submit(
        "toy",
        {4: labels(4), 16: labels(16, np.int64)},
        time_s={4: 23, 16: 1.5},
        hardware="M2",
        tags={"algorithm": "Lloyd"},
        progress=lambda done, total: progress.append((done, total)),
    )

    assert submission["status"] == "queued"
    sent = server.submissions[submission["id"]]
    assert sent["time_s"] == {"4": 23.0, "16": 1.5}
    assert (sent["hardware"], sent["tags"]) == ("M2", {"algorithm": "Lloyd"})
    # 2 arrays of 4000 bytes and the headers, by parts of 3000 bytes.
    assert sorted(sent["parts"]) == [1, 2, 3]
    assert [done for done, total in progress] == [3000, 6000, sent["size"]]
    (tmp_path / "received.npz").write_bytes(sent["content"])
    assert check_submission("toy", tmp_path / "received.npz") == [4, 16]
    with np.load(tmp_path / "received.npz") as received:
        assert received["16"].dtype == np.uint32
        np.testing.assert_array_equal(received["16"], labels(16))

    # The limit of the server.
    with pytest.raises(ServerError, match="One submission per 24 hours") as error:
        Client(server.url, "token-of-ada").submit("toy", {4: labels(4)})
    assert (error.value.status, error.value.retry_after) == (429, 3600)


@pytest.mark.parametrize(
    "arrays, match",
    [
        ({"4": labels(4, np.int32)}, "For 4 clusters: The labels must be a uint32"),
        ({"4": labels(4)[:-1]}, "For 4 clusters: The labels must have shape"),
        ({"4": labels(16)}, "For 4 clusters: The labels must be in"),
        ({"5": labels(4)}, "are among 4, 16"),
        ({"4": labels(4), "centers": np.zeros(3)}, "are among 4, 16"),
        ({}, "empty"),
    ],
)
def test_invalid_file_is_not_sent(server, tmp_path, arrays, match):
    path = tmp_path / "labels.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=match):
        Client(server.url, "token-of-ada").submit("toy", path)
    path.write_bytes(b"not a zip file")
    with pytest.raises(ValueError, match="not a .npz file"):
        Client(server.url, "token-of-ada").submit("toy", path)
    with pytest.raises(ValueError, match="Unknown dataset"):
        Client(server.url, "token-of-ada").submit("mnist", path)
    assert not server.requests


@pytest.mark.parametrize(
    "time_s, match",
    [(23.21, "for each number of clusters, as {4: 23.21}"), ({16: 1.5}, "no labels")],
)
def test_invalid_times_are_not_sent(server, time_s, match):
    with pytest.raises(ValueError, match=match):
        Client(server.url, "token-of-ada").submit("toy", {4: labels(4)}, time_s=time_s)
    assert not server.requests


def test_save_labels(tmp_path):
    for invalid in (labels(4).astype(float), -labels(4).astype(np.int64) - 1):
        with pytest.raises(ValueError, match="not non-negative integers"):
            save_labels(tmp_path / "labels.npz", {4: invalid})


def test_upload_is_resumed_and_retried(server, tmp_path):
    path = tmp_path / "labels.npz"
    save_labels(path, {4: labels(4), 16: labels(16)})
    user = Client(server.url, "token-of-ada")
    # The server fails 5 times: the client gives up at the first part.
    server.failures = [500, 500, 500, 500, 500]
    with pytest.raises(ServerError, match="The server failed"):
        user.submit("toy", path)
    (submission,) = server.submissions.values()
    assert sorted(submission["parts"]) == [] and submission["status"] == "uploading"

    server.failures = [500, 502]
    server.requests.clear()
    assert user.upload(submission["id"], path)["status"] == "queued"
    assert submission["content"] == path.read_bytes()
    parts = [path for method, path in server.requests if method == "PUT"]
    assert [int(part.rsplit("/", 1)[1]) for part in parts] == [1, 1, 1, 2, 3]

    # A refusal is not tried again, and a completed submission is left alone.
    other = tmp_path / "other.npz"
    save_labels(other, {4: labels(4)})
    assert user.upload(submission["id"], other)["status"] == "queued"
    with pytest.raises(ServerError, match="no submission with this id") as error:
        user.upload("unknown", path)
    assert error.value.status == 404


def test_login(server, tmp_path, capsys):
    user = Client(server.url, github=server.url)
    with pytest.raises(RuntimeError, match="Run: inertia-client login"):
        user.status("id-0")

    user.login()

    assert "enter the code ABCD-1234" in capsys.readouterr().out
    assert server.polls == 3
    saved = client.token_path()
    assert saved == tmp_path / "config" / "the-inertia-challenge" / "github-token"
    assert saved.read_text() == "token-of-ada\n"
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    # Another client finds the token.
    with pytest.raises(ServerError, match="no submission with this id"):
        Client(server.url).status("id-0")

    with pytest.raises(ServerError, match="Run: inertia-client login") as error:
        Client(server.url, "another token").status("id-0")
    assert error.value.status == 401


def test_server_in_https():
    with pytest.raises(ValueError, match="must be in https"):
        Client("http://example.com")
    assert Client().server == client.DEFAULT_SERVER


def test_command_line(server, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(client.TOKEN_VARIABLE, "token-of-ada")
    monkeypatch.setenv(client.SERVER_VARIABLE, server.url)
    path = tmp_path / "labels.npz"
    save_labels(path, {4: labels(4)})

    client.main(["check", "toy", str(path)])
    assert "is a valid submission for toy, for 4 clusters" in capsys.readouterr().out
    assert not server.requests

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    with pytest.raises(SystemExit, match="Not sent"):
        client.main(["submit", "toy", str(path)])
    assert not server.requests

    arguments = ["--time", "4=1.5", "--hardware", "M2", "--tag", "algorithm=a=b"]
    client.main(["submit", "toy", str(path), "--yes", *arguments])
    assert "id-0  toy  queued" in capsys.readouterr().out
    assert server.submissions["id-0"]["tags"] == {"algorithm": "a=b"}
    assert server.submissions["id-0"]["time_s"] == {"4": 1.5}

    server.submissions["id-0"] |= {
        "status": "scored",
        "scores": [{"n_clusters": 4, "inertia": 123.5, "outcome": "best"}],
    }
    client.main(["status", "id-0", "--wait"])
    output = capsys.readouterr().out
    assert "id-0  toy  scored" in output
    assert "4 clusters: inertia 1.235000000e+02 (best)" in output

    with pytest.raises(SystemExit, match="Error: A time is N_CLUSTERS=SECONDS"):
        client.main(["submit", "toy", str(path), "--yes", "--time", "1.5"])
    with pytest.raises(SystemExit, match="Error: A tag is KEY=VALUE"):
        client.main(["submit", "toy", str(path), "--yes", "--tag", "fast"])


def toy_data(tmp_path):
    X = np.random.default_rng(0).standard_normal((N_SAMPLES, 5)).astype(np.float32)
    dataset = challenge.dataset_info("toy")
    download._write_dense(dataset, tmp_path, [X[:3], X[3:]])
    challenge.write_meta(dataset, tmp_path, {"n_samples": N_SAMPLES, "n_features": 5})
    return Reader(dataset, tmp_path)


def test_submit_only_improvement(server, tmp_path):
    reader = toy_data(tmp_path)
    inertia = {k: score.score_labels(reader, labels(k), k).inertia for k in (4, 16)}
    user = Client(
        server.url,
        "token-of-ada",
        github_api=server.url,
        results_url=server.url + "/scores",
    )
    messages = []
    # Ada is better than the labels for 4 clusters, worse for 16; another user
    # is better than everyone.
    server.scores = [
        {"user": "ada", "n_clusters": 4, "inertia": inertia[4] * 0.9},
        {"user": "ada", "n_clusters": 4, "inertia": inertia[4] * 2},
        {"user": "ada", "n_clusters": 16, "inertia": inertia[16] * 1.1},
        {"user": "bob", "n_clusters": 16, "inertia": 0.0},
    ]
    submission = user.submit(
        "toy",
        {4: labels(4), 16: labels(16)},
        time_s={4: 1.0, 16: 2.0},
        submit_only_improvement=True,
        data_dir=tmp_path,
        log=messages.append,
    )

    assert submission["status"] == "queued"
    sent = server.submissions[submission["id"]]
    assert sent["time_s"] == {"16": 2.0}
    (tmp_path / "received.npz").write_bytes(sent["content"])
    assert check_submission("toy", tmp_path / "received.npz") == [16]
    assert messages[0].startswith("4 clusters: not sent")
    assert messages[1].startswith("16 clusters: sent, improves on")

    # Nothing improves: nothing is sent.
    server.submissions.clear()
    server.scores = [{"user": "ada", "n_clusters": 4, "inertia": 0.0}]
    assert (
        user.submit(
            "toy", {4: labels(4)}, submit_only_improvement=True, data_dir=tmp_path
        )
        is None
    )
    assert not server.submissions

    # No score of the user yet: it is sent.
    server.scores = []
    assert user.submit(
        "toy", {4: labels(4)}, submit_only_improvement=True, data_dir=tmp_path
    )
