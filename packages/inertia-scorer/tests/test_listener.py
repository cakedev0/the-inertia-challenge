from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import threading

import numpy as np
import pytest

from inertia_scorer import challenge, download, listener
from inertia_scorer.challenge import Dataset
from inertia_scorer.listener import DirectoryQueue, HttpQueue, Results

N_SAMPLES = 600


def git(directory, *arguments):
    return subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A toy dataset of well separated blobs, a queue, and the clone of a
    repository of results. Returns (X, blobs, queue, results, remote)."""
    for variable in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(variable, "listener")
    for variable in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(variable, "listener@example.com")

    rng = np.random.default_rng(0)
    blobs = rng.integers(0, 4, N_SAMPLES).astype(np.uint32)
    X = (1.5 * rng.standard_normal((4, 5))[blobs]).astype(np.float32)
    X += rng.standard_normal(X.shape).astype(np.float32)
    data = tmp_path / "data"
    data.mkdir()
    dataset = Dataset("toy", N_SAMPLES, 5, (4, 16), "")
    monkeypatch.setitem(challenge.DATASETS, "toy", dataset)
    monkeypatch.setenv(challenge.DATA_DIR_VARIABLE, str(data))
    download._write_dense(dataset, data, [X])
    challenge.write_meta(dataset, data, {"n_samples": N_SAMPLES, "n_features": 5})

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", remote], check=True)
    clone = tmp_path / "results"
    subprocess.run(["git", "clone", "-q", remote, clone], check=True)
    git(clone, "commit", "--allow-empty", "-m", "Start")
    git(clone, "push", "-u", "origin", "HEAD:main")
    return X, blobs, DirectoryQueue(tmp_path / "queue"), Results(clone), remote


def submit(queue, name, labels, dataset="toy", user="ada", **metadata):
    np.savez(
        queue.pending_dir / f"{name}.npz", **{str(k): v for k, v in labels.items()}
    )
    ticket = {"dataset": dataset, "user": user}
    ticket |= metadata
    (queue.pending_dir / f"{name}.json").write_text(json.dumps(ticket))


def result_of(queue, name):
    return json.loads((queue.done_dir / f"{name}.json").read_text())


def noisy(blobs, n_moved, seed):
    """`blobs` with `n_moved` samples in another cluster."""
    labels = blobs.copy()
    moved = np.random.default_rng(seed).choice(len(labels), n_moved, replace=False)
    labels[moved] = (labels[moved] + 1) % 4
    return labels


def test_best_recorded_discarded(setup):
    X, blobs, queue, results, remote = setup
    dataset = challenge.DATASETS["toy"]

    metadata = {"time_s": 23.21, "hardware": "M2", "tags": {"algorithm": "Lloyd"}}
    submit(queue, "first", {4: noisy(blobs, 4, 1)}, **metadata)
    listener.listen(queue, results, once=True)
    first = result_of(queue, "first")
    assert first["status"] == "scored"
    assert [score["outcome"] for score in first["scores"]] == ["best"]
    assert first["scores"][0]["previous_best"] is None
    assert not list(queue.pending_dir.iterdir())

    # Far from the best: nothing is published.
    submit(queue, "bad", {4: noisy(blobs, 300, 2)})
    # The best one, and a number of clusters without a score yet.
    submit(queue, "better", {4: blobs, 16: (blobs + 4 * (X[:, 0] > 0)).astype("u4")})
    # Above the best, by less than 5%.
    submit(queue, "close", {4: noisy(blobs, 1, 3)})
    listener.listen(queue, results, once=True)

    assert [score["outcome"] for score in result_of(queue, "bad")["scores"]] == [
        "discarded"
    ]
    better = result_of(queue, "better")
    assert [score["outcome"] for score in better["scores"]] == ["best", "best"]
    assert better["scores"][0]["previous_best"] == first["scores"][0]["inertia"]
    close = result_of(queue, "close")["scores"][0]
    assert close["outcome"] == "recorded"
    assert close["previous_best"] < close["inertia"] < 1.05 * close["previous_best"]

    lines = results.scores(dataset)
    assert [(line["submission"], line["n_clusters"]) for line in lines] == [
        ("first", 4),
        ("better", 4),
        ("better", 16),
        ("close", 4),
    ]
    assert lines[0]["user"] == "ada"
    assert {key: lines[0][key] for key in metadata} == metadata
    assert {key: lines[1][key] for key in metadata} == {
        "time_s": None,
        "hardware": None,
        "tags": {},
    }

    # Everything is pushed, one commit per submission with something published,
    # with the scores only.
    assert git(results.directory, "status", "--porcelain", "--branch").strip() == (
        "## main...origin/main"
    )
    subjects = git(remote, "log", "--format=%s", "main").splitlines()
    assert subjects == [
        "Score close on toy",
        "Score better on toy",
        "Score first on toy",
        "Start",
    ]
    files = git(remote, "ls-tree", "-r", "--name-only", "main").split()
    assert files == ["scores/toy.jsonl"]


@pytest.mark.parametrize(
    "labels, dataset, match",
    [
        ({4: np.zeros((N_SAMPLES, 3), dtype=np.int64)}, "toy", "too large"),
        ({4: np.zeros(N_SAMPLES, dtype=np.int32)}, "toy", "uint32 array"),
        ({4: np.full(N_SAMPLES, 4, dtype=np.uint32)}, "toy", "got a label 4"),
        ({5: np.zeros(N_SAMPLES, dtype=np.uint32)}, "toy", "are among 4, 16"),
        ({}, "toy", "empty"),
        ({4: np.zeros(N_SAMPLES, dtype=np.uint32)}, "other", "Unknown dataset"),
    ],
)
def test_rejected(setup, labels, dataset, match):
    X, blobs, queue, results, remote = setup
    head = git(results.directory, "rev-parse", "HEAD")
    # The valid array is not published either.
    labels = {16: blobs, **labels} if labels else {}
    submit(queue, "invalid", labels, dataset=dataset)

    listener.listen(queue, results, once=True)

    result = result_of(queue, "invalid")
    assert result["status"] == "rejected" and result["scores"] == []
    assert match in result["error"]
    assert git(results.directory, "rev-parse", "HEAD") == head
    assert not git(results.directory, "status", "--porcelain")


@pytest.mark.parametrize(
    "metadata, match",
    [
        ({"user": ""}, "The user is a string"),
        ({"user": "a" * 65}, "The user is a string"),
        ({"user": 3}, "The user is a string"),
        ({"user": "a\nb"}, "can't be printed"),
        ({"time_s": "23.21"}, "positive number of seconds"),
        ({"time_s": 0}, "positive number of seconds"),
        ({"time_s": True}, "positive number of seconds"),
        ({"time_s": float("inf")}, "positive number of seconds"),
        ({"hardware": ""}, "The hardware is a line of text"),
        ({"hardware": "a" * 201}, "The hardware is a line of text"),
        ({"hardware": ["M2"]}, "The hardware is a line of text"),
        ({"tags": {"threads": 8}}, "mapping of text to text"),
        ({"tags": ["fast"]}, "mapping of text to text"),
        ({"tags": {"notes": "é" * 2100}}, "the limit is 4096"),
    ],
)
def test_invalid_metadata(setup, metadata, match):
    X, blobs, queue, results, remote = setup
    submit(queue, "invalid", {4: blobs}, **metadata)

    listener.listen(queue, results, once=True)

    result = result_of(queue, "invalid")
    assert result["status"] == "rejected" and match in result["error"]
    assert not results.scores(challenge.DATASETS["toy"])


def test_not_a_submission(setup):
    X, blobs, queue, results, remote = setup
    (queue.pending_dir / "garbage.npz").write_bytes(b"not a zip file")
    (queue.pending_dir / "garbage.json").write_text('{"dataset": "toy", "user": "ada"}')
    (queue.pending_dir / "ticket.json").write_text("[1]")

    listener.listen(queue, results, once=True)

    assert result_of(queue, "garbage")["status"] == "rejected"
    assert result_of(queue, "ticket")["error"] == "The ticket is invalid."
    assert not list(queue.pending_dir.iterdir())


def test_processed_twice(setup):
    # The listener stopped after the push and before `done`.
    X, blobs, queue, results, remote = setup
    submit(queue, "first", {4: blobs})
    listener.process(queue.get(), results)
    head = git(results.directory, "rev-parse", "HEAD")

    listener.listen(queue, results, once=True)

    assert result_of(queue, "first")["scores"][0]["outcome"] == "best"
    assert git(results.directory, "rev-parse", "HEAD") == head
    assert len(results.scores(challenge.DATASETS["toy"])) == 1


def test_results_needs_a_clean_clone(setup, tmp_path):
    X, blobs, queue, results, remote = setup
    (results.directory / "draft.txt").write_text("")
    with pytest.raises(RuntimeError, match="not committed"):
        Results(results.directory)
    with pytest.raises(RuntimeError, match="not the root"):
        Results(tmp_path / "data", remote=False)


@pytest.fixture
def server():
    """A server with the API of `HttpQueue`. `server.submissions` are the queued
    ones, as (ticket, content of the .npz file), and `server.results` the results
    it got."""

    class Handler(BaseHTTPRequestHandler):
        def answer(self, status, body=b""):
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers["Authorization"] != "Bearer secret":
                return self.answer(403, b'{"error": "not the listener"}')
            if not server.submissions:
                return self.answer(204)
            ticket, labels = server.submissions[0]
            if self.path == "/queue/next":
                return self.answer(200, json.dumps(ticket).encode())
            assert self.path == f"/queue/{ticket['id']}/labels"
            self.answer(200, labels)

        def do_POST(self):
            ticket, _ = server.submissions.pop(0)
            assert self.path == f"/queue/{ticket['id']}/result"
            length = int(self.headers["Content-Length"])
            server.results.append(json.loads(self.rfile.read(length)))
            self.answer(200, b"{}")

        def log_message(self, *arguments):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.submissions, server.results = [], []
    server.url = f"http://127.0.0.1:{server.server_port}/"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def test_http_queue(setup, server, tmp_path):
    X, blobs, queue, results, remote = setup
    np.savez(tmp_path / "labels.npz", **{"4": blobs})
    ticket = {"id": "0b1e-first", "dataset": "toy", "user": "ada", "time_s": 23.21}
    server.submissions.append((ticket, (tmp_path / "labels.npz").read_bytes()))
    invalid = {"id": "second", "dataset": "toy", "user": "ada", "hardware": "M2"}
    server.submissions.append((invalid, b"not a zip file"))
    queue = HttpQueue(server.url, "secret", tmp_path / "downloads")

    listener.listen(queue, results, once=True)

    assert [result["status"] for result in server.results] == ["scored", "rejected"]
    assert server.results[0]["id"] == "0b1e-first"
    assert server.results[0]["scores"][0]["outcome"] == "best"
    (line,) = results.scores(challenge.DATASETS["toy"])
    assert (line["user"], line["time_s"], line["hardware"]) == ("ada", 23.21, None)
    assert not list((tmp_path / "downloads").iterdir())

    with pytest.raises(RuntimeError, match="answered 403 to /queue/next"):
        HttpQueue(server.url, "wrong").get()
