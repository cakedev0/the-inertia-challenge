"""Score the submissions of a queue and publish the results in a git repository.

    inertia-listener RESULTS --queue-url https://...
    inertia-listener RESULTS --queue-dir QUEUE

The listener does not depend on what the queue is: it takes any object with the
two methods of `Queue`. A submission is from a user, for a dataset: a `.npz`
file as for `inertia-score`, with one uint32 array per number of clusters, for
some or all the numbers of clusters of the dataset. The user can say how long
the clustering took for each number of clusters (`time_s`, in seconds) and on
what hardware, and add tags:
free text about the submission, as `{"algorithm": "Lloyd"}`, of at most 4 kB.
The listener publishes what the user says and can't check it.
`HttpQueue` is the queue of the server of the scoreboard, and `DirectoryQueue`
a queue in a directory, for tests and for a server on the same machine.

The results are a clone of a git repository that only the listener writes to:
`scores/<dataset>.jsonl` has one line per published score, with its user, time,
hardware and tags.

The labels are not kept.

For each number of clusters of a submission, with `best` the lowest inertia
published so far:

- inertia < 1.1 * best, or no score yet: the score is published;
- else nothing is published.

Each submission that publishes something is one commit, pushed right away.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Protocol
import urllib.error
import urllib.request
import zipfile

from .challenge import Dataset, Reader, dataset_info
from .score import (
    DEFAULT_CHUNK_MB,
    DEFAULT_THREADS,
    checksum_warning,
    read_labels,
    score_all,
)

# A score is published when its inertia is below (1 + RECORD_MARGIN) * best.
RECORD_MARGIN = 0.1
TOKEN_VARIABLE = "INERTIA_QUEUE_TOKEN"
MAX_USER_LENGTH = 64
MAX_HARDWARE_LENGTH = 200
# Of the tags of a submission, as JSON in UTF-8.
MAX_TAGS_BYTES = 4096
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


@dataclass
class Submission:
    id: str
    dataset: str
    labels: Path  # local .npz file
    # The queue says who the user is: the listener trusts it.
    user: str
    # What the user says about the submission, published with the scores.
    # Seconds of the clustering, for each number of clusters: {64: 23.21}. As
    # it comes from JSON, the keys can also be strings.
    time_s: dict = field(default_factory=dict)
    hardware: str | None = None  # as "2 x Xeon 6972P, 192 cores"
    tags: dict[str, str] = field(default_factory=dict)  # as {"algorithm": "Lloyd"}


class Queue(Protocol):
    def get(self) -> Submission | None:
        """The next submission to process, or None if there is none for now.

        A submission is returned again until `done` is called for it.
        """

    def done(self, submission: Submission, result: dict) -> None:
        """Remove `submission` from the queue and give `result` to its sender.

        `result` has the keys id, dataset, user, status ("scored" or "rejected"),
        error (if rejected) and scores: for each number of clusters, the fields
        of `Score`, outcome ("best", "recorded" or "discarded") and
        previous_best.
        """


class DirectoryQueue:
    """A queue in a directory.

    A submission `<id>` is two files in `directory/pending`: `<id>.npz`, and
    `<id>.json` with `{"dataset": "<name>", "user": "<name>", "time_s": {"64": 23.21},
    "hardware": "...", "tags": {...}}` (the last three are optional). The sender
    writes the `.json` last, with a rename. Once the submission is processed,
    the two files are removed and the result is in `directory/done/<id>.json`.
    """

    def __init__(self, directory):
        self.pending_dir = Path(directory) / "pending"
        self.done_dir = Path(directory) / "done"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.done_dir.mkdir(parents=True, exist_ok=True)

    def get(self) -> Submission | None:
        tickets = [
            (path.stat().st_mtime, path.stem, path)
            for path in self.pending_dir.glob("*.json")
            if ID_PATTERN.fullmatch(path.stem)
        ]
        for _, name, path in sorted(tickets):  # oldest first
            submission = Submission(name, "", path.with_suffix(".npz"), "")
            try:
                ticket = json.loads(path.read_text())
                submission.dataset = ticket["dataset"]
                submission.user = ticket["user"]
                submission.time_s = ticket.get("time_s", {})
                submission.hardware = ticket.get("hardware")
                submission.tags = ticket.get("tags", {})
            except (ValueError, KeyError, TypeError):
                self.done(submission, rejected(submission, "The ticket is invalid."))
                continue
            return submission
        return None

    def done(self, submission: Submission, result: dict) -> None:
        path = self.done_dir / f"{submission.id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        os.replace(tmp, path)
        (self.pending_dir / f"{submission.id}.json").unlink(missing_ok=True)
        submission.labels.unlink(missing_ok=True)


class HttpQueue:
    """The queue of the server of the scoreboard.

    The listener asks the server, so its machine needs no open port:

    - `GET <url>/queue/next` is the next submission, as JSON (id, dataset, user,
      time_s, hardware, tags), or 204 if there is none. The server returns it
      again until its result is sent;
    - `GET <url>/queue/<id>/labels` is its `.npz` file;
    - `POST <url>/queue/<id>/result` takes the result, as JSON.

    All with `Authorization: Bearer <token>`, the token of the listener.
    """

    def __init__(self, url: str, token: str, directory=None, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        # Where the `.npz` files are while they are scored.
        self.directory = Path(directory or tempfile.mkdtemp(prefix="inertia-queue-"))
        self.directory.mkdir(parents=True, exist_ok=True)

    def _open(self, path: str, data: bytes | None = None):
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "inertia-listener",
            },
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode(errors="replace")
            raise RuntimeError(
                f"The queue answered {error.code} to {path}: {detail}"
            ) from None

    def get(self) -> Submission | None:
        with self._open("/queue/next") as response:
            if response.status == 204:
                return None
            ticket = json.load(response)
        if not ID_PATTERN.fullmatch(str(ticket["id"])):
            raise RuntimeError(f"The queue gave an invalid id: {ticket['id']!r}.")
        labels = self.directory / f"{ticket['id']}.npz"
        with self._open(f"/queue/{ticket['id']}/labels") as response:
            with open(labels, "wb") as file:
                shutil.copyfileobj(response, file, 1 << 20)
        return Submission(
            ticket["id"],
            ticket["dataset"],
            labels,
            ticket["user"],
            ticket.get("time_s") or {},
            ticket.get("hardware"),
            ticket.get("tags", {}),
        )

    def done(self, submission: Submission, result: dict) -> None:
        data = json.dumps(result).encode()
        with self._open(f"/queue/{submission.id}/result", data) as response:
            response.read()
        submission.labels.unlink(missing_ok=True)


def check_times(time_s) -> dict[int, float]:
    """The times of a submission as {n_clusters: seconds}. Raise a ValueError if
    they are not a mapping from numbers of clusters to positive numbers."""
    error = ValueError(
        "The times are a mapping from a number of clusters to a positive number "
        f"of seconds, as {{64: 23.21}}, got {time_s!r}."
    )
    if not isinstance(time_s, dict):
        raise error
    times = {}
    for key, seconds in time_s.items():
        if (
            isinstance(key, bool)
            or not (isinstance(key, int) or (isinstance(key, str) and key.isdecimal()))
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not 0 < seconds < math.inf
        ):
            raise error
        times[int(key)] = float(seconds)
    return times


def check_metadata(user, hardware, tags) -> None:
    """Raise a ValueError if the metadata of a submission can't be published."""
    if not isinstance(user, str) or not 1 <= len(user) <= MAX_USER_LENGTH:
        raise ValueError(
            f"The user is a string of 1 to {MAX_USER_LENGTH} characters, got {user!r}."
        )
    if not user.isprintable():
        raise ValueError(f"The user has characters that can't be printed: {user!r}.")
    if hardware is not None and (
        not isinstance(hardware, str)
        or not 1 <= len(hardware) <= MAX_HARDWARE_LENGTH
        or not hardware.isprintable()
    ):
        raise ValueError(
            f"The hardware is a line of text of 1 to {MAX_HARDWARE_LENGTH} "
            f"characters, got {hardware!r}."
        )
    if not isinstance(tags, dict) or not all(
        isinstance(item, str) for tag in tags.items() for item in tag
    ):
        raise ValueError(
            'The tags are a mapping of text to text, as {"algorithm": "Lloyd"}.'
        )
    size = len(json.dumps(tags, ensure_ascii=False).encode())
    if size > MAX_TAGS_BYTES:
        raise ValueError(
            f"The tags take {size} bytes as JSON, the limit is {MAX_TAGS_BYTES}."
        )


def rejected(submission: Submission, error: str) -> dict:
    return {
        "id": submission.id,
        "dataset": submission.dataset,
        "user": submission.user,
        "status": "rejected",
        "error": error,
        "scores": [],
    }


class Results:
    """The clone of the git repository of the scores.

    With `remote=False` nothing is pulled or pushed.
    """

    def __init__(self, directory, remote: bool = True):
        self.directory = Path(directory).resolve()
        self.remote = remote
        try:
            root = self._git("rev-parse", "--show-toplevel")
        except RuntimeError:
            root = ""
        if not root or Path(root) != self.directory:
            raise RuntimeError(f"{self.directory} is not the root of a git repository.")
        if self._git("status", "--porcelain"):
            raise RuntimeError(
                f"{self.directory} has changes that are not committed. The listener "
                "needs a clone of its own."
            )
        self.pull()

    def _git(self, *arguments) -> str:
        process = subprocess.run(
            ["git", "-C", str(self.directory), *arguments],
            capture_output=True,
            text=True,
        )
        if process.returncode:
            raise RuntimeError(
                f"git {' '.join(arguments)} failed in {self.directory}:\n"
                f"{process.stderr.strip()}"
            )
        return process.stdout.strip()

    def pull(self) -> None:
        if self.remote:
            self._git("pull", "--ff-only")

    def scores_path(self, dataset: Dataset) -> Path:
        return self.directory / "scores" / f"{dataset.name}.jsonl"

    def scores(self, dataset: Dataset) -> list[dict]:
        path = self.scores_path(dataset)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def add_score(self, dataset: Dataset, score: dict) -> None:
        path = self.scores_path(dataset)
        path.parent.mkdir(exist_ok=True)
        with open(path, "a") as file:
            file.write(json.dumps(score) + "\n")

    def publish(self, message: str) -> None:
        """Commit and push what `add_score` wrote."""
        self._git("add", "--all", "scores")
        self._git("commit", "-m", message)
        if self.remote:
            self._git("push")

    def restore(self) -> None:
        """Drop what was written since the last push."""
        self._git("reset", "--hard", "@{upstream}" if self.remote else "HEAD")
        self._git("clean", "-fdq", "scores")


def process(
    submission: Submission,
    results: Results,
    *,
    data_dir=None,
    chunk_bytes: int = DEFAULT_CHUNK_MB << 20,
    threads: int = DEFAULT_THREADS,
) -> dict:
    """Score `submission` and publish what deserves it. Return what goes to
    `Queue.done`.

    An invalid submission is rejected. Other errors (a dataset that is not
    there, git) are raised, and the submission can be processed again.
    """
    try:
        dataset = dataset_info(submission.dataset)
        check_metadata(submission.user, submission.hardware, submission.tags)
        times = check_times(submission.time_s)
    except (ValueError, TypeError) as error:
        return rejected(submission, str(error))

    reader = Reader(dataset, data_dir)
    warning = checksum_warning(reader)
    if warning:
        raise RuntimeError(warning)

    results.pull()
    published = results.scores(dataset)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    outcomes = []
    try:
        scores = score_all(
            reader,
            dict(read_labels(dataset, submission.labels)),
            chunk_bytes=chunk_bytes,
            threads=threads,
        )
        for score in map(asdict, scores):
            n_clusters = score["n_clusters"]
            others = [
                line["inertia"]
                for line in published
                if line["n_clusters"] == n_clusters
                and line["submission"] != submission.id
            ]
            best = min(others, default=None)
            if best is None or score["inertia"] < best:
                outcome = "best"
            elif score["inertia"] < (1 + RECORD_MARGIN) * best:
                outcome = "recorded"
            else:
                outcome = "discarded"
            outcomes.append(score | {"outcome": outcome, "previous_best": best})
            time_s = times.pop(n_clusters, None)
            # A submission processed twice (the listener stopped between the push
            # and the end) is already there.
            already = any(
                line["n_clusters"] == n_clusters and line["submission"] == submission.id
                for line in published
            )
            if outcome == "discarded" or already:
                continue
            line = {"submission": submission.id, "user": submission.user, "date": now}
            line |= score
            line["time_s"] = time_s
            line["hardware"] = submission.hardware
            line["tags"] = submission.tags
            results.add_score(dataset, line)
        if times:
            raise ValueError(
                f"The submission has a time for {', '.join(map(str, sorted(times)))} "
                "clusters, but no labels."
            )
        if any(
            outcome["outcome"] != "discarded" for outcome in outcomes
        ) and results._git("status", "--porcelain"):
            results.publish(f"Score {submission.id} on {dataset.name}")
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        # Nothing of a submission with an invalid array is published.
        results.restore()
        return rejected(submission, str(error))
    except BaseException:
        results.restore()
        raise
    return {
        "id": submission.id,
        "dataset": dataset.name,
        "user": submission.user,
        "status": "scored",
        "scores": outcomes,
    }


def listen(queue: Queue, results: Results, *, poll=5.0, once=False, **options):
    """Process the submissions of `queue` as they come. With `once`, return when
    the queue is empty. `options` are the ones of `process`."""
    while True:
        submission = queue.get()
        if submission is None:
            if once:
                return
            time.sleep(poll)
            continue
        start = time.perf_counter()
        result = process(submission, results, **options)
        queue.done(submission, result)
        summary = result.get("error") or ", ".join(
            f"{score['n_clusters']}: {score['inertia']:.9e} ({score['outcome']})"
            for score in result["scores"]
        )
        elapsed = time.perf_counter() - start
        print(
            f"{submission.id} {submission.dataset} {result['status']} in "
            f"{elapsed:.1f} s. {summary}",
            flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score the submissions of a queue and publish the results in a "
        "git repository.",
    )
    parser.add_argument("results", help="clone of the git repository of the results")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--queue-url",
        help="URL of the server of the scoreboard, see HttpQueue. The token of "
        f"the listener is in ${TOKEN_VARIABLE}",
    )
    source.add_argument(
        "--queue-dir", help="directory of the queue, see DirectoryQueue"
    )
    parser.add_argument(
        "--no-remote", action="store_true", help="commit, but do not pull or push"
    )
    parser.add_argument(
        "--once", action="store_true", help="stop when the queue is empty"
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="time between two looks at an empty queue (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=DEFAULT_CHUNK_MB,
        metavar="MB",
        help="the datasets are read by chunks of this size (default: %(default)s)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="chunks processed at the same time (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        help="directory of the datasets (default: $INERTIA_CHALLENGE_DATA, "
        "else ~/.cache/the-inertia-challenge)",
    )
    args = parser.parse_args(argv)

    if args.queue_dir:
        queue = DirectoryQueue(args.queue_dir)
    else:
        token = os.environ.get(TOKEN_VARIABLE)
        if not token:
            sys.exit(f"Error: set ${TOKEN_VARIABLE} to the token of the listener.")
        queue = HttpQueue(args.queue_url, token)

    try:
        listen(
            queue,
            Results(args.results, remote=not args.no_remote),
            poll=args.poll,
            once=args.once,
            data_dir=args.data_dir,
            chunk_bytes=max(1, int(args.chunk_size * 2**20)),
            threads=max(1, args.threads),
        )
    except (RuntimeError, OSError) as error:
        sys.exit(f"Error: {error}")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
