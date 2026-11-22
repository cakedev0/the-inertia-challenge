"""Score the submissions of a queue and publish the results in a git repository.

    inertia-listener RESULTS --queue-dir QUEUE

The listener does not depend on what the queue is: it takes any object with the
two methods of `Queue`. A submission is from a user, for a dataset: a `.npz`
file as for `inertia-score`, with one uint32 array per number of clusters, for
some or all the numbers of clusters of the dataset. The user can say how long
the clustering took (`time_s`, in seconds) and on what hardware, and add tags:
free text about the submission, as `{"algorithm": "Lloyd"}`, of at most 4 kB.
The listener publishes what the user says and can't check it.
`DirectoryQueue` is a queue in a directory, for tests and for a server on the
same machine.

The results are a clone of a git repository that only the listener writes to:
`scores/<dataset>.jsonl` has one line per published score, with its user, time,
hardware and tags.

The labels are not kept.

For each number of clusters of a submission, with `best` the lowest inertia
published so far:

- inertia < 1.05 * best, or no score yet: the score is published;
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
import subprocess
import sys
import time
from typing import Protocol
import zipfile

import numpy as np

from .challenge import Dataset, Reader, get_dataset
from .score import DEFAULT_CHUNK_MB, DEFAULT_THREADS, checksum_warning, score_labels

# A score is published when its inertia is below (1 + RECORD_MARGIN) * best.
RECORD_MARGIN = 0.05
MAX_USER_LENGTH = 64
MAX_HARDWARE_LENGTH = 200
# Of the tags of a submission, as JSON in UTF-8.
MAX_TAGS_BYTES = 4096
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
# Bytes of a `.npy` file before the values, with a margin.
NPY_HEADER = 4096


@dataclass
class Submission:
    id: str
    dataset: str
    labels: Path  # local .npz file
    # The queue says who the user is: the listener trusts it.
    user: str
    # What the user says about the submission, published with the scores.
    time_s: float | None = None  # time of the clustering, in seconds
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
    `<id>.json` with `{"dataset": "<name>", "user": "<name>", "time_s": 23.21,
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
                submission.time_s = ticket.get("time_s")
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


def check_metadata(user, time_s, hardware, tags) -> None:
    """Raise a ValueError if the metadata of a submission can't be published."""
    if not isinstance(user, str) or not 1 <= len(user) <= MAX_USER_LENGTH:
        raise ValueError(
            f"The user is a string of 1 to {MAX_USER_LENGTH} characters, got {user!r}."
        )
    if not user.isprintable():
        raise ValueError(f"The user has characters that can't be printed: {user!r}.")
    if time_s is not None and (
        isinstance(time_s, bool)
        or not isinstance(time_s, (int, float))
        or not 0 < time_s < math.inf
    ):
        raise ValueError(f"The time is a positive number of seconds, got {time_s!r}.")
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


def read_labels(dataset: Dataset, path):
    """Yield (n_clusters, labels) for the arrays of the submission `path`.

    Nothing is read before the names and the sizes of all the arrays are checked,
    as the file comes from anyone.
    """
    with zipfile.ZipFile(path) as archive:
        keys = {}
        for info in archive.infolist():
            key = info.filename.removesuffix(".npy")
            if not key.isdecimal() or int(key) not in dataset.n_clusters:
                raise ValueError(
                    f"The keys of a submission for {dataset.name} are among "
                    f"{', '.join(map(str, dataset.n_clusters))}, got {key!r}."
                )
            if int(key) in keys:
                raise ValueError(f"The submission has {int(key)} clusters twice.")
            if info.file_size > 4 * dataset.n_samples + NPY_HEADER:
                raise ValueError(
                    f"The array {key!r} is too large for uint32 labels of shape "
                    f"({dataset.n_samples},)."
                )
            keys[int(key)] = key
    if not keys:
        raise ValueError("The submission is empty.")
    with np.load(path) as submission:
        for n_clusters in sorted(keys):
            yield n_clusters, submission[keys[n_clusters]]


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
        dataset = get_dataset(submission.dataset)
        check_metadata(
            submission.user, submission.time_s, submission.hardware, submission.tags
        )
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
        for n_clusters, labels in read_labels(dataset, submission.labels):
            score = asdict(
                score_labels(
                    reader, labels, n_clusters, chunk_bytes=chunk_bytes, threads=threads
                )
            )
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
            line["time_s"] = submission.time_s
            line["hardware"] = submission.hardware
            line["tags"] = submission.tags
            results.add_score(dataset, line)
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
    parser.add_argument(
        "--queue-dir",
        required=True,
        help="directory of the queue, see DirectoryQueue (the only queue for now)",
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

    try:
        listen(
            DirectoryQueue(args.queue_dir),
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
