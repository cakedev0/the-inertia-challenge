"""Submit labels to the inertia challenge.

    inertia-client login
    inertia-client submit sift_128 sift_128.npz --time 100=23.21 --hardware M2
    inertia-client status <id> --wait

A submission is a `.npz` file for one dataset, as for `inertia-score`: one
uint32 array of labels per number of clusters. The server takes one submission
per day and per dataset from a user, so `submit` first checks the file as the
server will, and asks before sending.

`login` identifies you with GitHub: it gives a code to enter on github.com, and
keeps a token that can only tell who you are (no access to your repositories).
From Python:

    import inertia_client

    inertia_client.submit("sift_128", {100: labels}, time_s={100: 23.21}, hardware="M2")
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import numpy as np

from inertia_scorer.challenge import get_dataset
from inertia_scorer.score import check_labels, read_labels

DEFAULT_SERVER = "https://the-inertia-challenge-server.arthur-lacote.workers.dev"
SERVER_VARIABLE = "INERTIA_SERVER"
TOKEN_VARIABLE = "INERTIA_GITHUB_TOKEN"
GITHUB = "https://github.com"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
# Times a part is sent again when the connection or the server fails.
RETRIES = 4


class ServerError(RuntimeError):
    """The server refused a request. `status` is the HTTP status."""

    def __init__(self, status: int, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def token_path() -> Path:
    config = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(config).expanduser() / "the-inertia-challenge" / "github-token"


def check_submission(dataset_name: str, path) -> list[int]:
    """Check the `.npz` file `path` as the server will, without the dataset.
    Return its numbers of clusters. Raise a ValueError if it would be rejected."""
    dataset = get_dataset(dataset_name)
    n_clusters = []
    try:
        for k, labels in read_labels(dataset, path):
            try:
                check_labels(labels, k, dataset.n_samples)
            except ValueError as error:
                raise ValueError(f"For {k} clusters: {error}") from None
            n_clusters.append(k)
    except zipfile.BadZipFile:
        raise ValueError(f"{path} is not a .npz file.") from None
    return n_clusters


def save_labels(path, labels: dict) -> None:
    """Write `labels`, {n_clusters: array of labels}, as a submission."""
    arrays = {}
    for n_clusters, array in labels.items():
        array = np.asarray(array)
        if array.dtype != np.uint32:
            if array.dtype.kind not in "iu" or (array.size and array.min() < 0):
                raise ValueError(
                    f"The labels for {n_clusters} clusters are not non-negative "
                    f"integers (dtype {array.dtype})."
                )
            array = array.astype(np.uint32)
        arrays[str(int(n_clusters))] = array
    np.savez(path, **arrays)


class Client:
    """The server of the challenge, for a user.

    The server is `server`, else $INERTIA_SERVER, else the one of the
    challenge. The GitHub token is `token`, else $INERTIA_GITHUB_TOKEN, else
    the one that `login` saved.
    """

    def __init__(self, server=None, token=None, github=GITHUB, timeout=120.0):
        server = server or os.environ.get(SERVER_VARIABLE) or DEFAULT_SERVER
        self.server = server.rstrip("/")
        url = urllib.parse.urlsplit(self.server)
        if url.scheme != "https" and url.hostname not in ("localhost", "127.0.0.1"):
            raise ValueError(f"The server must be in https, got {self.server}.")
        self._token = token or os.environ.get(TOKEN_VARIABLE)
        self.github = github
        self.timeout = timeout

    @property
    def token(self) -> str:
        if not self._token:
            path = token_path()
            if not path.is_file():
                raise RuntimeError("You are not logged in. Run: inertia-client login")
            self._token = path.read_text().strip()
        return self._token

    def _request(self, method, path, body=None, raw=None, auth=True) -> dict:
        headers = {"User-Agent": "inertia-client"}
        if auth:
            headers["Authorization"] = f"Bearer {self.token}"
        data = raw
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.server}{path}", data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try:
                message = json.load(error)["error"]
            except (ValueError, KeyError, TypeError):
                message = f"The server answered {error.code}."
            if error.code == 401:
                message += " Run: inertia-client login"
            retry_after = error.headers.get("Retry-After")
            raise ServerError(
                error.code, message, int(retry_after) if retry_after else None
            ) from None

    # Login

    def _github(self, path: str, **fields) -> dict:
        request = urllib.request.Request(
            f"{self.github}{path}",
            data=urllib.parse.urlencode(fields).encode(),
            headers={"Accept": "application/json", "User-Agent": "inertia-client"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def login(self, show=print, save=True) -> str:
        """Identify the user with the device flow of GitHub: `show` gets the
        code to enter on github.com. Return the token, saved for the next
        calls."""
        client_id = self._request("GET", "/", auth=False)["github_client_id"]
        # No scope: the token only says who the user is.
        device = self._github("/login/device/code", client_id=client_id)
        show(
            f"Open {device['verification_uri']} and enter the code "
            f"{device['user_code']}\nWaiting for you there..."
        )
        interval = device.get("interval", 5)
        deadline = time.monotonic() + device.get("expires_in", 900)
        while time.monotonic() < deadline:
            time.sleep(interval)
            answer = self._github(
                "/login/oauth/access_token",
                client_id=client_id,
                device_code=device["device_code"],
                grant_type=DEVICE_GRANT,
            )
            if "access_token" in answer:
                self._token = answer["access_token"]
                if save:
                    path = token_path()
                    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    # Created for the user only, before the token is in it.
                    descriptor = os.open(
                        path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
                    )
                    with os.fdopen(descriptor, "w") as file:
                        file.write(self._token + "\n")
                return self._token
            error = answer.get("error")
            if error == "slow_down":
                interval = answer.get("interval", interval + 5)
            elif error != "authorization_pending":
                raise RuntimeError(
                    f"GitHub refused the login: "
                    f"{answer.get('error_description') or error}"
                )
        raise RuntimeError("The code expired before it was entered.")

    # Submissions

    def submit(
        self,
        dataset: str,
        labels,
        *,
        time_s: dict[int, float] | None = None,
        hardware: str | None = None,
        tags: dict[str, str] | None = None,
        progress=None,
    ) -> dict:
        """Check and send a submission. Return it as the server sees it.

        `labels` is a `.npz` file, or {n_clusters: array of labels}. `time_s`,
        `hardware` and `tags` are what you say about it: how long the
        clustering took for each number of clusters, as {n_clusters: seconds},
        on what, and free text as {"algorithm": "Lloyd"}.
        `progress(done, total)` is called after each part, in bytes.
        """
        if isinstance(labels, dict):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"{dataset}.npz"
                save_labels(path, labels)
                return self.submit(
                    dataset,
                    path,
                    time_s=time_s,
                    hardware=hardware,
                    tags=tags,
                    progress=progress,
                )
        path = Path(labels)
        n_clusters = check_submission(dataset, path)
        body = {"dataset": dataset, "size": path.stat().st_size}
        if time_s:
            if not isinstance(time_s, dict):
                raise ValueError(
                    "`time_s` is the time for each number of clusters, as "
                    f"{{{n_clusters[0]}: 23.21}}, got {time_s!r}."
                )
            unknown = sorted(set(map(int, time_s)) - set(n_clusters))
            if unknown:
                raise ValueError(
                    f"There is a time for {', '.join(map(str, unknown))} clusters, "
                    "but no labels."
                )
            body["time_s"] = {str(int(k)): float(s) for k, s in time_s.items()}
        if hardware is not None:
            body["hardware"] = hardware
        if tags:
            body["tags"] = tags
        created = self._request("POST", "/submissions", body)
        return self.upload(created["id"], path, progress)

    def upload(self, submission_id: str, path, progress=None) -> dict:
        """Send the parts of `path` that the server does not have, and complete
        the submission. Also resumes an upload that was stopped."""
        path = Path(path)
        submission = self.status(submission_id)
        if submission["status"] != "uploading":
            return submission
        size, part_size = submission["size"], submission["part_size"]
        if path.stat().st_size != size:
            raise ValueError(
                f"{path} has {path.stat().st_size} bytes, the submission has {size}."
            )
        uploaded = set(submission["uploaded_parts"])
        done = sum(min(part_size, size - (part - 1) * part_size) for part in uploaded)
        with open(path, "rb") as file:
            for part in range(1, submission["n_parts"] + 1):
                if part in uploaded:
                    continue
                file.seek((part - 1) * part_size)
                data = file.read(part_size)
                self._upload_part(submission_id, part, data)
                done += len(data)
                if progress:
                    progress(done, size)
        return self._request("POST", f"/submissions/{submission_id}/complete")

    def _upload_part(self, submission_id: str, part: int, data: bytes) -> None:
        for attempt in range(RETRIES + 1):
            try:
                self._request(
                    "PUT", f"/submissions/{submission_id}/parts/{part}", raw=data
                )
                return
            except (ServerError, OSError) as error:
                refused = isinstance(error, ServerError) and error.status < 500
                if refused or attempt == RETRIES:
                    raise
                time.sleep(2**attempt)

    def status(self, submission_id: str) -> dict:
        return self._request("GET", f"/submissions/{submission_id}")

    def wait(self, submission_id: str, poll: float = 10.0, timeout=None) -> dict:
        """The submission once it is scored or rejected."""
        start = time.monotonic()
        while True:
            submission = self.status(submission_id)
            if submission["status"] not in ("uploading", "queued"):
                return submission
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(f"{submission_id} is still {submission['status']}.")
            time.sleep(poll)


def submit(dataset: str, labels, **options) -> dict:
    """`Client().submit`: check and send a submission, see `Client.submit`."""
    return Client().submit(dataset, labels, **options)


def _print_submission(submission: dict) -> None:
    print(f"{submission['id']}  {submission['dataset']}  {submission['status']}")
    if submission.get("error"):
        print(f"  {submission['error']}")
    for score in submission.get("scores", []):
        print(
            f"  {score['n_clusters']} clusters: inertia {score['inertia']:.9e} "
            f"({score['outcome']})"
        )
    if submission["status"] == "uploading":
        print(
            f"  {len(submission['uploaded_parts'])} of {submission['n_parts']} parts "
            f"uploaded. To resume: inertia-client resume {submission['id']} FILE"
        )


def _progress(done: int, total: int) -> None:
    unit, name = (1e6, "MB") if total >= 1e7 else (1e3, "kB")
    print(
        f"\r  uploaded {done / unit:.0f} of {total / unit:.0f} {name}",
        end="",
        flush=True,
    )
    if done == total:
        print()


def _tags(pairs: list[str]) -> dict[str, str]:
    tags = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ValueError(f"A tag is KEY=VALUE, got {pair!r}.")
        tags[key] = value
    return tags


def _times(pairs: list[str]) -> dict[int, float]:
    times = {}
    for pair in pairs:
        try:
            n_clusters, seconds = pair.split("=")
            times[int(n_clusters)] = float(seconds)
        except ValueError:
            raise ValueError(f"A time is N_CLUSTERS=SECONDS, got {pair!r}.") from None
    return times


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Submit labels to the inertia challenge.",
    )
    parser.add_argument(
        "--server",
        help=f"URL of the server (default: ${SERVER_VARIABLE}, else the one of the "
        "challenge)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("login", help="identify yourself with GitHub")

    check = commands.add_parser("check", help="check a submission, without sending it")
    check.add_argument("dataset")
    check.add_argument("file", help=".npz file, {n_clusters: uint32 labels}")

    send = commands.add_parser("submit", help="check and send a submission")
    send.add_argument("dataset")
    send.add_argument("file", help=".npz file, {n_clusters: uint32 labels}")
    send.add_argument(
        "--time",
        action="append",
        default=[],
        metavar="N_CLUSTERS=SECONDS",
        help="how long the clustering took for a number of clusters, as 100=23.21 "
        "(can be repeated)",
    )
    send.add_argument("--hardware", help='on what, as "M2, 8 cores"')
    send.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="free text about the submission, as algorithm=Lloyd (can be repeated)",
    )
    send.add_argument("--yes", action="store_true", help="do not ask before sending")
    send.add_argument("--wait", action="store_true", help="wait for the scores")

    resume = commands.add_parser("resume", help="resume an upload that was stopped")
    resume.add_argument("id")
    resume.add_argument("file")

    status = commands.add_parser("status", help="where a submission is, and its scores")
    status.add_argument("id")
    status.add_argument("--wait", action="store_true", help="wait for the scores")
    args = parser.parse_args(argv)

    try:
        client = Client(args.server)
        if args.command == "login":
            # Flushed: the code must show up when the output is not a terminal.
            client.login(show=lambda message: print(message, flush=True))
            print(f"Logged in. The token is in {token_path()}")
        elif args.command == "check":
            n_clusters = check_submission(args.dataset, args.file)
            print(
                f"{args.file} is a valid submission for {args.dataset}, for "
                f"{', '.join(map(str, n_clusters))} clusters."
            )
        elif args.command == "submit":
            n_clusters = check_submission(args.dataset, args.file)
            tags = _tags(args.tag)
            times = _times(args.time)
            print(
                f"{args.file} is a valid submission for {args.dataset}, for "
                f"{', '.join(map(str, n_clusters))} clusters."
            )
            if not args.yes:
                answer = input(
                    f"You can submit once per day for {args.dataset}. Send it? [y/N] "
                )
                if answer.strip().lower() not in ("y", "yes"):
                    sys.exit("Not sent.")
            submission = client.submit(
                args.dataset,
                args.file,
                time_s=times,
                hardware=args.hardware,
                tags=tags,
                progress=_progress,
            )
            if args.wait:
                submission = client.wait(submission["id"])
            _print_submission(submission)
        elif args.command == "resume":
            _print_submission(client.upload(args.id, args.file, _progress))
        elif args.command == "status":
            submission = client.wait(args.id) if args.wait else client.status(args.id)
            _print_submission(submission)
    except (ValueError, RuntimeError, OSError) as error:
        sys.exit(f"Error: {error}")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
