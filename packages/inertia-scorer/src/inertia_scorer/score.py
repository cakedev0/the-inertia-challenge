"""Score a submission: the inertia of its labels on a dataset.

    inertia-score deep_1M submission.npz

A submission is a `.npz` file with one array per number of clusters: the key is
the number of clusters `k` and the array is the label of each sample, of dtype
uint32, shape (n_samples,) and values in [0, k):

    np.savez("submission.npz", **{"10000": labels.astype(np.uint32)})

The score of labels is the sum of the squared distances of the samples to the
mean of their cluster. No centers are needed: for given labels, the means are
the centers with the lowest inertia.

It is computed in one pass on the data, read by chunks, with

    inertia = sum_i |x_i|^2 - sum_j |s_j|^2 / n_j

where `s_j` is the sum of the samples of cluster j and `n_j` their number. All
the sums are in float64.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import sys
import time
import zipfile

import numpy as np
import scipy.sparse as sp

from .challenge import SHA256, Reader, get_dataset

DEFAULT_CHUNK_MB = 100
DEFAULT_THREADS = 4


@dataclass
class Score:
    dataset: str
    n_clusters: int
    n_samples: int
    inertia: float  # sum of squared distances to the mean of the cluster
    mean_inertia: float  # inertia / n_samples
    n_nonempty: int  # clusters with at least one sample


def check_labels(labels, n_clusters: int, n_samples: int) -> None:
    """Raise a ValueError if `labels` is not a valid submission for `n_clusters`."""
    if not isinstance(labels, np.ndarray) or labels.dtype != np.uint32:
        dtype = getattr(labels, "dtype", type(labels).__name__)
        raise ValueError(
            f"The labels must be a uint32 array, got {dtype}. "
            "Use labels.astype(np.uint32)."
        )
    if labels.shape != (n_samples,):
        raise ValueError(
            f"The labels must have shape ({n_samples},), got {labels.shape}."
        )
    if n_clusters < 1:
        raise ValueError(f"The number of clusters must be positive, got {n_clusters}.")
    largest = int(labels.max())
    if largest >= n_clusters:
        raise ValueError(
            f"The labels must be in [0, {n_clusters}), got a label {largest}."
        )


def _chunk_sums(reader: Reader, labels, start: int, stop: int):
    """Per-cluster sums and counts of rows `start` to `stop`, for the clusters that
    have rows in this range, and the sum of the squared norms of the rows."""
    X = reader.read(start, stop).astype(np.float64)
    clusters, inverse, counts = np.unique(
        labels[start:stop], return_inverse=True, return_counts=True
    )
    n_rows = stop - start
    if sp.issparse(X):
        indicator = sp.csr_matrix(
            (np.ones(n_rows), (inverse, np.arange(n_rows))),
            shape=(len(clusters), n_rows),
        )
        sums = (indicator @ X).tocsr()
        squared_norms = float((X.data * X.data).sum())
    else:
        # One stored value per column: the product adds each row of X to the
        # row of its cluster, without sorting the rows.
        indicator = sp.csc_matrix(
            (np.ones(n_rows), inverse, np.arange(n_rows + 1)),
            shape=(len(clusters), n_rows),
        )
        sums = indicator @ X
        # Not a BLAS dot: its threads would add to the ones of the scorer.
        squared_norms = float(np.einsum("ij,ij->i", X, X).sum())
    return clusters, counts, sums, squared_norms


def _add_rows(sums, rows, values):
    sums[rows] += values


def _in_order(executor, function, arguments, window):
    """Results of `function(*args)` in the order of `arguments`, with at most
    `window` calls running or waiting to be consumed."""
    pending = deque()
    for args in arguments:
        pending.append(executor.submit(function, *args))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def score_labels(
    reader: Reader,
    labels,
    n_clusters: int,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_MB << 20,
    threads: int = DEFAULT_THREADS,
) -> Score:
    """Inertia of `labels` on the dataset of `reader`.

    The dataset is read by chunks of at most `chunk_bytes`. The chunks are added
    in order, so the result does not depend on `threads`.
    """
    n_samples, n_features = reader.shape
    check_labels(labels, n_clusters, n_samples)

    sums = np.zeros((n_clusters, n_features), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    squared_norms = 0.0
    arguments = [
        (reader, labels, start, stop) for start, stop in reader.chunks(chunk_bytes)
    ]
    with ThreadPoolExecutor(threads) as executor, ThreadPoolExecutor(threads) as adders:
        for clusters, chunk_counts, chunk_sums, chunk_norms in _in_order(
            executor, _chunk_sums, arguments, window=threads
        ):
            if sp.issparse(chunk_sums):
                rows = np.repeat(clusters, np.diff(chunk_sums.indptr))
                sums[rows, chunk_sums.indices] += chunk_sums.data
            else:
                # With many clusters, adding to `sums` costs as much as the
                # rest. Each thread adds a range of clusters, so they write to
                # different rows.
                bounds = np.linspace(0, len(clusters), threads + 1).astype(int)
                additions = [
                    adders.submit(_add_rows, sums, clusters[a:b], chunk_sums[a:b])
                    for a, b in zip(bounds, bounds[1:])
                ]
                for addition in additions:
                    addition.result()
            counts[clusters] += chunk_counts
            squared_norms += chunk_norms

    nonempty = counts > 0
    sums_squared_norms = np.einsum("ij,ij->i", sums, sums)
    between = float((sums_squared_norms[nonempty] / counts[nonempty]).sum())
    inertia = max(0.0, squared_norms - between)
    return Score(
        dataset=reader.dataset.name,
        n_clusters=n_clusters,
        n_samples=n_samples,
        inertia=inertia,
        mean_inertia=inertia / n_samples,
        n_nonempty=int(nonempty.sum()),
    )


def score_submission(
    dataset_name: str,
    path,
    *,
    data_dir=None,
    chunk_bytes: int = DEFAULT_CHUNK_MB << 20,
    threads: int = DEFAULT_THREADS,
) -> list[Score]:
    """Scores of all the arrays of the submission `path`, by number of clusters."""
    reader = Reader(get_dataset(dataset_name), data_dir)
    scores = []
    submission = np.load(path)
    if not isinstance(submission, np.lib.npyio.NpzFile):
        raise ValueError(f"A submission is a .npz file, {path} is not one.")
    with submission:
        keys = {}
        for key in submission.files:
            if not key.isdecimal():
                raise ValueError(
                    f"The keys of a submission are numbers of clusters, got {key!r}."
                )
            keys[int(key)] = key
        for n_clusters in sorted(keys):
            labels = submission[keys[n_clusters]]
            scores.append(
                score_labels(
                    reader,
                    labels,
                    n_clusters,
                    chunk_bytes=chunk_bytes,
                    threads=threads,
                )
            )
    return scores


def checksum_warning(reader: Reader) -> str | None:
    name = reader.dataset.name
    reference = SHA256.get(name)
    local = reader.meta.get("sha256", {}).get(name)
    if reference is not None and local != reference:
        return (
            f"Warning: the local copy of {name} is not the reference one (sha256 "
            f"{local}, expected {reference}). Its scores can't be compared to the "
            "ones of other machines."
        )
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inertia of the labels of a submission.",
    )
    parser.add_argument("dataset", help="name of the dataset, see inertia-download")
    parser.add_argument("submission", help=".npz file, {n_clusters: uint32 labels}")
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=DEFAULT_CHUNK_MB,
        metavar="MB",
        help="the dataset is read by chunks of this size (default: %(default)s)",
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
    parser.add_argument("--json", action="store_true", help="print one JSON per score")
    args = parser.parse_args(argv)

    try:
        reader = Reader(get_dataset(args.dataset), args.data_dir)
        warning = checksum_warning(reader)
        if warning:
            print(warning, file=sys.stderr)
        start = time.perf_counter()
        scores = score_submission(
            args.dataset,
            args.submission,
            data_dir=args.data_dir,
            chunk_bytes=max(1, int(args.chunk_size * 2**20)),
            threads=max(1, args.threads),
        )
        elapsed = time.perf_counter() - start
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        sys.exit(f"Error: {error}")

    if args.json:
        for score in scores:
            print(json.dumps(asdict(score)))
    else:
        header = (
            "dataset",
            "n_clusters",
            "non-empty",
            "inertia",
            "inertia / n_samples",
        )
        rows = [
            (
                score.dataset,
                str(score.n_clusters),
                str(score.n_nonempty),
                f"{score.inertia:.9e}",
                f"{score.mean_inertia:.9e}",
            )
            for score in scores
        ]
        widths = [max(len(row[i]) for row in [header, *rows]) for i in range(5)]
        for row in [header, *rows]:
            print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))
    print(f"Scored in {elapsed:.1f} s.", file=sys.stderr)


if __name__ == "__main__":
    main()
