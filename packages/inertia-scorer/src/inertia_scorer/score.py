"""Score a submission: the inertia of its labels on a dataset.

    inertia-score deep_1M submission.npz

A submission is a `.npz` file with one array per number of clusters: the key is
the number of clusters `k` and the array is the label of each sample, of dtype
uint32, shape (n_samples,) and values in [0, k):

    np.savez("submission.npz", **{"10000": labels.astype(np.uint32)})

The score of labels is the sum of the squared distances of the samples to the
mean of their cluster. No centers are needed: for given labels, the means are
the centers with the lowest inertia.

All the arrays of a submission are scored in one pass on the data, read by
chunks, with

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
import struct
import sys
import time
import zipfile

import numpy as np
import scipy.sparse as sp

from . import _sums
from .challenge import SHA256, Dataset, Reader, dataset_info

DEFAULT_CHUNK_MB = 100
DEFAULT_THREADS = 4
# Bytes of a `.npy` file before the values, with a margin.
NPY_HEADER = 4096


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


def read_labels(dataset: Dataset, path):
    """Yield (n_clusters, labels) for the arrays of the submission `path`.

    Unlike `score_submission`, it only takes the numbers of clusters of the
    dataset, and nothing is read before the names and the sizes of all the arrays
    are checked: this is for a file that comes from anyone.
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
    yield from _arrays(path, keys)


def _arrays(path, keys: dict[int, str]):
    """Yield (n_clusters, array) for the arrays `keys` of the `.npz` file `path`,
    by increasing number of clusters.

    The arrays that are not compressed, as `np.savez` writes them, are mapped
    and not loaded: all the labels of a submission for a large dataset are then
    scored together without holding them in memory.
    """
    with np.load(path) as submission, zipfile.ZipFile(path) as archive:
        for n_clusters in sorted(keys):
            mapped = _map_array(path, archive, keys[n_clusters] + ".npy")
            if mapped is None:
                mapped = submission[keys[n_clusters]]
            yield n_clusters, mapped


def _map_array(path, archive: zipfile.ZipFile, name: str):
    """Read-only map of the array `name` of the open `.npz` file `path`, or None
    if it can't be mapped: `np.load` then reads it, or says why it can't."""
    try:
        info = archive.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
            return None
        with archive.open(info) as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                header = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                header = np.lib.format.read_array_header_2_0(member)
            else:
                return None
            shape, fortran_order, dtype = header
            header_bytes = member.tell()
        if dtype.hasobject or fortran_order or len(shape) != 1 or shape[0] == 0:
            return None
        if header_bytes + shape[0] * dtype.itemsize != info.file_size:
            return None
        # The name and the extra field of the local header of the member can
        # differ from the ones of the directory of the archive: their lengths
        # are read from the file.
        with open(path, "rb") as file:
            file.seek(info.header_offset + 26)
            name_bytes, extra_bytes = struct.unpack("<HH", file.read(4))
        offset = info.header_offset + 30 + name_bytes + extra_bytes + header_bytes
        return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
    except (KeyError, ValueError, OSError, struct.error):
        return None


def _read_chunk(reader: Reader, all_labels, start: int, stop: int):
    """Rows `start` to `stop` and their labels, one row per array of labels."""
    X = reader.read(start, stop)
    labels = np.empty((len(all_labels), stop - start), dtype=np.uint32)
    for row, array in zip(labels, all_labels):
        row[:] = array[start:stop]
    if sp.issparse(X):
        # scipy chooses the dtype of the indices from their values.
        X = (
            np.ascontiguousarray(X.data, dtype=np.float32),
            X.indices.astype(np.int64),
            X.indptr.astype(np.int64),
        )
    return X, labels


def _squared_norms(X) -> float:
    """Sum of the squared norms of the rows of a chunk."""
    if isinstance(X, tuple):
        data, _, indptr = X
        norms = np.empty(len(indptr) - 1)
        _sums.squared_norms_sparse(data, indptr, norms)
    else:
        norms = np.empty(len(X))
        _sums.squared_norms_dense(X, norms)
    return float(norms.sum())


def _add_chunk(X, *arguments) -> bool:
    if isinstance(X, tuple):
        return _sums.add_sparse(*X, *arguments)
    return _sums.add_dense(X, *arguments)


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


def score_all(
    reader: Reader,
    labels: dict,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_MB << 20,
    threads: int = DEFAULT_THREADS,
) -> list[Score]:
    """Inertia of each array of `labels`, {n_clusters: labels}, on the dataset of
    `reader`, by increasing number of clusters.

    The dataset is read once, by chunks of at most `chunk_bytes`. Each row is
    added to the sum of its cluster, in the order of the rows, so the sums of
    the clusters depend neither on `threads`, nor on `chunk_bytes`, nor on the
    other arrays of `labels`.
    """
    n_samples, n_features = reader.shape
    labels = dict(sorted(labels.items()))
    for n_clusters, array in labels.items():
        check_labels(array, n_clusters, n_samples)

    # The sums of all the arrays of labels, one after the other.
    offsets = np.cumsum([0, *labels], dtype=np.int64)
    sums = np.zeros((offsets[-1], n_features), dtype=np.float64)
    counts = np.zeros(offsets[-1], dtype=np.int64)
    squared_norms = 0.0
    arguments = [
        (reader, list(labels.values()), start, stop)
        for start, stop in reader.chunks(chunk_bytes)
    ]
    with ThreadPoolExecutor(threads) as readers, ThreadPoolExecutor(threads) as adders:
        for X, chunk_labels in _in_order(
            readers, _read_chunk, arguments, window=threads
        ):
            chunk_norms = adders.submit(_squared_norms, X)
            # Each thread adds the rows of its clusters.
            parts = [
                adders.submit(
                    _add_chunk, X, chunk_labels, offsets, sums, counts, part, threads
                )
                for part in range(threads)
            ]
            if not all([part.result() for part in parts]):
                raise ValueError("The labels changed while they were scored.")
            squared_norms += chunk_norms.result()

    scores = []
    for n_clusters, start, stop in zip(labels, offsets, offsets[1:]):
        nonempty = counts[start:stop] > 0
        sums_squared_norms = np.einsum("ij,ij->i", sums[start:stop], sums[start:stop])
        between = float(
            (sums_squared_norms[nonempty] / counts[start:stop][nonempty]).sum()
        )
        inertia = max(0.0, squared_norms - between)
        scores.append(
            Score(
                dataset=reader.dataset.name,
                n_clusters=n_clusters,
                n_samples=n_samples,
                inertia=inertia,
                mean_inertia=inertia / n_samples,
                n_nonempty=int(nonempty.sum()),
            )
        )
    return scores


def score_labels(
    reader: Reader,
    labels,
    n_clusters: int,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_MB << 20,
    threads: int = DEFAULT_THREADS,
) -> Score:
    """Inertia of `labels` on the dataset of `reader`. See `score_all`."""
    return score_all(
        reader, {n_clusters: labels}, chunk_bytes=chunk_bytes, threads=threads
    )[0]


def score_submission(
    dataset_name: str,
    path,
    *,
    data_dir=None,
    chunk_bytes: int = DEFAULT_CHUNK_MB << 20,
    threads: int = DEFAULT_THREADS,
) -> list[Score]:
    """Scores of all the arrays of the submission `path`, by number of clusters."""
    reader = Reader(dataset_info(dataset_name), data_dir)
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
    return score_all(
        reader, dict(_arrays(path, keys)), chunk_bytes=chunk_bytes, threads=threads
    )


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
        reader = Reader(dataset_info(args.dataset), args.data_dir)
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
