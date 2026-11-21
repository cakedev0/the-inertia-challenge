"""Download the datasets of the challenge, if they are not already there.

    python download.py                      # list the datasets and what is there
    python download.py sift_128 deep_1M     # download these two
    python download.py all                  # everything: about 200 GB

The datasets go to `--data-dir`, else to $INERTIA_CHALLENGE_DATA, else to
~/.cache/the-inertia-challenge. See `challenge.py` for the format of the files.

The loaders are the ones of the k-means benchmark of
https://github.com/probabl-ai/scikit-learn-benchmarks/pull/187, plus the first
rows of DEEP1B. They need h5py, pandas, pyarrow and scikit-learn.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from challenge import (
    DATASETS,
    SHA256,
    Dataset,
    data_dir,
    data_files,
    get_dataset,
    is_present,
    read_meta,
    write_meta,
)

BLOCK = 1 << 20
USER_AGENT = "the-inertia-challenge"
DEEP_URL = (
    "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin"
)
# `base.1B.fbin` starts with the numbers of rows and of features, as two int32.
DEEP_HEADER = 8
DEEP_SEGMENT = 128 << 20
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"


def _open(url, first_byte=0, last_byte=None):
    # ann-benchmarks.com refuses the default user agent of urllib.
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if first_byte or last_byte is not None:
        last = "" if last_byte is None else last_byte
        request.add_header("Range", f"bytes={first_byte}-{last}")
    return urllib.request.urlopen(request, timeout=120)


def fetch(url: str, path: Path, retries: int = 5) -> Path:
    """Download `url` to `path`, unless it is there. A download that was cut
    resumes from `<path>.part`."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    print(f"  downloading {url}", flush=True)
    failures = 0
    while True:
        have = part.stat().st_size if part.is_file() else 0
        try:
            with _open(url, first_byte=have) as response:
                if have and response.status != 206:
                    # The server ignored the range: start again.
                    have = 0
                with open(part, "ab" if have else "wb") as file:
                    shutil.copyfileobj(response, file, BLOCK)
                length = response.headers.get("Content-Length")
                received = part.stat().st_size - have
                if length is not None and received != int(length):
                    raise OSError(f"got {received} bytes of {length}")
            break
        except urllib.error.HTTPError as error:
            if error.code == 416 and have:
                # Nothing after `have`: the file was complete.
                break
            failures += 1
            if failures > retries:
                raise
        except OSError:
            failures += 1
            if failures > retries:
                raise
        time.sleep(2 * failures)
    os.replace(part, path)
    return path


def _write_dense(dataset: Dataset, directory: Path, arrays) -> None:
    """Write the arrays, one after the other, as the float32 rows of `dataset`."""
    (path,) = data_files(dataset, directory)
    part = path.with_name(path.name + ".part")
    n_samples = 0
    with open(part, "wb") as file:
        for array in arrays:
            array = np.ascontiguousarray(array, dtype="<f4")
            assert array.ndim == 2 and array.shape[1] == dataset.n_features
            array.tofile(file)
            n_samples += array.shape[0]
    assert n_samples == dataset.n_samples, (n_samples, dataset.n_samples)
    os.replace(part, path)


def _write_sparse(dataset: Dataset, directory: Path, X) -> None:
    import scipy.sparse as sp

    X = sp.csr_matrix(X, dtype=np.float32)
    assert X.shape == (dataset.n_samples, dataset.n_features), X.shape
    arrays = (X.data, X.indices.astype("<i4"), X.indptr.astype("<i8"))
    for path, array in zip(data_files(dataset, directory), arrays):
        part = path.with_name(path.name + ".part")
        array.tofile(part)
        os.replace(part, path)


def _ann_benchmarks_train(url, raw):
    import h5py

    path = fetch(url, raw / url.rsplit("/", 1)[1])
    with h5py.File(path, "r") as file:
        return np.asarray(file["train"], dtype=np.float32)


def build_sift_128(raw: Path):
    yield _ann_benchmarks_train(
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5", raw
    )


def build_glove_200(raw: Path):
    yield _ann_benchmarks_train("http://ann-benchmarks.com/glove-200-angular.hdf5", raw)


def build_clip_512(raw: Path):
    """Parts 00 to 19 of Qdrant/wolt-food-clip-ViT-B-32-embeddings (4.6 GB of
    parquet): 1M rows."""
    import pyarrow.parquet as pq

    url = (
        "https://huggingface.co/datasets/Qdrant/wolt-food-clip-ViT-B-32-embeddings/"
        "resolve/main/data/part-{:02d}-wolt-clip-ViT-B-32.parquet"
    )
    for i in range(20):
        path = fetch(url.format(i), raw / url.format(i).rsplit("/", 1)[1])
        # The vectors are JSON strings, "[x0,x1,...]".
        column = pq.read_table(path, columns=["vector"]).column("vector")
        strings = column.cast(column.type.storage_type).to_pylist()
        text = ",".join(string.strip()[1:-1] for string in strings)
        yield np.array(text.split(","), dtype=np.float32).reshape(len(strings), -1)


def build_susy(raw: Path):
    import pandas as pd

    path = fetch(f"{UCI}/00279/SUSY.csv.gz", raw / "SUSY.csv.gz")
    data = pd.read_csv(path, delimiter=",", header=None, compression="gzip")
    # The first column is the class.
    yield data[data.columns[1:]].to_numpy()


def build_year_prediction_msd_standardized(raw: Path):
    """Each feature is standardized over all the rows: their raw standard
    deviations range from 4.4 to 1,750."""
    import pandas as pd

    path = fetch(
        f"{UCI}/00203/YearPredictionMSD.txt.zip", raw / "YearPredictionMSD.txt.zip"
    )
    data = pd.read_csv(path, header=None)
    # The first column is the year.
    x = data.iloc[:, 1:].to_numpy(dtype=np.float64)
    std = x.std(axis=0)
    std[std == 0] = 1
    yield (x - x.mean(axis=0)) / std


def build_road_network_km_x10(raw: Path):
    """Longitude and latitude are projected to local kilometers (equirectangular
    around the mean latitude) and the altitude is converted to kilometers. Each of
    the 434,874 points then comes with 9 copies jittered by a Gaussian noise of
    5 m (seed 0)."""
    import pandas as pd

    path = fetch(f"{UCI}/00246/3D_spatial_network.txt", raw / "3D_spatial_network.txt")
    data = pd.read_csv(
        path, header=None, names=["osm_id", "longitude", "latitude", "altitude"]
    )
    lon, lat, alt = data[["longitude", "latitude", "altitude"]].to_numpy(np.float64).T
    earth_km = 6371.0088
    lat0 = np.deg2rad(lat.mean())
    km = np.column_stack(
        [
            np.deg2rad(lon - lon.mean()) * earth_km * np.cos(lat0),
            np.deg2rad(lat - lat.mean()) * earth_km,
            (alt - alt.mean()) / 1000,
        ]
    )
    copies, jitter_km = 10, 0.005
    x = np.repeat(km[:, None, :], copies, axis=1)
    x[:, 1:] += np.random.default_rng(0).normal(
        scale=jitter_km, size=(len(km), copies - 1, 3)
    )
    yield x.reshape(-1, 3)


def build_20newsgroups(raw: Path):
    """Train set then test set of `fetch_20newsgroups_vectorized`."""
    import scipy.sparse as sp
    from sklearn.datasets import fetch_20newsgroups_vectorized

    train = fetch_20newsgroups_vectorized(subset="train", data_home=raw)
    test = fetch_20newsgroups_vectorized(subset="test", data_home=raw)
    return sp.vstack([train.data, test.data], format="csr")


def build_rcv1(raw: Path):
    from sklearn.datasets import fetch_rcv1

    return fetch_rcv1(data_home=raw).data.tocsr()


def _fetch_range(file_descriptor, start, stop, stopped, retries=20):
    """Bytes `start` to `stop` of the rows of DEEP1B, written at the same place
    of the local file."""
    position, failures = start, 0
    while position < stop:
        try:
            with _open(DEEP_URL, DEEP_HEADER + position, DEEP_HEADER + stop - 1) as r:
                if r.status != 206:
                    raise OSError(f"expected a partial content, got {r.status}")
                while position < stop:
                    if stopped.is_set():
                        return
                    block = r.read(min(BLOCK, stop - position))
                    if not block:
                        break
                    os.pwrite(file_descriptor, block, position)
                    position += len(block)
                    failures = 0
        except OSError:
            failures += 1
            if failures > retries:
                raise
            time.sleep(min(30, 2 * failures))


def download_deep(dataset: Dataset, directory: Path, jobs: int) -> None:
    """Extend `deep.f32` to the rows of `dataset`, with `jobs` connections.

    `deep.json` follows the rows that are complete, so a download that was
    stopped resumes from there.
    """
    with _open(DEEP_URL, 0, DEEP_HEADER - 1) as response:
        n_samples, n_features = np.frombuffer(response.read(), dtype="<i4")
    assert n_features == dataset.n_features and n_samples >= dataset.n_samples

    (path,) = data_files(dataset, directory)
    path.touch()
    meta = read_meta(dataset, directory)
    row_bytes = 4 * dataset.n_features
    have = meta.get("n_samples", 0) * row_bytes
    target = dataset.n_samples * row_bytes
    segments = [
        (start, min(start + DEEP_SEGMENT, target))
        for start in range(have, target, DEEP_SEGMENT)
    ]
    print(
        f"  downloading {(target - have) / 1e9:.1f} GB of {DEEP_URL} "
        f"with {jobs} connections",
        flush=True,
    )
    meta.update(n_samples=have // row_bytes, n_features=dataset.n_features)
    stopped = threading.Event()
    file_descriptor = os.open(path, os.O_RDWR)
    executor = ThreadPoolExecutor(jobs)
    try:
        futures = [
            executor.submit(_fetch_range, file_descriptor, start, stop, stopped)
            for start, stop in segments
        ]
        last_print = time.perf_counter()
        for (_, stop), future in zip(segments, futures):
            future.result()
            meta["n_samples"] = stop // row_bytes
            write_meta(dataset, directory, meta)
            if time.perf_counter() - last_print > 30:
                last_print = time.perf_counter()
                print(f"  {stop / 1e9:.1f} / {target / 1e9:.1f} GB", flush=True)
    finally:
        stopped.set()
        executor.shutdown(wait=True, cancel_futures=True)
        os.fsync(file_descriptor)
        os.close(file_descriptor)


BUILDERS = {
    "sift_128": build_sift_128,
    "glove_200": build_glove_200,
    "clip_512": build_clip_512,
    "susy": build_susy,
    "year_prediction_msd_standardized": build_year_prediction_msd_standardized,
    "road_network_km_x10": build_road_network_km_x10,
    "20newsgroups": build_20newsgroups,
    "rcv1": build_rcv1,
}


def checksum(dataset: Dataset, directory: Path, sizes=None) -> dict[int, str]:
    """sha256 of the files of `dataset`, one after the other.

    For a dense dataset, `sizes` gives numbers of rows: the result has the
    checksum of the first rows of the file, for each of them.
    """
    files = data_files(dataset, directory)
    digest = hashlib.sha256()
    if dataset.sparse:
        for path in files:
            with open(path, "rb") as file:
                while block := file.read(64 * BLOCK):
                    digest.update(block)
        return {dataset.n_samples: digest.hexdigest()}

    row_bytes = 4 * dataset.n_features
    checksums, position = {}, 0
    with open(files[0], "rb") as file:
        for n_samples in sorted(sizes or [dataset.n_samples]):
            while position < n_samples * row_bytes:
                block = file.read(min(64 * BLOCK, n_samples * row_bytes - position))
                if not block:
                    raise OSError(f"{files[0]} is shorter than {n_samples} rows")
                digest.update(block)
                position += len(block)
            checksums[n_samples] = digest.copy().hexdigest()
    return checksums


def record_checksums(dataset: Dataset, directory: Path) -> None:
    """Write in `<stem>.json` the checksum of each dataset that the files hold
    and that has none yet, and compare them to the reference ones."""
    meta = read_meta(dataset, directory)
    recorded = meta.setdefault("sha256", {})
    missing = [
        other
        for other in DATASETS.values()
        if other.stem == dataset.stem
        and other.n_samples <= meta["n_samples"]
        and other.name not in recorded
    ]
    if not missing:
        return
    names = ", ".join(other.name for other in missing)
    print(f"  computing the checksum of {names}", flush=True)
    checksums = checksum(dataset, directory, [other.n_samples for other in missing])
    for other in missing:
        recorded[other.name] = checksums[other.n_samples]
        reference = SHA256.get(other.name)
        if reference is not None and recorded[other.name] != reference:
            print(
                f"  Warning: {other.name} has the sha256 {recorded[other.name]}, "
                f"not the reference one ({reference}). Scores on this copy can't "
                "be compared to the ones of other machines.",
                file=sys.stderr,
            )
    write_meta(dataset, directory, meta)


def prepare(dataset: Dataset, directory: Path, keep_raw=False, jobs=16) -> None:
    """Download `dataset` to `directory`, unless it is there."""
    directory.mkdir(parents=True, exist_ok=True)
    if is_present(dataset, directory):
        print(f"{dataset.name}: already there")
    else:
        print(f"{dataset.name}:", flush=True)
        if dataset.stem == "deep":
            download_deep(dataset, directory, jobs)
        else:
            raw = directory / "raw" / dataset.name
            raw.mkdir(parents=True, exist_ok=True)
            built = BUILDERS[dataset.name](raw)
            if dataset.sparse:
                _write_sparse(dataset, directory, built)
            else:
                _write_dense(dataset, directory, built)
            write_meta(
                dataset,
                directory,
                {"n_samples": dataset.n_samples, "n_features": dataset.n_features},
            )
            if not keep_raw:
                shutil.rmtree(raw)
                if not any(raw.parent.iterdir()):
                    raw.parent.rmdir()
    record_checksums(dataset, directory)


def _size(dataset: Dataset, directory: Path) -> str:
    if dataset.sparse:
        if not is_present(dataset, directory):
            return ""
        size = sum(path.stat().st_size for path in data_files(dataset, directory))
    else:
        size = 4 * dataset.n_samples * dataset.n_features
    return f"{size / 1e9:.2f} GB" if size < 1e10 else f"{size / 1e9:.0f} GB"


def print_status(directory: Path) -> None:
    header = ("dataset", "n_samples", "n_features", "n_clusters", "size", "status")
    rows = []
    for dataset in DATASETS.values():
        local = read_meta(dataset, directory).get("sha256", {}).get(dataset.name)
        if not is_present(dataset, directory):
            status = "missing"
        elif dataset.name in SHA256 and local != SHA256[dataset.name]:
            status = "not the reference copy"
        else:
            status = "ok"
        rows.append(
            (
                dataset.name,
                f"{dataset.n_samples:,}",
                f"{dataset.n_features:,}",
                ", ".join(f"{k:,}" for k in dataset.n_clusters),
                _size(dataset, directory),
                status,
            )
        )
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    print(f"Datasets in {directory}:\n")
    for row in [header, *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download the datasets of the challenge. Without a dataset, "
        "list them and say which ones are there.",
    )
    parser.add_argument(
        "datasets", nargs="*", help="names of datasets, or 'all' (about 200 GB)"
    )
    parser.add_argument(
        "--data-dir",
        help="directory of the datasets (default: $INERTIA_CHALLENGE_DATA, "
        "else ~/.cache/the-inertia-challenge)",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="keep the downloaded files in <data-dir>/raw once converted",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=16,
        help="connections for DEEP (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    directory = data_dir(args.data_dir)
    if not args.datasets:
        print_status(directory)
        return
    names = list(DATASETS) if args.datasets == ["all"] else args.datasets
    try:
        datasets = [get_dataset(name) for name in names]
    except ValueError as error:
        sys.exit(f"Error: {error}")
    for dataset in datasets:
        prepare(dataset, directory, keep_raw=args.keep_raw, jobs=max(1, args.jobs))


if __name__ == "__main__":
    main()
