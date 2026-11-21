"""Datasets of the inertia challenge: the list, their files and how to read them.

A dense dataset is one file of float32 rows (`<stem>.f32`, C order, little
endian, no header). A sparse one is a CSR matrix in three files
(`<stem>.data.f32`, `<stem>.indices.i32`, `<stem>.indptr.i64`). In both cases
`<stem>.json` is written last, and says how many rows are there.

The DEEP datasets are the first rows of the same file, `deep.f32`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

DATA_DIR_VARIABLE = "INERTIA_CHALLENGE_DATA"
DEFAULT_DATA_DIR = "~/.cache/the-inertia-challenge"


@dataclass(frozen=True)
class Dataset:
    name: str
    n_samples: int
    n_features: int
    n_clusters: tuple[int, ...]
    description: str
    sparse: bool = False
    # Datasets with the same stem share their files.
    stem: str = ""

    def __post_init__(self):
        if not self.stem:
            object.__setattr__(self, "stem", self.name)


_PARETO_N_CLUSTERS = (64, 256, 1024)

DATASETS = {
    dataset.name: dataset
    for dataset in [
        Dataset(
            "sift_128",
            1_000_000,
            128,
            _PARETO_N_CLUSTERS,
            "SIFT descriptors (train set of ann-benchmarks' sift-128-euclidean)",
        ),
        Dataset(
            "glove_200",
            1_183_514,
            200,
            _PARETO_N_CLUSTERS,
            "GloVe word vectors, not normalized (train set of ann-benchmarks' "
            "glove-200-angular)",
        ),
        Dataset(
            "clip_512",
            1_000_000,
            512,
            _PARETO_N_CLUSTERS,
            "CLIP ViT-B/32 image embeddings (Qdrant/wolt-food-clip-ViT-B-32-"
            "embeddings), L2-normalized, ~23% of exact duplicates",
        ),
        Dataset(
            "susy",
            5_000_000,
            18,
            _PARETO_N_CLUSTERS,
            "SUSY (UCI), simulated particle collisions, raw features",
        ),
        Dataset(
            "year_prediction_msd_standardized",
            515_345,
            90,
            _PARETO_N_CLUSTERS,
            "YearPredictionMSD (UCI), audio features, standardized",
        ),
        Dataset(
            "road_network_km_x10",
            4_348_740,
            3,
            _PARETO_N_CLUSTERS,
            "3D Road Network (UCI) in kilometers, each point with 9 copies "
            "jittered by 5 m",
        ),
        Dataset(
            "20newsgroups",
            18_846,
            130_107,
            _PARETO_N_CLUSTERS,
            "20 Newsgroups, L2-normalized token counts",
            sparse=True,
        ),
        Dataset(
            "rcv1",
            804_414,
            47_236,
            _PARETO_N_CLUSTERS,
            "RCV1-v2 news stories, cosine-normalized log TF-IDF",
            sparse=True,
        ),
        Dataset(
            "deep_1M", 1_000_000, 96, (10_000,), "first 1M rows of DEEP1B", stem="deep"
        ),
        Dataset(
            "deep_5M", 5_000_000, 96, (50_000,), "first 5M rows of DEEP1B", stem="deep"
        ),
        Dataset(
            "deep_20M",
            20_000_000,
            96,
            (20_000,),
            "first 20M rows of DEEP1B",
            stem="deep",
        ),
        Dataset(
            "deep_100M",
            100_000_000,
            96,
            (10_000, 1_000_000),
            "first 100M rows of DEEP1B",
            stem="deep",
        ),
        Dataset(
            "deep_500M",
            500_000_000,
            96,
            (500_000, 5_000_000),
            "first 500M rows of DEEP1B",
            stem="deep",
        ),
    ]
}

# sha256 of the files written by `download.py` (see `download.checksum`). A copy
# with another checksum has other rows, or rows that differ by rounding: its
# scores can't be compared to the ones of the reference copy.
SHA256 = {
    "sift_128": "ac8e11ed111b09e882bcf0ef1a77ebdd36c89f495bce8fd596603f69c35148e6",
    "glove_200": "da82748627405673d89db7b41bbe7ff2c23ed1e3efd988237b15b6e6dcb3c00f",
    "clip_512": "a27fa70aa15532b0a067a6111b1b6e8be9a54664e977c6b4bd13bfebd1d4de08",
    "susy": "b37595b167e0915dfec2a954d0c781630c36491ca077b9fc1c2c68399a54f5f7",
    "year_prediction_msd_standardized": (
        "84d3062735fe3df40d6b41388a36e9c223d42ae756d5283e15557efe8cc07f3f"
    ),
    "road_network_km_x10": (
        "c0db247b04bc9ee2afea0667ea5f6a852904f14935cf00b65f88ccf6d1bb85cf"
    ),
    "20newsgroups": "be771135f62fef7f2ddd5ea73c46a12fa97523ea126e0b6e26ad63e56a496bd3",
    "rcv1": "e91c6eec5e67eef418b70aca970ca76cbcd276232f73b5611036efff23c4f3ff",
    "deep_1M": "8dca9378a247cbcd1f781fffa10b5b8b542fd4a7a470f9063fbdb9f4f2a9ede2",
    "deep_5M": "1f728feec2efa803b2848f5c4ed3fd547b61955dda5635398cbe2d2093de435a",
    "deep_20M": "98b8979c174ae97540bd336ec9ebc47792b956992c0c991c76c6dfd7c020648d",
    "deep_100M": "f8d787b68f1548414468c07ce49670db8a747e367c7d23969148c76da44f30c8",
    "deep_500M": "6600cbfe75f43a2421f11c5214705ce3a65c511a0062e68e195983fe8fce006f",
}


class DatasetNotFound(FileNotFoundError):
    pass


def data_dir(path=None) -> Path:
    """The directory of the datasets: `path`, else $INERTIA_CHALLENGE_DATA, else
    ~/.cache/the-inertia-challenge."""
    path = path or os.environ.get(DATA_DIR_VARIABLE) or DEFAULT_DATA_DIR
    return Path(path).expanduser()


def get_dataset(name: str) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Datasets: {', '.join(DATASETS)}")
    return DATASETS[name]


def data_files(dataset: Dataset, directory: Path) -> list[Path]:
    if dataset.sparse:
        suffixes = ("data.f32", "indices.i32", "indptr.i64")
    else:
        suffixes = ("f32",)
    return [directory / f"{dataset.stem}.{suffix}" for suffix in suffixes]


def read_meta(dataset: Dataset, directory: Path) -> dict:
    """The content of `<stem>.json`, or {} if the files are missing or too short."""
    path = directory / f"{dataset.stem}.json"
    files = data_files(dataset, directory)
    if not path.is_file() or not all(file.is_file() for file in files):
        return {}
    meta = json.loads(path.read_text())
    if meta.get("n_features") != dataset.n_features:
        return {}
    if not dataset.sparse:
        row_bytes = 4 * dataset.n_features
        if files[0].stat().st_size < meta.get("n_samples", 0) * row_bytes:
            return {}
    return meta


def write_meta(dataset: Dataset, directory: Path, meta: dict) -> None:
    path = directory / f"{dataset.stem}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=1) + "\n")
    os.replace(tmp, path)


def is_present(dataset: Dataset, directory: Path) -> bool:
    return read_meta(dataset, directory).get("n_samples", 0) >= dataset.n_samples


class Reader:
    """Reads the rows of a dataset by ranges, as float32.

    `read` opens the files at each call, so it can be called from several
    threads.
    """

    def __init__(self, dataset: Dataset, directory=None):
        self.dataset = dataset
        self.directory = data_dir(directory)
        self.meta = read_meta(dataset, self.directory)
        if self.meta.get("n_samples", 0) < dataset.n_samples:
            raise DatasetNotFound(
                f"{dataset.name} is not in {self.directory}. "
                f"Run: python download.py {dataset.name}"
            )
        self.shape = (dataset.n_samples, dataset.n_features)
        self.files = data_files(dataset, self.directory)
        if dataset.sparse:
            self.indptr = np.fromfile(
                self.files[2], dtype="<i8", count=dataset.n_samples + 1
            )

    def chunks(self, chunk_bytes: int) -> list[tuple[int, int]]:
        """Ranges of rows (start, stop) of at most `chunk_bytes` each on disk, and
        of at least one row."""
        n_samples, n_features = self.shape
        if not self.dataset.sparse:
            step = max(1, chunk_bytes // (4 * n_features))
            return [
                (start, min(start + step, n_samples))
                for start in range(0, n_samples, step)
            ]
        # 8 bytes per stored value: a float32 and its int32 column.
        max_nnz = max(1, chunk_bytes // 8)
        ranges, start = [], 0
        while start < n_samples:
            stop = np.searchsorted(self.indptr, self.indptr[start] + max_nnz, "right")
            stop = min(n_samples, max(int(stop) - 1, start + 1))
            ranges.append((start, stop))
            start = stop
        return ranges

    def read(self, start: int, stop: int):
        """Rows `start` to `stop`: an array, or a CSR matrix for a sparse dataset."""
        n_features = self.shape[1]
        if not self.dataset.sparse:
            rows = np.fromfile(
                self.files[0],
                dtype="<f4",
                count=(stop - start) * n_features,
                offset=4 * start * n_features,
            )
            return rows.reshape(stop - start, n_features)

        import scipy.sparse as sp

        indptr = self.indptr[start : stop + 1]
        first, nnz = int(indptr[0]), int(indptr[-1] - indptr[0])
        data = np.fromfile(self.files[0], dtype="<f4", count=nnz, offset=4 * first)
        indices = np.fromfile(self.files[1], dtype="<i4", count=nnz, offset=4 * first)
        return sp.csr_matrix(
            (data, indices, indptr - first), shape=(stop - start, n_features)
        )
