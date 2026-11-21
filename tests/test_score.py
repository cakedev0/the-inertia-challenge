import json

import numpy as np
import pytest
import scipy.sparse as sp

import challenge
from challenge import Dataset, Reader
import download
import score


def brute_force_inertia(X, labels):
    X = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)
    inertia = 0.0
    for cluster in np.unique(labels):
        rows = X[labels == cluster]
        inertia += ((rows - rows.mean(axis=0)) ** 2).sum()
    return inertia


@pytest.fixture
def make_dataset(tmp_path, monkeypatch):
    """Write X as a dataset of the challenge, in a temporary data directory."""

    def make(X, name="toy"):
        dataset = Dataset(name, X.shape[0], X.shape[1], (4,), "", sparse=sp.issparse(X))
        monkeypatch.setitem(challenge.DATASETS, name, dataset)
        if dataset.sparse:
            download._write_sparse(dataset, tmp_path, X)
        else:
            download._write_dense(dataset, tmp_path, [X[:3], X[3:]])
        challenge.write_meta(
            dataset, tmp_path, {"n_samples": X.shape[0], "n_features": X.shape[1]}
        )
        return dataset

    return make


def dense_data(n_samples=1000, n_features=7, seed=0):
    rng = np.random.default_rng(seed)
    # Far from the origin: the two terms of the score are then much larger
    # than the score itself.
    return (rng.standard_normal((n_samples, n_features)) + 100).astype(np.float32)


@pytest.mark.parametrize("chunk_bytes", [1, 28, 1000, 5000, 1 << 20])
@pytest.mark.parametrize("threads", [1, 3])
def test_dense_against_brute_force(make_dataset, tmp_path, chunk_bytes, threads):
    X = dense_data()
    reader = Reader(make_dataset(X), tmp_path)
    labels = np.random.default_rng(1).integers(0, 13, len(X)).astype(np.uint32)

    result = score.score_labels(
        reader, labels, 20, chunk_bytes=chunk_bytes, threads=threads
    )

    assert result.inertia == pytest.approx(brute_force_inertia(X, labels), rel=1e-9)
    assert result.mean_inertia == result.inertia / len(X)
    assert result.n_nonempty == 13
    assert result.n_clusters == 20


@pytest.mark.parametrize("chunk_bytes", [1, 100, 3000, 1 << 20])
def test_sparse_against_brute_force(make_dataset, tmp_path, chunk_bytes):
    X = sp.random(300, 50, density=0.1, format="lil", dtype=np.float32, random_state=0)
    # A row without any stored value.
    X[5] = 0
    X = X.tocsr()
    reader = Reader(make_dataset(X), tmp_path)
    labels = np.random.default_rng(1).integers(0, 9, X.shape[0]).astype(np.uint32)

    result = score.score_labels(reader, labels, 9, chunk_bytes=chunk_bytes)

    assert result.inertia == pytest.approx(brute_force_inertia(X, labels), rel=1e-9)
    assert result.n_nonempty == 9


def test_chunks_cover_the_rows(make_dataset, tmp_path):
    dense = Reader(make_dataset(dense_data(), "dense"), tmp_path)
    X = sp.random(300, 50, density=0.1, format="csr", dtype=np.float32, random_state=0)
    sparse = Reader(make_dataset(X, "sparse"), tmp_path)
    for reader, chunk_bytes in [(dense, 300), (sparse, 300), (dense, 1 << 30)]:
        ranges = reader.chunks(chunk_bytes)
        assert ranges[0][0] == 0 and ranges[-1][1] == reader.shape[0]
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
        assert all(start < stop for start, stop in ranges)
    assert max(stop - start for start, stop in dense.chunks(300)) == 300 // (4 * 7)
    np.testing.assert_array_equal(sparse.read(10, 20).toarray(), X[10:20].toarray())


def test_one_sample_per_cluster(make_dataset, tmp_path):
    X = dense_data(n_samples=50)
    reader = Reader(make_dataset(X), tmp_path)
    labels = np.random.default_rng(0).permutation(50).astype(np.uint32)
    assert score.score_labels(reader, labels, 50).mean_inertia < 1e-8


def test_prefix_of_shared_files(make_dataset, tmp_path, monkeypatch):
    # As the DEEP datasets: a dataset that is the first rows of a longer file.
    X = dense_data()
    make_dataset(X, "long")
    short = Dataset("short", 400, X.shape[1], (4,), "", stem="long")
    monkeypatch.setitem(challenge.DATASETS, "short", short)
    labels = np.random.default_rng(1).integers(0, 5, 400).astype(np.uint32)

    result = score.score_labels(Reader(short, tmp_path), labels, 5, chunk_bytes=999)

    assert result.inertia == pytest.approx(
        brute_force_inertia(X[:400], labels), rel=1e-9
    )

    download.record_checksums(short, tmp_path)
    recorded = json.loads((tmp_path / "long.json").read_text())["sha256"]
    assert recorded["long"] == download.checksum(short, tmp_path, [1000])[1000]
    assert recorded["short"] != recorded["long"]


def test_missing_dataset(make_dataset, tmp_path):
    dataset = make_dataset(dense_data())
    longer = Dataset("longer", 2000, 7, (4,), "", stem=dataset.stem)
    with pytest.raises(challenge.DatasetNotFound, match="python download.py longer"):
        Reader(longer, tmp_path)
    (tmp_path / "toy.json").unlink()
    with pytest.raises(challenge.DatasetNotFound):
        Reader(dataset, tmp_path)


@pytest.mark.parametrize(
    "labels, match",
    [
        (np.zeros(1000, dtype=np.int32), "uint32 array, got int32"),
        (np.zeros(999, dtype=np.uint32), "shape"),
        (np.zeros((1000, 1), dtype=np.uint32), "shape"),
        (np.full(1000, 4, dtype=np.uint32), "got a label 4"),
    ],
)
def test_invalid_labels(make_dataset, tmp_path, labels, match):
    reader = Reader(make_dataset(dense_data()), tmp_path)
    with pytest.raises(ValueError, match=match):
        score.score_labels(reader, labels, 4)


def test_command_line(make_dataset, tmp_path, capsys):
    X = dense_data()
    make_dataset(X)
    rng = np.random.default_rng(1)
    labels = {k: rng.integers(0, k, len(X)).astype(np.uint32) for k in (4, 16)}
    path = tmp_path / "submission.npz"
    np.savez(path, **{str(k): array for k, array in labels.items()})

    score.main(["toy", str(path), "--data-dir", str(tmp_path), "--json"])
    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert [result["n_clusters"] for result in results] == [4, 16]
    for result in results:
        expected = brute_force_inertia(X, labels[result["n_clusters"]])
        assert result["inertia"] == pytest.approx(expected, rel=1e-9)

    score.main(["toy", str(path), "--data-dir", str(tmp_path), "--chunk-size", "0.01"])
    assert "inertia / n_samples" in capsys.readouterr().out

    np.savez(path, centers=labels[4])
    with pytest.raises(SystemExit, match="numbers of clusters, got 'centers'"):
        score.main(["toy", str(path), "--data-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="Unknown dataset"):
        score.main(["nope", str(path), "--data-dir", str(tmp_path)])
