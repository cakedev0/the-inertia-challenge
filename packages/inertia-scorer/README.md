# inertia-scorer

The datasets of [the inertia challenge](https://github.com/cakedev0/the-inertia-challenge)
and the scoring of a submission: `inertia-download`, `inertia-score`,
`inertia-listener` and the `inertia_scorer` Python package.

```bash
pip install "inertia-scorer[download]"
```

It only needs NumPy and SciPy. The `download` extra adds what `inertia-download`
needs: h5py, pandas, pyarrow and scikit-learn.

## Datasets

```bash
inertia-download                      # list the datasets and which ones are there
inertia-download sift_128 deep_1M     # download these two
inertia-download all                  # everything: about 200 GB
```

The datasets go to `--data-dir`, else to `$INERTIA_CHALLENGE_DATA`, else to
`~/.cache/the-inertia-challenge`. A dataset that is already there is not
downloaded again, and a download of DEEP that was stopped resumes where it
was. `deep_500M` holds the four other DEEP datasets. `inertia-download` checks
the sha256 of what it wrote against the reference copy, and warns when they
differ.

All the values are float32. A dense dataset is one file of rows without a
header (`<name>.f32`), so `np.memmap` and `np.fromfile` read it. A sparse one
is a CSR matrix in three files. `load_dataset` reads all the rows, and `Reader`
reads them by ranges, for the datasets that don't fit in memory:

```python
from inertia_scorer import Reader, dataset_info, load_dataset

X = load_dataset("sift_128")  # array, or scipy.sparse.csr_matrix

reader = Reader(dataset_info("deep_100M"))
X = reader.read(0, 1_000_000)  # the first million rows
```

## Score

```bash
inertia-score sift_128 sift_128.npz
```

```
dataset   n_clusters  non-empty  inertia          inertia / n_samples
sift_128  10          10         8.149170491e+10  8.149170491e+04
sift_128  100         100        6.319282816e+10  6.319282816e+04
sift_128  1000        1000       5.089552628e+10  5.089552628e+04
sift_128  10000       10000      4.061009915e+10  4.061009915e+04
Scored in 1.0 s.
```

Options: `--chunk-size` (in MB, 100 by default), `--threads` (4 by default),
`--data-dir`, and `--json` for one JSON object per line. From Python:
`inertia_scorer.score_submission("sift_128", "sift_128.npz")`. The format of
the `.npz` file is in [inertia-client](../inertia-client/README.md).

Each cluster gets its mean as center, the center with the lowest inertia for
these labels. So labels from an approximate assignment are scored as the
partition they are. The dataset is read once, by chunks, for all the numbers of
clusters of the submission, and

```
inertia = sum_i |x_i|^2 - sum_j |s_j|^2 / n_j
```

where `s_j` is the sum of the samples of cluster `j` and `n_j` their number.
All the sums are in float64, and each cluster belongs to one thread, so the
result does not depend on `--threads`. On `deep_1M` with 10,000 clusters,
chunks of 0.01 MB to 1 GB give the same inertia up to a relative difference of
1e-14, and so does a two-pass float64 computation (the means, then the
distances).

Memory: the sums (`8 * n_clusters * n_features` bytes, for all the numbers of
clusters of the submission) and about one chunk per thread.

## Listener

`inertia-listener` is the service behind the scoreboard. It takes the
submissions of a queue, scores them, and publishes the scores in
`scores/<dataset>.jsonl` of a git repository (the `results` branch of the
challenge), when they are less than 10% above the best one. The labels are not
kept.

```bash
export INERTIA_QUEUE_TOKEN=...
inertia-listener path/to/results --queue-url https://the-server
```

`listener.listen` takes any object with the methods `get` and `done` of
`listener.Queue`: `listener.HttpQueue` asks the server over HTTPS
(`--queue-url`), and `listener.DirectoryQueue` is a directory (`--queue-dir`).

The dashboard, `docs/index.html` in the repository, reads these files with no
build step. To try it with other scores, serve a directory with the two files of
`docs` and a `scores` directory (`python -m http.server --bind 127.0.0.1`), and
open `index.html?scores=scores`.

## Development

From a clone, with a C compiler (the loops are in Cython):

```bash
pip install -e "packages/inertia-scorer[download]" -e packages/inertia-client
```
