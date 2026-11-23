# the-inertia-challenge

A scoreboard for k-means: run your algorithm on your side, submit the labels,
and this repository scores them. The score is the inertia of the labels: the
sum of the squared distances of the samples to the mean of their cluster.

The repository has the datasets, the scoring, the service that scores the
submissions and the page of the scoreboard. The server that takes the
submissions is not there yet.

## Quick start

The easiest task is `year_prediction_msd_standardized` with 5 clusters (515,345
samples, 90 features, 185 MB). Install the packages, download the dataset, run
k-means with a random initialization, and submit the labels:

```bash
pip install inertia-client "inertia-scorer[download]" scikit-learn
inertia-download year_prediction_msd_standardized
inertia-client login  # once, with GitHub
```

```python
from sklearn.cluster import KMeans

import inertia_client
from inertia_scorer import load_dataset

X = load_dataset("year_prediction_msd_standardized")  # float32, (515345, 90)

labels = KMeans(5, init="random", n_init=1).fit_predict(X)

inertia_client.submit("year_prediction_msd_standardized", {5: labels})
```

The server takes one submission per day and per dataset. To only send the
numbers of clusters that beat your best score on the scoreboard (the labels are
scored on your side, so the dataset must be downloaded), add
`submit_only_improvement=True`, or `--submit-only-improvement` on the command
line.

## Tasks

A task is a dataset and a number of clusters. The numbers of clusters of a
dataset are the ones that give clusters of 100, 1,000, 10,000 and 100,000
samples on average: `n_samples / size`, rounded, when it is at least 5.

| dataset | n_samples | n_features | n_clusters | |
| --- | --- | --- | --- | --- |
| `sift_128` | 1,000,000 | 128 | 10, 100, 1,000, 10,000 | SIFT descriptors (train set of ann-benchmarks' `sift-128-euclidean`) |
| `glove_200` | 1,183,514 | 200 | 12, 118, 1,184, 11,835 | GloVe word vectors, not normalized (train set of ann-benchmarks' `glove-200-angular`) |
| `clip_512` | 1,000,000 | 512 | 10, 100, 1,000, 10,000 | CLIP ViT-B/32 image embeddings ([Qdrant/wolt-food-clip-ViT-B-32-embeddings](https://huggingface.co/datasets/Qdrant/wolt-food-clip-ViT-B-32-embeddings)), L2-normalized, ~23% of exact duplicates |
| `susy` | 5,000,000 | 18 | 50, 500, 5,000, 50,000 | [SUSY](https://archive.ics.uci.edu/dataset/279/susy) (UCI), raw features |
| `year_prediction_msd_standardized` | 515,345 | 90 | 5, 52, 515, 5,153 | [YearPredictionMSD](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd) (UCI), standardized |
| `road_network_km_x10` | 4,348,740 | 3 | 43, 435, 4,349, 43,487 | [3D Road Network](https://archive.ics.uci.edu/dataset/246/3d+road+network+north+jutland+denmark) (UCI) in kilometers, each point with 9 copies jittered by 5 m |
| `20newsgroups` | 18,846 | 130,107 | 19, 188 | sparse, L2-normalized token counts (`fetch_20newsgroups_vectorized`) |
| `rcv1` | 804,414 | 47,236 | 8, 80, 804, 8,044 | sparse, cosine-normalized log TF-IDF (`fetch_rcv1`) |
| `deep_1M` | 1,000,000 | 96 | 10, 100, 1,000, 10,000 | first rows of [DEEP1B](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search) |
| `deep_5M` | 5,000,000 | 96 | 50, 500, 5,000, 50,000 | |
| `deep_20M` | 20,000,000 | 96 | 200, 2,000, 20,000, 200,000 | |
| `deep_100M` | 100,000,000 | 96 | 1,000, 10,000, 100,000, 1,000,000 | |
| `deep_500M` | 500,000,000 | 96 | 5,000, 50,000, 500,000, 5,000,000 | |

The first eight datasets are the ones of the k-means benchmark of
[scikit-learn-benchmarks#187](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/187),
with the same loaders. The DEEP datasets are the ones of the user guide of
[sklearn-contrib-emeans](https://github.com/cakedev0/sklearn-contrib-emeans).

## Install

```bash
git clone https://github.com/cakedev0/the-inertia-challenge
cd the-inertia-challenge
pip install -e "packages/inertia-scorer[download]" -e packages/inertia-client
```

The repository has two packages, in `packages/`:

- `inertia-scorer`: the datasets and the scoring, with the commands
  `inertia-download`, `inertia-score` and `inertia-listener`. It only needs NumPy and SciPy, and a C
  compiler to install it from the sources (its loops are in Cython). The
  `download` extra adds what `inertia-download` needs: h5py, pandas, pyarrow
  and scikit-learn.
- `inertia-client`: the client of the online scoreboard: `inertia-client` and
  the `inertia_client` Python package.

## Download the datasets

```bash
inertia-download                      # list the datasets and which ones are there
inertia-download sift_128 deep_1M     # download these two
inertia-download all                  # everything: about 200 GB
```

The datasets go to `--data-dir`, else to `$INERTIA_CHALLENGE_DATA`, else to
`~/.cache/the-inertia-challenge`. A dataset that is already there is not
downloaded again, and a download of DEEP that was stopped resumes where it
was. The DEEP datasets are the first rows of one file: `deep_500M` holds the
four others.

All the values are float32. A dense dataset is one file of rows without a
header (`<name>.f32`), so `np.memmap` and `np.fromfile` read it. A sparse one
is a CSR matrix in three files. `inertia_scorer.load_dataset` reads all the
rows of either, and `inertia_scorer.Reader` reads them by ranges, for the
datasets that don't fit in memory:

```python
from inertia_scorer import Reader, dataset_info, load_dataset

X = load_dataset("sift_128")  # array, or scipy.sparse.csr_matrix

reader = Reader(dataset_info("deep_100M"))
X = reader.read(0, 1_000_000)  # the first million rows
```

`inertia-download` checks the sha256 of what it wrote against the reference copy,
and warns when they differ.

## Submission

A submission is a `.npz` file for one dataset. Each key is a number of clusters
`k`, and its array has the label of each sample: dtype uint32, shape
`(n_samples,)`, values in `[0, k)`.

```python
import numpy as np
from sklearn.cluster import KMeans

from inertia_scorer import dataset_info, load_dataset

X = load_dataset("sift_128")
submission = {}
for k in dataset_info("sift_128").n_clusters:  # 10, 100, 1000, 10000
    labels = KMeans(n_clusters=k, n_init=1, max_iter=10).fit(X).labels_
    submission[str(k)] = labels.astype(np.uint32)
np.savez("sift_128.npz", **submission)
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
`inertia_scorer.score_submission("sift_128", "sift_128.npz")`.

### How the score is computed

The score only depends on the labels. Each cluster gets its mean as center,
which is the center with the lowest inertia for these labels. So labels from
an approximate assignment are scored as the partition they are, and no
centers are submitted.

The dataset is read once, by chunks of at most `--chunk-size`, for all the
numbers of clusters of the submission, and

```
inertia = sum_i |x_i|^2 - sum_j |s_j|^2 / n_j
```

where `s_j` is the sum of the samples of cluster `j` and `n_j` their number.
All the sums are in float64. Each sample is added to the sum of its cluster in
the order of the dataset, and each cluster belongs to one thread, so the result
does not depend on `--threads`. It depends on `--chunk-size` by the rounding of
`sum_i |x_i|^2` only: on `deep_1M` with 10,000 clusters, chunks of 0.01 MB to
1 GB give the same inertia up to a relative difference of 1e-14, and so does a
two-pass float64 computation (the means, then the distances).

Memory: the sums (`8 * n_clusters * n_features` bytes, for all the numbers of
clusters of the submission) and about one chunk per thread. The labels are
read from the file as they are needed.

## Listener

`inertia-listener` is the service behind the scoreboard. It takes the
submissions of a queue, scores them, and publishes the results in a git
repository:

```bash
export INERTIA_QUEUE_TOKEN=...
inertia-listener path/to/results --queue-url https://the-server
```

A submission is from one user and for one dataset, with labels for some or all
of its numbers of clusters. The user can say how long the clustering took
(`time_s`, in seconds) and on what hardware, and add tags: free text as
`{"algorithm": "Lloyd"}`, of at most 4 kB. All of it is published with the
scores, as the user gave it: the listener can't check it. For each of them, with `best` the lowest inertia published so far:

- below `1.1 * best`, or no score yet: the score is published, in
  `scores/<dataset>.jsonl`;
- else nothing is published.

The labels are not kept.

The listener does not depend on what the queue is: `listener.listen` takes any
object with the methods `get` and `done` of `listener.Queue`. There are two:
`listener.HttpQueue` asks the server of the scoreboard over HTTPS
(`--queue-url`), and `listener.DirectoryQueue` is a directory (`--queue-dir`). `path/to/results` is a
clone that only the listener writes to.

## Dashboard

`docs/index.html` is the page of the scoreboard, for GitHub Pages. It reads
`scores/<dataset>.jsonl` on the `results` branch of this repository, where the
listener pushes, so it needs no build: a new score shows up when the page is
loaded again. `docs/datasets.json` is the list of the datasets, written from
`inertia_scorer.challenge.DATASETS` (a test checks that it is up to date).

The leader board is above the tabs of the datasets. It ranks the users by the mean, over the
tasks that have a score, of how far their best inertia is above the best one,
in %. The weight of a task is `sqrt(n_samples * n_clusters)`, and a task
without a score of the user counts as 10% above the best.

To try it with other scores, serve a directory that has the two files of
`docs` and a `scores` directory, and open `index.html?scores=scores`:

```bash
python -m http.server --bind 127.0.0.1
```

## License

BSD 3-Clause, as scikit-learn. See [LICENSE](LICENSE).
