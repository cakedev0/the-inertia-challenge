# the-inertia-challenge

A scoreboard for k-means: run your algorithm on your side, submit the labels,
and this repository scores them. The score is the inertia of the labels: the
sum of the squared distances of the samples to the mean of their cluster.

For now the repository has the datasets and the scoring script. The scoreboard
itself is not there yet.

## Tasks

A task is a dataset and a number of clusters.

| dataset | n_samples | n_features | n_clusters | |
| --- | --- | --- | --- | --- |
| `sift_128` | 1,000,000 | 128 | 64, 256, 1,024 | SIFT descriptors (train set of ann-benchmarks' `sift-128-euclidean`) |
| `glove_200` | 1,183,514 | 200 | 64, 256, 1,024 | GloVe word vectors, not normalized (train set of ann-benchmarks' `glove-200-angular`) |
| `clip_512` | 1,000,000 | 512 | 64, 256, 1,024 | CLIP ViT-B/32 image embeddings ([Qdrant/wolt-food-clip-ViT-B-32-embeddings](https://huggingface.co/datasets/Qdrant/wolt-food-clip-ViT-B-32-embeddings)), L2-normalized, ~23% of exact duplicates |
| `susy` | 5,000,000 | 18 | 64, 256, 1,024 | [SUSY](https://archive.ics.uci.edu/dataset/279/susy) (UCI), raw features |
| `year_prediction_msd_standardized` | 515,345 | 90 | 64, 256, 1,024 | [YearPredictionMSD](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd) (UCI), standardized |
| `road_network_km_x10` | 4,348,740 | 3 | 64, 256, 1,024 | [3D Road Network](https://archive.ics.uci.edu/dataset/246/3d+road+network+north+jutland+denmark) (UCI) in kilometers, each point with 9 copies jittered by 5 m |
| `20newsgroups` | 18,846 | 130,107 | 64, 256, 1,024 | sparse, L2-normalized token counts (`fetch_20newsgroups_vectorized`) |
| `rcv1` | 804,414 | 47,236 | 64, 256, 1,024 | sparse, cosine-normalized log TF-IDF (`fetch_rcv1`) |
| `deep_1M` | 1,000,000 | 96 | 10,000 | first rows of [DEEP1B](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search) |
| `deep_5M` | 5,000,000 | 96 | 50,000 | |
| `deep_20M` | 20,000,000 | 96 | 20,000 | |
| `deep_100M` | 100,000,000 | 96 | 10,000 and 1,000,000 | |
| `deep_500M` | 500,000,000 | 96 | 500,000 and 5,000,000 | |

The first eight datasets and their numbers of clusters are the ones of the
k-means benchmark of
[scikit-learn-benchmarks#187](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/187),
with the same loaders. The DEEP tasks are the ones of the user guide of
[sklearn-contrib-emeans](https://github.com/cakedev0/sklearn-contrib-emeans).

## Install

```bash
git clone https://github.com/cakedev0/the-inertia-challenge
cd the-inertia-challenge
pip install -r requirements.txt
```

`score.py` only needs NumPy and SciPy. `download.py` also needs h5py, pandas,
pyarrow and scikit-learn.

## Download the datasets

```bash
python download.py                      # list the datasets and which ones are there
python download.py sift_128 deep_1M     # download these two
python download.py all                  # everything: about 200 GB
```

The datasets go to `--data-dir`, else to `$INERTIA_CHALLENGE_DATA`, else to
`~/.cache/the-inertia-challenge`. A dataset that is already there is not
downloaded again, and a download of DEEP that was stopped resumes where it
was. The DEEP datasets are the first rows of one file: `deep_500M` holds the
four others.

All the values are float32. A dense dataset is one file of rows without a
header (`<name>.f32`), so `np.memmap` and `np.fromfile` read it. A sparse one
is a CSR matrix in three files. `challenge.Reader` reads both by ranges of
rows:

```python
from challenge import Reader, get_dataset

reader = Reader(get_dataset("sift_128"))
X = reader.read(0, reader.shape[0])  # array, or scipy.sparse.csr_matrix
```

`download.py` checks the sha256 of what it wrote against the reference copy,
and warns when they differ.

## Submission

A submission is a `.npz` file for one dataset. Each key is a number of clusters
`k`, and its array has the label of each sample: dtype uint32, shape
`(n_samples,)`, values in `[0, k)`.

```python
import numpy as np
from sklearn.cluster import KMeans

from challenge import Reader, get_dataset

reader = Reader(get_dataset("sift_128"))
X = reader.read(0, reader.shape[0])
submission = {}
for k in (64, 256, 1024):
    labels = KMeans(n_clusters=k, n_init=1, max_iter=10).fit(X).labels_
    submission[str(k)] = labels.astype(np.uint32)
np.savez("sift_128.npz", **submission)
```

## Score

```bash
python score.py sift_128 sift_128.npz
```

```
dataset   n_clusters  non-empty  inertia          inertia / n_samples
sift_128  64          64         6.622908544e+10  6.622908544e+04
sift_128  256         256        5.773387353e+10  5.773387353e+04
sift_128  1024        1024       5.076856687e+10  5.076856687e+04
Scored in 0.7 s.
```

Options: `--chunk-size` (in MB, 100 by default), `--threads` (4 by default),
`--data-dir`, and `--json` for one JSON object per line. From Python:
`score.score_submission("sift_128", "sift_128.npz")`.

### How the score is computed

The score only depends on the labels. Each cluster gets its mean as center,
which is the center with the lowest inertia for these labels. So labels from
an approximate assignment are scored as the partition they are, and no
centers are submitted.

The dataset is read once, by chunks of at most `--chunk-size`, and

```
inertia = sum_i |x_i|^2 - sum_j |s_j|^2 / n_j
```

where `s_j` is the sum of the samples of cluster `j` and `n_j` their number.
All the sums are in float64. The chunks are added in order, so the result does
not depend on `--threads`. It depends on `--chunk-size` by rounding only: on
`deep_1M` with 10,000 clusters, chunks of 0.01 MB to 1 GB give the same
inertia up to a relative difference of 1e-13, and so does a two-pass float64
computation (the means, then the distances).

Memory: the labels, the sums (`8 * n_clusters * n_features` bytes) and about 5
times the chunk size per thread.
