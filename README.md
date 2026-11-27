<img src="docs/logo.svg" alt="" width="96" align="right">

# the-inertia-challenge

A scoreboard for k-means: run your algorithm on your side, submit the labels,
and compare your inertia to the others on
[the dashboard](https://cakedev0.github.io/the-inertia-challenge/).

## Quick start

```bash
pip install inertia-client "inertia-scorer[download]" scikit-learn
inertia-download year_prediction_msd_standardized  # 185 MB
inertia-client login  # once, with GitHub
```

```python
from sklearn.cluster import KMeans

import inertia_client
from inertia_scorer import load_dataset

X = load_dataset("year_prediction_msd_standardized")  # float32, (515345, 90)
labels = KMeans(5, init="random", n_init=1).fit_predict(X)
inertia_client.submit(
    "year_prediction_msd_standardized", {5: labels}, submit_only_improvement=True
)
```

You can submit once per day per dataset. `submit_only_improvement=True` scores
the labels on your side first, and only sends the numbers of clusters that beat
your best score and would be published (see the rules).

## Dashboard

[cakedev0.github.io/the-inertia-challenge](https://cakedev0.github.io/the-inertia-challenge/)
lists the scores of each task, with the time, hardware and tags of each
submission. The leader board ranks the users by their score: how far their
best inertia is above the best one, in %, averaged over the tasks that have a
score, with the weights `sqrt(n_samples * n_clusters)`. A task without a score
of the user counts as 10% above the best. It also shows the same mean with no
weights, over the tasks of the user only.

## Rules

- A submission is the label of each sample of a dataset, for one or more of its
  numbers of clusters. No centers: the score is the inertia of the labels, the
  sum of the squared distances of the samples to the mean of their cluster.
- A score is published if it is less than 10% above the best published one.
- One submission per day and per dataset.
- The time, hardware and tags are what you say about the submission. They are
  shown as you give them, not checked.

`inertia-score` computes the same score on your side. See
[inertia-scorer](packages/inertia-scorer/README.md) for the datasets and the
scoring, and [inertia-client](packages/inertia-client/README.md) for the
submissions.

## Tasks

A task is a dataset and a number of clusters. The numbers of clusters give
clusters of 100, 1,000, 10,000 and 100,000 samples on average (at least 5).

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

The first eight datasets are the ones of the k-means benchmark of
[scikit-learn-benchmarks#187](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/187).
The DEEP datasets are the first rows of one file, as in the user guide of
[sklearn-contrib-emeans](https://github.com/cakedev0/sklearn-contrib-emeans).

BSD 3-Clause license, see [LICENSE](LICENSE).
