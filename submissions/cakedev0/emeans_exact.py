"""Submissions of EMeans on the full data, for the cases it fits in the budget.

A case is a dataset and a number of clusters. For each case, EMeans is fitted
with several seeds and numbers of epochs for as long as the total stays in the
budget (100 s), and the labels of the lowest inertia are kept. A case is too
slow when its first fit (5 epochs, seed 0) takes more than the budget: it gets
no labels. A case with `n_samples * n_features * n_clusters >= MAX_COST` is
not tried.

    pip install -e packages/inertia-scorer -e packages/inertia-client \
        sklearn-contrib-emeans==0.4.1
    python submissions/cakedev0/emeans_exact.py fit OUT --hardware "..."
    python submissions/cakedev0/emeans_exact.py submit OUT

`fit` writes `OUT/<dataset>.npz` and `OUT/<dataset>.json`, and skips the
datasets that already have their `.json`. `submit` sends them: it needs
`inertia-client login`, not the datasets.
"""

import argparse
import json
import os
from pathlib import Path
import time

from inertia_scorer import DATASETS, DatasetNotFound, Reader

PARAMS = {"init": "k-means||", "revival": "greedy"}
# The tries of a case, as (n_epochs, seed), in this order: 5 and 10 epochs for
# each seed, then 20 epochs for each seed.
SEEDS = range(10)
TRIES = [(n_epochs, seed) for seed in SEEDS for n_epochs in (5, 10)]
TRIES += [(20, seed) for seed in SEEDS]
# Cases of at least this many n_samples * n_features * n_clusters are not tried.
# On one socket of the Xeon 6787P (86 cores), the first fit goes through about
# 1e11 of them per second on the dense datasets with ~100 features or more, so
# 100 s is around 1e13.
MAX_COST = 3e13
# A try runs when `MARGIN` times its expected time fits in what is left.
MARGIN = 1.2


def fit_case(X, n_clusters, budget):
    """Return the tries of a case and the labels of the best one, or None when
    the case is too slow."""
    from sklearn_emeans import EMeans

    tries, labels, spent = [], None, 0.0
    for n_epochs, seed in TRIES:
        if tries:
            # Above the real time for a first try with this many epochs: the
            # initialization does not get longer with the epochs.
            same = [t["time_s"] for t in tries if t["n_epochs"] == n_epochs]
            expected = tries[0]["time_s"] * n_epochs / TRIES[0][0]
            if spent + MARGIN * (same[-1] if same else expected) > budget:
                continue
        start = time.perf_counter()
        model = EMeans(
            n_clusters=n_clusters, n_epochs=n_epochs, random_state=seed, **PARAMS
        ).fit(X)
        elapsed = time.perf_counter() - start
        spent += elapsed
        inertia = float(model.inertia_)
        print(
            f"  {n_clusters} clusters, {n_epochs} epochs, seed {seed}: "
            f"{elapsed:.2f} s, inertia {inertia:.6e}",
            flush=True,
        )
        if labels is None or inertia < min(t["inertia"] for t in tries):
            labels = model.labels_
        tries.append(
            {
                "n_epochs": n_epochs,
                "seed": seed,
                "time_s": elapsed,
                "inertia": inertia,
            }
        )
        if spent > budget and len(tries) == 1:
            return tries, None
    return tries, labels


def fit(args):
    import sklearn_emeans

    from inertia_client import save_labels

    args.out.mkdir(parents=True, exist_ok=True)
    # Costs (stored values of the data * number of clusters) of the cases that
    # were too slow, for the dense and for the sparse datasets: a case that
    # costs as much as one of them is not tried.
    too_slow = {False: [], True: []}
    for path in args.out.glob("*.json"):
        result = json.loads(path.read_text())
        too_slow[DATASETS[result["dataset"]].sparse] += result["too_slow"].values()

    names = args.datasets or sorted(
        DATASETS, key=lambda name: DATASETS[name].n_samples * DATASETS[name].n_features
    )
    for name in names:
        dataset, path = DATASETS[name], args.out / f"{name}.json"
        if path.exists():
            print(f"{name}: already done")
            continue
        try:
            reader = Reader(dataset)
        except DatasetNotFound as error:
            print(error)
            continue
        n_samples, n_features = reader.shape
        n_values = int(reader.indptr[-1]) if dataset.sparse else n_samples * n_features
        slow = too_slow[dataset.sparse]
        result = {
            "dataset": name,
            "hardware": args.hardware,
            "n_cpus": len(os.sched_getaffinity(0)),
            "version": sklearn_emeans.__version__,
            "params": PARAMS,
            "budget_s": args.budget,
            "cases": {},
            "too_slow": {},
        }
        labels, X = {}, None
        for n_clusters in dataset.n_clusters:
            cost = n_values * n_clusters
            if cost >= MAX_COST or any(c <= cost for c in slow):
                print(f"{name}, {n_clusters} clusters: skipped, too slow")
                result["too_slow"][n_clusters] = cost
                continue
            if X is None:
                print(f"{name}: reading", flush=True)
                X = reader.read(0, n_samples)
            tries, best = fit_case(X, n_clusters, args.budget)
            if best is None:
                print(f"{name}, {n_clusters} clusters: too slow")
                result["too_slow"][n_clusters] = cost
                slow.append(cost)
                continue
            labels[n_clusters] = best
            result["cases"][n_clusters] = tries
        if labels:
            save_labels(args.out / f"{name}.npz", labels)
        path.write_text(json.dumps(result, indent=1))


def metadata(result):
    """The times and the tags of a submission, from the `.json` of `fit`."""
    time_s, best = {}, []
    for n_clusters, tries in result["cases"].items():
        time_s[int(n_clusters)] = round(sum(t["time_s"] for t in tries), 2)
        kept = min(tries, key=lambda t: t["inertia"])
        best.append(
            f"{n_clusters}: n_epochs={kept['n_epochs']} seed={kept['seed']} "
            f"({kept['time_s']:.2f} s), {len(tries)} tries"
        )
    params = ", ".join(f"{key}={value!r}" for key, value in result["params"].items())
    tags = {
        "algorithm": f"EMeans({params})",
        "library": f"sklearn-contrib-emeans {result['version']}",
        "n_cpus": str(result["n_cpus"]),
        "selection": (
            "Lowest inertia over several seeds and numbers of epochs, as many "
            f"as fit in {result['budget_s']:g} s for each number of clusters. "
            "The time is the total of the tries, without reading the data."
        ),
        "best": "; ".join(best),
    }
    return time_s, tags


def submit(args):
    from inertia_client import Client, ServerError

    client = Client()
    for path in sorted(args.out.glob("*.json")):
        result = json.loads(path.read_text())
        if not result["cases"] or (args.datasets and path.stem not in args.datasets):
            continue
        time_s, tags = metadata(result)
        if args.dry_run:
            print(path.stem, time_s, json.dumps(tags, indent=1))
            continue
        try:
            submission = client.submit(
                result["dataset"],
                path.with_suffix(".npz"),
                time_s=time_s,
                hardware=result["hardware"],
                tags=tags,
            )
        except ServerError as error:
            print(f"{path.stem}: {error}")
            continue
        print(f"{path.stem}: {submission['id']} {submission['status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("fit")
    command.add_argument("out", type=Path)
    command.add_argument("--hardware", required=True)
    command.add_argument("--budget", type=float, default=100.0, help="seconds")
    command.add_argument("--datasets", nargs="+", choices=DATASETS)
    command.set_defaults(run=fit)
    command = commands.add_parser("submit")
    command.add_argument("out", type=Path)
    command.add_argument("--datasets", nargs="+", choices=DATASETS)
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(run=submit)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
