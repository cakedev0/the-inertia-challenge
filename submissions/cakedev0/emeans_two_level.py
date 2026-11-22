"""Submissions of EMeans with the two-level assignment, on all the cases.

A case is a dataset and a number of clusters. For each case, EMeans is fitted
with the seeds 0, 1, 2... until the fits have taken `--min-time` seconds
(100 s) in total, and the labels of the lowest inertia are kept. A case
therefore gets at least one fit, however long it is.

The sparse datasets are fitted with `assignment="exact"`: "two-level" does not
support them.

    pip install -e packages/inertia-scorer -e packages/inertia-client \
        sklearn-contrib-emeans==0.5.0
    python submissions/cakedev0/emeans_two_level.py fit OUT --hardware "..."
    python submissions/cakedev0/emeans_two_level.py submit OUT

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

PARAMS = {
    "init": "random",
    "n_epochs": 5,
    "revival": "cost-based",
    "assignment": "two-level",
}


def fit(args):
    import sklearn_emeans
    from sklearn_emeans import EMeans

    from inertia_client import save_labels

    args.out.mkdir(parents=True, exist_ok=True)
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
        params = PARAMS | ({"assignment": "exact"} if dataset.sparse else {})
        result = {
            "dataset": name,
            "hardware": args.hardware,
            "n_cpus": len(os.sched_getaffinity(0)),
            "version": sklearn_emeans.__version__,
            "params": params,
            "min_time_s": args.min_time,
            "cases": {},
        }
        print(f"{name}: reading", flush=True)
        X = reader.read(0, dataset.n_samples)
        labels = {}
        for n_clusters in dataset.n_clusters:
            tries, spent = [], 0.0
            while spent < args.min_time:
                seed = len(tries)
                start = time.perf_counter()
                model = EMeans(n_clusters=n_clusters, random_state=seed, **params)
                model.fit(X)
                elapsed = time.perf_counter() - start
                spent += elapsed
                inertia = float(model.inertia_)
                if not tries or inertia < min(t["inertia"] for t in tries):
                    labels[n_clusters] = model.labels_
                    print(
                        f"  {n_clusters} clusters, seed {seed}: {elapsed:.2f} s, "
                        f"inertia {inertia:.6e}",
                        flush=True,
                    )
                tries.append({"seed": seed, "time_s": elapsed, "inertia": inertia})
            print(f"  {n_clusters} clusters: {len(tries)} fits in {spent:.1f} s")
            result["cases"][n_clusters] = tries
            # After each case: the last ones of a large dataset can be long.
            save_labels(args.out / f"{name}.npz", labels)
        path.write_text(json.dumps(result, indent=1))


def metadata(result):
    """The times and the tags of a submission, from the `.json` of `fit`."""
    time_s, best = {}, []
    for n_clusters, tries in result["cases"].items():
        time_s[int(n_clusters)] = round(sum(t["time_s"] for t in tries), 2)
        kept = min(tries, key=lambda t: t["inertia"])
        best.append(
            f"{n_clusters}: seed={kept['seed']} ({kept['time_s']:.2f} s), "
            f"{len(tries)} fits"
        )
    params = ", ".join(f"{key}={value!r}" for key, value in result["params"].items())
    tags = {
        "algorithm": f"EMeans({params})",
        "library": f"sklearn-contrib-emeans {result['version']}",
        "n_cpus": str(result["n_cpus"]),
        "selection": (
            "Lowest inertia over the seeds 0, 1, 2... until the fits have taken "
            f"{result['min_time_s']:g} s in total, for each number of clusters. "
            "The time is the total of the fits, without reading the data."
        ),
        "best": "; ".join(best),
    }
    return time_s, tags


def submit(args):
    from inertia_client import Client, ServerError

    client = Client()
    for path in sorted(args.out.glob("*.json")):
        if args.datasets and path.stem not in args.datasets:
            continue
        result = json.loads(path.read_text())
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
    command.add_argument("--min-time", type=float, default=100.0, help="seconds")
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
