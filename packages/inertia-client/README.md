# inertia-client

Submit labels to
[the inertia challenge](https://github.com/cakedev0/the-inertia-challenge):
`inertia-client` and the `inertia_client` Python package.

```bash
inertia-client login
inertia-client submit sift_128 sift_128.npz --time 100=23.21 --hardware "M2, 8 cores" --tag algorithm=Lloyd
inertia-client status <id> --wait
```

`login` identifies you with GitHub. It gives a code to enter on github.com and
keeps a token in `~/.config/the-inertia-challenge/github-token`. The token can
only tell who you are: it gives no access to your repositories. GitHub only
accepts it for 8 hours, so the client renews it when needed: you log in again
only after 6 months without using the client.

`submit` checks the file as the server will, without the dataset, then asks
before sending it: the server takes one submission per day and per dataset.
`check` only does the check. An upload that was stopped resumes with
`inertia-client resume <id> FILE`, for 6 hours.

The file is a `.npz` for one dataset. Each key is a number of clusters `k`, and
its array has the label of each sample: dtype uint32, shape `(n_samples,)`,
values in `[0, k)`. From Python, `submit` takes `{k: labels}` and writes the
file.

`--time`, `--hardware` and `--tag` are what you say about the submission. They
are shown on the scoreboard as you give them. The time is for one number of
clusters: give `--time` once for each of them.

`--submit-only-improvement` scores the labels on your side (with
`inertia-scorer`, so the dataset must be downloaded, see `--data-dir`) and
compares each number of clusters to the scoreboard. It is sent if it beats your
best score (or you have none yet), and if it is less than 10% above the best
score of everyone, else the listener would not publish it. Nothing is sent when
no number of clusters is. Your scores are the ones of your GitHub login.

From Python:

```python
import inertia_client

inertia_client.submit(
    "sift_128", {100: labels}, time_s={100: 23.21}, hardware="M2, 8 cores"
)
# Only the numbers of clusters worth sending; None if there is none.
inertia_client.submit("sift_128", {100: labels}, submit_only_improvement=True)
```
