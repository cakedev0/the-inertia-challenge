# inertia-client

Submit labels to [the inertia challenge](../../README.md): `inertia-client` and
the `inertia_client` Python package.

```bash
inertia-client login
inertia-client submit sift_128 sift_128.npz --time 100=23.21 --hardware "M2, 8 cores" --tag algorithm=Lloyd
inertia-client status <id> --wait
```

`login` identifies you with GitHub. It gives a code to enter on github.com and
keeps a token in `~/.config/the-inertia-challenge/github-token`. The token can
only tell who you are: it gives no access to your repositories.

`submit` checks the file as the server will, without the dataset, then asks
before sending it: the server takes one submission per day and per dataset.
`check` only does the check. An upload that was stopped resumes with
`inertia-client resume <id> FILE`, for 6 hours.

`--time`, `--hardware` and `--tag` are what you say about the submission. They
are shown on the scoreboard as you give them. The time is for one number of
clusters: give `--time` once for each of them.

From Python:

```python
import inertia_client

inertia_client.submit(
    "sift_128", {100: labels}, time_s={100: 23.21}, hardware="M2, 8 cores"
)
```
