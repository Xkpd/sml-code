# Working together with GitHub

This repository contains the shared modelling code and the fixed processed dataset.
Keep the GitHub repository private and only add approved project members. The data
contains participant-level coded identifiers and timestamps, so do not make the
repository public, fork it publicly, or share it outside the authorised group.

## Recommended workflow

1. Before starting work, fetch and pull the latest `main` branch.
2. Create a short-lived branch for one task, such as `xingkun/lightgbm-update`.
3. Make a small, focused change and run the relevant tests.
4. Commit with a clear message, then push the branch.
5. Open a pull request on GitHub. Ask one group member to review it before merging.
6. After it is merged, switch back to `main`, pull, and delete the old task branch.

In GitHub Desktop, use **Fetch origin / Pull origin**, **Current branch > New
branch**, **Commit to ...**, **Push origin**, and then **Create Pull Request**.

The matching terminal commands are:

```bash
git switch main
git pull --ff-only
git switch -c your-name/short-task
# edit and test
git add path/to/changed-file
git commit -m "Describe the change"
git push -u origin your-name/short-task
```

After the pull request is merged:

```bash
git switch main
git pull --ff-only
git branch -d your-name/short-task
```

## Project-specific rules

- Treat `data/` and `splits/` as frozen shared inputs. Do not replace or regenerate
  them without group agreement.
- Coordinate before changing shared files such as `run.py`, `data.py`,
  `metrics.py`, `results.py`, `experiment.json`, or `experiment.lock.json`.
- Prefer editing only the model file you own under `models/`.
- Generated outputs, recovery databases, environments, and model checkpoints are
  intentionally ignored. Share approved final results separately rather than
  committing participant-level output data.
- Never commit passwords, access tokens, API keys, or `.env` files.

If two people changed the same lines, do not guess through the conflict. Compare
both versions together, keep the intended scientific behaviour, rerun the tests,
and then commit the resolved file.
