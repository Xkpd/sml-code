# FT integration and inner-only calibration

## Handoff status (2026-10-05)

This branch is based on `62c505f` (shared protocol v3), the version used for the
completed 72-fit calibration. Origin/main has since moved to protocol v4. This
handoff does not overwrite the team's v4 settings or lock; integration with that
version and a new formal freeze must be reviewed separately.

All eight calibration conditions completed. The learning rates 0.0001, 0.0003,
and 0.001 were selected in 2, 5, and 1 conditions respectively. The provisional
recommendation is to retain the grid, not fix one learning rate for all runs.
Five fits reached the 50-epoch cap, including one fit for a selected candidate;
the stopping policy still requires review before formal evaluation. No outer
test was evaluated. Generated logs and participant data are not part of this
code commit.

Validation: adapter refit/serialization tests and the nine-fit one-epoch smoke
completed successfully. The v3 shared suite had 23 passing tests and two Windows
SQLite temporary-file cleanup errors (WinError 32); do not describe the full
suite as passing. The calibration notebook is saved with outputs cleared and
the long-run switch disabled.

## Running the code

### v4 alignment and epoch diagnostic (2026-10-05)

The working integration branch now incorporates origin/main at aff250b (protocol
v4), retaining the implemented FT adapter in place of the remote placeholder.
Shared runner, data, metrics, results, experiment.json and experiment.lock.json
match that revision. Formal FT status remains pending. The preserved v3 commit
and original analysis/ft_calibration_v1 results remain available.

The default calibration output is now analysis/ft_calibration_v4, preventing
accidental reuse of the v3 recovery database under a new shared protocol.
`python diagnose_ft_epochs.py` runs a separate 12-fit diagnostic into
analysis/ft_epoch100_v4: outer7/n6/Lab, all three rates and all three inner folds;
outer0/n18/Lab, LR=0.0001 and all three inner folds. It raises only max_epochs
to 100, preserving patience=8 and the original model settings. The process uses
below-normal Windows priority and the adapter's four CPU threads, sequentially.
The original calibration recovery database is opened read-only. Reruns may
vary on GPU; the report records common-prefix curve differences, rather than
assuming that every score difference is caused by the higher epoch limit.

The adapter implements the shared runner's fit, predict_proba, serialize and
deserialize contracts. It uses the project's compact numerical FT architecture:
64-dimensional tokens, 2 blocks, 4 heads, ReGLU, batch size 512, AdamW,
weight decay 1e-5, attention/FFN dropout 0.1, BF16 and gradient clipping at 1.
Training weights are supplied by the shared runner, preprocessing fits only on
training data, and checkpoint selection uses shared participant Macro-F1.
Refit epochs are mandatory and are supplied by the shared runner.

Use Python 3.11 with the shared requirements and CUDA PyTorch (tested with
2.11.0+cu128). Run `python test_ft_adapter.py` for adapter checks.

`python calibrate_ft.py` checks the frozen shared inputs and prints the plan.
`python calibrate_ft.py --execute` runs 72 inner fits: outer folds 0 and 7,
n=6 and n=18, FL and Formal_Lab, seed 17, three learning rates from the current
experiment.json, and three inner folds. Folds 0 and 7 are preselected to align
with the earlier LR calibration, not selected based on FT outcomes.

`python calibrate_ft.py --execute --smoke --output analysis/ft_smoke_v1`
runs only the n6 Lab outer0 condition for one epoch per candidate/fold. Smoke
scores must not be used to decide the learning-rate grid.

The calibration invokes the existing Runner.tune; it does not duplicate CV or
selection rules. All outer_test calls are blocked. There are no final refits,
test predictions or formal result exports. Completed fits resume from a separate
SQLite recovery database. A currently interrupted fit restarts from scratch.
Logs print per epoch; details including timings and epoch histories are saved
in that database. Selection summaries are in analysis/ft_calibration_v1/calibration.json.

The 50-epoch cap and patience=8 are pilot settings. Inspect whether Lab runs
reach the cap and whether boundary grid choices improve materially before
claiming the grid is adequate. Additional calibration must use a distinct
configuration/output directory and must not inspect outer-test performance.

experiment.json and experiment.lock.json remain unchanged: FT is still pending
for formal evaluation. The calibration builds a separate in-memory candidate
configuration and records its fingerprint, not a new formal lock. After team
agreement, put the final FT settings/grid into experiment.json, explicitly
accept matched_epoch_policy=reuse_seed17_refit_epochs, and freeze through the
shared process. Do not mix these calibration outputs with formal results.
