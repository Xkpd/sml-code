# FT calibration results handoff

Packaged on 2026-10-06. These are existing completed experiments, not a new training run.

## Files and experiment status

- [Original calibration](calibration_50epochs_v3/calibration.json): protocol v3, max_epochs=50, patience=8; 8 conditions x 3 learning rates x 3 inner folds = 72 fits. Learning rates: [0.0001, 0.0003, 0.001]. The middle value was selected in 5/8 conditions. See the accompanying calibration_report.md.
- [Epoch-limit diagnostic](diagnostic_100epochs_v4/diagnostic.json): protocol v4, max_epochs=100, patience=8; 12 selected Lab fits, not a complete 72-fit recalibration. See diagnostic_report.md and conclusions_zh.md.
- [Diagnostic curves](diagnostic_100epochs_v4/epoch_curves.png): all 12 diagnostic curves.
- manifest.json records original local paths and SHA-256 checksums. Copied result files are unchanged.

Both experiments excluded outer-test evaluation. No raw data, per-participant predictions, checkpoints, or recovery databases are included. Fingerprints retain input filenames and hashes for provenance.

## Interpretation and remaining work

One diagnostic fit achieved its best epoch at 54; its within-run improvement after epoch 50 was about 0.004574. GPU reruns differed even before epoch 50, so old-versus-new score differences cannot all be attributed to the epoch cap.

The diagnostic supports considering a cap of 100 with patience=8. A complete 72-fit calibration under that cap has NOT been run. The model default remains 50 and the formal experiment configuration remains pending. Do not describe these files as a completed 100-epoch calibration or formal outer-test results.

If the team adopts the 100-epoch cap, run all 8 conditions under the same settings into a new output directory, then review the learning-rate grid and freeze the formal configuration through the shared process.

The copied reports describe the state when written; any historical statement about not having pushed refers to that time. This handoff adds reports only and does not alter model settings or the shared lock.
