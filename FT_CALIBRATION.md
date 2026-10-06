# FT submission implementation and running guide

## Final settings (2026-10-06)

FT is now `ready` in `experiment.json` and included in `experiment.lock.json`.
Only the FT model record is added to the lock: the shared protocol, participant
manifests, data, LR and LightGBM fingerprints remain the same. Existing LR and
LightGBM outputs remain compatible.

The model is a compact numerical FT-Transformer: 135 feature tokens plus CLS,
64-dimensional tokens, 2 blocks, 4 attention heads, ReGLU with hidden width 128,
attention/FFN dropout 0.1, batch size 512, AdamW with weight decay 1e-5 and gradient
clipping at norm 1. Numeric tokens, CLS, biases and normalization parameters are
excluded from weight decay. This retains the pilot architecture; it is not a
claim of reproducing every initialization detail of the published implementation.
Reference: [Revisiting Deep Learning Models for Tabular Data](https://github.com/yandex-research/rtdl-revisiting-models).

The learning-rate grid remains `[0.0001, 0.0003, 0.001]`. Inner training now has a
100-epoch maximum and patience 8. Training uses CUDA BF16 autocast, with the loss
and output probabilities calculated in float32. Explicit deterministic
algorithms are required, TF32 is disabled and cuDNN benchmarking is disabled.
Unsupported deterministic operations raise an error rather than silently
relaxing the policy. CPU threads remain 4; FT uses one GPU worker.

## Scientific flow

The shared runner provides the same 10 participant outer folds, 3 inner folds,
6/12/18 training participants, seeds 17/42/73 and domains as LR/LightGBM. FT does
not generate independent splits or scores. For each inner fit:

1. Fit the standardizer on the current training fold only.
2. Use the shared training-fold inverse-frequency sample weights. The loss is
   the minibatch mean of weighted per-window cross-entropy; do not apply a
   second class weight or normalize weights independently within each batch.
3. Score validation using the shared, unweighted-within-participant Macro-F1.
   Save the best checkpoint, retaining the earlier epoch for exact ties, and
   stop after 8 epochs without improvement or the epoch cap.
4. The runner chooses the learning rate with the greatest mean across 3 inner
   folds; exact rate ties prefer the lower rate.
5. Refit a fresh model on all selected training participants for exactly the
   median of the selected candidate's 3 best epochs. Save/reload the checkpoint
   before the runner opens the held-out FL test windows.

Matched analysis tunes on matched seed 17 separately for each outer fold/domain.
Seeds 42/73 reuse that rate and epoch count but receive independent refits on
their own registered matched rows. Testing still uses the full, same FL windows.
`best_epoch` is filled for each inner FT fit in `tuning.csv`. Shared output column
names and class order remain unchanged.

## Setup and checks on the CUDA desktop

Use Python 3.11 in an isolated environment. From the shared project folder:

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python run.py check --model ft
python -m unittest test_experiment
python -m unittest test_ft_adapter
python test_ft_adapter.py --cuda-preflight
python calibrate_ft.py --execute --smoke
```

The pinned CUDA wheel follows the [official PyTorch 2.11 installation instructions](https://pytorch.org/get-started/previous-versions/#v2110)
and the existing pilot's CUDA build. A compatible NVIDIA driver and BF16-capable
GPU are required. The Mac CPU adapter tests use `amp=False` only in synthetic test
fixtures; do not change the frozen formal setting to run on the Mac.

The explicit CUDA preflight fails if the GPU/BF16 is unavailable. It checks the
frozen architecture with a full-size minibatch and a remainder, repeated
same-seed predictions, best-checkpoint restoration and save/load consistency.
The ordinary test suite skips its CUDA test on a CPU machine; a skipped test is
not GPU approval. The real-data smoke runs 9 one-epoch inner fits on outer0/n6
Formal_Lab. All outer-test access is blocked. It writes only to
`analysis/ft_smoke_final_v4`; smoke scores are not scientific results or grid
selection evidence. Do not start the formal run if these checks fail.

Run the suites as two separate commands as shown above. On the verification Mac,
loading PyTorch and LightGBM in one combined test process caused a native crash;
both suites passed in separate processes. Formal execution imports only the
chosen model adapter.

Historical Windows tests reported two SQLite temporary-file cleanup errors.
The test-owned connections now close explicitly; the suite still needs to pass
on the target Windows computer. The frozen shared runner was not modified.

## Full run and the two result files

After the checks pass:

```bash
python run.py run --model ft
python run.py validate --model ft
python run.py evaluate --model ft
```

The full plan is 1,800 inner fits plus 240 final refits. Do not add `--workers 3`:
that option is for LightGBM, not FT. Early stopping usually ends inner fits before
100 epochs; final refits use their inner-selected count, not automatically 100.

The formal results folder contains only:

- `outputs/ft_transformer/tuning.csv`
- `outputs/ft_transformer/predictions.csv.gz`

Timing, epoch histories, stopping-cap flags, runtime details and checkpoints are
kept in the separate local `recovery/ft_transformer` folder. Repeat the same run
command to resume saved work. An interrupted, unfinished fit restarts; completed
fits are reused. Keep the same output path, GPU, Python/PyTorch/CUDA environment,
driver and cuBLAS workspace setting when resuming. Hardware details are recorded
per fit, but the shared recovery identity does not itself enforce GPU/driver
identity. Historical pilot checkpoints (`ft_v1`) are deliberately incompatible
with the formal `ft_v2` checkpoint and must not be copied into formal recovery.

Deterministic settings support repeatability in a fixed environment, not exact
agreement across different hardware or releases; see [PyTorch reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
Loading a checkpoint on CPU is supported for inspection using float32 inference;
it is not interchangeable with CUDA BF16 for producing the formal results.

## Local verification (2026-10-06)

On macOS with PyTorch 2.11.0: 27 shared tests and 8 FT tests passed in separate
processes; the CUDA-only test was skipped. A real outer0/n6/Formal_Lab inner0
smoke used 1,611 training and 961 validation windows, with disjoint participants,
for one CPU float32 epoch. Its saved/restored checkpoint produced identical
validation probabilities. It did not open any outer-test data or write formal
results. This is a correctness smoke, not performance evidence or a CUDA test.

## Evidence for the chosen settings

The unchanged `reports/ft_calibration_handoff/` folder preserves two historical
stages, neither of which evaluated outer-test performance:

- **50-epoch calibration:** 72 inner fits across 8 preselected conditions
  (folds 0/7, n=6/18, both domains, seed 17). Learning-rate winners were 2/5/1;
  the middle value won most often. Five fits reached the cap. One winning
  condition was nearly tied, so the endpoint result is not strong evidence to
  move the grid.
- **100-epoch diagnostic:** 12 selected Lab fits rerun with patience 8. One best
  epoch was 54; no fit reached 100. Earlier-epoch trajectories also changed on
  GPU, so differences between these two runs cannot all be attributed to the cap.

The submission settings retain the grid and raise the cap to 100 on this limited
inner-only evidence. Deterministic controls are also new. A complete 72-fit
recalibration under these final settings has **not** been performed; do not label
historical scores as results from the final frozen implementation. No outer-test
scores were used to choose these changes. Further optional calibration is
available through `python calibrate_ft.py --execute`, isolated in
`analysis/ft_calibration_final_v4`, and never replaces the historical reports.
Do not adjust the grid or stopping policy after inspecting formal outer scores.

### Upper learning-rate check after the final calibration

If the completed final calibration selects its upper endpoint (`0.001`), test
`0.003` only on those winning conditions before any formal outer evaluation:

```bash
python calibrate_ft.py --upper-lr-diagnostic
python calibrate_ft.py --execute --upper-lr-diagnostic
```

The first command must print the planned number of conditions/fits and performs
no training. For the current final calibration it prints 4 conditions and 12
inner fits. The second command writes to
`analysis/ft_upper_lr_diagnostic_v4`; it neither edits `experiment.json` nor the
lock and cannot access outer-test data. It reads the baseline from
`analysis/ft_calibration_final_v4/calibration.json`. If that file is elsewhere,
pass `--baseline /path/to/calibration.json`.

The output `calibration.json` contains `diagnostic.comparisons`, including the
mean participant Macro-F1 for `0.001`, the new mean for `0.003`, and their
difference for every checked condition. Keep `0.003` only if it improves
repeatedly by a meaningful amount. Decide and freeze the formal grid before
starting `python run.py run --model ft`; diagnostic results are not formal model
outputs and must not be placed under `outputs/ft_transformer`.
