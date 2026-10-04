# PAAWS Shared Machine-Learning Experiment

## Project overview

This project investigates how the number and source of labelled training participants affect human activity recognition for unseen people in naturalistic settings.

The central research question is:

> How does increasing the number of labelled training participants affect four-class activity-recognition performance on unseen naturalistic participants when models are trained using naturalistic versus laboratory data?

The experiment compares models trained on free-living (`FL`) data with models trained on formal laboratory (`Formal_Lab`) data. Every final evaluation is performed on held-out free-living data so that performance reflects generalisation to unseen participants in realistic conditions.

## Dataset

The repository contains a processed, participant-level activity-recognition dataset:

- 20 pseudonymised participants represented across 40 participant/domain feature shards.
- Two training domains: free-living and formal laboratory.
- Non-overlapping 10-second windows sampled at 80 Hz, with 800 samples per window.
- 135 engineered time-domain, frequency-domain, correlation, orientation, and movement features.
- Four primary activity classes: Sitting, Standing, Lying Down, and Walking.
- Biking observations are retained in the stored data for sensitivity purposes but excluded from the primary four-class analysis.

The processed files include coded participant identifiers, window indices, labels, and timestamps. The repository must remain private and access should be limited to authorised project collaborators.

## Experimental design

The study uses nested participant-level cross-validation with frozen splits. Participant-level separation prevents windows from the same person appearing in both training and evaluation data.

### Outer evaluation

- Ten outer folds are used.
- Each outer fold holds out two participants.
- Both free-living and laboratory observations from the held-out participants are excluded from training.
- Final evaluation uses all eligible free-living windows belonging to the two held-out participants.

### Training-set sizes and repetitions

The main analysis evaluates participant training-set sizes of **6, 12, and 18**. This is the predeclared shorter schedule, adopted for the team's computing budget in protocol `paaws_shared_v3`. Each size is repeated using subset seeds 17, 42, and 73. The participant subsets and their ordering are fixed in the supplied split manifests. The original manifests still contain sizes 9 and 15; the runner does not schedule them.

The complete main schedule covers both training domains, ten outer folds, three participant sizes, and three subset seeds: 180 conditions. A matched sensitivity analysis adds 60 conditions at a participant size of 18 to compare free-living and laboratory training using matched participant-by-class window lists. With three hyperparameter candidates this requires **2,040 fits per model**, compared with 3,240 for the five-size schedule (about 37% fewer fits). `python run.py plan` calculates the count from each model's actual grid.

### Inner model selection

For every main-analysis model, training domain, outer fold, participant size, and subset seed:

1. The selected training participants are loaded in their frozen order.
2. Each hyperparameter candidate is evaluated using three inner participant folds.
3. Preprocessing and class weights are fitted using the inner training fold only.
4. Validation Macro-F1 is calculated separately for each validation participant.
5. Participant scores are averaged so that participants contribute equally regardless of their number of windows.
6. A hyperparameter is selected using the model's predefined selection rule.
7. A new model is fitted on all selected training participants.
8. The refitted model predicts the complete eligible free-living data for the held-out outer-fold participants.

For matched analysis, tune separately on matched seed 17 within each outer fold and training domain. Seeds 42 and 73 reuse that selected hyperparameter, then each receives its own fresh refit on its registered matched rows. The main-analysis choice is not reused for matched analysis. FT integration must also explicitly accept reuse of the seed-17 selected refit epoch count.

This ordering prevents outer-test information from affecting training or selection. Inner validation affects hyperparameter selection (and FT checkpoint selection) only; preprocessing and class weights use the current training fold.

## Models

### Multinomial logistic regression

- Candidate regularisation values: `C = 0.03, 0.1, 0.3`.
- Features are standardised using statistics calculated from the current training fold only.
- The `lbfgs` solver is used with `max_iter=2000` and `tol=0.0001`.
- Candidates within 0.001 Macro-F1 of the best score use the middle-grid preference rule.

### LightGBM

- Candidate leaf counts: `num_leaves = 127, 255, 511`.
- Raw engineered features are used without standardisation.
- Training uses a learning rate of 0.05 and a maximum of 300 boosting rounds.
- The candidate with the highest validation score is selected; exact ties prefer fewer leaves.

### FT-Transformer

The shared runner contains an integration interface for an FT-Transformer model. Its configuration remains marked as pending until the architecture, training settings, checkpoint selection, and final-refit epoch policy are fully implemented and frozen.

## Class weighting and metrics

Training-only inverse-frequency class weights are calculated within each training fold and normalised to have mean one. Validation and test metrics are unweighted within each participant.

The primary selection metric is participant-balanced Macro-F1. Additional summaries include balanced accuracy, weighted F1 and accuracy. Saved predictions also support per-participant scores, per-class precision/recall/F1 and supports, individual fold scores, and confusion matrices. A class absent from a participant's true labels is NA and excluded from that participant's class average, matching the original scoring convention.

Final results are aggregated in the following order:

1. Windows within each participant.
2. The two held-out participants within each outer fold.
3. The three subset seeds.
4. The ten outer folds.

Approximate 95% t-intervals are calculated across the ten outer-fold means using 9 degrees of freedom. Comparisons between free-living and laboratory training use paired fold-level differences.

## Reproducibility and integrity

The experiment is frozen by `experiment.lock.json`. The lock records hashes for the shared source files, settings, split manifests, feature metadata, and all 40 data shards. The runner checks these hashes before model execution so that results from different experiment versions cannot be mixed accidentally.

The supplied split files must not be regenerated independently. Any intended protocol change should be released as a coordinated new experiment version rather than mixed into an existing result directory.

Interrupted runs are recoverable. The separate local `recovery/<model>/progress.sqlite3` stores completed inner-fit scores, model checkpoints, tuning selections, timing information and prediction chunks. `recovery/<model>/run.lock` prevents simultaneous writers using that recovery folder. These are additional internal files; only the two CSV files belong in `outputs/<model>/`. Repeating the same command with the same output and recovery locations resumes completed work. An interrupted in-progress fit restarts; completed saved fits are reused.

Recovery is bound to the experiment, model, installed environment and resolved output path. Saved export hashes prevent missing, unidentified or older recovery data from overwriting newer results. CSVs are prepared in the recovery folder; a recorded transition permits recovery even if export stops between replacing the two files. Keep both locations in place on the same filesystem while running, and do not start concurrent jobs writing the same output folder. If only the two result files are available, `validate` and `evaluate` still work; `run`/`export` refuse to invent replacement recovery data. For a separate new run, choose both a new `--output` and a new `--recovery-dir`. Do not mix v2 recovery or results with this v3 package.

After every fit the runner prints elapsed training time and the model's reported iterations, actual boosting rounds or epochs. This information is retained in the recovery database without adding columns to the agreed result files.

### Shared loader and scoring origin

All three adapters must use this folder's `data.py` and `metrics.py` through `run.py`. The consolidated loader carries forward the corrected team `paaws_pipeline` data loader and registered split-manifest logic from `team_shared_code_and_manifests_v1.zip` and the supplied corrected `data.py`. The manual metrics retain the original confusion-matrix formulas, absent-class convention, participant averaging, training-only class weights and ten-fold t-interval. There is no need to restore the old import path or maintain a second loader.

`test_experiment.py` retains regression checks for original manifest membership and nested subsets, participant exclusion, matched row selection, training-only scaling, independently calculated scoring/weighting formulas and the final aggregation. The feature shards and split files are unchanged. Git attributes preserve the original plain split-CSV bytes so another computer can verify the same frozen hashes.

## Repository structure

```text
experiment.json             Study settings and model grids
experiment.lock.json        Frozen hashes for reproducibility
run.py                      Shared tune, select, refit, test, and export runner
data.py                     Data and frozen-split loading
metrics.py                  Metrics and training class weights
results.py                  Output validation and result summaries
graph/plot.py               Three research figures from completed model results
models/                     Model adapters
data/shards/                Processed participant/domain feature files
splits/                     Frozen outer, inner, and matched manifests
test_experiment.py          Synthetic integrity and recovery tests
requirements.txt            Python dependencies
outputs/<model>/            Only tuning.csv and predictions.csv.gz (generated)
recovery/<model>/           Local progress database and process lock (generated)
```

## Setup

Python 3.11 and a separate virtual environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python test_experiment.py
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Checking the frozen experiment

These commands verify the package without fitting models:

```bash
python run.py plan
python run.py check --model lr
python run.py check --model lightgbm
```

## Running models

Run a model's complete frozen schedule with:

```bash
python run.py run --model lr
python run.py run --model lightgbm
```

A single condition can be run or resumed by specifying its analysis, domain, outer fold, participant size, and subset seed:

```bash
python run.py run --model lightgbm \
  --analysis main \
  --domain Formal_Lab \
  --outer-fold 0 \
  --subset-size 6 \
  --subset-seed 17
```

Formal test scores must not be used to revise model settings or hyperparameter grids. Use `test_experiment.py` for implementation checks.

## Outputs and validation

Each completed model produces exactly two handoff result files:

```text
outputs/<model>/tuning.csv
outputs/<model>/predictions.csv.gz
```

The tuning file records candidate scores for the inner folds and identifies the selected setting. The prediction file contains held-out window predictions, probabilities, labels, and experimental condition identifiers.

Validate a complete result set with:

```bash
python run.py validate --model lightgbm
```

For an intentionally incomplete run:

```bash
python run.py validate --model lightgbm --allow-partial
```

After the full result set passes validation, print the study summaries with:

```bash
python run.py evaluate --model lightgbm
```

`summarise` is an alias for `evaluate`. Neither command creates another CSV or requires the recovery database. Select additional detail as needed:

```bash
python run.py evaluate --model lightgbm --detail participants
python run.py evaluate --model lightgbm --detail classes
python run.py evaluate --model lightgbm --detail folds
python run.py evaluate --model lightgbm --detail confusion --analysis main --domain FL --subset-size 6 --outer-fold 0 --subset-seed 17
```

`--detail all` prints everything. Filters restrict displayed rows after full-study aggregation; choosing a fold/seed does not recalculate study means or confidence intervals from that smaller selection. Evaluation requires the complete frozen result set; `validate --allow-partial` checks progress during a run.

Confusion matrices show true classes in rows and predicted classes in columns. Pooled window counts are descriptive; the accompanying normalized matrix gives each eligible participant equal weight. Per-class summaries report the number of contributing folds; a class with fewer than ten contributing folds has no ten-fold confidence interval.

Generated outputs, local recovery files, environments, and model checkpoints are excluded from version control. Participant-level predictions should be handled using the project's approved secure sharing process.

## Research graphs

The `graph` folder needs only one source file, `plot.py`. After the model runs are complete, collect each model's existing `tuning.csv` and `predictions.csv.gz` under `outputs/<model>/`, where the model folders are `multinomial_logistic_regression`, `lightgbm` and `ft_transformer`.

Install the plotting dependency separately, then run from this project folder:

```bash
python -m pip install matplotlib==3.11.2
python graph/plot.py
```

This creates only three figure files alongside the script:

| Figure | Research question addressed |
|---|---|
| `graph/learning_curves.png` | How does participant Macro-F1 change with 6/12/18 labelled training participants, for each model and training domain? |
| `graph/domain_comparison.png` | At 18 participants, how do FL and Lab compare before and after matching participant-by-class window counts? Includes the paired FL-minus-Lab differences. |
| `graph/class_f1.png` | At 18 participants in the main analysis, which of the four activities are difficult for each model and training domain? |

All figures evaluate unseen FL participants. The script validates complete outputs against the frozen experiment and uses `results.py` for every score and confidence interval. It does not fit models or write extra CSV files. Incomplete conditions are rejected. A subset of completed models can be drawn explicitly, for example:

```bash
python graph/plot.py --models lr lightgbm
```

The figures name the models shown; missing models are never treated as zero scores. The default command requires all three models. Optional `--results-dir PATH` changes the parent input folder; `--output-dir PATH` changes the image destination. Images cannot be saved inside the model results, recovery or input-data folders. Running again replaces the same three images, so there is no accumulation of dated copies.

Error bars are the existing approximate 95% t intervals across ten outer-fold means after seed averaging. They are not intervals across individual windows or thirty independent seed runs, and are not silently clipped to 0–1. The paired difference uses the shared paired-fold calculation, rather than subtracting separate confidence-interval endpoints. Matched comparisons control counts but do not establish a causal domain effect.

Class F1 follows the agreed absent-class convention: unsupported values are shown as NA; fewer than ten contributing folds are labelled and have no ten-fold interval. Do not average these plotted class means to reconstruct primary participant Macro-F1. More detailed precision, recall, accuracy and confusion matrices remain available through `run.py evaluate --detail ...`; they are not extra default graphs.

Matplotlib is only a plotting dependency. The frozen training requirements and experiment lock are unchanged by adding this folder.

## Current implementation status

The logistic-regression and LightGBM adapters are implemented and frozen. The FT-Transformer integration remains pending. The package includes automated checks for data leakage guards, split integrity, interruption recovery, model serialisation, selection rules, output validation, and participant-level aggregation.
