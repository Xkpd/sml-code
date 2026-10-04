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

The main analysis evaluates participant training-set sizes of 6, 9, 12, 15, and 18. Each size is repeated using subset seeds 17, 42, and 73. The participant subsets and their ordering are fixed in the supplied split manifests.

The complete main schedule covers both training domains, ten outer folds, five participant sizes, and three subset seeds. A matched sensitivity analysis is also included at a participant size of 18 to compare free-living and laboratory training using matched participant-by-class window lists.

### Inner model selection

For every model, training domain, outer fold, participant size, and subset seed:

1. The selected training participants are loaded in their frozen order.
2. Each hyperparameter candidate is evaluated using three inner participant folds.
3. Preprocessing and class weights are fitted using the inner training fold only.
4. Validation Macro-F1 is calculated separately for each validation participant.
5. Participant scores are averaged so that participants contribute equally regardless of their number of windows.
6. A hyperparameter is selected using the model's predefined selection rule.
7. A new model is fitted on all selected training participants.
8. The refitted model predicts the complete eligible free-living data for the held-out outer-fold participants.

This ordering prevents information from the held-out participants or inner validation folds from affecting preprocessing, class weighting, hyperparameter selection, or model fitting.

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

The primary selection metric is participant-balanced Macro-F1. Additional summaries include balanced accuracy and weighted F1.

Final results are aggregated in the following order:

1. Windows within each participant.
2. The two held-out participants within each outer fold.
3. The three subset seeds.
4. The ten outer folds.

Approximate 95% t-intervals are calculated across the ten outer-fold means using 9 degrees of freedom. Comparisons between free-living and laboratory training use paired fold-level differences.

## Reproducibility and integrity

The experiment is frozen by `experiment.lock.json`. The lock records hashes for the shared source files, settings, split manifests, feature metadata, and all 40 data shards. The runner checks these hashes before model execution so that results from different experiment versions cannot be mixed accidentally.

The supplied split files must not be regenerated independently. Any intended protocol change should be released as a coordinated new experiment version rather than mixed into an existing result directory.

Interrupted runs are recoverable. Each model output directory uses a SQLite progress database for completed fits, model checkpoints, tuning selections, timing information, and prediction chunks. Repeating the same command resumes completed work instead of starting again.

## Repository structure

```text
experiment.json             Study settings and model grids
experiment.lock.json        Frozen hashes for reproducibility
run.py                      Shared tune, select, refit, test, and export runner
data.py                     Data and frozen-split loading
metrics.py                  Metrics and training class weights
results.py                  Output validation and result summaries
models/                     Model adapters
data/shards/                Processed participant/domain feature files
splits/                     Frozen outer, inner, and matched manifests
test_experiment.py          Synthetic integrity and recovery tests
requirements.txt            Python dependencies
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

Each completed model produces two main handoff files:

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

After a complete result set passes validation, generate fold-level summaries with:

```bash
python run.py summarise --model lightgbm
```

Generated outputs, recovery databases, environments, and model checkpoints are excluded from version control. Participant-level predictions should be handled using the project's approved secure sharing process.

## Current implementation status

The logistic-regression and LightGBM adapters are implemented and frozen. The FT-Transformer integration remains pending. The package includes automated checks for data leakage guards, split integrity, interruption recovery, model serialisation, selection rules, output validation, and participant-level aggregation.
