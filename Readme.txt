PAAWS SHARED EXPERIMENT
=======================

Start here. Every model owner uses a copy of this same folder.
The shared experiment is implemented once. Each owner maintains one model file.
This package contains the processed data, existing frozen splits, common runner,
LR and LightGBM adapters, and an explicit integration point for Person 3's FT.

WHAT TO OPEN

  experiment.json             The study settings and each model's final grid.
  run.py                      The common experiment: tune -> select -> refit -> test.
  data.py                     Loads the existing participant/window splits.
  metrics.py                  The shared handwritten metrics and class weights.
  results.py                  Saves/checks the two outputs; calculates summaries.
  models/logistic_regression.py  Person 1's model only.
  models/lightgbm.py             Your model only.
  models/ft_transformer.py       Person 3's documented integration point, not ready.
  test_experiment.py          Synthetic checks; no real-data model fitting.
  requirements.txt           Tested LR/LightGBM dependencies.
  experiment.lock.json        Automatic hashes of shared code, settings and inputs.
  splits/                    Six original manifests. Do not regenerate them.
  data/                      40 original feature shards + feature/class definitions.

Suggested reading order: experiment.json, run.py Runner.tune and run_condition,
data.py load_split, your models/*.py, then results.py.

THE STUDY IN ONE PASS

For each model, training domain, outer fold, participant size and subset seed:
1. Hold out two participants. Exclude BOTH their FL and Lab data from training.
2. Load the selected training participants in their frozen order.
3. Try each of the three hyperparameter values on all three inner participant folds.
4. Fit preprocessing and class weights on that inner training fold only.
5. Score each validation participant separately, then take their equal mean.
6. Choose one candidate from the three-fold scores using that model's fixed rule.
7. Fit a NEW model on all selected training participants. Save it and verify reload.
8. Only now predict the two held-out participants' complete eligible FL windows.

Repeat across 10 outer folds, seeds 17/42/73, and training domains FL/Formal_Lab.
The original five sizes 6/9/12/15/18 are included: 360 final conditions and 3,240
fits per model, including matched sensitivity. The predeclared shorter schedule
6/12/18 would give 240 conditions and 2,040 fits, but would require a new common
version before anyone begins formal training. Do not change sizes independently.

Main analysis uses all eligible primary-class windows of the selected people.
Matched sensitivity uses n=18 and the original participant-by-class window lists.
For each matched outer fold/domain, tune only seed17. Seeds42/73 reuse its chosen
hyperparameter and fit fresh models on their own matched windows. Main selections
are never reused for matched. All tests still use the complete eligible FL windows.

Class IDs: 0 Sitting; 1 Standing; 2 Lying_Down; 3 Walking. The stored data also has
Biking rows; data.py applies primary_mask to exclude those from this study.
135 engineered features are used. LR standardises using training data only;
LightGBM uses the raw feature values. Training-only inverse-frequency weights
are mean-one. Validation and test metrics are unweighted within each participant.

LR: C = 0.03/0.1/0.3, lbfgs, max_iter=2000, tol=0.0001. Among candidates within
0.001 Macro-F1 of the best, prefer the middle grid value. Nonconvergence stops
the fit; it is not silently counted as a completed result.
LightGBM: num_leaves = 127/255/511, otherwise the supplied fixed settings, with a
300-round cap. Choose maximum validation score; exact ties choose fewer leaves.
Valid exhaustion of useful splits before 300 rounds is accepted. CPU thread count
uses the library default, as in the supplied model. There is no early stopping.

SETUP AND RUN

Use Python 3.11 with a separate environment. In a terminal in this folder:

  python -m pip install -r requirements.txt
  python test_experiment.py
  python run.py plan
  python run.py check --model lr
  python run.py check --model lightgbm

The provided experiment.lock.json freezes the common package and ready models.
check verifies the actual data, manifests, source and settings; it does not fit.
Use lr or lightgbm below (ft remains blocked until its adapter is completed):

  python run.py run --model lr
  python run.py run --model lightgbm

Each command runs that model's complete schedule. Run one process per model output
folder. Different people run their models on their own computers.

To run/resume one formal condition instead:
  python run.py run --model lightgbm --analysis main --domain Formal_Lab --outer-fold 0 --subset-size 6 --subset-seed 17

This is a REAL formal condition with full inner tuning and outer predictions,
not a smoke test. Do not use its test score to adjust hyperparameters or grids.
For a safe implementation check, use test_experiment.py instead.

Repeat the SAME command to resume after interruption. Completed inner fits and
final models are recovered. Only a fit interrupted before it was saved is repeated.
Changing the environment/source/settings within the same result folder is rejected.

OUTPUTS: ONLY TWO FILES TO HAND OVER PER MODEL

  outputs/<model>/tuning.csv
  outputs/<model>/predictions.csv.gz

Each file combines all completed conditions for that model. The runner writes
them automatically on successful exit, Ctrl-C and ordinary fitting errors. If the
process was forcibly killed, the recovery file still contains committed work. Regenerate
the two files without training with:
  python run.py export --model lightgbm

There are also two automatic hidden files in the model's output folder:
.progress.sqlite3 stores fit scores, selected settings, final model checkpoints,
fit timing/counts and prediction chunks. .run.lock prevents concurrent writes.
Keep these locally for recovery. The group needs only the two CSV files for
performance comparisons, plus this shared code/configuration. Do not delete the
recovery file until the team has safely accepted the results.

tuning.csv columns:
model,analysis,training_domain,outer_fold,subset_seed,subset_size,
hyperparameter_name,hyperparameter_value,inner_fold,
participant_balanced_validation_macro_f1,selected,best_epoch

One row = one candidate x one inner fold. Normally nine rows per tuned condition.
selected is True for all three rows of the winning candidate. best_epoch is blank
for LR/LightGBM and the selected checkpoint epoch for each FT inner fit. Matched
tuning rows exist ONLY for seed17. Never duplicate these scores as seeds42/73.

predictions.csv.gz columns:
model,training_domain,analysis,outer_fold,subset_size,subset_seed,
hyperparameter_name,hyperparameter_value,test_participant,test_window,
true_activity,predicted_activity,sitting_prob,standing_prob,lying_prob,walking_prob

One row = one held-out FL window in one condition. test_window is the original
within-participant FL window index. true_activity/predicted_activity use IDs0..3.
Probabilities retain full precision. analysis is main or matched; training_domain
is FL or Formal_Lab. outer_fold identifies a pair of held-out people, not one person.

CHECKING AND ANALYSIS

  python run.py validate --model lightgbm

This requires the entire frozen schedule and checks tuning choices plus every
expected test window, label and probability. For deliberately partial output:
  python run.py validate --model lightgbm --allow-partial
Partial output is explicitly reported as incomplete.

After a model's full results validate:
  python run.py summarise --model lightgbm

This optionally writes analysis/lightgbm_summary.csv for graph creation, including
Macro-F1, balanced accuracy and weighted-F1, with approximate 95% t intervals and
paired FL-minus-Lab differences. It averages within this order: each participant,
the two people in an outer fold, the three seeds, then the ten outer folds (df9).
Windows and the 30 fold/seed combinations are not independent error-bar replicates.
The two handoff files preserve the data needed to produce the final comparison
graphs later; this package does not generate presentation figures automatically.

PERSON 3: CONNECT YOUR FT MODEL

Edit only models/ft_transformer.py and your entry under models in experiment.json.
Keep the architecture and training implementation in that model file so the
model's frozen source hash covers them; do not import an old benchmark runner.
The module documents four functions: fit, predict_proba, serialize, deserialize.
There is no second FT-specific CV runner, split generator, metric or output writer.
fit receives training data, optional inner validation, candidate learning rate,
seed, settings, precomputed training weights and final-refit epochs.
The fitting state must retain its training-only scaler and model parameters.

For inner fits, return the best validation checkpoint and its positive best_epoch.
The common runner chooses a learning rate across all three folds and uses the
median best_epoch for that rate as the fresh final-refit duration. For a final fit,
validation is None: train a NEW model for exactly epochs and return epochs_run.
Document/install the appropriate PyTorch build on the FT machine separately.

FT currently has status=pending, empty final settings and no accepted matched epoch
policy. Its grid is the benchmark grid and remains subject to Person 3's final
inner-only decision. Before it can run, implement and test the adapter, confirm
the full settings, set status=ready and explicitly set matched_epoch_policy to
reuse_seed17_refit_epochs. This extends the seed17 reuse rule to the final duration;
it must be accepted by the FT owner before using it. If another policy is needed,
coordinate a revised shared runner before formal FT results are produced.

Then the coordinator runs python run.py freeze and shares the updated package.
Adding the FT adapter/settings preserves the already-frozen shared protocol and
LR/LightGBM hashes. Changing existing frozen models or shared code is rejected.
Do not simply delete the lock to mix a different experiment into existing results.

WHERE THESE FILES CAME FROM

Data and the six used split manifests are copied unchanged from the original
paaws_model_v1_1 and paaws_experiment_design_v1. No windows or participant splits
were regenerated. data.py merges the supplied corrected loader with load_split;
it preserves participant order. LR is adapted from the latest supplied shared ZIP,
not the older LR module left in the original project. LightGBM is adapted from the
supplied lightbgm/model.py, including the corrected 1..300 realised-round check.
The larger old output kit is no longer a runtime dependency of this folder.

Original shared ZIP SHA256:
aaac28d875c6025cbaeed545cd9a922afbadb4010524fb84326b718f9933798b

VERIFICATION BEFORE HANDOFF (2026-10-04)

14 automated synthetic tests passed, covering leakage guards, matched reuse,
interruption/recovery, exact outputs, malformed-output rejection, model reloads,
LightGBM split exhaustion, LR convergence, FT epoch handoff, freeze protection,
and participant/seed/fold aggregation with paired confidence intervals.
Both actual LR and LightGBM estimators also completed a synthetic condition each
(nine inner fits plus one fresh refit) through the common runner and output checker.
All 40 feature files and six split files match their originals. All 360 condition
participant assignments, nested subsets and matched FL/Lab counts were checked.
The reference contains exactly 450,475 eligible FL windows with original labels.
Real training/validation loads match the corrected original loader on four
conditions; metric calculations match the originals on 50 confusion matrices.
No full real-data model runs were started. FT remains an unimplemented adapter.
Execution was checked on this Mac; the Windows file-lock branch was not executed.

This is the shared MODELLING package. For the final assignment submission, include
the original feature-construction/preprocessing source as well as this modelling
code, and follow the four-page report requirements. Do not submit data/, model
checkpoints, results containing per-window data, or Python environment folders.
The data folder here is for internal team use, not the code-only submission ZIP.
