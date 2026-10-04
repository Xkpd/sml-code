"""FT-Transformer integration interface for the shared experiment runner.

Implement these four functions without adding a separate CV or output-writing loop.
The common runner supplies the fixed splits, tuning candidate, seed, settings and
training-only mean-one class weights. It also writes and checks both result files.

For each fit:
* Fit data.fit_standardizer(train.X) using only train and retain it in state.
* Apply the supplied sample_weight to each example's training loss.
* With validation present, select checkpoints using the unweighted, participant-
  balanced score from metrics.participant_macro_f1. Return best_epoch in info.
* With validation=None, train a NEW model for exactly the runner's epochs value.
  Do not invent a validation split or use outer-test data. Return epochs_run in info.
* predict_proba applies the saved scaler and returns N x 4 probabilities in class
  ID order 0, 1, 2, 3 (Sitting, Standing, Lying_Down, Walking), without rounding.
* serialize must retain the scaler, architecture and all fitted parameters. Move
  tensors to CPU for portable checkpoints. deserialize must reproduce predictions.

The runner owns learning-rate selection across three inner folds and the agreed
final/matched epoch policy. Hardware choices must not change that study protocol.
This placeholder deliberately does not import PyTorch or reuse a partial benchmark.
"""


def _unavailable():
    raise NotImplementedError(
        "FT-Transformer is not integrated yet. Implement models/ft_transformer.py "
        "and confirm its settings before a full FT run."
    )


def fit(train, validation, *, value, seed, settings, sample_weight, epochs=None):
    """Return (fitted state, {'best_epoch': ...[, 'epochs_run': ...]})."""
    _unavailable()


def predict_proba(state, X):
    _unavailable()


def serialize(state):
    _unavailable()


def deserialize(payload):
    _unavailable()
