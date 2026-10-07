"""Inner-only FT calibration using the shared Runner.tune implementation.

Leaves experiment.json and experiment.lock.json untouched. Calibration results
are isolated from formal model outputs and cannot authorize formal outer runs.
"""
import argparse
import copy
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from data import Condition, ExperimentData
import run

ROOT = Path(__file__).resolve().parent
UPPER_LR = 0.01


def upper_lr_plan(baseline_path, base, current_fingerprint):
    """Select only conditions whose original calibration chose its upper endpoint."""
    baseline_path = Path(baseline_path).resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f'Completed calibration not found: {baseline_path}')
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if baseline.get('status') != 'COMPLETE' or baseline.get('outer_test_loaded') is not False:
        raise ValueError('Upper-LR diagnostic requires a complete inner-only calibration')
    spec = base['models']['ft_transformer']
    if baseline.get('settings') != spec['settings'] or baseline.get('grid') != spec['grid']:
        raise ValueError('Baseline FT settings/grid do not match the frozen FT configuration')
    recorded = baseline.get('fingerprint', {})
    if (recorded.get('shared_sha256') != current_fingerprint['shared_sha256']
            or recorded.get('models', {}).get('ft_transformer')
            != current_fingerprint['models']['ft_transformer']):
        raise ValueError('Baseline does not match the frozen shared protocol and FT model')
    upper = max(float(value) for value in spec['grid'])
    if UPPER_LR <= upper:
        raise ValueError('Diagnostic learning rate must be above the frozen grid')
    chosen = [selection for selection in baseline.get('selections', [])
              if float(selection['value']) == upper]
    if not chosen:
        raise ValueError('The original upper endpoint did not win any calibration condition')
    conditions = [Condition(**selection['condition']) for selection in chosen]
    if len({condition.key for condition in conditions}) != len(conditions):
        raise ValueError('Baseline contains duplicate selected conditions')
    baseline_means = {}
    for selection, condition in zip(chosen, conditions):
        scores = [float(row['participant_balanced_validation_macro_f1'])
                  for row in selection['rows']
                  if float(row['hyperparameter_value']) == upper]
        if len(scores) != 3:
            raise ValueError('Baseline upper endpoint must contain all three inner folds')
        baseline_means[condition.key] = statistics.mean(scores)
    return baseline_path, upper, conditions, baseline_means


def calibrate(output=None, *, execute=False, smoke=False, upper_lr=False, baseline=None):
    if smoke and upper_lr:
        raise ValueError('Choose either smoke or upper-LR diagnostic, not both')
    default = ('analysis/ft_upper_lr_diagnostic_v5' if upper_lr else
               'analysis/ft_smoke_final_v5' if smoke else 'analysis/ft_calibration_final_v5')
    output = Path(output or ROOT / default).resolve()
    base = run.read_config(ROOT)
    # Verify the final adapter/settings as well as the common frozen inputs.
    run.verify_frozen(ROOT, base, 'ft_transformer')
    current_fingerprint = run.fingerprint(ROOT, base)
    config = copy.deepcopy(base)
    settings = copy.deepcopy(base['models']['ft_transformer']['settings'])
    if smoke:
        settings['max_epochs'] = 1
    spec = config['models']['ft_transformer']
    spec.update(status='ready', settings=settings, matched_epoch_policy='reuse_seed17_refit_epochs')
    conditions = [Condition('main', domain, fold, 17, size)
                  for fold in (0, 7) for size in (6, 18) for domain in ('FL', 'Formal_Lab')]
    if smoke:
        conditions = [Condition('main', 'Formal_Lab', 0, 17, 6)]
    baseline_path = baseline_upper = baseline_means = None
    if upper_lr:
        baseline_path, baseline_upper, conditions, baseline_means = upper_lr_plan(
            baseline or ROOT / 'analysis/ft_calibration_final_v5/calibration.json',
            base, current_fingerprint)
        spec['grid'] = [UPPER_LR]
    print(f'{len(conditions)} conditions; {len(conditions)*3*len(spec["grid"])} inner fits; NO outer test/refit', flush=True)
    if not execute:
        return [asdict(c) for c in conditions]
    frozen = run.fingerprint(ROOT, config)
    # Include calibration orchestration in the resume identity without changing the shared lock.
    calibration_source = run.file_digest(Path(__file__))
    runtime = {**run.environment('ft_transformer'), 'calibration_source_sha256': calibration_source,
               'conditions': [asdict(c) for c in conditions], 'smoke_only': smoke,
               'upper_lr_diagnostic': upper_lr}
    recovery = output / 'recovery'
    output.mkdir(parents=True, exist_ok=True)
    recovery.mkdir(parents=True, exist_ok=True)
    def no_test(*args, **kwargs):
        raise RuntimeError('Outer-test access is prohibited in FT calibration')
    with run.single_writer(recovery), patch.object(ExperimentData, 'outer_test', no_test):
        runner = run.Runner(ROOT, config, 'ft_transformer', output / 'unused_formal_outputs', frozen,
                            recovery_dir=recovery, runtime=runtime)
        try:
            selected = []
            for condition in conditions:
                selected.append(runner.tune(condition))
                comparisons = []
                if upper_lr:
                    for result in selected:
                        key = Condition(**result['condition']).key
                        candidate = statistics.mean(float(row['participant_balanced_validation_macro_f1'])
                                                    for row in result['rows'])
                        comparisons.append({
                            'condition': result['condition'],
                            'baseline_learning_rate': baseline_upper,
                            'baseline_mean_macro_f1': baseline_means[key],
                            'candidate_learning_rate': UPPER_LR,
                            'candidate_mean_macro_f1': candidate,
                            'candidate_minus_baseline': candidate - baseline_means[key],
                        })
                run.atomic_json(output / 'calibration.json', {
                    'status': 'SMOKE_ONLY' if smoke else ('COMPLETE' if len(selected)==len(conditions) else 'IN_PROGRESS'),
                    'outer_test_loaded': False, 'settings': settings, 'grid': spec['grid'],
                    'planned_conditions': [asdict(c) for c in conditions], 'selections': selected,
                    'source_sha256': calibration_source, 'fingerprint': frozen,
                    'diagnostic': ({'kind': 'upper_learning_rate',
                                    'baseline_path': str(baseline_path),
                                    'baseline_sha256': run.file_digest(baseline_path),
                                    'comparisons': comparisons} if upper_lr else None)})
            return selected
        finally:
            runner.progress.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--upper-lr-diagnostic', action='store_true',
                        help='Run 0.01 only if the frozen upper endpoint wins a future inner-only calibration')
    parser.add_argument('--baseline', type=Path,
                        help='Completed final calibration.json; default: analysis/ft_calibration_final_v5')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    calibrate(args.output, execute=args.execute, smoke=args.smoke,
              upper_lr=args.upper_lr_diagnostic, baseline=args.baseline)
