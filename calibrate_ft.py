"""Inner-only FT calibration using the shared Runner.tune implementation.

Leaves experiment.json and experiment.lock.json untouched. Calibration results
are isolated from formal model outputs and cannot authorize formal outer runs.
"""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from data import Condition, ExperimentData
import run

ROOT = Path(__file__).resolve().parent


def calibrate(output=None, *, execute=False, smoke=False):
    default = 'analysis/ft_smoke_final_v4' if smoke else 'analysis/ft_calibration_final_v4'
    output = Path(output or ROOT / default).resolve()
    base = run.read_config(ROOT)
    # Verify the final adapter/settings as well as the common frozen inputs.
    run.verify_frozen(ROOT, base, 'ft_transformer')
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
    print(f'{len(conditions)} conditions; {len(conditions)*3*len(spec["grid"])} inner fits; NO outer test/refit', flush=True)
    if not execute:
        return [asdict(c) for c in conditions]
    frozen = run.fingerprint(ROOT, config)
    # Include calibration orchestration in the resume identity without changing the shared lock.
    calibration_source = run.file_digest(Path(__file__))
    runtime = {**run.environment('ft_transformer'), 'calibration_source_sha256': calibration_source,
               'conditions': [asdict(c) for c in conditions], 'smoke_only': smoke}
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
                run.atomic_json(output / 'calibration.json', {
                    'status': 'SMOKE_ONLY' if smoke else ('COMPLETE' if len(selected)==len(conditions) else 'IN_PROGRESS'),
                    'outer_test_loaded': False, 'settings': settings, 'grid': spec['grid'],
                    'planned_conditions': [asdict(c) for c in conditions], 'selections': selected,
                    'source_sha256': calibration_source, 'fingerprint': frozen})
            return selected
        finally:
            runner.progress.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    calibrate(args.output, execute=args.execute, smoke=args.smoke)
