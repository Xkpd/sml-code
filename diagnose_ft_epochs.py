"""Isolated, sequential 100-epoch inner-only diagnostic on the v4 runner."""
import copy
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import run
from data import Condition, ExperimentData, load_split
from models.ft_transformer import FTConfig

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'analysis/ft_epoch100_v4'
BASELINE = ROOT / 'analysis/ft_calibration_v1/recovery/progress.sqlite3'


def main():
    process = psutil.Process()
    if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'):
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    base = run.read_config(ROOT)
    run.verify_frozen(ROOT, base, 'lightgbm')
    cfg = copy.deepcopy(base)
    settings = asdict(FTConfig(max_epochs=100))
    settings.pop('seed')
    cfg['models']['ft_transformer'].update(status='ready', settings=settings,
        matched_epoch_policy='reuse_seed17_refit_epochs')
    plan = [(Condition('main', 'Formal_Lab', 7, 17, 6), lr, fold)
            for lr in base['models']['ft_transformer']['grid'] for fold in range(3)]
    plan += [(Condition('main', 'Formal_Lab', 0, 17, 18), .0001, fold) for fold in range(3)]
    frozen = run.fingerprint(ROOT, cfg)
    runtime = {**run.environment('ft_transformer'), 'diagnostic_source': run.file_digest(Path(__file__)),
               'baseline_sha256': run.file_digest(BASELINE),
               'plan': [[asdict(c), lr, fold] for c, lr, fold in plan]}
    OUT.mkdir(parents=True, exist_ok=True)
    recovery = OUT / 'recovery'
    recovery.mkdir(exist_ok=True)
    def forbidden(*args, **kwargs):
        raise RuntimeError('Outer testing is forbidden during epoch diagnostics')
    db = sqlite3.connect(BASELINE.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        baseline = {key: json.loads(value) for key, value in db.execute(
            'SELECT key,value FROM items WHERE kind=?', ('inner',))}
    finally:
        db.close()
    with run.single_writer(recovery), patch.object(ExperimentData, 'outer_test', forbidden):
        runner = run.Runner(ROOT, cfg, 'ft_transformer', OUT / 'unused_formal_outputs', frozen,
                            recovery_dir=recovery, runtime=runtime)
        try:
            comparisons = []
            for condition, lr, fold in plan:
                key = f'{condition.key}__inner{fold}__value{lr}'
                saved = runner.progress.get('epoch_diagnostic', key)
                if saved is None:
                    print('DIAGNOSTIC', key, flush=True)
                    split = load_split(runner.data, condition, inner_fold=fold)
                    state, info = runner.fit(split, lr, condition.subset_seed)
                    del state, split
                    runner.progress.put('epoch_diagnostic', key, run.json_bytes(info))
                else:
                    info = json.loads(saved)
                old = baseline[key]['fit']
                old_history, new_history = old['history'], info['history']
                old_score = max(x['validation_macro_f1'] for x in old_history)
                new_score = max(x['validation_macro_f1'] for x in new_history)
                prefix = new_history[:min(50, len(new_history), len(old_history))]
                prefix_delta = max(abs(a['validation_macro_f1']-b['validation_macro_f1'])
                                   for a, b in zip(old_history, prefix))
                comparisons.append(dict(key=key, condition=asdict(condition), learning_rate=lr,
                    inner_fold=fold, old_best_epoch=old['best_epoch'], new_best_epoch=info['best_epoch'],
                    old_epochs_run=old['epochs_run'], new_epochs_run=info['epochs_run'],
                    old_score=old_score, new_score=new_score, delta=new_score-old_score,
                    max_common_prefix_score_difference=prefix_delta,
                    fit_seconds=info['fit_seconds'], history=new_history))
                run.atomic_json(OUT / 'diagnostic.json', dict(
                    status='COMPLETE' if len(comparisons)==len(plan) else 'IN_PROGRESS',
                    outer_test_loaded=False, max_epochs=100, patience=8,
                    fingerprint=frozen, runtime=runtime, comparisons=comparisons))
                time.sleep(.25)
            report(comparisons)
        finally:
            runner.progress.close()


def report(rows):
    lines = ['# FT epoch-limit diagnostic (v4)\n',
        '12 inner fits; max_epochs=100; patience=8; no outer-test access. Original v3 results preserved.\n',
        '| Condition / LR / fold | Best epoch: old → new | Epochs run | F1 change |',
        '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['key']} | {r['old_best_epoch']} → {r['new_best_epoch']} | {r['new_epochs_run']} | {r['delta']:+.6f} |")
    lines += ['', f"Best epoch beyond 50: {sum(r['new_best_epoch'] > 50 for r in rows)}/12.",
              f"Hit 100-epoch cap: {sum(r['new_epochs_run']==100 for r in rows)}/12.",
              f"Cumulative fit seconds: {sum(r['fit_seconds'] for r in rows):.2f}.",
              '', 'This is a rerun, not checkpoint continuation. Common-prefix score differences are recorded to distinguish stochastic rerun variation from evidence after epoch 50. Do not replace the original 8-condition calibration scores with these selectively repeated results.']
    (OUT / 'diagnostic_report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
