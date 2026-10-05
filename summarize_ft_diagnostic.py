"""Render the completed diagnostic and distinguish within-run epoch gains."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / 'analysis/ft_epoch100_v4'
data = json.loads((root / 'diagnostic.json').read_text(encoding='utf-8'))
assert data['status'] == 'COMPLETE' and len(data['comparisons']) == 12
fig, axes = plt.subplots(4, 3, figsize=(15, 14), constrained_layout=True)
gains = []
for ax, r in zip(axes.flat, data['comparisons']):
    h = r['history']
    first50 = max(e['validation_macro_f1'] for e in h if e['epoch'] <= 50)
    gain = max(e['validation_macro_f1'] for e in h) - first50
    gains.append(gain)
    ax.plot([e['epoch'] for e in h], [e['validation_macro_f1'] for e in h])
    ax.axvline(50, color='grey', linestyle='--', linewidth=1)
    ax.axvline(r['new_best_epoch'], color='green', linestyle=':', linewidth=1)
    c = r['condition']
    ax.set_title(f"outer{c['outer_fold']} n={c['subset_size']} Lab, lr={r['learning_rate']}\ninner{r['inner_fold']}, best={r['new_best_epoch']}, post50 gain={gain:.4f}", fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Participant Macro-F1')
    ax.grid(alpha=.2)
fig.savefig(root / 'epoch_curves.png', dpi=130)
plt.close(fig)
lines = ['# 100-epoch 诊断结论（2026-10-05）', '',
    '已对齐远程 main 的 aff250b（v4）。12 次诊断完成，未读取 outer test；原 72 次校准保留。', '',
    '| outer / n / inner / LR | 原 best epoch | 新 best epoch | 本次实际轮数 | 本次 50 轮后的最佳分数增益 |',
    '|---|---:|---:|---:|---:|']
for r, gain in zip(data['comparisons'], gains):
    c = r['condition']
    lines.append(f"| {c['outer_fold']} / {c['subset_size']} / {r['inner_fold']} / {r['learning_rate']} | {r['old_best_epoch']} | {r['new_best_epoch']} | {r['new_epochs_run']} | {gain:.6f} |")
lines += ['', f"累计 fit 时间：{sum(r['fit_seconds'] for r in data['comparisons']):.1f} 秒（低优先级运行，不宜与原计时作严格性能比较）。", '',
    '建议将 FT 的候选正式 max_epochs 提高至 100，patience 仍为 8。至少一条曲线在 50 轮后出现更优点；增加上限并不意味着每次都训练 100 轮。',
    '学习率网格暂保留 [0.0001, 0.0003, 0.001]。此前 8 条件有 5 次选中间值；不能强制每个条件选中间值。若正式采用 100 上限，应以同一上限复查完整 8 条件再更新最终校准结论，不能把本次选择性补测与原分数混合。',
    '这次从头重跑。前 50 轮与原运行存在随机/数值差异，因此上表用本次运行内部的 post-50 gain 作为更直接证据。',
    '正式 experiment.json 与锁文件保持 Git v4 原样，FT 仍 pending；没有启动正式 outer 实验或推送远程。',
    'v4 共享测试 27 项：25 通过、2 项 Windows SQLite 文件占用错误。FT 适配器保存/恢复和精确 refit 轮数测试通过。',
    '', '曲线：epoch_curves.png；完整结果：diagnostic.json。']
(root / '结论与后续.md').write_text('\n'.join(lines), encoding='utf-8')
print(root / '结论与后续.md')
