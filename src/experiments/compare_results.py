"""
Load all experiment results and generate comparison plots.

Usage:
    python compare_results.py
    python compare_results.py --output-dir results/plots
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from train_experiment import _get_accuracy, _get_metric, ROOT_DIR

REFERENCE_LABEL = 'baseline (ref)'


def _load_reference_accuracy(results_root: pathlib.Path) -> float:
    p = results_root / 'baseline' / 'metrics.json'
    if p.exists():
        return _get_accuracy(json.loads(p.read_text())) * 100
    return 0.0

PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]


def load_results(results_root: pathlib.Path) -> dict:
    results = {}
    for metrics_file in sorted(results_root.glob('*/metrics.json')):
        name = metrics_file.parent.name
        data = json.loads(metrics_file.read_text())
        results[name] = data
    return results


def _val_acc_percent(result: dict) -> list[float]:
    return [v * 100 for v in result['history'].get('val_accuracy', [])]


def plot_val_accuracy_curves(results: dict, out_path: pathlib.Path, ref_acc: float = 0.0):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, data) in enumerate(results.items()):
        val_accs = _val_acc_percent(data)
        if not val_accs:
            continue
        epochs = list(range(1, len(val_accs) + 1))
        color = PALETTE[i % len(PALETTE)]
        ax.plot(epochs, val_accs, label=name, color=color, linewidth=2)
        # Mark best point
        best_epoch = int(np.argmax(val_accs)) + 1
        best_acc = max(val_accs)
        ax.plot(best_epoch, best_acc, 'o', color=color, markersize=6)

    max_epochs = max(
        (len(_val_acc_percent(d)) for d in results.values()), default=30)
    if ref_acc > 0:
        ax.axhline(ref_acc, color='black', linestyle='--',
                   linewidth=1.5, label=f'{REFERENCE_LABEL} ({ref_acc:.2f}%)')
        ax.fill_between([0, max_epochs + 1], ref_acc, 100,
                        color='green', alpha=0.05)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('VWW Person Detection — Validation Accuracy by Epoch', fontsize=14)
    ax.set_xlim(1, max_epochs)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_final_accuracy_bar(results: dict, out_path: pathlib.Path, ref_acc: float = 0.0):
    names = list(results.keys())
    accs = [_get_accuracy(results[n]) * 100 for n in names]

    # Sort descending
    order = np.argsort(accs)[::-1]
    names = [names[i] for i in order]
    accs = [accs[i] for i in order]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=0.8)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    if ref_acc > 0:
        ax.axhline(ref_acc, color='black', linestyle='--',
                   linewidth=1.5, label=f'{REFERENCE_LABEL} ({ref_acc:.2f}%)')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('VWW Person Detection — Final Model Accuracy Comparison', fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

    y_min = max(0, min(accs) - 2)
    y_max = max(max(accs), ref_acc if ref_acc > 0 else max(accs)) + 2
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_training_curves(results: dict, out_path: pathlib.Path, ref_acc: float = 0.0):
    """4-panel plot: val accuracy, val loss, precision, recall."""
    metrics = ['val_accuracy', 'val_loss', 'val_precision', 'val_recall']
    titles = ['Validation Accuracy', 'Validation Loss', 'Precision', 'Recall']
    ylabels = ['Accuracy', 'Loss', 'Precision', 'Recall']

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        for i, (name, data) in enumerate(results.items()):
            values = data['history'].get(metric, [])
            if not values:
                continue
            if metric == 'val_accuracy':
                values = [v * 100 for v in values]
            ax.plot(range(1, len(values) + 1), values,
                    label=name, color=PALETTE[i % len(PALETTE)], linewidth=1.8)

        if metric == 'val_accuracy' and ref_acc > 0:
            ax.axhline(ref_acc, color='black', linestyle='--',
                       linewidth=1.2, label=f'{REFERENCE_LABEL} ({ref_acc:.2f}%)')

        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('VWW Training Curves — All Experiments', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_tflite_size_vs_accuracy(results: dict, out_path: pathlib.Path, ref_acc: float = 0.0):
    """Scatter plot of TFLite model size (KB) vs final val accuracy."""
    names, accs, sizes = [], [], []
    for name, data in results.items():
        tflite_path = ROOT_DIR / 'exported_models' / name / 'model.tflite'
        if not tflite_path.exists():
            continue
        size_kb = tflite_path.stat().st_size / 1024
        acc = _get_accuracy(data) * 100
        names.append(name)
        accs.append(acc)
        sizes.append(size_kb)

    if not names:
        print('No .tflite files found — skipping size vs accuracy plot.')
        print('Run convert_tf_lite.py for each model first.')
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (name, size, acc) in enumerate(zip(names, sizes, accs)):
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(size, acc, color=color, s=100, zorder=3)
        ax.annotate(name, (size, acc), textcoords='offset points',
                    xytext=(6, 4), fontsize=9, color=color)

    if ref_acc > 0:
        ax.axhline(ref_acc, color='black', linestyle='--',
                   linewidth=1.2, label=f'{REFERENCE_LABEL} ({ref_acc:.2f}%)')
    ax.set_xlabel('TFLite Model Size (KB)', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Model Size vs. Accuracy (INT8 quantized)', fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


def print_summary_table(results: dict, ref_acc: float = 0.0):
    print('\n' + '=' * 75)
    print(f'{"Experiment":<22} {"Val Acc":>9} {"AUC":>8} {"Precision":>10} {"Recall":>8}')
    print('-' * 75)
    if ref_acc > 0:
        print(f'{REFERENCE_LABEL:<22} {ref_acc:>8.2f}% {"—":>8} {"—":>10} {"—":>8}')

    rows = []
    for name, data in results.items():
        rows.append((
            name,
            _get_accuracy(data) * 100,
            _get_metric(data, 'auc'),
            _get_metric(data, 'precision'),
            _get_metric(data, 'recall'),
        ))

    for name, acc, auc, prec, rec in sorted(rows, key=lambda r: -r[1]):
        beat = ' *' if ref_acc > 0 and acc > ref_acc and name != 'baseline' else ''
        print(f'{name:<22} {acc:>8.2f}% {auc:>8.4f} {prec:>10.4f} {rec:>8.4f}{beat}')
    print('=' * 75)
    print('  * = beats baseline\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare VWW experiment results')
    parser.add_argument('--results-dir', default=str(ROOT_DIR / 'results'),
                        help='Directory containing experiment subdirs with metrics.json')
    parser.add_argument('--output-dir', default=str(ROOT_DIR / 'results' / 'plots'),
                        help='Directory to save plots')
    args = parser.parse_args()

    results_root = pathlib.Path(args.results_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_root)
    if not results:
        print(f'No results found in {results_root}. Run run_experiments.py first.')
        raise SystemExit(1)

    ref_acc = _load_reference_accuracy(results_root)
    print(f'Found {len(results)} experiments: {", ".join(results)}')
    print_summary_table(results, ref_acc)

    plot_val_accuracy_curves(results, output_dir / 'val_accuracy_curves.png', ref_acc)
    plot_final_accuracy_bar(results, output_dir / 'final_accuracy_bar.png', ref_acc)
    plot_training_curves(results, output_dir / 'training_curves_4panel.png', ref_acc)
    plot_tflite_size_vs_accuracy(results, output_dir / 'size_vs_accuracy.png', ref_acc)
