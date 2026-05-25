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

REFERENCE_ACCURACY = 89.9
REFERENCE_LABEL = 'TFLite Micro ref'

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


def plot_val_accuracy_curves(results: dict, out_path: pathlib.Path):
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
    ax.axhline(REFERENCE_ACCURACY, color='black', linestyle='--',
               linewidth=1.5, label=REFERENCE_LABEL)
    ax.fill_between([0, max_epochs + 1], REFERENCE_ACCURACY, 100,
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


def plot_final_accuracy_bar(results: dict, out_path: pathlib.Path):
    names = list(results.keys())
    accs = [results[n]['final_metrics'].get('accuracy', 0) * 100 for n in names]

    # Sort descending
    order = np.argsort(accs)[::-1]
    names = [names[i] for i in order]
    accs = [accs[i] for i in order]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=0.8)

    # Annotate bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(REFERENCE_ACCURACY, color='black', linestyle='--',
               linewidth=1.5, label=f'{REFERENCE_LABEL} ({REFERENCE_ACCURACY}%)')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('VWW Person Detection — Final Model Accuracy Comparison', fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

    y_min = max(0, min(accs) - 2)
    y_max = max(max(accs), REFERENCE_ACCURACY) + 2
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_training_curves(results: dict, out_path: pathlib.Path):
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

        if metric == 'val_accuracy':
            ax.axhline(REFERENCE_ACCURACY, color='black', linestyle='--',
                       linewidth=1.2, label=REFERENCE_LABEL)

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


def plot_tflite_size_vs_accuracy(results: dict, out_path: pathlib.Path):
    """Scatter plot of TFLite model size (KB) vs final val accuracy."""
    names, accs, sizes = [], [], []
    for name, data in results.items():
        tflite_path = pathlib.Path(f'exported_models/{name}/model.tflite')
        if not tflite_path.exists():
            continue
        size_kb = tflite_path.stat().st_size / 1024
        acc = data['final_metrics'].get('accuracy', 0) * 100
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

    ax.axhline(REFERENCE_ACCURACY, color='black', linestyle='--',
               linewidth=1.2, label=f'{REFERENCE_LABEL} ({REFERENCE_ACCURACY}%)')
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


def print_summary_table(results: dict):
    print('\n' + '=' * 75)
    print(f'{"Experiment":<22} {"Val Acc":>9} {"AUC":>8} {"Precision":>10} {"Recall":>8}')
    print('-' * 75)
    print(f'{"[TFLite Micro ref]":<22} {REFERENCE_ACCURACY:>8.2f}% {"—":>8} {"—":>10} {"—":>8}')

    rows = []
    for name, data in results.items():
        fm = data['final_metrics']
        rows.append((
            name,
            fm.get('accuracy', 0) * 100,
            fm.get('auc', 0),
            fm.get('precision', 0),
            fm.get('recall', 0),
        ))

    for name, acc, auc, prec, rec in sorted(rows, key=lambda r: -r[1]):
        beat = ' *' if acc > REFERENCE_ACCURACY else ''
        print(f'{name:<22} {acc:>8.2f}% {auc:>8.4f} {prec:>10.4f} {rec:>8.4f}{beat}')
    print('=' * 75)
    print('  * = beats TFLite Micro reference\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare VWW experiment results')
    parser.add_argument('--results-dir', default='results',
                        help='Directory containing experiment subdirs with metrics.json')
    parser.add_argument('--output-dir', default='results/plots',
                        help='Directory to save plots')
    args = parser.parse_args()

    results_root = pathlib.Path(args.results_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_root)
    if not results:
        print(f'No results found in {results_root}. Run run_experiments.py first.')
        raise SystemExit(1)

    print(f'Found {len(results)} experiments: {", ".join(results)}')
    print_summary_table(results)

    plot_val_accuracy_curves(results, output_dir / 'val_accuracy_curves.png')
    plot_final_accuracy_bar(results, output_dir / 'final_accuracy_bar.png')
    plot_training_curves(results, output_dir / 'training_curves_4panel.png')
    plot_tflite_size_vs_accuracy(results, output_dir / 'size_vs_accuracy.png')
