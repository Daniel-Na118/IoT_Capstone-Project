"""
Run all VWW model experiments sequentially and print a summary table.

Usage:
    python run_experiments.py
    python run_experiments.py --only baseline finetune   # run specific ones
    python run_experiments.py --skip baseline             # skip specific ones
"""
import argparse
import json
import pathlib
import traceback

from experiment_configs import EXPERIMENTS, REFERENCE_ACCURACY
from train_experiment import train


def run_all(names_to_run: list[str], names_to_skip: list[str]):
    summary = []

    for config in EXPERIMENTS:
        if names_to_run and config.name not in names_to_run:
            continue
        if config.name in names_to_skip:
            print(f'Skipping {config.name}')
            continue

        # Resume: skip if metrics already saved
        metrics_path = pathlib.Path('results') / config.name / 'metrics.json'
        if metrics_path.exists():
            print(f'\nLoading existing results for {config.name} ...')
            result = json.loads(metrics_path.read_text())
        else:
            try:
                result = train(config)
            except Exception:
                print(f'ERROR training {config.name}:')
                traceback.print_exc()
                continue

        acc = result['final_metrics'].get('accuracy', 0) * 100
        auc = result['final_metrics'].get('auc', 0)
        elapsed = result.get('training_time_seconds', 0)
        summary.append((config.name, acc, auc, elapsed))

    print('\n' + '=' * 65)
    print(f'{"Experiment":<22} {"Val Acc %":>10} {"AUC":>8} {"Time (min)":>12}')
    print('-' * 65)
    print(f'{"[TFLite Micro ref]":<22} {REFERENCE_ACCURACY:>10.2f} {"—":>8} {"—":>12}')
    for name, acc, auc, elapsed in sorted(summary, key=lambda r: -r[1]):
        beat = ' *' if acc > REFERENCE_ACCURACY else ''
        print(f'{name:<22} {acc:>10.2f} {auc:>8.4f} {elapsed/60:>10.1f}m{beat}')
    print('=' * 65)
    print('  * = beats TFLite Micro reference')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VWW experiments')
    parser.add_argument('--only', nargs='+', default=[],
                        help='Run only these experiment names')
    parser.add_argument('--skip', nargs='+', default=[],
                        help='Skip these experiment names')
    args = parser.parse_args()
    run_all(args.only, args.skip)
