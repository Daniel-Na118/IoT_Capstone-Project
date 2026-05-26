from train_experiment import ModelConfig

# REFERENCE_ACCURACY is loaded dynamically from results/baseline/metrics.json
# at runtime in run_experiments.py and compare_results.py.
# This fallback is only used if baseline hasn't been run yet.
REFERENCE_ACCURACY_FALLBACK = 0.0

EXPERIMENTS = [
    ModelConfig(
        name='baseline',
        architecture='mobilenetv1',
        alpha=0.25,
        input_height=96,
        input_width=96,
        epochs=30,
        batch_size=128,
        learning_rate=1e-3,
        augmentation='basic',
        loss='bce',
        extra_notes='MobileNet V1 0.25 @ 96x96, basic aug — the reference to beat',
    ),

    ModelConfig(
        name='aug_v2',
        architecture='mobilenetv1',
        alpha=0.25,
        input_height=96,
        input_width=96,
        epochs=30,
        batch_size=128,
        learning_rate=1e-3,
        augmentation='strong',
        loss='bce',
        extra_notes='Same as baseline but with moderate color jitter + small random crop',
    ),

    ModelConfig(
        name='v2_basic',
        architecture='mobilenetv2',
        alpha=0.35,
        input_height=96,
        input_width=96,
        epochs=30,
        batch_size=128,
        learning_rate=1e-3,
        augmentation='basic',
        loss='bce',
        extra_notes='MobileNet V2 0.35 with basic aug — isolates architecture gain from augmentation',
    ),

    ModelConfig(
        name='mobilenetv2_035',
        architecture='mobilenetv2',
        alpha=0.35,
        input_height=96,
        input_width=96,
        epochs=30,
        batch_size=128,
        learning_rate=1e-3,
        augmentation='strong',
        loss='bce',
        extra_notes='MobileNet V2 0.35 + moderate strong aug',
    ),

    ModelConfig(
        name='v1_128',
        architecture='mobilenetv1',
        alpha=0.25,
        input_height=128,
        input_width=128,
        epochs=30,
        batch_size=64,
        learning_rate=1e-3,
        augmentation='basic',
        loss='bce',
        extra_notes='MobileNet V1 0.25 at 128x128 — tests whether larger input helps',
    ),

    ModelConfig(
        name='v1_alpha05',
        architecture='mobilenetv1',
        alpha=0.5,
        input_height=96,
        input_width=96,
        epochs=30,
        batch_size=128,
        learning_rate=1e-3,
        augmentation='strong',
        loss='bce',
        extra_notes='MobileNet V1 alpha=0.5 — more capacity at same resolution',
    ),
]
