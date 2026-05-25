from train_experiment import ModelConfig

# Reference: TF Lite Micro person detection model achieves ~89.9% val accuracy
# using MobileNet V1 alpha=0.25 at 96x96 on the VWW/COCO dataset.
REFERENCE_ACCURACY = 89.9

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
        extra_notes='Replication of original simple_train.py',
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
        extra_notes='Strong augmentation: flip+crop+hue+saturation',
    ),

    ModelConfig(
        name='finetune',
        architecture='mobilenetv1',
        alpha=0.25,
        input_height=96,
        input_width=96,
        epochs=40,
        batch_size=128,
        learning_rate=1e-3,
        augmentation='strong',
        loss='bce',
        finetune_after_epoch=20,
        finetune_layers=14,   # unfreeze last 14 layers of MobileNetV1
        finetune_lr_scale=0.1,
        extra_notes='Phase 1: frozen backbone 20 epochs; Phase 2: unfreeze top layers 20 epochs',
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
        extra_notes='MobileNet V2 alpha=0.35 — inverted residuals, better acc/size tradeoff',
    ),

    ModelConfig(
        name='focal_loss',
        architecture='mobilenetv1',
        alpha=0.25,
        input_height=96,
        input_width=96,
        epochs=30,
        batch_size=128,
        learning_rate=1e-3,
        augmentation='strong',
        loss='focal',
        focal_gamma=2.0,
        extra_notes='Focal loss (gamma=2) to down-weight easy negatives',
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
        extra_notes='MobileNet V1 alpha=0.5 — higher capacity ceiling test',
    ),
]
