import tensorflow as tf
import pathlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

tf.get_logger().setLevel('ERROR')

TFRECORD_DIR = pathlib.Path('data/vww_tfrecord/coco2017')


@dataclass
class ModelConfig:
    name: str
    architecture: str        # 'mobilenetv1' | 'mobilenetv2'
    alpha: float
    input_height: int
    input_width: int
    epochs: int
    batch_size: int
    learning_rate: float
    augmentation: str        # 'basic' | 'strong'
    loss: str                # 'bce' | 'focal'
    focal_gamma: float = 2.0
    finetune_after_epoch: Optional[int] = None  # None = no fine-tuning
    finetune_layers: int = 0                    # layers to unfreeze from top
    finetune_lr_scale: float = 0.1
    dropout_rate: float = 0.0
    label_smoothing: float = 0.0
    extra_notes: str = ''


def _parse_record(example, height, width):
    parsed = tf.io.parse_example(
        example[tf.newaxis], {
            'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'image/class':   tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        })
    img = tf.io.decode_jpeg(parsed['image/encoded'][0])
    img = tf.image.resize(img, (height, width))
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    label = parsed['image/class']
    return img, label


def _augment_basic(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def _augment_strong(image, label):
    # Convert from [-1,1] to [0,1] for color ops, then back
    img01 = (image + 1.0) / 2.0
    img01 = tf.image.random_flip_left_right(img01)
    img01 = tf.image.random_flip_up_down(img01)
    img01 = tf.image.random_brightness(img01, 0.3)
    img01 = tf.image.random_contrast(img01, 0.7, 1.3)
    img01 = tf.image.random_hue(img01, 0.1)
    img01 = tf.image.random_saturation(img01, 0.7, 1.3)
    img01 = tf.clip_by_value(img01, 0.0, 1.0)
    # Random crop: pad by 12px then crop back to original size
    h = tf.shape(img01)[0]
    w = tf.shape(img01)[1]
    img01 = tf.image.resize_with_crop_or_pad(img01, h + 12, w + 12)
    img01 = tf.image.random_crop(img01, (h, w, 3))
    image = img01 * 2.0 - 1.0
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def load_dataset(config: ModelConfig, split: str) -> tf.data.Dataset:
    filenames = sorted(str(p) for p in TFRECORD_DIR.glob(f'coco_{split}.record*'))
    if not filenames:
        raise FileNotFoundError(f'No TFRecords found at {TFRECORD_DIR}/coco_{split}.record*')
    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.map(lambda ex: _parse_record(ex, config.input_height, config.input_width),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x, y: tf.shape(x)[2] == 3)
    if split == 'train':
        aug_fn = _augment_strong if config.augmentation == 'strong' else _augment_basic
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def _focal_loss(gamma=2.0):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')

    def loss_fn(y_true, y_pred):
        bce_val = bce(y_true, y_pred)
        p_t = tf.exp(-bce_val)
        return tf.reduce_mean((1.0 - p_t) ** gamma * bce_val)

    loss_fn.__name__ = f'focal_loss_g{gamma}'
    return loss_fn


def build_model(config: ModelConfig) -> tf.keras.Model:
    shape = (config.input_height, config.input_width, 3)

    if config.architecture == 'mobilenetv1':
        backbone = tf.keras.applications.MobileNet(
            input_shape=shape, alpha=config.alpha,
            include_top=False, weights='imagenet')
    elif config.architecture == 'mobilenetv2':
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=shape, alpha=config.alpha,
            include_top=False, weights='imagenet')
    else:
        raise ValueError(f'Unknown architecture: {config.architecture}')

    backbone.trainable = False

    inputs = tf.keras.Input(shape)
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if config.dropout_rate > 0:
        x = tf.keras.layers.Dropout(config.dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation=None)(x)
    return tf.keras.Model(inputs, outputs)


def _compile(model: tf.keras.Model, config: ModelConfig, lr: float):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=10000,
        t_mul=1.0,
        m_mul=0.95,
        alpha=0.0,
    )
    if config.loss == 'focal':
        loss = _focal_loss(gamma=config.focal_gamma)
    else:
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=config.label_smoothing)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=lr_schedule, momentum=0.9, nesterov=True),
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )


def train(config: ModelConfig) -> dict:
    results_dir = pathlib.Path('results') / config.name
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = pathlib.Path('models') / config.name
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(weights_dir / 'best_val.weights.h5')

    print(f'\n{"="*60}')
    print(f'  Experiment: {config.name}')
    print(f'  {config.architecture} alpha={config.alpha}  '
          f'{config.input_height}x{config.input_width}  '
          f'aug={config.augmentation}  loss={config.loss}')
    if config.extra_notes:
        print(f'  Notes: {config.extra_notes}')
    print(f'{"="*60}')

    print('Loading datasets...')
    train_ds = (load_dataset(config, 'train')
                .shuffle(2048)
                .batch(config.batch_size)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (load_dataset(config, 'val')
              .batch(config.batch_size)
              .prefetch(tf.data.AUTOTUNE))

    print('Building model...')
    model = build_model(config)
    _compile(model, config, config.learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, save_weights_only=True,
            monitor='val_accuracy', mode='max', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=7,
            restore_best_weights=True, mode='max'),
    ]

    all_history = {}

    # --- Phase 1: frozen backbone ---
    phase1_epochs = config.finetune_after_epoch if config.finetune_after_epoch else config.epochs
    print(f'Phase 1: training head ({phase1_epochs} epochs, backbone frozen)...')
    t0 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=phase1_epochs, callbacks=callbacks, verbose=1)
    _merge_history(all_history, h1.history)

    # --- Phase 2: optional fine-tuning ---
    if config.finetune_after_epoch and config.finetune_layers > 0:
        backbone = model.layers[1]  # backbone is the second layer
        backbone.trainable = True
        for layer in backbone.layers[:-config.finetune_layers]:
            layer.trainable = False

        fine_lr = config.learning_rate * config.finetune_lr_scale
        _compile(model, config, fine_lr)

        remaining = config.epochs - phase1_epochs
        print(f'Phase 2: fine-tuning top {config.finetune_layers} backbone layers '
              f'({remaining} epochs, lr={fine_lr:.2e})...')
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=remaining,
                       initial_epoch=0,
                       callbacks=callbacks, verbose=1)
        _merge_history(all_history, h2.history)

    elapsed = time.time() - t0

    # Evaluate final best model
    print('Evaluating best checkpoint...')
    model.load_weights(ckpt_path)
    eval_results = model.evaluate(val_ds, verbose=0)
    metric_names = model.metrics_names
    final_metrics = dict(zip(metric_names, eval_results))

    output = {
        'config': asdict(config),
        'history': all_history,
        'final_metrics': final_metrics,
        'training_time_seconds': elapsed,
    }
    metrics_path = results_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(output, indent=2))
    print(f'Results saved to {metrics_path}')
    print(f'Final val accuracy: {final_metrics.get("accuracy", 0):.4f}')

    # Export SavedModel
    export_path = f'exported_models/{config.name}/saved_model'
    pathlib.Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    model.export(export_path)
    print(f'Model exported to {export_path}')

    return output


def _merge_history(target: dict, source: dict):
    for k, v in source.items():
        target.setdefault(k, []).extend(v)
