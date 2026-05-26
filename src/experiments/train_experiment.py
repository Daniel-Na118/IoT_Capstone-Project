import tensorflow as tf
import pathlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

tf.get_logger().setLevel('ERROR')

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]

TFRECORD_ROOTS = {
    'coco2017':    ROOT_DIR / 'data/vww_tfrecord/coco2017',
    'wake_vision': ROOT_DIR / 'data/vww_tfrecord/wake_vision',
}
TFRECORD_PREFIX = {
    'coco2017':    'coco',
    'wake_vision': 'wake_vision',
}


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
    dataset: str = 'coco2017'          # 'coco2017' | 'wake_vision'
    optimizer: str = 'sgd'             # 'sgd' | 'adam'
    weights: str = 'imagenet'          # 'imagenet' | 'none'
    focal_gamma: float = 2.0
    finetune_after_epoch: Optional[int] = None  # None = no fine-tuning
    finetune_layers: int = 0           # layers to unfreeze; -1 = unfreeze all
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


def _augment_strong(image, label, height: int, width: int):
    # Convert from [-1,1] to [0,1] for color ops, then back
    img01 = (image + 1.0) / 2.0
    img01 = tf.image.random_flip_left_right(img01)
    # No vertical flip
    img01 = tf.image.random_brightness(img01, 0.2)
    img01 = tf.image.random_contrast(img01, 0.8, 1.2)
    img01 = tf.image.random_hue(img01, 0.05)
    img01 = tf.image.random_saturation(img01, 0.8, 1.2)
    img01 = tf.clip_by_value(img01, 0.0, 1.0)
    # Random crop: pad by 6px
    img01 = tf.image.resize_with_crop_or_pad(img01, height + 6, width + 6)
    img01 = tf.image.random_crop(img01, [height, width, 3])
    image = img01 * 2.0 - 1.0
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def load_dataset(config: ModelConfig, split: str) -> tf.data.Dataset:
    root = TFRECORD_ROOTS.get(config.dataset)
    prefix = TFRECORD_PREFIX.get(config.dataset)
    if root is None:
        raise ValueError(f'Unknown dataset: {config.dataset}')
    filenames = sorted(str(p) for p in root.glob(f'{prefix}_{split}.record*'))
    if not filenames:
        raise FileNotFoundError(
            f'No TFRecords found at {root}/{prefix}_{split}.record*\n'
            f'  For wake_vision, run: python download_wake_vision.py'
        )
    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.map(lambda ex: _parse_record(ex, config.input_height, config.input_width),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x, y: tf.shape(x)[2] == 3)
    if split == 'train':
        if config.augmentation == 'strong':
            h, w = config.input_height, config.input_width
            aug_fn = lambda img, lbl: _augment_strong(img, lbl, h, w)
        else:
            aug_fn = _augment_basic
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
    weights = None if config.weights == 'none' else config.weights

    if config.architecture == 'mobilenetv1':
        backbone = tf.keras.applications.MobileNet(
            input_shape=shape, alpha=config.alpha,
            include_top=False, weights=weights)
    elif config.architecture == 'mobilenetv2':
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=shape, alpha=config.alpha,
            include_top=False, weights=weights)
    else:
        raise ValueError(f'Unknown architecture: {config.architecture}')

    inputs = tf.keras.Input(shape)
    x = backbone(inputs)
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
    if config.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule, momentum=0.9, nesterov=True)

    if config.loss == 'focal':
        loss = _focal_loss(gamma=config.focal_gamma)
    else:
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=config.label_smoothing)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )


def train(config: ModelConfig) -> dict:
    results_dir = ROOT_DIR / 'results' / config.name
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = ROOT_DIR / 'models' / config.name
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
    t0 = time.time()

    if config.finetune_after_epoch and config.finetune_layers != 0:
        # Freeze backbone, warm up head only
        model.layers[1].trainable = False
        _compile(model, config, config.learning_rate)
        phase1_epochs = config.finetune_after_epoch
        print(f'Phase 1: warming up head ({phase1_epochs} epochs, backbone frozen)...')
        h1 = model.fit(train_ds, validation_data=val_ds,
                       epochs=phase1_epochs, callbacks=callbacks, verbose=1)
        _merge_history(all_history, h1.history)

        # Unfreeze backbone for training
        backbone = model.layers[1]
        backbone.trainable = True
        if config.finetune_layers != -1:
            for layer in backbone.layers[:-config.finetune_layers]:
                layer.trainable = False
        fine_lr = config.learning_rate * config.finetune_lr_scale
        _compile(model, config, fine_lr)
        remaining = config.epochs - phase1_epochs
        layers_desc = 'all layers' if config.finetune_layers == -1 else f'top {config.finetune_layers} layers'
        print(f'Phase 2: fine-tuning backbone {layers_desc} '
              f'({remaining} epochs, lr={fine_lr:.2e})...')
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=remaining,
                       initial_epoch=0,
                       callbacks=callbacks, verbose=1)
        _merge_history(all_history, h2.history)
    else:
        # --- Full end-to-end training (backbone fully trainable) ---
        print(f'Training end-to-end ({config.epochs} epochs, backbone trainable)...')
        h1 = model.fit(train_ds, validation_data=val_ds,
                       epochs=config.epochs, callbacks=callbacks, verbose=1)
        _merge_history(all_history, h1.history)

    elapsed = time.time() - t0

    print('Evaluating best checkpoint...')
    model.load_weights(ckpt_path)
    final_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)

    # Best-epoch metrics from history as a reliable fallback / cross-check
    best_epoch = int(max(range(len(all_history['val_accuracy'])),
                         key=lambda i: all_history['val_accuracy'][i]))
    best_epoch_metrics = {
        k.removeprefix('val_'): v[best_epoch]
        for k, v in all_history.items() if k.startswith('val_')
    }

    output = {
        'config': asdict(config),
        'history': all_history,
        'final_metrics': final_metrics,
        'best_epoch_metrics': best_epoch_metrics,
        'best_epoch': best_epoch + 1,
        'training_time_seconds': elapsed,
    }
    metrics_path = results_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(output, indent=2))
    print(f'Results saved to {metrics_path}')
    acc = _get_accuracy(output)
    print(f'Final val accuracy: {acc:.4f}')

    # Export SavedModel
    export_path = ROOT_DIR / 'exported_models' / config.name / 'saved_model'
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(export_path))
    print(f'Model exported to {export_path}')

    return output


def _merge_history(target: dict, source: dict):
    for k, v in source.items():
        target.setdefault(k, []).extend(v)


def _get_accuracy(result: dict) -> float:
    bm = result.get('best_epoch_metrics', {})
    if bm.get('accuracy', 0) > 0:
        return bm['accuracy']
    # Fall back to final_metrics
    fm = result.get('final_metrics', {})
    if fm.get('accuracy', 0) > 0:
        return fm['accuracy']
    # fallback 
    if fm.get('compile_metrics', 0) > 0:
        return fm['compile_metrics']
    # Final fallback: best value from raw history
    hist_vals = result.get('history', {}).get('val_accuracy', [])
    return max(hist_vals) if hist_vals else 0.0


def _get_metric(result: dict, name: str) -> float:
    # Return a named metric from best_epoch_metrics, final_metrics, or history
    bm = result.get('best_epoch_metrics', {})
    if name in bm:
        return bm[name]
    fm = result.get('final_metrics', {})
    if name in fm:
        return fm[name]
    hist_vals = result.get('history', {}).get(f'val_{name}', [])
    return max(hist_vals) if hist_vals else 0.0
