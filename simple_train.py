import tensorflow as tf
import pathlib
from collections import namedtuple

tf.get_logger().setLevel('ERROR')
ImageShape = namedtuple('ImageShape', 'height width channels')

# Training configuration
DATASET_NAME = "coco2017"
INPUT_HEIGHT = 96
INPUT_WIDTH = 96
ALPHA = 0.25  # MobileNet depth multiplier
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MODEL_NAME = f"vww_mobilenet_simple_{ALPHA}_{INPUT_HEIGHT}_{INPUT_WIDTH}"

def _example_to_tensors(example, input_shape):
    """Parse TFRecord and extract image and label."""
    example = tf.io.parse_example(
        example[tf.newaxis], {
            'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'image/class': tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        })
    img_tensor = tf.io.decode_jpeg(example['image/encoded'][0])
    img_tensor = tf.image.resize(img_tensor, size=(input_shape.height, input_shape.width))
    # Normalize to [-1, 1]
    img_tensor = img_tensor / 127.5 - 1.0
    label = example['image/class']
    return img_tensor, label

def augment_image(image, label):
    """Apply data augmentation."""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    
    image = tf.image.random_brightness(image, 0.2)
    
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label

def load_dataset(dataset_name, input_shape, split="train"):
    """Load and prepare dataset."""
    datadir = pathlib.Path('data/vww_tfrecord') / dataset_name
    filenames = [str(p) for p in datadir.glob(f"coco_{split}.record*")]
    dataset = tf.data.TFRecordDataset(filenames)
    
    def _map_fn(example):
        return _example_to_tensors(example, input_shape)
    
    dataset = dataset.map(_map_fn)
    dataset = dataset.filter(lambda x, y: tf.shape(x)[2] == 3)
    
    # Apply augmentation only for training
    if split == "train":
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

def build_model(input_shape, alpha):
    """Build MobileNet model."""
    input_shape_tuple = (input_shape.height, input_shape.width, input_shape.channels)
    
    # Load pretrained MobileNet backbone
    backbone = tf.keras.applications.MobileNet(
        input_shape=input_shape_tuple, 
        alpha=alpha, 
        include_top=False, 
        weights='imagenet'
    )
    
    classifier = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation=None)
    ])
    
    inputs = tf.keras.Input(input_shape_tuple)
    x = backbone(inputs)
    outputs = classifier(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

# Main training code
print("Loading model...")
input_shape = ImageShape(height=INPUT_HEIGHT, width=INPUT_WIDTH, channels=3)
model = build_model(input_shape, ALPHA)

print("Loading datasets...")
train_dataset = load_dataset(DATASET_NAME, input_shape, split="train").shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(DATASET_NAME, input_shape, split="val").batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Compiling model...")
# Cosine decay learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=LEARNING_RATE,
    first_decay_steps=10000,
    t_mul=1.0,
    m_mul=0.95,
    alpha=0.0
)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("Setting up callbacks...")
ckpt_path = f'models/{MODEL_NAME}/best_val.weights.h5'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, 
        save_weights_only=True, 
        monitor='val_loss',
        mode='min',
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
]

print(f"Training {MODEL_NAME}...")
history = model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    verbose=1
)

print(f"Saving model to exported_models/{MODEL_NAME}/saved_model...")
model.export(f'exported_models/{MODEL_NAME}/saved_model')
print("Finish training and saving model")
