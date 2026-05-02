"""
Training script using FiftyOne to import and manage the COCO dataset.
Automatically downloads and handles the COCO dataset, converts to person detection task.
"""
import tensorflow as tf
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
from pathlib import Path

tf.get_logger().setLevel('ERROR')

# Training configuration
COCO_SPLIT = "train"  # 'train' or 'validation'
INPUT_HEIGHT = 96
INPUT_WIDTH = 96
ALPHA = 0.25  # MobileNet depth multiplier
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MODEL_NAME = f"vww_mobilenet_fiftyone_{ALPHA}_{INPUT_HEIGHT}_{INPUT_WIDTH}"

def load_coco_dataset(split="train", max_samples=None):
    """
    Load COCO dataset using FiftyOne.
    
    Args:
        split: 'train' or 'validation'
        max_samples: Limit number of samples (for testing)
    
    Returns:
        FiftyOne Dataset
    """
    print(f"Loading COCO {split} dataset...")
    
    try:
        # Try to load existing dataset
        dataset = fo.load_dataset(f"coco_{split}_vww")
        print(f"Loaded existing dataset: coco_{split}_vww")
    except:
        print(f"Downloading COCO {split} dataset...")
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=split,
            label_types=["detections"],
            classes=["person"],
            max_samples=max_samples
        )
        
        # Create binary classification labels (person present: 1, no person: 0)
        for sample in dataset:
            detections = sample.detections
            if detections and len(detections.detections) > 0:
                sample["has_person"] = 1
            else:
                sample["has_person"] = 0
            sample.save()
        
        # Persist the dataset
        dataset.name = f"coco_{split}_vww"
        dataset.save()
        print(f"Created and saved dataset: coco_{split}_vww")
    
    return dataset

def create_tf_dataset(fiftyone_dataset, split="train"):
    """
    Convert FiftyOne dataset to TensorFlow dataset.
    
    Args:
        fiftyone_dataset: FiftyOne Dataset
        split: 'train' or 'validation'
    
    Returns:
        tf.data.Dataset with (image, label) pairs
    """
    images = []
    labels = []
    
    print(f"Processing {len(fiftyone_dataset)} samples...")
    
    for idx, sample in enumerate(fiftyone_dataset):
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(fiftyone_dataset)}")
        
        # Load image
        img = tf.io.decode_image(tf.io.read_file(sample.filepath), channels=3)
        img = tf.image.resize(img, size=(INPUT_HEIGHT, INPUT_WIDTH))
        img = tf.cast(img, tf.float32)
        
        # Get label (person present: 1, no person: 0)
        label = sample["has_person"]
        
        images.append(img)
        labels.append(float(label))
    
    # Convert to tensors
    images = tf.stack(images)
    labels = tf.constant(labels, dtype=tf.float32)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset

def augment_image(image, label):
    """Apply data augmentation."""
    # Normalize to [-1, 1]
    image = image / 127.5 - 1.0
    
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    
    image = tf.image.random_brightness(image, 0.2)
    
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label

def build_model(input_height, input_width, alpha):
    """Build MobileNet model."""
    input_shape = (input_height, input_width, 3)
    
    # Load pretrained MobileNet backbone
    backbone = tf.keras.applications.MobileNet(
        input_shape=input_shape, 
        alpha=alpha, 
        include_top=False, 
        weights='imagenet'
    )
    
    # Add classification head
    classifier = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation=None)
    ])
    
    # Connect backbone to classifier
    inputs = tf.keras.Input(input_shape)
    x = backbone(inputs)
    outputs = classifier(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

# Main training code
print("=" * 60)
print("Training with FiftyOne COCO Dataset")
print("=" * 60)

# Load datasets
train_dataset_fo = load_coco_dataset(split="train")
val_dataset_fo = load_coco_dataset(split="validation")

# Convert to TensorFlow datasets
print("\nConverting training dataset to TensorFlow format...")
train_data = create_tf_dataset(train_dataset_fo, split="train")
print("✓ Training data ready")

print("\nConverting validation dataset to TensorFlow format...")
val_data = create_tf_dataset(val_dataset_fo, split="validation")
print("✓ Validation data ready")

# Prepare datasets for training
train_dataset = train_data.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE).shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build and compile model
print("\nBuilding model...")
model = build_model(INPUT_HEIGHT, INPUT_WIDTH, ALPHA)

print("Compiling model...")
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

# Setup callbacks
print("Setting up callbacks...")
ckpt_dir = Path(f'models/{MODEL_NAME}')
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_dir / 'best_val.ckpt'

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(ckpt_path),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# Train model
print(f"\nTraining {MODEL_NAME}...")
print("=" * 60)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save model
print("\n" + "=" * 60)
export_dir = Path(f'exported_models/{MODEL_NAME}/saved_model')
export_dir.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving model to {export_dir}...")
model.save(str(export_dir))
print("Training complete")
