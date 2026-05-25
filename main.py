import tensorflow as tf
from collections import namedtuple

# 1. Redefine the architecture (Standalone)
ImageShape = namedtuple('ImageShape', 'height width channels')

def build_model(input_shape, alpha):
    backbone = tf.keras.applications.MobileNet(
        input_shape=(input_shape.height, input_shape.width, input_shape.channels), 
        alpha=alpha, 
        include_top=False, 
        weights='imagenet'
    )
    classifier = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation=None)
    ])
    inputs = tf.keras.Input((input_shape.height, input_shape.width, input_shape.channels))
    x = backbone(inputs)
    outputs = classifier(x)
    return tf.keras.Model(inputs, outputs)

# 2. Setup paths and params
INPUT_HEIGHT = 96
INPUT_WIDTH = 96
ALPHA = 0.25
MODEL_NAME = f"vww_mobilenet_simple_{ALPHA}_{INPUT_HEIGHT}_{INPUT_WIDTH}"
WEIGHTS_PATH = f'models/{MODEL_NAME}/best_val.weights.h5'
EXPORT_PATH = f'exported_models/{MODEL_NAME}/saved_model'

# 3. Execution
print("Building model...")
input_shape = ImageShape(height=INPUT_HEIGHT, width=INPUT_WIDTH, channels=3)
model = build_model(input_shape, ALPHA)

print(f"Loading weights from {WEIGHTS_PATH}...")
model.load_weights(WEIGHTS_PATH)

print(f"Exporting to {EXPORT_PATH}...")
model.export(EXPORT_PATH)
print("Done! You can now convert this to TFLite.")