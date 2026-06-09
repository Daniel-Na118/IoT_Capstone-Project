import tensorflow as tf
import argparse
import numpy as np
import pathlib
import sys

from PIL import Image
from collections import namedtuple

# Reuse the training data pipeline so calibration sees images preprocessed
# exactly like training does (decode -> resize -> normalize to [-1, +1]).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'experiments'))
from train_experiment import ModelConfig, TFRECORD_ROOTS, load_dataset

parser = argparse.ArgumentParser(description="Convert a TF SavedModel to a TFLite model")
parser.add_argument("--model-name",
                    help="Name of the model under exported_models/<name>/saved_model")
parser.add_argument("--dataset", default="coco2014",
                    help="Folder under data/raw/<name>/Images for calibration, "
                         "a TFRecord dataset name (coco2017, wake_vision), or 'fake' "
                         "for zero-input calibration.")
parser.add_argument("--num-samples", type=int, default=100,
                    help="Number of samples to calibrate on")
parser.add_argument("--input-height", type=int, default=96)
parser.add_argument("--input-width",  type=int, default=96)
parser.add_argument("--input-channels", type=int, default=3, choices=[1, 3],
                    help="1 for grayscale models, 3 for RGB")

ImgShape = namedtuple('ImageShape', 'height width channels')


def _normalize(arr: np.ndarray) -> np.ndarray:
    # Match training preprocessing: float32 in [-1, +1]
    return arr.astype(np.float32) / 127.5 - 1.0


def fake_data_gen(num_samples, input_shape):
    def representative_dataset_gen():
        for _ in range(num_samples):
            # Zeros = neutral mid-range input in the [-1, +1] domain
            yield [np.zeros((1, input_shape.height, input_shape.width, input_shape.channels),
                            dtype=np.float32)]
    return representative_dataset_gen


def tfrecord_data_gen(dataset_name, num_samples, input_shape):
    """Calibration generator that pulls from the existing val TFRecords —
    avoids re-downloading raw images and matches training preprocessing exactly."""
    cfg = ModelConfig(
        name='_calibration', architecture='mobilenetv1', alpha=0.25,
        input_height=input_shape.height, input_width=input_shape.width,
        input_channels=input_shape.channels,
        epochs=0, batch_size=1, learning_rate=0,
        augmentation='basic', loss='bce', dataset=dataset_name,
    )
    ds = load_dataset(cfg, 'val').take(num_samples)

    def representative_dataset_gen():
        for image, _ in ds:
            yield [image[tf.newaxis, ...].numpy()]

    return representative_dataset_gen


def make_data_gen(dataset_name, num_samples, input_shape):
    """Calibration generator. Reads images from data/raw/<dataset_name>/Images,
    a TFRecord val split, or yields zeros for 'fake'."""
    if dataset_name == "fake":
        return fake_data_gen(num_samples, input_shape)

    if dataset_name in TFRECORD_ROOTS:
        return tfrecord_data_gen(dataset_name, num_samples, input_shape)

    imgdir = pathlib.Path('data/raw') / dataset_name / 'Images'
    pil_mode = 'L' if input_shape.channels == 1 else 'RGB'

    def representative_dataset_gen():
        count = 0
        for filename in imgdir.iterdir():
            if filename.suffix.lower() not in ['.jpeg', '.jpg', '.png']:
                continue
            image = Image.open(str(filename.resolve())).convert(pil_mode)
            # PIL.Image.resize takes (width, height) ;; order vs TF
            image = image.resize((input_shape.width, input_shape.height))
            arr = np.array(image)
            if input_shape.channels == 1:
                arr = arr[..., np.newaxis]  # (H, W) -> (H, W, 1)
            arr = _normalize(arr)[np.newaxis, ...]  # add batch dim
            yield [arr]
            count += 1
            if count >= num_samples:
                break

    return representative_dataset_gen


args = parser.parse_args()
input_shape = ImgShape(
    height=args.input_height,
    width=args.input_width,
    channels=args.input_channels,
)
model_savedir = f'exported_models/{args.model_name}/saved_model'

converter = tf.lite.TFLiteConverter.from_saved_model(model_savedir, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = make_data_gen(args.dataset, args.num_samples, input_shape)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
quantized_model = converter.convert()

out_path = pathlib.Path('exported_models') / args.model_name / 'model.tflite'
out_path.write_bytes(quantized_model)
print(f"Wrote {out_path}  ({len(quantized_model)} bytes)")
