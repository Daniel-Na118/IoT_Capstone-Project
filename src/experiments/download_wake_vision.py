"""
Usage:
    python download_wake_vision.py
    python download_wake_vision.py --train-samples 120000 --val-samples 20000

Requirements
    pip install datasets huggingface_hub

Output:
    data/vww_tfrecord/wake_vision/wake_vision_train.record-XXXXX-of-00010
    data/vww_tfrecord/wake_vision/wake_vision_val.record-00000-of-00001
"""
import argparse
import io
import math
import pathlib
import sys

import tensorflow as tf

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit(
        'Missing dependency. Run:\n'
        '  pip install datasets huggingface_hub'
    )

DATASET_ID = 'Harvard-Edge/Wake-Vision'
OUTPUT_DIR = ROOT_DIR / 'data' / 'vww_tfrecord' / 'wake_vision'


def _make_example(image_bytes: bytes, label: int) -> bytes:
    feature = {
        'image/encoded': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes])),
        'image/class': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[label])),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feature)).SerializeToString()


def _image_to_jpeg_bytes(pil_image) -> bytes:
    buf = io.BytesIO()
    pil_image.convert('RGB').save(buf, format='JPEG', quality=90)
    return buf.getvalue()


# Wake Vision split names on HuggingFace : local file prefix used by train_experiment.py -- fixed to match download names
HF_SPLIT_MAP = {
    'train': 'train_quality',   # curated high-quality subset
    'val': 'validation',
}


def write_split(split: str, max_samples: int, num_shards: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hf_split = HF_SPLIT_MAP[split]
    print(f'\nLoading Wake Vision {hf_split} split (streaming, max {max_samples:,} samples)...')
    ds = load_dataset(DATASET_ID, split=hf_split, streaming=True)

    writers = []
    paths = []
    for shard in range(num_shards):
        p = OUTPUT_DIR / f'wake_vision_{split}.record-{shard:05d}-of-{num_shards:05d}'
        paths.append(p)
        writers.append(tf.io.TFRecordWriter(str(p)))

    counts = {'person': 0, 'no_person': 0, 'skipped': 0}
    total = 0

    for example in ds:
        if total >= max_samples:
            break

        label = int(example.get('person', example.get('label', -1)))
        if label not in (0, 1):
            counts['skipped'] += 1
            continue

        try:
            img_bytes = _image_to_jpeg_bytes(example['image'])
        except Exception:
            counts['skipped'] += 1
            continue

        shard_idx = total % num_shards
        writers[shard_idx].write(_make_example(img_bytes, label))
        total += 1
        counts['person' if label == 1 else 'no_person'] += 1

        if total % 5000 == 0:
            print(f'  {total:>7,} / {max_samples:,}  '
                  f'person={counts["person"]:,}  no_person={counts["no_person"]:,}')

    for w in writers:
        w.close()

    print(f'\n{hf_split} → wake_vision_{split}: {total:,} examples written to {num_shards} shards')
    print(f'  person={counts["person"]:,}  no_person={counts["no_person"]:,}  '
          f'skipped={counts["skipped"]:,}')
    for p in paths:
        size_mb = p.stat().st_size / 1024 / 1024
        print(f'  {p.name}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Wake Vision → TFRecords')
    parser.add_argument('--train-samples', type=int, default=120000,
                        help='Max training examples to download (default: 120000)')
    parser.add_argument('--val-samples', type=int, default=20000,
                        help='Max validation examples to download (default: 20000)')
    parser.add_argument('--train-shards', type=int, default=10,
                        help='Number of train TFRecord shards (default: 10)')
    args = parser.parse_args()

    write_split('train', args.train_samples, args.train_shards)
    write_split('val', args.val_samples, num_shards=1)

    print('\nWake Vision TFRecords ready.')
    print('Train with:  python run_experiments.py --only scratch_v2 transfer_v2')
