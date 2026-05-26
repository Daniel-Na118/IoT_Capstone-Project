bash download_coco.sh data/raw/coco2014 2014

Train Model with: python train.py

Normal Model -> TF Lite: python convert_tf_lite.py --model-name MODEL_NAME
TF Lite -> TF Lite Micro: bash tf_micro.sh MODEL_NAME



To edit model hyperparameters, set the corresponding flags in train_vww_model.py:

python src/train_vww_model.py --help

```
usage: train_vww_model.py [-h] [--dataset DATASET]
                          [--input-height INPUT_HEIGHT]
                          [--input-width INPUT_WIDTH]
                          [--model-prefix MODEL_PREFIX] [--alpha ALPHA]
                          [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                          [--verbose VERBOSE] [--learning-rate LEARNING_RATE]
                          [--decay-rate DECAY_RATE] [--deploy]
```

Train a model to predict whether an image contains a person

```
optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of dataset. Subdirectory of data/vww_tfrecord
  --input-height INPUT_HEIGHT
                        Height of input
  --input-width INPUT_WIDTH
                        Width of input
  --model-prefix MODEL_PREFIX
                        Prefix to be used in naming the model
  --alpha ALPHA         Depth multiplier. The smaller it is, the smaller the
                        resulting model.
  --epochs EPOCHS       Training procedure runs through the whole dataset once
                        per epoch.
  --batch-size BATCH_SIZE
                        Number of examples to process concurrently
  --verbose VERBOSE     Printing verbosity of Tensorflow model.fit()Set
                        --verbose=1 for per-batch progress bar, --verbose=2
                        for per-epoch
  --learning-rate LEARNING_RATE
                        Initial learning rate of SGD training
  --decay-rate DECAY_RATE
                        Number of steps to decay learning rate after
  --deploy              Set flag to skip training and simply export the
                        trained model
```


Custom deployment
To edit conversion hyperparameters, set the corresponding flags in convert_tf_model_to_tf_lite.py:

```
usage: convert_tf_model_to_tf_lite.py [-h] [--model-name MODEL_NAME]
                                      [--dataset DATASET]
                                      [--num-samples NUM_SAMPLES]
                                      [--input-height INPUT_HEIGHT]
                                      [--input-width INPUT_WIDTH]
```

Convert a TF SavedModel to a TFLite model

```
optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Name of the model. See tools/train_model.sh for
                        semantics of model name
  --dataset DATASET     Name of the TFRecord dataset that should be used for
                        quantization
  --num-samples NUM_SAMPLES
                        Number of samples to calibrate on
  --input-height INPUT_HEIGHT
  --input-width INPUT_WIDTH
```