/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  ==============================================================================*/

#include "image_provider.h"

#ifndef ARDUINO_EXCLUDE_CODE

#include "Arduino.h"
#include <Arduino_OV767X.h>
#include <math.h>

// Maps each possible 8-bit camera pixel value to the int8 the model expects
// Default is the legacy "pixel - 128" mapping. Call QuantizeInit() from setup()
static int8_t quant_lut[256];
static bool   quant_lut_ready = false;

static void _installDefaultLUT() {
  for (int i = 0; i < 256; ++i) quant_lut[i] = (int8_t)(i - 128);
  quant_lut_ready = true;
}

void QuantizeInit(float scale, int zero_point) {
  // Training pipeline: float_val = pixel/127.5 - 1.0  (range [-1, +1])
  // TFLite int8:      q = round(float_val / scale) + zero_point
  for (int i = 0; i < 256; ++i) {
    float f = (float)i / 127.5f - 1.0f;
    int   q = (int)roundf(f / scale) + zero_point;
    if (q < -128) q = -128;
    if (q >  127) q =  127;
    quant_lut[i] = (int8_t)q;
  }
  quant_lut_ready = true;
}

// Source frame is QCIF grayscale (176x144)
//   Capture QCIF
//   Center-crop horizontally to 144x144 (square; drops 16px each side)
//   Nearest-neighbor downsample 144x144 -> image_width x image_height
//   Map each pixel through quant_lut to int8
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data) {
  constexpr int kSrcW    = 176;
  constexpr int kSrcH    = 144;
  constexpr int kCropSide = 144;
  constexpr int kCropX0  = (kSrcW - kCropSide) / 2;  // = 16

  static byte raw[kSrcW * kSrcH];
  static bool camera_ok = false;

  if (!camera_ok) {
    if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize camera!");
      return kTfLiteError;
    }
    camera_ok = true;
  }
  if (!quant_lut_ready) _installDefaultLUT();

  Camera.readFrame(raw);

  int idx = 0;
  for (int oy = 0; oy < image_height; ++oy) {
    const int sy = (oy * kCropSide) / image_height;
    const byte* row = raw + sy * kSrcW;
    for (int ox = 0; ox < image_width; ++ox) {
      const int sx = kCropX0 + (ox * kCropSide) / image_width;
      image_data[idx++] = quant_lut[row[sx]];
    }
  }
  return kTfLiteOk;
}

#endif  // ARDUINO_EXCLUDE_CODE
