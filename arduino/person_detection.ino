/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "ir_sender.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "pir.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define PIR_PIN 2

// ARMED:    PIR inactive, light off — waiting for motion.
// LIGHT_ON: person detected, light on — waiting for PIR to clear.
enum LightState { STATE_ARMED, STATE_LIGHT_ON };

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

LightState g_light_state = STATE_ARMED;

constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model schema version %d != supported %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);

  PIR_init(PIR_PIN);
  IR_init();
}

void loop() {
  bool pir_active = PIR_motionDetected();

  // Only run inference when a state transition may be needed:
  //   ARMED + motion    → try to confirm a person (turn light on)
  //   LIGHT_ON + no motion → verify no person remains (turn light off)
  bool should_infer = (g_light_state == STATE_ARMED   &&  pir_active) ||
                      (g_light_state == STATE_LIGHT_ON && !pir_active);

  if (!should_infer) return;

  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
    return;
  }

  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    return;
  }

  TfLiteTensor* output = interpreter->output(0);
  int8_t person_score    = output->data.uint8[kPersonIndex];
  int8_t no_person_score = output->data.uint8[kNotAPersonIndex];

  RespondToDetection(error_reporter, person_score, no_person_score);

  if (g_light_state == STATE_ARMED && person_score > no_person_score) {
    IR_sendOn();
    g_light_state = STATE_LIGHT_ON;
    TF_LITE_REPORT_ERROR(error_reporter, "Person detected — light ON");
  } else if (g_light_state == STATE_LIGHT_ON && no_person_score >= person_score) {
    IR_sendOff();
    g_light_state = STATE_ARMED;
    TF_LITE_REPORT_ERROR(error_reporter, "No person — light OFF");
  }
}
