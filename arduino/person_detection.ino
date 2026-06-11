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

#define PIR_PIN 12

// ARMED:          PIR inactive, light off — waiting for motion
// CHECKING:       PIR active, light off — retrying inference for a person
// LIGHT_ON:       person confirmed, light on — PIR active, no inference
// CONFIRMING_OFF: PIR cleared, light on — confirming no person before turning off
// COOLDOWN:       cooldown to give up finding a person — ignore PIR for a while (false-alarm prevention)
enum SystemState {
  STATE_ARMED,
  STATE_CHECKING,
  STATE_LIGHT_ON,
  STATE_CONFIRMING_OFF,
  STATE_COOLDOWN,
};

constexpr unsigned long kCheckIntervalMs = 3000;  // time between inference retries
constexpr int kMaxCheckAttempts = 7;              // give up after this many "no person" checks
constexpr unsigned long kCooldownMs = 10000;      // ignore PIR this long after giving up
constexpr int kOffConfirmCount = 2;               // consecutive "no person" checks before turning off

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

SystemState g_state = STATE_ARMED;
int g_check_attempts = 0;
int g_off_confirm_count = 0;
unsigned long g_last_check_ms = 0;
unsigned long g_cooldown_start_ms = 0;
unsigned long g_last_heartbeat_ms = 0;

const char* StateName(SystemState state) {
  switch (state) {
    case STATE_ARMED:          return "ARMED";
    case STATE_CHECKING:       return "CHECKING";
    case STATE_LIGHT_ON:       return "LIGHT_ON";
    case STATE_CONFIRMING_OFF: return "CONFIRMING_OFF";
    case STATE_COOLDOWN:       return "COOLDOWN";
  }
  return "UNKNOWN";
}

// Tuned for 128x128 grayscale MobileNet v1 alpha=0.25 with the tile-to-3
// Concatenation prefix. Log interpreter->arena_used_bytes() after AllocateTensors
constexpr int kTensorArenaSize = 180 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup() {
  Serial.begin(9600);
  unsigned long serial_wait_start = millis();
  while (!Serial && millis() - serial_wait_start < 3000) {}

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model schema version %d != supported %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Ops needed by the 128x128 grayscale MobileNet v1 with tile-to-3 prefix:
  //   Concatenation  - tile 1ch -> 3ch
  //   Conv2D / DepthwiseConv2D - MobileNet backbone
  //   Mean           - GlobalAveragePooling2D usually lowers to MEAN
  //   FullyConnected - final Dense(1)
  // AveragePool2D / Reshape / Softmax kept as defensive extras
  static tflite::MicroMutableOpResolver<8> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConcatenation();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMean();
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
  TF_LITE_REPORT_ERROR(error_reporter, "Arena: %u / %u bytes used",
                       (unsigned)interpreter->arena_used_bytes(),
                       (unsigned)kTensorArenaSize);

  input = interpreter->input(0);

  QuantizeInit(input->params.scale, input->params.zero_point);

  PIR_init(PIR_PIN);
  IR_init();
}

void loop() {
  bool pir_active = PIR_motionDetected();
  unsigned long now = millis();

  if (now - g_last_heartbeat_ms >= 1000) {
    g_last_heartbeat_ms = now;
    TF_LITE_REPORT_ERROR(error_reporter, "[heartbeat] PIR=%d state=%s",
                         pir_active ? 1 : 0, StateName(g_state));
  }

  // PIR state transitions
  switch (g_state) {
    case STATE_ARMED:
      if (pir_active) {
        g_state = STATE_CHECKING;
        g_check_attempts = 0;
        g_last_check_ms = now - kCheckIntervalMs;
        TF_LITE_REPORT_ERROR(error_reporter, "Motion detected — checking for person");
      }
      break;

    case STATE_CHECKING:
      if (!pir_active) {
        g_state = STATE_ARMED;
        TF_LITE_REPORT_ERROR(error_reporter, "Motion stopped before person confirmed — re-armed");
      }
      break;

    case STATE_LIGHT_ON:
      if (!pir_active) {
        g_state = STATE_CONFIRMING_OFF;
        g_off_confirm_count = 0;
        g_last_check_ms = now - kCheckIntervalMs; 
        TF_LITE_REPORT_ERROR(error_reporter, "Motion stopped — confirming no person");
      }
      break;

    case STATE_CONFIRMING_OFF:
      if (pir_active) {
        g_state = STATE_LIGHT_ON;
        TF_LITE_REPORT_ERROR(error_reporter, "Motion resumed — staying on");
      }
      break;

    case STATE_COOLDOWN:
      if (now - g_cooldown_start_ms >= kCooldownMs) {
        g_state = STATE_ARMED;
        TF_LITE_REPORT_ERROR(error_reporter, "Cooldown finished — re-armed");
      }
      break;
  }

  // Only STATE_CHECKING and STATE_CONFIRMING_OFF run inference, on a fixed interval
  bool should_infer = (g_state == STATE_CHECKING || g_state == STATE_CONFIRMING_OFF) &&
                      (now - g_last_check_ms >= kCheckIntervalMs);
  if (!should_infer) return;
  g_last_check_ms = now;

  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
    return;
  }

  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    return;
  }

  // Single-logit BCE output: int8 value > output zero_point  :  sigmoid > 0.5  :  person
  TfLiteTensor* output = interpreter->output(0);
  bool is_person = output->data.int8[0] > output->params.zero_point;
  // Adapt to RespondToDetection's two-score interface
  int8_t person_score    = is_person ? 1 : 0;
  int8_t no_person_score = is_person ? 0 : 1;

  RespondToDetection(error_reporter, person_score, no_person_score);

  if (g_state == STATE_CHECKING) {
    g_check_attempts++;
    TF_LITE_REPORT_ERROR(error_reporter, "Check %d/%d: %s", g_check_attempts,
                         kMaxCheckAttempts, is_person ? "person" : "no person");
    if (is_person) {
      IR_sendOn();
      g_state = STATE_LIGHT_ON;
      TF_LITE_REPORT_ERROR(error_reporter, "Person confirmed — light ON");
    } else if (g_check_attempts >= kMaxCheckAttempts) {
      g_state = STATE_COOLDOWN;
      g_cooldown_start_ms = now;
      TF_LITE_REPORT_ERROR(error_reporter,
                           "No person after %d checks — cooldown %lus",
                           kMaxCheckAttempts, kCooldownMs / 1000);
    }
  } else if (g_state == STATE_CONFIRMING_OFF) {
    if (is_person) {
      g_off_confirm_count = 0;
      TF_LITE_REPORT_ERROR(error_reporter, "Still a person — staying on");
    } else {
      g_off_confirm_count++;
      TF_LITE_REPORT_ERROR(error_reporter, "No person (%d/%d)", g_off_confirm_count,
                           kOffConfirmCount);
      if (g_off_confirm_count >= kOffConfirmCount) {
        IR_sendOff();
        g_state = STATE_ARMED;
        TF_LITE_REPORT_ERROR(error_reporter, "Confirmed empty — light OFF");
      }
    }
  }
}
